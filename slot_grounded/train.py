from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from torchvision.ops import roi_align

from slot_grounded.model import SlotGroundedVLM
from utils.data import EncoderDecoderCollate, encoder_decoder_process


DATASET_NAME = "thirdExec/synthetic-seismic-vlm"
SAVE_DIR = Path("slot_grounded/checkpoints")
BATCH_SIZE = 1
MAX_EPOCHS = 300
LEARNING_RATE = 1e-5
NUM_WORKERS = 4
MASK_LOSS_WEIGHT = 1.0
SLOT_LOSS_WEIGHT = 1.0
FAST_DEV_RUN = False
GRADIENT_CLIP_VAL = 1.0


def build_dataset():
    raw = load_dataset(DATASET_NAME)["train"]
    rows = []
    for example in raw:
        processed = encoder_decoder_process(example)
        if isinstance(processed, list):
            rows.extend(processed)
        else:
            rows.append(processed)

    dataset = Dataset.from_list(rows)
    split = dataset.train_test_split(test_size=0.2, seed=42)
    holdout = split["test"].train_test_split(test_size=0.5, seed=42)
    return split["train"], holdout["train"], holdout["test"]


class SlotGroundedTrainer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = SlotGroundedVLM()

    @property
    def tokenizer(self):
        return self.model.tokenizer

    def forward(self, batch):
        heights = [size[0] for size in batch["sizes"]]
        widths = [size[1] for size in batch["sizes"]]
        return self.model(
            pixel_values=batch["pixel_values"],
            tiles=batch["tiles"],
            boxes=batch["boxes"],
            class_ids=batch["label"],
            heights=heights,
            widths=widths,
            input_ids=batch["input_ids"].to(self.device),
            attention_mask=batch["attention_mask"].to(self.device),
            labels=batch["labels"].to(self.device),
            row_image_counts=batch["row_image_counts"],
        )

    def get_slot_hidden(self, outputs, batch, token_name):
        token_id = batch["slot_token_ids"][token_name]
        input_ids = batch["input_ids"].to(self.device)
        assistant_mask = batch["labels"].to(self.device) != -100
        batch_idx, token_idx = ((input_ids == token_id) & assistant_mask).nonzero(as_tuple=True)
        if batch_idx.numel() == 0:
            return None, None
        hidden = outputs["decoder_outputs"].hidden_states[-1]
        return hidden[batch_idx, token_idx + outputs["visual_len"]], batch_idx

    def compute_slot_loss(self, outputs, batch):
        zero = outputs["loss"].new_zeros(())
        losses = []
        logs = {
            "slot_class_loss": zero.detach(),
            "slot_bbox_loss": zero.detach(),
            "slot_center_loss": zero.detach(),
            "slot_reg_loss": zero.detach(),
        }

        obj_hidden, _ = self.get_slot_hidden(outputs, batch, "obj")
        if obj_hidden is not None:
            count = min(obj_hidden.shape[0], batch["class_slot_targets"].numel())
            if count:
                logits = self.model.object_head(
                    obj_hidden[:count].to(dtype=self.model.object_head.weight.dtype)
                )
                target = batch["class_slot_targets"][:count].to(self.device)
                loss = F.cross_entropy(logits, target)
                losses.append(loss)
                logs["slot_class_loss"] = loss.detach()

        bbox_hidden, _ = self.get_slot_hidden(outputs, batch, "bbox")
        if bbox_hidden is not None:
            count = min(bbox_hidden.shape[0], batch["bbox_slot_targets"].shape[0])
            if count:
                pred = self.model.slot_bbox_to_xyxy(bbox_hidden[:count])
                target = batch["bbox_slot_targets"][:count].to(self.device, dtype=pred.dtype)
                loss = F.smooth_l1_loss(pred, target)
                losses.append(loss)
                logs["slot_bbox_loss"] = loss.detach()

        center_hidden, _ = self.get_slot_hidden(outputs, batch, "center")
        if center_hidden is not None:
            count = min(center_hidden.shape[0], batch["center_slot_targets"].shape[0])
            if count:
                pred = self.model.slot_center_to_xy(center_hidden[:count])
                target = batch["center_slot_targets"][:count].to(self.device, dtype=pred.dtype)
                loss = F.smooth_l1_loss(pred, target)
                losses.append(loss)
                logs["slot_center_loss"] = loss.detach()

        reg_hidden, _ = self.get_slot_hidden(outputs, batch, "reg")
        if reg_hidden is not None:
            count = min(reg_hidden.shape[0], batch["numeric_targets"].numel())
            if count:
                hidden = reg_hidden[:count].to(dtype=self.model.numeric_head[0].weight.dtype)
                pred = self.model.numeric_head(hidden).squeeze(-1)
                target = batch["numeric_targets"][:count].to(self.device, dtype=pred.dtype)
                loss = F.smooth_l1_loss(pred, target)
                losses.append(loss)
                logs["slot_reg_loss"] = loss.detach()

        if not losses:
            return zero, logs
        return torch.stack(losses).mean(), logs

    def crop_mask_target(self, mask_value, box, output_size, device):
        mask_value = mask_value.to(device)
        boxes = torch.as_tensor(box, dtype=torch.float32, device=device)
        if boxes.ndim == 1:
            boxes = boxes.unsqueeze(0)
        batch_index = torch.zeros((boxes.shape[0], 1), dtype=boxes.dtype, device=device)
        rois = torch.cat([batch_index, boxes], dim=-1)
        return roi_align(
            input=mask_value,
            boxes=rois,
            output_size=output_size,
            spatial_scale=1,
            sampling_ratio=2,
        ).clamp(0, 1)

    def compute_mask_loss(self, outputs, batch):
        seg_hidden, _ = self.get_slot_hidden(outputs, batch, "seg")
        if seg_hidden is None:
            zero = outputs["loss"].new_zeros(())
            return zero, {"mask_bce_loss": zero.detach(), "mask_dice_loss": zero.detach()}

        spatial = torch.cat(
            [output["region_spatial_features"] for output in outputs["visual_outputs"]],
            dim=0,
        )
        count = min(seg_hidden.shape[0], spatial.shape[0], len(batch["boxes"]))
        if count == 0:
            zero = outputs["loss"].new_zeros(())
            return zero, {"mask_bce_loss": zero.detach(), "mask_dice_loss": zero.detach()}

        logits = self.model.mask_decoder(seg_hidden[:count], spatial[:count])
        targets = []
        for idx in range(count):
            targets.append(
                self.crop_mask_target(
                    batch["mask_values"][idx],
                    batch["boxes"][idx],
                    logits.shape[-2:],
                    logits.device,
                )
            )
        target = torch.cat(targets, dim=0).to(dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, target)
        probs = logits.sigmoid()
        intersection = (probs * target).sum(dim=(-1, -2, -3))
        denom = probs.sum(dim=(-1, -2, -3)) + target.sum(dim=(-1, -2, -3))
        dice = 1.0 - ((2.0 * intersection + 1.0) / (denom + 1.0)).mean()
        return bce + dice, {
            "mask_bce_loss": bce.detach(),
            "mask_dice_loss": dice.detach(),
        }

    def shared_step(self, batch, stage):
        outputs = self(batch)
        text_loss = outputs["loss"]
        slot_loss, slot_logs = self.compute_slot_loss(outputs, batch)
        mask_loss, mask_logs = self.compute_mask_loss(outputs, batch)
        loss = text_loss + SLOT_LOSS_WEIGHT * slot_loss + MASK_LOSS_WEIGHT * mask_loss

        batch_size = batch["input_ids"].shape[0]
        self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}/text_loss", text_loss.detach(), batch_size=batch_size)
        self.log(f"{stage}/slot_loss", slot_loss.detach(), batch_size=batch_size)
        self.log(f"{stage}/mask_loss", mask_loss.detach(), batch_size=batch_size)
        for key, value in slot_logs.items():
            self.log(f"{stage}/{key}", value, batch_size=batch_size)
        for key, value in mask_logs.items():
            self.log(f"{stage}/{key}", value, batch_size=batch_size)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def configure_optimizers(self):
        params = [param for param in self.parameters() if param.requires_grad]
        return torch.optim.AdamW(params, lr=LEARNING_RATE)


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    model = SlotGroundedTrainer()
    collator = EncoderDecoderCollate(model.tokenizer)
    train_dataset, eval_dataset, _ = build_dataset()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )

    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=SAVE_DIR,
        filename="slot-grounded-{epoch:02d}-{val_loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        default_root_dir=SAVE_DIR,
        callbacks=[checkpoint],
        log_every_n_steps=10,
        accelerator="auto",
        devices=1,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        fast_dev_run=FAST_DEV_RUN,
    )
    ckpt_path = SAVE_DIR / "last.ckpt"
    trainer.fit(
        model,
        train_loader,
        eval_loader,
        ckpt_path=ckpt_path.as_posix() if ckpt_path.exists() else None,
    )


if __name__ == "__main__":
    main()
