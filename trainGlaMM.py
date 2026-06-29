from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from torchvision.ops import roi_align

from utils.GLaMM import SeismicGLaMM
from utils.data import BBOX_SLOT, OBJ_SLOT, REG_SLOT, EncoderDecoderCollate, encoder_decoder_process
from utils.train_encoders import box_coverage, xyxy_abs_to_norm


DATASET_NAME = "thirdExec/synthetic-seismic-vlm"
SAVE_DIR = Path("GLaMM")
BATCH_SIZE = 1
MAX_EPOCHS = 600
LEARNING_RATE = 1e-5
NUM_WORKERS = 0
POSITIVE_COVERAGE_THRESHOLD = 0.25
GROUNDING_LOSS_WEIGHT = 1.0
MASK_LOSS_WEIGHT = 1.0
SLOT_LOSS_WEIGHT = 1.0


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


class GLaMMTrainer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = SeismicGLaMM()
        hidden_size = self.model.lang_decoder.model.get_input_embeddings().embedding_dim
        self.slot_class_head = torch.nn.Linear(hidden_size, 7)
        self.slot_bbox_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, 4),
        )
        self.slot_reg_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, 1),
        )

    @property
    def tokenizer(self):
        return self.model.tokenizer

    def forward(self, batch):
        heights = [size[0] for size in batch["sizes"]]
        widths = [size[1] for size in batch["sizes"]]
        return self.model(
            pixel_values=batch["pixel_values"],
            tiles=batch["tiles"],
            bbox=batch["boxes"],
            class_ids=batch["label"],
            H=heights,
            W=widths,
            input_ids=batch["input_ids"].to(self.device),
            attention_mask=batch["attention_mask"].to(self.device),
            labels=batch["labels"].to(self.device),
        )

    def compute_sample_grounding_loss(self, output, box, label, height, width):
        proposal = output["proposal"]
        tile_boxes = output["tile_bbox_abs"]
        objectness_logits = proposal["objectness_logits"]
        class_logits = proposal["class_logits"]
        pred_boxes = proposal["boxes"]

        coverage = box_coverage(box, tile_boxes)
        positive = coverage >= POSITIVE_COVERAGE_THRESHOLD
        if not positive.any():
            positive[coverage.argmax()] = True

        objectness_target = positive.to(objectness_logits.dtype)
        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness_logits,
            objectness_target,
        )

        gt_box_norm = xyxy_abs_to_norm(
            box,
            width=width,
            height=height,
            device=pred_boxes.device,
            dtype=pred_boxes.dtype,
        )
        bbox_target = gt_box_norm.unsqueeze(0).expand_as(pred_boxes)
        bbox_loss = F.smooth_l1_loss(
            pred_boxes[positive],
            bbox_target[positive],
        )

        label = label.to(class_logits.device)
        class_loss = F.cross_entropy(
            class_logits[positive],
            label.expand(positive.sum()),
        )

        total = objectness_loss + class_loss + bbox_loss
        return {
            "loss": total,
            "objectness_loss": objectness_loss.detach(),
            "class_loss": class_loss.detach(),
            "bbox_loss": bbox_loss.detach(),
            "positive_tiles": positive.sum().detach().float(),
        }

    def compute_grounding_loss(self, outputs, batch):
        losses = []
        logs = {
            "objectness_loss": [],
            "class_loss": [],
            "bbox_loss": [],
            "positive_tiles": [],
        }

        for idx, output in enumerate(outputs):
            height, width = batch["sizes"][idx]
            sample_losses = self.compute_sample_grounding_loss(
                output=output,
                box=batch["boxes"][idx],
                label=batch["label"][idx],
                height=height,
                width=width,
            )
            losses.append(sample_losses["loss"])
            for key in logs:
                logs[key].append(sample_losses[key])

        grounding_loss = torch.stack(losses).mean()
        log_values = {
            key: torch.stack(values).mean()
            for key, values in logs.items()
        }
        return grounding_loss, log_values

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
        losses = []
        bce_losses = []
        dice_losses = []

        for idx, output in enumerate(outputs):
            logits = output["mask_logits"]
            target = self.crop_mask_target(
                mask_value=batch["mask_values"][idx],
                box=batch["boxes"][idx],
                output_size=logits.shape[-2:],
                device=logits.device,
            )
            bce = F.binary_cross_entropy_with_logits(logits, target)
            probs = logits.sigmoid()
            intersection = (probs * target).sum(dim=(-1, -2, -3))
            denom = probs.sum(dim=(-1, -2, -3)) + target.sum(dim=(-1, -2, -3))
            dice = 1.0 - ((2.0 * intersection + 1.0) / (denom + 1.0)).mean()
            losses.append(bce + dice)
            bce_losses.append(bce.detach())
            dice_losses.append(dice.detach())

        return torch.stack(losses).mean(), {
            "mask_bce_loss": torch.stack(bce_losses).mean(),
            "mask_dice_loss": torch.stack(dice_losses).mean(),
        }

    def slot_bbox_to_xyxy(self, hidden):
        hidden = hidden.to(dtype=self.slot_bbox_head[0].weight.dtype)
        pred = self.slot_bbox_head(hidden).sigmoid()
        cx = pred[..., 0]
        cy = pred[..., 1]
        w = pred[..., 2]
        h = pred[..., 3]
        x1 = (cx - 0.5 * w).clamp(0, 1)
        y1 = (cy - 0.5 * h).clamp(0, 1)
        x2 = (cx + 0.5 * w).clamp(0, 1)
        y2 = (cy + 0.5 * h).clamp(0, 1)
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def get_slot_hidden(self, decoder_outputs, batch, token_name):
        token_id = batch["slot_token_ids"][token_name]
        input_ids = batch["input_ids"].to(self.device)
        batch_idx, token_idx = (input_ids == token_id).nonzero(as_tuple=True)
        if batch_idx.numel() == 0:
            return None, None

        hidden = decoder_outputs.hidden_states[-1]
        visual_len = hidden.shape[1] - input_ids.shape[1]
        return hidden[batch_idx, token_idx + visual_len], batch_idx

    def boxes_for_batch_indices(self, batch, batch_idx, dtype):
        boxes = []
        for idx in batch_idx.detach().cpu().tolist():
            height, width = batch["sizes"][idx]
            box = xyxy_abs_to_norm(
                batch["boxes"][idx],
                width=width,
                height=height,
                device=self.device,
                dtype=dtype,
            )
            boxes.append(box)
        return torch.stack(boxes, dim=0)

    def compute_slot_loss(self, outputs, batch):
        decoder_outputs = outputs["decoder_outputs"]
        zero = outputs["loss"].new_zeros(())
        logs = {
            "slot_class_loss": zero.detach(),
            "slot_bbox_loss": zero.detach(),
            "slot_reg_loss": zero.detach(),
        }
        losses = []

        obj_hidden, obj_batch_idx = self.get_slot_hidden(decoder_outputs, batch, "obj")
        if obj_hidden is not None:
            usable_count = min(obj_hidden.shape[0], batch["class_slot_targets"].numel())
            obj_hidden = obj_hidden[:usable_count].to(dtype=self.slot_class_head.weight.dtype)
            class_targets = batch["class_slot_targets"][:usable_count].to(self.device)
            class_loss = F.cross_entropy(self.slot_class_head(obj_hidden), class_targets)
            losses.append(class_loss)
            logs["slot_class_loss"] = class_loss.detach()

        bbox_hidden, bbox_batch_idx = self.get_slot_hidden(decoder_outputs, batch, "bbox")
        if bbox_hidden is not None:
            usable_count = min(bbox_hidden.shape[0], batch["bbox_slot_targets"].shape[0])
            bbox_hidden = bbox_hidden[:usable_count]
            bbox_pred = self.slot_bbox_to_xyxy(bbox_hidden)
            bbox_target = batch["bbox_slot_targets"][:usable_count].to(
                device=self.device,
                dtype=bbox_pred.dtype,
            )
            bbox_loss = F.smooth_l1_loss(bbox_pred, bbox_target)
            losses.append(bbox_loss)
            logs["slot_bbox_loss"] = bbox_loss.detach()

        reg_hidden, reg_batch_idx = self.get_slot_hidden(decoder_outputs, batch, "reg")
        if reg_hidden is not None:
            usable_count = min(reg_hidden.shape[0], batch["numeric_targets"].numel())
            if usable_count > 0:
                reg_input = reg_hidden[:usable_count].to(dtype=self.slot_reg_head[0].weight.dtype)
                reg_pred = self.slot_reg_head(reg_input).squeeze(-1)
                reg_target = batch["numeric_targets"][:usable_count].to(
                    device=self.device,
                    dtype=reg_pred.dtype,
                )
                reg_loss = F.smooth_l1_loss(reg_pred, reg_target)
                losses.append(reg_loss)
                logs["slot_reg_loss"] = reg_loss.detach()

        if not losses:
            return zero, logs
        return torch.stack(losses).mean(), logs

    def shared_step(self, batch, stage):
        outputs = self(batch)
        text_loss = outputs["loss"]
        grounding_loss, grounding_logs = self.compute_grounding_loss(
            outputs["dual_outputs"],
            batch,
        )
        mask_loss, mask_logs = self.compute_mask_loss(outputs["dual_outputs"], batch)
        slot_loss, slot_logs = self.compute_slot_loss(outputs, batch)
        loss = (
            text_loss
            + GROUNDING_LOSS_WEIGHT * grounding_loss
            + MASK_LOSS_WEIGHT * mask_loss
            + SLOT_LOSS_WEIGHT * slot_loss
        )

        batch_size = len(batch["boxes"])
        self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}/text_loss", text_loss.detach(), prog_bar=False, batch_size=batch_size)
        self.log(f"{stage}/grounding_loss", grounding_loss.detach(), prog_bar=False, batch_size=batch_size)
        self.log(f"{stage}/mask_loss", mask_loss.detach(), prog_bar=False, batch_size=batch_size)
        self.log(f"{stage}/slot_loss", slot_loss.detach(), prog_bar=False, batch_size=batch_size)
        for key, value in grounding_logs.items():
            self.log(f"{stage}/{key}", value, prog_bar=False, batch_size=batch_size)
        for key, value in mask_logs.items():
            self.log(f"{stage}/{key}", value, prog_bar=False, batch_size=batch_size)
        for key, value in slot_logs.items():
            self.log(f"{stage}/{key}", value, prog_bar=False, batch_size=batch_size)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def configure_optimizers(self):
        trainable_params = [
            param for param in self.parameters()
            if param.requires_grad
        ]
        return torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    model = GLaMMTrainer()
    collator = EncoderDecoderCollate(model.tokenizer)
    train_dataset, eval_dataset, _ = build_dataset()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collator,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collator,
    )

    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=SAVE_DIR,
        filename="glamm-{epoch:02d}-{val_loss:.4f}",
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
