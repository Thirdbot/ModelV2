from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

from utils.data import bcx_process, encoder_collator
from utils.global_region_encoder import DualEncoder


DATASET_NAME = "thirdExec/synthetic-seismic-vlm"
SAVE_DIR = Path("encoder")
BATCH_SIZE = 2
MAX_EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
POSITIVE_COVERAGE_THRESHOLD = 0.25
ALIGN_LOSS_WEIGHT = 0.1


def build_dataset():
    raw = load_dataset(DATASET_NAME)["train"]
    rows = []
    for example in raw:
        processed = bcx_process(example)
        if isinstance(processed, list):
            rows.extend(processed)
        else:
            rows.append(processed)

    dataset = Dataset.from_list(rows)
    split = dataset.train_test_split(test_size=0.2, seed=42)
    holdout = split["test"].train_test_split(test_size=0.5, seed=42)
    return split["train"], holdout["train"], holdout["test"]


def box_coverage(region_box, tile_boxes):
    region = torch.as_tensor(region_box, dtype=tile_boxes.dtype, device=tile_boxes.device)
    if region.ndim == 1:
        region = region.unsqueeze(0)

    rx1, ry1, rx2, ry2 = region[0]
    tx1, ty1, tx2, ty2 = tile_boxes.unbind(dim=-1)

    ix1 = torch.maximum(rx1, tx1)
    iy1 = torch.maximum(ry1, ty1)
    ix2 = torch.minimum(rx2, tx2)
    iy2 = torch.minimum(ry2, ty2)

    inter_w = (ix2 - ix1).clamp(min=0)
    inter_h = (iy2 - iy1).clamp(min=0)
    inter = inter_w * inter_h

    region_area = ((rx2 - rx1).clamp(min=0) * (ry2 - ry1).clamp(min=0)).clamp(min=1)
    return inter / region_area


def xyxy_abs_to_norm(box, width, height, device, dtype):
    box = torch.as_tensor(box, dtype=dtype, device=device)
    out = box.clone()
    out[..., [0, 2]] /= width
    out[..., [1, 3]] /= height
    return out


class EncoderTrainer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = DualEncoder(is_train=True)
        self.global_align = torch.nn.Linear(768, 256)
        self.region_align = torch.nn.Linear(768, 256)

    def forward(self, batch):
        heights = [size[0] for size in batch["sizes"]]
        widths = [size[1] for size in batch["sizes"]]
        return self.encoder(
            pixel_values=batch["pixel_values"],
            tiles=batch["tiles"],
            bbox=batch["boxes"],
            class_ids=batch["label"],
            H=heights,
            W=widths,
        )

    def compute_sample_loss(self, output, box, label, height, width):
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

        pos_global = output["global_tiles"][positive]
        region_feature = output["region_feature"].expand(pos_global.shape[0], -1)
        global_proj = F.normalize(self.global_align(pos_global), dim=-1)
        region_proj = F.normalize(self.region_align(region_feature), dim=-1)
        align_loss = 1.0 - (global_proj * region_proj).sum(dim=-1).mean()

        total = objectness_loss + class_loss + bbox_loss + ALIGN_LOSS_WEIGHT * align_loss
        losses = {
            "loss": total,
            "objectness_loss": objectness_loss.detach(),
            "class_loss": class_loss.detach(),
            "bbox_loss": bbox_loss.detach(),
            "align_loss": align_loss.detach(),
            "positive_tiles": positive.sum().detach().float(),
        }
        return losses

    def shared_step(self, batch, stage):
        outputs = self(batch)
        losses = []
        logs = {
            "objectness_loss": [],
            "class_loss": [],
            "bbox_loss": [],
            "align_loss": [],
            "positive_tiles": [],
        }

        for idx, output in enumerate(outputs):
            height, width = batch["sizes"][idx]
            sample_losses = self.compute_sample_loss(
                output=output,
                box=batch["boxes"][idx],
                label=batch["label"][idx],
                height=height,
                width=width,
            )
            losses.append(sample_losses["loss"])
            for key in logs:
                logs[key].append(sample_losses[key])

        loss = torch.stack(losses).mean()
        self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=len(outputs))
        for key, values in logs.items():
            self.log(
                f"{stage}/{key}",
                torch.stack(values).mean(),
                prog_bar=False,
                batch_size=len(outputs),
            )
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, "val")

    def configure_optimizers(self):
        trainable_params = [
            param for param in self.parameters()
            if param.requires_grad
        ]
        return torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = SAVE_DIR.joinpath("lightning_logs","last.ckpt")
    train_dataset, eval_dataset, _ = build_dataset()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=encoder_collator,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=encoder_collator,
    )

    model = EncoderTrainer()
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=SAVE_DIR,
        filename="encoder-{epoch:02d}-{val_loss:.4f}",
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
    trainer.fit(model, train_loader, eval_loader,
                ckpt_path=ckpt_path.as_posix() if ckpt_path.exists() else None,)


if __name__ == "__main__":
    main()
