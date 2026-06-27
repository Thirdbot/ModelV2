from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

from utils.GLaMM import SeismicGLaMM
from utils.data import EncoderDecoderCollate, encoder_decoder_process
from utils.train_encoders import box_coverage, xyxy_abs_to_norm


DATASET_NAME = "thirdExec/synthetic-seismic-vlm"
SAVE_DIR = Path("GLaMM")
BATCH_SIZE = 1
MAX_EPOCHS = 50
LEARNING_RATE = 1e-5
NUM_WORKERS = 0
POSITIVE_COVERAGE_THRESHOLD = 0.25
GROUNDING_LOSS_WEIGHT = 1.0


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

    def shared_step(self, batch, stage):
        outputs = self(batch)
        text_loss = outputs["loss"]
        grounding_loss, grounding_logs = self.compute_grounding_loss(
            outputs["dual_outputs"],
            batch,
        )
        loss = text_loss + GROUNDING_LOSS_WEIGHT * grounding_loss

        batch_size = len(batch["boxes"])
        self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}/text_loss", text_loss.detach(), prog_bar=False, batch_size=batch_size)
        self.log(f"{stage}/grounding_loss", grounding_loss.detach(), prog_bar=False, batch_size=batch_size)
        for key, value in grounding_logs.items():
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
