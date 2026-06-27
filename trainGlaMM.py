from pathlib import Path

import pytorch_lightning as pl
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

from utils.GLaMM import SeismicGLaMM
from utils.data import EncoderDecoderCollate, encoder_decoder_process


DATASET_NAME = "thirdExec/synthetic-seismic-vlm"
SAVE_DIR = Path("GLaMM")
BATCH_SIZE = 1
MAX_EPOCHS = 3
LEARNING_RATE = 1e-5
NUM_WORKERS = 0


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

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs["loss"]
        self.log("train/loss", loss, prog_bar=True, batch_size=len(batch["boxes"]))
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs["loss"]
        self.log("val/loss", loss, prog_bar=True, batch_size=len(batch["boxes"]))
        return loss

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
