from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoProcessor, Trainer, TrainingArguments

from Collator import SeismicVlmCollator
from VLM import VLM
from WandbOffline import WandbOffline


class SeismicTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        if not torch.isfinite(loss):
            details = {}
            if isinstance(outputs, dict):
                for key in ("text_loss", "grounding_loss", "bbox_loss", "mask_loss", "mask_bce_loss", "mask_dice_loss"):
                    value = outputs.get(key)
                    if value is not None:
                        details[key] = value.detach().float().cpu().item()
            raise FloatingPointError(f"Non-finite training loss: {details}")

        if model.training and isinstance(outputs, dict):
            logs = {}
            for key in (
                "text_loss",
                "grounding_loss",
                "bbox_loss",
                "mask_loss",
                "mask_bce_loss",
                "mask_dice_loss",
            ):
                value = outputs.get(key)
                if value is not None:
                    logs[key] = value.detach().float().item()
            if logs:
                self.log(logs)

        return (loss, outputs) if return_outputs else loss


@dataclass
class TrainConfig:
    dataset: str = "thirdExec/synthetic-seismic-vlm"
    encoder: str = "NorskRegnesentralSTI/NCS-v1-2d-base"
    decoder: str = "HuggingFaceTB/SmolLM-360M-Instruct"
    save_root: str = "outputs"
    run_name: str = "test_drive"
    test_size: float = 0.2
    seed: int = 42
    max_objects: int = 10
    max_length: int = 256
    text_loss_weight: float = 1.0
    bbox_loss_weight: float = 1.0
    mask_loss_weight: float = 1.0
    include_empty_rows: bool = False
    max_train_samples: int | None = 1
    max_eval_samples: int | None = 1
    filter_empty_region_rows: bool = True
    num_train_epochs: float = 1.0
    max_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    logging_steps: int = 1
    eval_steps: int = 1
    save_steps: int = 1
    save_total_limit: int = 2
    save_only_model: bool = True
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
    freeze_encoder: bool = True
    freeze_decoder: bool = False
    wandb_project: str = "seismic-vlm"
    wandb_dir: str = "runs/wandb"
    use_wandb: bool = True
    resume_from_checkpoint: str | None = None


def load_splits(config: TrainConfig):
    dataset = load_dataset(config.dataset)["train"]
    split = dataset.train_test_split(test_size=config.test_size, seed=config.seed, shuffle=True)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    if config.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(config.max_train_samples, len(train_dataset))))
    if config.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(config.max_eval_samples, len(eval_dataset))))

    return train_dataset, eval_dataset


def build_tokenizer_and_model(config: TrainConfig, special_tokens: list[str] | None = None) -> tuple[Any, VLM]:
    tokenizer = AutoProcessor.from_pretrained(config.decoder)
    model = VLM.from_encoder_decoder_pretrained(
        encoder_name_or_path=config.encoder,
        decoder_name_or_path=config.decoder,
        tokenizer=tokenizer,
        max_detection_slots=config.max_objects,
        text_loss_weight=config.text_loss_weight,
        bbox_loss_weight=config.bbox_loss_weight,
        mask_loss_weight=config.mask_loss_weight,
    )

    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    special_tokens = special_tokens or [
        "<image>",
        "<SEG>",
        "<region>",
        "</region>",
        "<image_index>",
        "</image_index>",
        "<object>",
        "</object>",
        "<class_id>",
        "</class_id>",
        "<color>",
        "</color>",
        "<evidence>",
        "</evidence>",
        "<bbox>",
        "</bbox>",
        "<think>",
        "</think>",
        "<answer>",
        "</answer>",
    ]
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    apply_freezing(model, freeze_encoder=config.freeze_encoder, freeze_decoder=config.freeze_decoder)
    return tokenizer, model


def apply_freezing(model: VLM, freeze_encoder: bool, freeze_decoder: bool) -> None:
    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False

    if freeze_decoder:
        for param in model.decoder.parameters():
            param.requires_grad = False


def build_collator(config: TrainConfig, tokenizer: Any):
    return SeismicVlmCollator(
        tokenizer=tokenizer,
        image_token="<image>",
        include_empty_rows=config.include_empty_rows,
        max_objects=config.max_objects,
        max_length=config.max_length,
    )


def build_training_arguments(config: TrainConfig, training_type: str) -> TrainingArguments:
    output_dir = Path(config.save_root) / training_type / config.run_name
    report_to = WandbOffline(
        project=config.wandb_project,
        root_dir=config.wandb_dir,
        enabled=config.use_wandb,
    ).setup(config.run_name)

    return TrainingArguments(
        output_dir=str(output_dir),
        run_name=config.run_name,
        report_to=report_to,
        remove_unused_columns=False,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_grad_norm=config.max_grad_norm,
        logging_steps=config.logging_steps,
        eval_strategy="epoch",
        eval_steps=config.eval_steps,
        save_strategy="epoch",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        save_only_model=config.save_only_model,
        fp16=config.fp16,
        bf16=config.bf16,
        dataloader_num_workers=0,
        seed=config.seed,
    )


def build_trainer(config: TrainConfig, training_type: str) -> Trainer:
    train_dataset, eval_dataset = load_splits(config)
    tokenizer, model = build_tokenizer_and_model(config)
    collator = build_collator(config, tokenizer)
    if config.filter_empty_region_rows:
        train_dataset = filter_empty_region_rows(train_dataset, collator)
        eval_dataset = filter_empty_region_rows(eval_dataset, collator)
    training_args = build_training_arguments(config, training_type)

    return SeismicTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )


def filter_empty_region_rows(dataset, collator):
    def has_regions(example):
        return len(collator._example_to_full_record(example, example_index=0)["regions"]) > 0

    filtered = dataset.filter(has_regions)
    if len(filtered) == 0:
        raise ValueError("No dataset rows contain regions with the current collator settings.")
    return filtered


def train_from_config(config: TrainConfig, training_type: str) -> None:
    trainer = build_trainer(config, training_type=training_type)
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    trainer.save_model()
    trainer.save_state()
