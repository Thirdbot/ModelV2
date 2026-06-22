"""
Collator for seismic Grounding VQA.

The dataset keeps full images, masks, and structured `regions`.
This collator keeps rows as full images/masks/regions. The model is
responsible for resize/pad processing internally, while the collator prepares
question-answer text labels and structured grounding supervision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from GroundingVQAFormatter import GroundingVQAFormatter, as_list


def parse_regions(value: Any) -> list[dict[str, Any]]:
    """Parse and normalize the dataset `regions` column."""
    if value is None:
        return []
    if isinstance(value, str):
        value = json.loads(value)
    return [normalize_region(region) for region in as_list(value)]


def normalize_region(region: dict[str, Any]) -> dict[str, Any]:
    """Normalize the fixed `thirdExec/synthetic-seismic-vlm` region schema."""
    return {
        **region,
        "region_index": int(region["region_idx"]),
        "image_index": int(region["image_idx"]),
        "mask_index": int(region["mask_idx"]),
        "region_id": str(region["object_id"]),
        "base_image_id": f"image_{int(region['image_idx'])}",
        "object": str(region["object_type"]),
        "class_id": int(region["class_id"]),
        "color": str(region["class_color"]),
        "bbox": [int(v) for v in region["bbox"]],
        "evidence": as_list(region.get("evidence", "")),
    }


def image_to_tensor(image: Any) -> torch.Tensor:
    """Return RGB image tensor as float [3, H, W] in [0, 1]."""
    if isinstance(image, torch.Tensor):
        tensor = image
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3:
            raise ValueError(f"Expected image tensor [C,H,W] or [H,W], got {tuple(tensor.shape)}")
        if tensor.shape[0] == 4:
            tensor = tensor[:3]
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        return tensor.float() / 255.0 if tensor.dtype == torch.uint8 else tensor.float()

    if isinstance(image, Image.Image):
        return TF.to_tensor(image.convert("RGB"))

    raise TypeError(f"Unsupported image type: {type(image)!r}")


def mask_to_tensor(mask: Any) -> torch.Tensor:
    """Return binary mask tensor as float [H, W]."""
    if isinstance(mask, torch.Tensor):
        tensor = mask
        if tensor.ndim == 3:
            tensor = tensor[0]
        if tensor.ndim != 2:
            raise ValueError(f"Expected mask tensor [H,W] or [1,H,W], got {tuple(tensor.shape)}")
        return (tensor > 0).float()

    if isinstance(mask, Image.Image):
        return (TF.to_tensor(mask.convert("L"))[0] > 0).float()

    raise TypeError(f"Unsupported mask type: {type(mask)!r}")


@dataclass
class SeismicVlmCollator:
    tokenizer: Any
    image_token: str = "<image>"
    include_empty_rows: bool = False
    max_objects: int | None = None
    max_length: int | None = 1024
    formatter: GroundingVQAFormatter = field(init=False)

    def __post_init__(self) -> None:
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.formatter = GroundingVQAFormatter(tokenizer=self.tokenizer, image_token=self.image_token)

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        full_records: list[dict[str, Any]] = []

        for example_index, example in enumerate(examples):
            record = self._example_to_full_record(example, example_index)
            if not record["regions"] and not self.include_empty_rows:
                continue
            full_records.append(record)

        if not full_records:
            raise ValueError("Collator produced no full-image records. Check regions or set include_empty_rows=True.")

        tokenized = self._tokenize_records(full_records)

        return {
            "pixel_values": [record["images"] for record in full_records],
            "masks": [record["masks"] for record in full_records],
            "regions": [record["regions"] for record in full_records],
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["labels"],
            "meta": [record["meta"] for record in full_records],
            "text": [record["text"] for record in full_records],
            "prompt_text": [record["prompt_text"] for record in full_records],
            "target_xml": [record["target_xml"] for record in full_records],
        }

    def _tokenize_records(self, records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_length = self.max_length or 1024
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer needs either pad_token_id or eos_token_id.")

        batch_input_ids = []
        batch_labels = []
        for record in records:
            prompt_ids = self.tokenizer(
                record["prompt_text"],
                add_special_tokens=False,
            )["input_ids"]
            target_text = record["target_xml"]
            if self.tokenizer.eos_token:
                target_text = f"{target_text}{self.tokenizer.eos_token}"
            target_ids = self.tokenizer(
                target_text,
                add_special_tokens=False,
            )["input_ids"]

            if not target_ids:
                raise ValueError("Grounding VQA target produced no tokens.")

            if len(target_ids) >= max_length:
                target_ids = target_ids[:max_length]
                prompt_ids = []
            else:
                prompt_budget = max_length - len(target_ids)
                prompt_ids = prompt_ids[-prompt_budget:]

            input_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids
            if all(label == -100 for label in labels):
                raise ValueError(
                    "All text labels were masked. Increase max_length or reduce max_objects/evidence length."
                )
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)

        batch_max_length = max(len(input_ids) for input_ids in batch_input_ids)
        input_tensor = torch.full((len(records), batch_max_length), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(records), batch_max_length), dtype=torch.long)
        label_tensor = torch.full((len(records), batch_max_length), -100, dtype=torch.long)

        for row_idx, (input_ids, labels) in enumerate(zip(batch_input_ids, batch_labels)):
            length = len(input_ids)
            input_tensor[row_idx, :length] = torch.tensor(input_ids, dtype=torch.long)
            attention_mask[row_idx, :length] = 1
            label_tensor[row_idx, :length] = torch.tensor(labels, dtype=torch.long)

        return {
            "input_ids": input_tensor,
            "attention_mask": attention_mask,
            "labels": label_tensor,
        }

    def _example_to_full_record(self, example: dict[str, Any], example_index: int) -> dict[str, Any]:
        images = [image_to_tensor(image) for image in as_list(example.get("images"))]
        masks = [mask_to_tensor(mask) for mask in as_list(example.get("masks"))]
        regions = parse_regions(example.get("regions"))
        if self.max_objects is not None:
            regions = regions[:self.max_objects]

        answer = example.get("answer", "")
        reason = example.get("reason", "")
        target_xml = self.formatter.build_target(regions, reason=reason, answer=answer)
        prompt = example.get("prompt") or example.get("question") or ""
        prompt_text, text = self.formatter.build_text(
            prompt=prompt,
            target=target_xml,
            images=images,
        )

        source_row = example.get("source_row", example_index)
        return {
            "images": images,
            "masks": masks,
            "regions": regions,
            "prompt_text": prompt_text,
            "text": text,
            "target_xml": target_xml,
            "meta": {
                "source_row": source_row,
                "image_sizes": [
                    {"image_index": idx, "height": int(image.shape[1]), "width": int(image.shape[2])}
                    for idx, image in enumerate(images)
                ],
            },
        }
