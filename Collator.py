"""
Collator for the seismic VLM dataset.

The dataset keeps full images, masks, and structured `regions`.
This collator keeps rows as full images/masks/regions. The model is
responsible for tiling internally, while the collator prepares text labels
and structured supervision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image
from torchvision.transforms import functional as TF


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def parse_regions(value: Any) -> list[dict[str, Any]]:
    """Parse and normalize the dataset `regions` column."""
    if value is None:
        return []
    if isinstance(value, str):
        value = json.loads(value)
    return [normalize_region(region) for region in _as_list(value)]


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
        "evidence": _as_list(region.get("evidence", "")),
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


def _xml_escape(value: Any) -> str:
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _already_tagged(text: str, tag: str) -> bool:
    text = text.strip()
    return text.startswith(f"<{tag}>") and text.endswith(f"</{tag}>")


def _region_evidence(region: dict[str, Any]) -> list[str]:
    evidence = [str(item) for item in _as_list(region.get("evidence")) if str(item).strip()]
    if evidence:
        return evidence

    obj = str(region.get("object", "region")).capitalize()
    color = region.get("color")
    bbox = region.get("bbox")
    center = region.get("center")

    generated = []
    if bbox is not None:
        generated.append(f"{obj} occupies the area from x={bbox[0]} to {bbox[2]} and y={bbox[1]} to {bbox[3]}")
    if center is not None:
        generated.append(f"{obj} sits near x={center[0]} and y={center[1]}")
    if color:
        generated.append(f"{obj} is highlighted in {color}")
    return generated or [f"{obj} supports the answer"]


def build_region_xml(
    regions: list[dict[str, Any]],
    reason: str | None = None,
    answer: str | None = None,
) -> str:
    parts: list[str] = []
    for region in regions:
        parts.append("<region>")
        parts.append(f"<object>{_xml_escape(region.get('object', ''))}</object>")
        parts.append(f"<class_id>{_xml_escape(region.get('class_id', ''))}</class_id>")
        parts.append(f"<color>{_xml_escape(region.get('color', ''))}</color>")
        for evidence in _region_evidence(region):
            parts.append(f"<evidence>{_xml_escape(evidence)}</evidence>")
        bbox = region["bbox"]
        parts.append(f"<bbox>[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]</bbox>")
        parts.append("<SEG>")
        parts.append("</region>")

    if reason is not None and str(reason).strip():
        reason_text = str(reason).strip()
        if _already_tagged(reason_text, "think"):
            parts.append(reason_text)
        else:
            parts.append(f"<think>{_xml_escape(reason_text)}</think>")

    if answer is not None:
        answer_text = str(answer).strip()
        if _already_tagged(answer_text, "answer"):
            parts.append(answer_text)
        else:
            parts.append(f"<answer>{_xml_escape(answer_text)}</answer>")
    return "\n".join(parts)


@dataclass
class SeismicVlmCollator:
    tokenizer: Any
    image_token: str = "<image>"
    include_empty_tiles: bool = False
    max_objects: int | None = None
    max_length: int | None = 1024

    def __post_init__(self) -> None:
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        full_records: list[dict[str, Any]] = []
        texts: list[str] = []
        prompt_texts: list[str] = []

        for example_index, example in enumerate(examples):
            record = self._example_to_full_record(example, example_index)
            if not record["regions"] and not self.include_empty_tiles:
                continue
            full_records.append(record)
            texts.append(record["text"])
            prompt_texts.append(record["prompt_text"])

        if not full_records:
            raise ValueError("Collator produced no full-image records. Check regions or set include_empty_tiles=True.")

        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = tokenized["input_ids"].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        prompt_lengths = [
            len(self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
            for prompt_text in prompt_texts
        ]
        for row_idx, prompt_length in enumerate(prompt_lengths):
            labels[row_idx, :prompt_length] = -100

        return {
            "pixel_values": [record["images"] for record in full_records],
            "masks": [record["masks"] for record in full_records],
            "regions": [record["regions"] for record in full_records],
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
            "meta": [record["meta"] for record in full_records],
            "text": texts,
            "prompt_text": prompt_texts,
            "target_xml": [record["target_xml"] for record in full_records],
        }

    def _example_to_full_record(self, example: dict[str, Any], example_index: int) -> dict[str, Any]:
        images = [image_to_tensor(image) for image in _as_list(example.get("images"))]
        masks = [mask_to_tensor(mask) for mask in _as_list(example.get("masks"))]
        regions = parse_regions(example.get("regions"))
        if self.max_objects is not None:
            regions = regions[:self.max_objects]

        answer = example.get("answer", "")
        reason = example.get("reason", "")
        target_xml = build_region_xml(regions, reason=reason, answer=answer)
        prompt = example.get("prompt") or example.get("question") or ""
        prompt_text, text = self._build_full_image_text(
            prompt=prompt,
            target_xml=target_xml,
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

    def _build_full_image_text(
        self,
        prompt: str,
        target_xml: str,
        images: list[torch.Tensor],
    ) -> tuple[str, str]:
        image_tokens = "\n".join([self.image_token for _ in images]) or self.image_token
        image_sizes = "\n".join(
            [
                f"Image {idx}: width={int(image.shape[2])}, height={int(image.shape[1])}."
                for idx, image in enumerate(images)
            ]
        )
        prompt_content = (
            f"{image_tokens}\n"
            "These are full seismic images. The model will tile them internally for the vision encoder.\n"
            f"{image_sizes}\n"
            "Return all <bbox> coordinates in full-image coordinates.\n"
            f"{prompt}"
        )
        prompt_messages = [
            {
                "role": "user",
                "content": prompt_content,
            },
        ]
        full_messages = [
            *prompt_messages,
            {
                "role": "assistant",
                "content": target_xml,
            },
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        full_text = self.tokenizer.apply_chat_template(
            full_messages,
            add_generation_prompt=False,
            tokenize=False,
        )
        return prompt_text, full_text
