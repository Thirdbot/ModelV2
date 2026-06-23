from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


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
    evidence = [str(item) for item in as_list(region.get("evidence")) if str(item).strip()]
    if evidence:
        return evidence

    obj = str(region.get("object", "region")).capitalize()
    color = region.get("color")
    center = region.get("center")

    generated = []
    if center is not None:
        generated.append(f"{obj} is visible near the interpreted region")
    if color:
        generated.append(f"{obj} is highlighted in {color}")
    return generated or [f"{obj} supports the answer"]


@dataclass
class GroundingVQAFormatter:
    tokenizer: Any
    image_token: str = "<image>"

    def build_target(
        self,
        regions: list[dict[str, Any]],
        reason: str | None = None,
        answer: str | None = None,
    ) -> str:
        parts: list[str] = []
        for region in regions:
            parts.append("<region>")
            parts.append(f"<image_index>{_xml_escape(region.get('image_index', 0))}</image_index>")
            parts.append(f"<object>{_xml_escape(region.get('object', ''))}</object>")
            parts.append(f"<class_id>{_xml_escape(region.get('class_id', ''))}</class_id>")
            parts.append(f"<color>{_xml_escape(region.get('color', ''))}</color>")
            for evidence in _region_evidence(region):
                parts.append(f"<evidence>{_xml_escape(evidence)}</evidence>")
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

    def build_prompt(self, prompt: str, images: list[torch.Tensor]) -> str:
        prompt_content = self.build_prompt_content(prompt=prompt, images=images)
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_content}],
            add_generation_prompt=True,
            tokenize=False,
        )

    def build_prompt_content(self, prompt: str, images: list[torch.Tensor]) -> str:
        image_tokens = "\n".join([self.image_token for _ in images]) or self.image_token
        image_sizes = "\n".join(
            [
                f"Image {idx}: width={int(image.shape[2])}, height={int(image.shape[1])}."
                for idx, image in enumerate(images)
            ]
        )
        prompt_content = (
            f"{image_tokens}\n"
            "Task: answer the seismic interpretation question and ground each visual evidence region.\n"
            f"{image_sizes}\n"
            "Use <region> blocks for grounded evidence, include <image_index> for the source image, "
            "and place one segmentation marker inside each grounded region.\n"
            f"{prompt}"
        )
        return prompt_content

    def build_text(self, prompt: str, target: str, images: list[torch.Tensor]) -> tuple[str, str]:
        prompt_text = self.build_prompt(prompt=prompt, images=images)
        prompt_content = self.build_prompt_content(prompt=prompt, images=images)
        full_text = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt_content},
                {"role": "assistant", "content": target},
            ],
            add_generation_prompt=False,
            tokenize=False,
        )
        return prompt_text, full_text
