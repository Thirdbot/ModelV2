from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoProcessor

from Collator import image_to_tensor
from GroundingVQAFormatter import GroundingVQAFormatter
from train_common import TrainConfig
from VLM import VLM


CONFIG = TrainConfig(
    dataset="thirdExec/synthetic-seismic-vlm",
    encoder="NorskRegnesentralSTI/NCS-v1-2d-base",
    decoder="HuggingFaceTB/SmolLM-360M-Instruct",
    test_size=0.2,
    seed=42,
    max_objects=10,
    max_length=512,
)

CHECKPOINT_DIR = "outputs/masks/text_mask_output/train_100/checkpoint-1900"
TEST_INDEX = 0
MAX_NEW_TOKENS = 64
OUTPUT_JSON = "inference_outputs.json"
OUTPUT_DIR = "inference_rendered"
MASK_THRESHOLD = 0.5
OVERLAY_ALPHA = 110
FORCE_CPU = False
DEVICE = "cpu" if FORCE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")

SPECIAL_TOKENS = [
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
    "<think>",
    "</think>",
    "<answer>",
    "</answer>",
]


def resolve_checkpoint(path: str) -> Path | None:
    checkpoint = Path(path)
    if (checkpoint / "model.safetensors").exists():
        return checkpoint

    candidates = sorted(Path("outputs").glob("**/model.safetensors"))
    if not candidates:
        print("No checkpoint found. Using base pretrained weights.")
        return None

    latest = max(candidates, key=lambda item: item.stat().st_mtime).parent
    print(f"Checkpoint {checkpoint} has no model.safetensors. Using {latest}.")
    return latest


def load_test_row(config: TrainConfig, test_index: int) -> dict[str, Any]:
    dataset = load_dataset(config.dataset)["train"]
    split = dataset.train_test_split(test_size=config.test_size, seed=config.seed, shuffle=True)
    test_dataset = split["test"]
    if test_index >= len(test_dataset):
        raise IndexError(f"TEST_INDEX={test_index} but test split has {len(test_dataset)} rows.")
    return test_dataset[test_index]


def load_model_and_tokenizer(config: TrainConfig) -> tuple[VLM, Any]:
    checkpoint = resolve_checkpoint(CHECKPOINT_DIR)
    tokenizer_source = str(checkpoint) if checkpoint is not None else config.decoder
    tokenizer = AutoProcessor.from_pretrained(tokenizer_source)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    model = VLM.from_encoder_decoder_pretrained(
        encoder_name_or_path=config.encoder,
        decoder_name_or_path=config.decoder,
        tokenizer=tokenizer,
        max_detection_slots=config.max_objects,
    )
    if model.decoder.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    if checkpoint is not None:
        state_dict = load_file(str(checkpoint / "model.safetensors"))
        model.load_state_dict(state_dict, strict=True)

    model.to(DEVICE)
    model.eval()
    return model, tokenizer


def build_prompt(tokenizer: Any, row: dict[str, Any], image_tensors: list[torch.Tensor]) -> str:
    formatter = GroundingVQAFormatter(tokenizer=tokenizer, image_token="<image>")
    question = row.get("question") or "Find and describe the seismic evidence."
    return formatter.build_prompt(prompt=question, images=image_tensors)


def parse_grounding_image_indices(text: str, max_objects: int, default_index: int = 0) -> list[int]:
    matches = re.findall(r"<image_index>\s*(\d+)\s*</image_index>", text)
    indices = [int(value) for value in matches[:max_objects]]
    if not indices:
        indices = [default_index]
    while len(indices) < max_objects:
        indices.append(indices[-1])
    return indices


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().cpu().clamp(0, 1)
    image = (image * 255).to(torch.uint8)
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    return Image.fromarray(image.permute(1, 2, 0).numpy(), mode="RGB")


def render_mask_overlays(
    image_tensors: list[torch.Tensor],
    mask_logits: torch.Tensor,
    image_meta: list[dict[str, Any]],
) -> list[str]:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    colors = [
        (255, 0, 0, OVERLAY_ALPHA),
        (0, 180, 255, OVERLAY_ALPHA),
        (255, 220, 0, OVERLAY_ALPHA),
        (0, 220, 120, OVERLAY_ALPHA),
        (180, 90, 255, OVERLAY_ALPHA),
        (255, 120, 0, OVERLAY_ALPHA),
    ]

    base_images = [tensor_to_pil(image).convert("RGBA") for image in image_tensors]
    overlays = [Image.new("RGBA", image.size, (0, 0, 0, 0)) for image in base_images]

    mask_probs = mask_logits.detach().sigmoid().cpu()
    for slot_idx, mask_prob in enumerate(mask_probs):
        if slot_idx >= len(image_meta):
            continue
        meta = image_meta[slot_idx]
        image_index = int(meta["image_index"])
        if image_index >= len(overlays):
            continue

        resized_w = int(meta["resized_w"])
        resized_h = int(meta["resized_h"])
        orig_w = int(meta["orig_w"])
        orig_h = int(meta["orig_h"])
        mask_prob = mask_prob[:resized_h, :resized_w]
        mask = (mask_prob > MASK_THRESHOLD).to(torch.uint8) * 255
        mask_img = Image.fromarray(mask.numpy(), mode="L").resize((orig_w, orig_h), resample=Image.Resampling.NEAREST)
        color = colors[slot_idx % len(colors)]
        color_img = Image.new("RGBA", mask_img.size, color)
        overlays[image_index].paste(color_img, (0, 0), mask_img)

    saved_paths = []
    for image_index, base in enumerate(base_images):
        original_path = output_dir / f"test_{TEST_INDEX}_image_{image_index}_original.png"
        overlay_path = output_dir / f"test_{TEST_INDEX}_image_{image_index}_overlay.png"
        base.convert("RGB").save(original_path)
        composed = Image.alpha_composite(base, overlays[image_index])
        composed.convert("RGB").save(overlay_path)
        saved_paths.extend([str(original_path), str(overlay_path)])
    return saved_paths


@torch.no_grad()
def run_inference() -> dict[str, Any]:
    model, tokenizer = load_model_and_tokenizer(CONFIG)
    row = load_test_row(CONFIG, TEST_INDEX)
    image_tensors = [image_to_tensor(image).to(DEVICE) for image in row["images"]]
    prompt = build_prompt(tokenizer, row, image_tensors)

    tokenized = tokenizer(
        [prompt],
        padding=True,
        truncation=True,
        max_length=CONFIG.max_length,
        return_tensors="pt",
    )
    tokenized = {key: value.to(DEVICE) for key, value in tokenized.items()}

    generated_ids = model.generate(
        pixel_values=[image_tensors],
        input_ids=tokenized["input_ids"],
        attention_mask=tokenized["attention_mask"],
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0].strip()
    grounding_image_indices = parse_grounding_image_indices(generated_text, max_objects=CONFIG.max_objects)
    grounding_text = prompt + generated_text
    grounding_tokens = tokenizer(
        [grounding_text],
        padding=True,
        truncation=True,
        max_length=CONFIG.max_length + MAX_NEW_TOKENS,
        return_tensors="pt",
    )
    grounding_tokens = {key: value.to(DEVICE) for key, value in grounding_tokens.items()}

    grounding = model.predict_grounding_from_text(
        pixel_values=[image_tensors],
        input_ids=grounding_tokens["input_ids"],
        attention_mask=grounding_tokens["attention_mask"],
        grounding_image_indices=[grounding_image_indices],
    )
    image_meta = grounding.get("image_meta", [[]])[0]
    mask_logits = grounding["mask_logits"][0].detach().cpu()
    rendered_paths = render_mask_overlays(
        image_tensors=image_tensors,
        mask_logits=mask_logits,
        image_meta=image_meta,
    )

    result = {
        "test_index": TEST_INDEX,
        "question": row.get("question", ""),
        "generated_text": generated_text,
        "image_meta": image_meta,
        "grounding_image_indices": grounding_image_indices,
        "rendered_paths": rendered_paths,
        "target_answer": row.get("answer", ""),
        "target_evidence": row.get("evidence", ""),
    }
    return result


result = run_inference()

print("\n--- test split full multimodal inference ---")
print(f"test_index: {result['test_index']}")
print(result["generated_text"])
if result["rendered_paths"]:
    print("rendered_images:")
    for path in result["rendered_paths"]:
        print(path)

with open(OUTPUT_JSON, "w", encoding="utf-8") as file:
    json.dump(result, file, indent=2)
print(f"\nWrote {OUTPUT_JSON}")
