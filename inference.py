from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from safetensors.torch import load_file
from transformers import AutoProcessor

from Collator import image_to_tensor
from train_common import TrainConfig
from VLM import VLM


CONFIG = TrainConfig(
    dataset="thirdExec/synthetic-seismic-vlm",
    encoder="NorskRegnesentralSTI/NCS-v1-2d-base",
    decoder="HuggingFaceTB/SmolLM-360M-Instruct",
    test_size=0.2,
    seed=42,
    max_objects=10,
    max_internal_tiles=8,
    internal_tile_stride=112,
    max_length=512,
)

CHECKPOINT_DIR = "outputs/masks/text_mask_output/train_100/checkpoint-1900"
TEST_INDEX = 0
MAX_NEW_TOKENS = 64
OUTPUT_JSON = "inference_outputs.json"
FORCE_CPU = False
DEVICE = "cpu" if FORCE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")

SPECIAL_TOKENS = [
    "<image>",
    "<SEG>",
    "<region>",
    "</region>",
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
        max_internal_tiles=config.max_internal_tiles,
        internal_tile_stride=config.internal_tile_stride,
    )
    if model.decoder.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    if checkpoint is not None:
        state_dict = load_file(str(checkpoint / "model.safetensors"))
        state_dict, skipped = filter_loadable_state_dict(model, state_dict)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if skipped:
            print(f"Skipped incompatible checkpoint keys: {len(skipped)}")
        if missing:
            print(f"Missing checkpoint keys: {len(missing)}")
        if unexpected:
            print(f"Unexpected checkpoint keys: {len(unexpected)}")

    model.to(DEVICE)
    model.eval()
    return model, tokenizer


def filter_loadable_state_dict(model: VLM, state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], list[str]]:
    model_state = model.state_dict()
    loadable = {}
    skipped = []
    for key, value in state_dict.items():
        if key not in model_state or model_state[key].shape != value.shape:
            skipped.append(key)
            continue
        loadable[key] = value
    return loadable, skipped


def build_prompt(tokenizer: Any, row: dict[str, Any], image_tensors: list[torch.Tensor]) -> str:
    image_tokens = "\n".join(["<image>" for _ in image_tensors])
    image_sizes = "\n".join(
        [
            f"Image {idx}: width={int(image.shape[2])}, height={int(image.shape[1])}."
            for idx, image in enumerate(image_tensors)
        ]
    )
    question = row.get("question") or "Find and describe the seismic evidence. Return XML regions with evidence, bbox, and <SEG>."
    prompt_content = (
        f"{image_tokens}\n"
        "These are full seismic images. The model will tile them internally for the vision encoder.\n"
        f"{image_sizes}\n"
        "Return all <bbox> coordinates in full-image coordinates.\n"
        f"{question}"
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_content}],
        add_generation_prompt=True,
        tokenize=False,
    )


def local_bbox_to_global(local_bbox: list[float], tile_meta: dict[str, Any]) -> list[int]:
    x0 = int(tile_meta["x0"])
    y0 = int(tile_meta["y0"])
    orig_w = int(tile_meta["orig_w"])
    orig_h = int(tile_meta["orig_h"])
    return [
        max(0, min(orig_w, round(local_bbox[0] + x0))),
        max(0, min(orig_h, round(local_bbox[1] + y0))),
        max(0, min(orig_w, round(local_bbox[2] + x0))),
        max(0, min(orig_h, round(local_bbox[3] + y0))),
    ]


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

    grounding = model.predict_grounding(
        pixel_values=[image_tensors],
        num_objects=CONFIG.max_objects,
    )
    local_bboxes = grounding["bbox_preds"][0].detach().cpu().tolist()
    tile_meta = grounding.get("tile_meta", [[]])[0]
    global_bboxes = [
        local_bbox_to_global(bbox, tile_meta[min(idx, len(tile_meta) - 1)])
        for idx, bbox in enumerate(local_bboxes)
        if tile_meta
    ]

    result = {
        "test_index": TEST_INDEX,
        "question": row.get("question", ""),
        "generated_text": generated_text,
        "local_bboxes": local_bboxes,
        "global_bboxes": global_bboxes,
        "tile_meta": tile_meta,
        "target_answer": row.get("answer", ""),
        "target_evidence": row.get("evidence", ""),
    }
    return result


result = run_inference()

print("\n--- test split full multimodal inference ---")
print(f"test_index: {result['test_index']}")
print(result["generated_text"])
if result["global_bboxes"]:
    print("first_global_bbox_from_head:", result["global_bboxes"][0])

with open(OUTPUT_JSON, "w", encoding="utf-8") as file:
    json.dump(result, file, indent=2)
print(f"\nWrote {OUTPUT_JSON}")
