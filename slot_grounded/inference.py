from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from slot_grounded.train import SlotGroundedTrainer, build_dataset
from utils.data import (
    BBOX_SLOT,
    CENTER_SLOT,
    GENERIC_NUM_SCALE,
    OBJ_SLOT,
    REG_SLOT,
    SEG_TOKEN,
    EncoderDecoderCollate,
)


CHECKPOINT_DIR = Path("slot_grounded/checkpoints")
OUTPUT_DIR = Path("slot_grounded/inference_outputs")
NUM_SAMPLES = 5
MAX_NEW_TOKENS = 180
CLASS_NAMES = {
    0: "no_object",
    1: "fault",
    2: "closure",
    3: "salt",
    4: "onlap",
    5: "lithology",
}
CLASS_COLORS = {
    0: (160, 160, 160),
    1: (255, 0, 0),
    2: (0, 128, 255),
    3: (255, 128, 0),
    4: (255, 220, 0),
    5: (0, 200, 80),
}


def find_checkpoint():
    last = CHECKPOINT_DIR / "last.ckpt"
    if last.exists():
        return last

    checkpoints = sorted(CHECKPOINT_DIR.glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {CHECKPOINT_DIR}")
    return checkpoints[-1]


def format_bbox(box):
    return [int(round(value)) for value in box]


def strip_prompt_echo(text):
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant", 1)[-1]
    return text.replace("<|im_end|>", "").strip()


def clamp_bbox_to_image(bbox, width, height):
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def norm_box_to_abs(box, width, height):
    x1, y1, x2, y2 = box
    return [
        x1 * width,
        y1 * height,
        x2 * width,
        y2 * height,
    ]


def roi_mask_to_full_canvas(mask_logits, bbox, width, height):
    x1, y1, x2, y2 = clamp_bbox_to_image(bbox, width, height)
    crop_h = y2 - y1
    crop_w = x2 - x1

    roi_prob = mask_logits.detach().float().sigmoid().unsqueeze(0)
    roi_prob = F.interpolate(
        roi_prob,
        size=(crop_h, crop_w),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    canvas = torch.zeros((height, width), dtype=roi_prob.dtype)
    canvas[y1:y2, x1:x2] = roi_prob.cpu()
    return canvas


def save_mask_overlay(image, full_mask, class_id, output_path, alpha=0.45):
    base = image.convert("RGBA")
    width, height = base.size
    color = CLASS_COLORS.get(class_id, (255, 0, 0))

    mask_byte = (full_mask.clamp(0, 1) * 255).byte().numpy()
    mask_image = Image.fromarray(mask_byte, mode="L").resize((width, height))

    overlay = Image.new("RGBA", (width, height), color + (0,))
    colored = Image.new("RGBA", (width, height), color + (int(255 * alpha),))
    overlay.paste(colored, mask=mask_image)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(base, overlay).save(output_path)
    mask_image.save(output_path.with_name(f"{output_path.stem}_mask.png"))


def render_slots(text, object_names, bboxes, centers, numbers):
    object_index = 0
    while OBJ_SLOT in text and object_names:
        text = text.replace(OBJ_SLOT, object_names[min(object_index, len(object_names) - 1)], 1)
        object_index += 1

    bbox_index = 0
    while BBOX_SLOT in text and bboxes:
        text = text.replace(BBOX_SLOT, str(format_bbox(bboxes[min(bbox_index, len(bboxes) - 1)])), 1)
        bbox_index += 1

    center_index = 0
    while CENTER_SLOT in text and centers:
        text = text.replace(CENTER_SLOT, str(format_bbox(centers[min(center_index, len(centers) - 1)])), 1)
        center_index += 1

    number_index = 0
    while REG_SLOT in text and number_index < len(numbers):
        text = text.replace(REG_SLOT, f"{numbers[number_index]:.4g}", 1)
        number_index += 1
    return text


def build_prompt_visual_tokens(model, batch, device):
    heights = [size[0] for size in batch["sizes"]]
    widths = [size[1] for size in batch["sizes"]]

    visual_outputs = model.model.encode_images(
        pixel_values=batch["pixel_values"],
        tiles=batch["tiles"],
        boxes=batch["boxes"],
        class_ids=batch["label"],
        heights=heights,
        widths=widths,
    )
    visual_tokens = model.model.group_visual_tokens(
        visual_outputs,
        row_image_counts=batch["row_image_counts"],
    )
    visual_tokens = torch.stack(visual_tokens, dim=0).to(device)
    return visual_outputs, visual_tokens


def rerun_decoder_for_generated(model, full_text_ids, visual_tokens, device):
    text_embeds = model.model.decoder.get_input_embeddings()(full_text_ids)
    visual_tokens = visual_tokens.to(device=device, dtype=text_embeds.dtype)
    inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)
    attention_mask = torch.ones(
        inputs_embeds.shape[:2],
        device=device,
        dtype=torch.long,
    )
    return model.model.decoder(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )


def hidden_for_token(model, token_text, full_text_ids, decoder_outputs, visual_len):
    token_id = model.tokenizer.convert_tokens_to_ids(token_text)
    batch_idx, token_idx = (full_text_ids == token_id).nonzero(as_tuple=True)
    if batch_idx.numel() == 0:
        return None
    hidden = decoder_outputs.hidden_states[-1]
    return hidden[batch_idx, token_idx + visual_len]


def crop_spatial_features_from_predicted_boxes(model, batch, bboxes_abs, device):
    spatial_features = []
    for image, pixel_value, bbox in zip(batch["images"], batch["pixel_values"], bboxes_abs):
        width, height = image.size
        crop, _ = model.model.roi(
            pixel_value.to(device),
            bbox,
            height,
            width,
        )
        _, spatial = model.model.ncs_encoder(crop, return_spatial=True)
        spatial_features.append(spatial)
    return torch.cat(spatial_features, dim=0)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = find_checkpoint()
    model = SlotGroundedTrainer.load_from_checkpoint(
        checkpoint.as_posix(),
        map_location="cpu",
    )
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    collator = EncoderDecoderCollate(model.tokenizer)
    _, _, test_dataset = build_dataset()
    sample_count = min(NUM_SAMPLES, len(test_dataset))
    loader = DataLoader(
        test_dataset.select(range(sample_count)),
        batch_size=1,
        shuffle=False,
        collate_fn=collator,
    )

    print(f"Loaded checkpoint: {checkpoint}")

    with torch.no_grad():
        for sample_idx, batch in enumerate(loader):
            input_ids = batch["prompt_input_ids"].to(device)
            attention_mask = batch["prompt_attention_mask"].to(device)
            visual_outputs, visual_tokens = build_prompt_visual_tokens(model, batch, device)

            text_embeds = model.model.decoder.get_input_embeddings()(input_ids)
            visual_tokens = visual_tokens.to(dtype=text_embeds.dtype)
            inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)
            visual_attention = torch.ones(
                (1, visual_tokens.shape[1]),
                device=device,
                dtype=attention_mask.dtype,
            )
            full_attention_mask = torch.cat([visual_attention, attention_mask], dim=1)

            generated = model.model.decoder.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=model.tokenizer.pad_token_id,
                eos_token_id=model.tokenizer.eos_token_id,
            )

            generated_text = model.tokenizer.decode(
                generated[0],
                skip_special_tokens=False,
            )

            if (
                generated.shape[1] >= input_ids.shape[1]
                and torch.equal(generated[:, :input_ids.shape[1]].to(input_ids.device), input_ids)
            ):
                full_text_ids = generated.to(device)
            else:
                full_text_ids = torch.cat([input_ids, generated.to(device)], dim=1)

            reg_outputs = rerun_decoder_for_generated(
                model=model,
                full_text_ids=full_text_ids,
                visual_tokens=visual_tokens,
                device=device,
            )
            visual_len = visual_tokens.shape[1]

            obj_hidden = hidden_for_token(model, OBJ_SLOT, full_text_ids, reg_outputs, visual_len)
            bbox_hidden = hidden_for_token(model, BBOX_SLOT, full_text_ids, reg_outputs, visual_len)
            center_hidden = hidden_for_token(model, CENTER_SLOT, full_text_ids, reg_outputs, visual_len)
            reg_hidden = hidden_for_token(model, REG_SLOT, full_text_ids, reg_outputs, visual_len)
            seg_hidden = hidden_for_token(model, SEG_TOKEN, full_text_ids, reg_outputs, visual_len)

            object_names = []
            class_ids = []
            if obj_hidden is not None:
                obj_logits = model.model.object_head(
                    obj_hidden.to(dtype=model.model.object_head.weight.dtype)
                )
                class_ids = obj_logits.argmax(dim=-1).detach().cpu().tolist()
                object_names = [CLASS_NAMES.get(class_id, f"class_{class_id}") for class_id in class_ids]

            predicted_bboxes = []
            if bbox_hidden is not None:
                bbox_norm = model.model.slot_bbox_to_xyxy(bbox_hidden).detach().cpu().tolist()
                for box_norm in bbox_norm:
                    height, width = batch["sizes"][0]
                    predicted_bboxes.append(norm_box_to_abs(box_norm, width, height))

            predicted_centers = []
            if center_hidden is not None:
                center_norm = model.model.slot_center_to_xy(center_hidden).detach().cpu().tolist()
                for center in center_norm:
                    height, width = batch["sizes"][0]
                    predicted_centers.append([center[0] * width, center[1] * height])
            else:
                predicted_centers = [
                    [
                        (bbox[0] + bbox[2]) / 2,
                        (bbox[1] + bbox[3]) / 2,
                    ]
                    for bbox in predicted_bboxes
                ]

            predicted_numbers = []
            if reg_hidden is not None:
                values = (
                    model.model.numeric_head(
                        reg_hidden.to(dtype=model.model.numeric_head[0].weight.dtype)
                    )
                    .squeeze(-1)
                    .detach()
                    .cpu()
                    .tolist()
                )
                predicted_numbers = [value * GENERIC_NUM_SCALE for value in values]

            overlay_paths = []
            if seg_hidden is not None and predicted_bboxes:
                usable_count = min(seg_hidden.shape[0], len(predicted_bboxes), len(batch["images"]))
                overlay_bboxes = predicted_bboxes[:usable_count]
                spatial_features = crop_spatial_features_from_predicted_boxes(
                    model=model,
                    batch=batch,
                    bboxes_abs=overlay_bboxes,
                    device=device,
                )
                mask_logits = model.model.mask_decoder(
                    seg_hidden[:usable_count],
                    spatial_features[:usable_count],
                )
                for mask_idx in range(usable_count):
                    image = batch["images"][mask_idx]
                    width, height = image.size
                    class_id = class_ids[min(mask_idx, len(class_ids) - 1)] if class_ids else 1
                    full_mask = roi_mask_to_full_canvas(
                        mask_logits[mask_idx],
                        bbox=overlay_bboxes[mask_idx],
                        width=width,
                        height=height,
                    )
                    overlay_path = OUTPUT_DIR / f"sample_{sample_idx:03d}_image_{mask_idx:03d}_overlay.png"
                    save_mask_overlay(
                        image=image,
                        full_mask=full_mask,
                        class_id=class_id,
                        output_path=overlay_path,
                    )
                    overlay_paths.append(overlay_path.as_posix())

            rendered_text = render_slots(
                strip_prompt_echo(generated_text),
                object_names=object_names,
                bboxes=predicted_bboxes,
                centers=predicted_centers,
                numbers=predicted_numbers,
            )
            valid_target = batch["labels"][0] != -100
            target_text = model.tokenizer.decode(
                batch["labels"][0][valid_target],
                skip_special_tokens=False,
            )

            print("=" * 80)
            print(f"sample: {sample_idx}")
            print("TARGET:")
            print(target_text.strip())
            print(f"PREDICTED CLASSES: {list(zip(class_ids, object_names))}")
            print(f"PREDICTED BBOXES: {[format_bbox(bbox) for bbox in predicted_bboxes]}")
            if predicted_numbers:
                print(f"PREDICTED NUMS: {[round(value, 4) for value in predicted_numbers]}")
            print(f"OVERLAYS: {overlay_paths}")
            print("GENERATED:")
            print(generated_text.strip())
            print("RENDERED:")
            print(rendered_text)


if __name__ == "__main__":
    main()
