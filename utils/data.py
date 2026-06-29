import json
import re
from PIL import ImageOps
import torch
from torchvision.transforms.functional import pil_to_tensor

OBJ_SLOT = "<OBJ_SLOT>"
BBOX_SLOT = "<BBOX_SLOT>"
CENTER_SLOT = "<CENTER_SLOT>"
REG_SLOT = "<REG_SLOT>"
SEG_TOKEN = "<SEG>"
SPECIAL_TOKENS = [OBJ_SLOT, BBOX_SLOT, CENTER_SLOT, REG_SLOT, SEG_TOKEN]
NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")
GENERIC_NUM_SCALE = 10.0


def simple_tiling(img,H,W,tile_size,stride):

    if W < tile_size:
        xs = [0]
    else:
        xs = list(range(0,max(W-tile_size+1,1),stride))
        if not xs or xs[-1] != W-tile_size:
            xs.append(max(W - tile_size, 0))

    if H < tile_size:
        ys = [0]
    else:
        ys = list(range(0,max(H-tile_size+1,1),stride))

        if not ys or ys[-1] != H-tile_size:
            ys.append(max(H - tile_size, 0))

    tiles = []
    for y1 in ys:
        for x1 in xs:
            x2 = min(x1 + tile_size, W)
            y2 = min(y1 + tile_size, H)

            crop = img.crop((x1, y1, x2, y2))

            pad_w = tile_size - crop.size[0]
            pad_h = tile_size - crop.size[1]

            padded = ImageOps.expand(
                crop,
                border=(0, 0, pad_w, pad_h),
            )

            tiles.append({
                "image": padded,
                "bbox_abs": [x1, y1, x2, y2],
                "bbox_norm": [
                    x1 / W,
                    y1 / H,
                    x2 / W,
                    y2 / H,
                ],
                "pad": [0, 0, pad_w, pad_h],
                "orig_size": [W, H],
            })

    return tiles


def extract_regions(text):
    return re.findall(r"<region>.*?</region>", text or "", flags=re.DOTALL)


def extract_numbers(value):
    return [float(match.group(0)) for match in NUMBER_RE.finditer(value or "")]


def replace_numeric_tags(text):
    numeric_targets = []

    def replace_match(match):
        tag = match.group(1)
        values = extract_numbers(match.group(2))
        numeric_targets.extend(values)
        if not values:
            return match.group(0)
        slots = ", ".join(REG_SLOT for _ in values)
        return f"<{tag}>{slots}</{tag}>"

    text = re.sub(
        r"<(NUM|nums)>(.*?)</\1>",
        replace_match,
        text or "",
        flags=re.DOTALL,
    )
    return text, numeric_targets


def clean_evidence_text(text):
    text = re.sub(r"</?region>", "", text or "", flags=re.DOTALL)
    text = re.sub(re.escape(SEG_TOKEN), "", text, flags=re.DOTALL)
    text = text.strip()
    return text


def replace_structured_values(text):
    text = re.sub(r"<bbox>.*?</bbox>", f"<bbox>{BBOX_SLOT}</bbox>", text or "", flags=re.DOTALL)
    text = re.sub(r"<center>.*?</center>", f"<center>{CENTER_SLOT}</center>", text, flags=re.DOTALL)
    return replace_numeric_tags(text)


def build_region_target(region, evidence_text):
    body = clean_evidence_text(evidence_text)
    body, numeric_targets = replace_structured_values(body)
    if not body:
        body = f"The region contains a visible {region.get('object_type', 'geological')} feature."
    if BBOX_SLOT not in body:
        body = f"{body}\n<bbox>{BBOX_SLOT}</bbox>"

    target = (
        "<region>\n"
        f"<object>{OBJ_SLOT}</object>\n"
        f"<class_id>{OBJ_SLOT}</class_id>\n"
        f"{body}\n"
        f"{SEG_TOKEN}\n"
        "</region>"
    )
    return target, numeric_targets


def build_row_target(regions, evidence_text, answer_text):
    region_blocks = extract_regions(evidence_text)
    rendered_regions = []
    numeric_targets = []
    class_slot_targets = []
    bbox_slot_targets = []
    bbox_slot_region_indices = []
    center_slot_targets = []
    center_slot_region_indices = []

    for idx, region in enumerate(regions):
        evidence_block = region_blocks[idx] if idx < len(region_blocks) else ""
        rendered_region, region_nums = build_region_target(region, evidence_block)
        rendered_regions.append(rendered_region)
        numeric_targets.extend(region_nums)
        class_slot_targets.extend([region["class_id"]] * rendered_region.count(OBJ_SLOT))
        bbox_slot_targets.extend([region["bbox"]] * rendered_region.count(BBOX_SLOT))
        bbox_slot_region_indices.extend([idx] * rendered_region.count(BBOX_SLOT))
        center = region.get("center")
        center_slot_targets.extend([center] * rendered_region.count(CENTER_SLOT))
        center_slot_region_indices.extend([idx] * rendered_region.count(CENTER_SLOT))

    answer, answer_nums = replace_structured_values(answer_text)
    if regions:
        answer_bbox_count = answer.count(BBOX_SLOT)
        if answer_bbox_count:
            bbox_slot_targets.extend([regions[0]["bbox"]] * answer_bbox_count)
            bbox_slot_region_indices.extend([0] * answer_bbox_count)

        answer_center_count = answer.count(CENTER_SLOT)
        center = regions[0].get("center")
        if answer_center_count and center is not None:
            center_slot_targets.extend([center] * answer_center_count)
            center_slot_region_indices.extend([0] * answer_center_count)

    numeric_targets.extend(answer_nums)
    return (
        "\n".join(rendered_regions + [answer.strip()]),
        numeric_targets,
        class_slot_targets,
        bbox_slot_targets,
        bbox_slot_region_indices,
        center_slot_targets,
        center_slot_region_indices,
    )

def bcx_process(example):
    """
    this is format for vlm to lean to extract evidences from image, by using regions data 1 image contain multiple evidences all difference depends on question
    :param example:
    :return:
    """
    images = example['images']
    masks = example['masks']
    regions = example['regions']
    regions = json.loads(regions)

    # map image and region together
    info = []

    for data in regions:

        # map index individual
        image_idx = data['image_idx']
        mask_idx = data['mask_idx']
        assert(image_idx == mask_idx) # check if it same indexes from original dataset
        bbox = data['bbox']

        label = data['class_id']

        # this will be broad low level evidence-image mapping since question and answer can mislead 1-Many problem.

        W,H = images[image_idx].size
        info.append({
            "i":images[image_idx],
            "m":masks[mask_idx],
            "label":label,
            "bbox":bbox,
            "H":H,
            "W":W
        })
    return info

def encoder_collator(examples):
    return {
          "images":[ex['i'] for ex in examples],
          "pixel_values": [pil_to_tensor(ex["i"].convert("RGB")).float().unsqueeze(0) / 255.0 for ex in examples],  # pixel_values
          "tiles": [simple_tiling(ex["i"],ex["H"],ex["W"],224,112) for ex in examples],
          "boxes": [ex["bbox"] for ex in examples],
          "label": torch.tensor([ex["label"] for ex in examples], dtype=torch.long),
          "sizes": [(ex["H"], ex["W"]) for ex in examples],
      }

def encoder_decoder_process(example):
    """
    this is format for vlm to lean to extract evidences from image, by using regions data 1 image contain multiple evidences all difference depends on question
    :param example:
    :return:
    """
    images = example['images']
    masks = example['masks']
    regions = example['regions']
    regions = json.loads(regions)
    evidence_str = example['evidence']
    answer_str = example.get("answer", "")
    instruction = example.get("instruction", "")
    question = example.get("question", "")

    (
        target_text,
        numeric_targets,
        class_slot_targets,
        bbox_slot_targets,
        bbox_slot_region_indices,
        center_slot_targets,
        center_slot_region_indices,
    ) = build_row_target(regions, evidence_str, answer_str)
    user_text = f"{instruction}\n{question}".strip()

    row_images = []
    row_masks = []
    boxes = []
    labels = []
    sizes = []

    for data in regions:
        image_idx = data['image_idx']
        mask_idx = data['mask_idx']
        assert(image_idx == mask_idx) # check if it same indexes from original dataset
        W, H = images[image_idx].size
        row_images.append(images[image_idx])
        row_masks.append(masks[mask_idx])
        boxes.append(data["bbox"])
        labels.append(data["class_id"])
        sizes.append((H, W))

    return {
        "i": row_images,
        "m": row_masks,
        "label": labels,
        "bbox": boxes,
        "numeric_targets": numeric_targets,
        "class_slot_targets": class_slot_targets,
        "bbox_slot_targets": bbox_slot_targets,
        "bbox_slot_region_indices": bbox_slot_region_indices,
        "center_slot_targets": center_slot_targets,
        "center_slot_region_indices": center_slot_region_indices,
        "sizes": sizes,
        "message": [
            {
                "role": "user",
                "content": [{"type": "text", "text": user_text}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target_text}],
            },
        ],
    }


class EncoderDecoderCollate:
    def __init__(self,tokenizer):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _text_messages(self, messages):
        cleaned = []
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                text = "".join(
                    item.get("text", "")
                    for item in content
                    if item.get("type") == "text"
                )
            else:
                text = content
            cleaned.append({"role": message["role"], "content": text})
        return cleaned

    def __call__(self, examples):
        messages = [self._text_messages(ex["message"]) for ex in examples]
        texts = [
            self.tokenizer.apply_chat_template(message,tokenize=False,add_generation_prompt=False) for message in messages]
        prompt_text = [
            self.tokenizer.apply_chat_template(message[:1],tokenize=False,add_generation_prompt=True) for message in messages
        ]

        text_batch = self.tokenizer(texts,padding=True,return_tensors="pt")
        prompt_batch = self.tokenizer(prompt_text,padding=True,return_tensors="pt")

        labels = text_batch["input_ids"].clone()

        prompt_lens = prompt_batch["attention_mask"].sum(dim=1)
        for i, prompt_len in enumerate(prompt_lens):
            labels[i, :prompt_len] = -100

        labels[labels == self.tokenizer.pad_token_id] = -100
        slot_token_ids = {
            "obj": self.tokenizer.convert_tokens_to_ids(OBJ_SLOT),
            "bbox": self.tokenizer.convert_tokens_to_ids(BBOX_SLOT),
            "center": self.tokenizer.convert_tokens_to_ids(CENTER_SLOT),
            "reg": self.tokenizer.convert_tokens_to_ids(REG_SLOT),
            "seg": self.tokenizer.convert_tokens_to_ids(SEG_TOKEN),
        }
        images = []
        pixel_values = []
        mask_values = []
        tiles = []
        boxes = []
        labels_per_region = []
        sizes = []
        row_image_counts = []
        numeric_targets = []
        class_slot_targets = []
        bbox_slot_targets = []
        center_slot_targets = []

        for example in examples:
            row_image_counts.append(len(example["i"]))
            row_start = len(boxes)
            images.extend(example["i"])
            boxes.extend(example["bbox"])
            labels_per_region.extend(example["label"])
            sizes.extend(example["sizes"])
            class_slot_targets.extend(example["class_slot_targets"])
            numeric_targets.extend(example["numeric_targets"])

            for img, mask, size in zip(example["i"], example["m"], example["sizes"]):
                pixel_values.append(pil_to_tensor(img.convert("RGB")).float().unsqueeze(0) / 255.0)
                mask_values.append(pil_to_tensor(mask.convert("L")).float().unsqueeze(0) / 255.0)
                tiles.append(simple_tiling(img, size[0], size[1], 224, 112))

            for box, local_region_index in zip(example["bbox_slot_targets"], example["bbox_slot_region_indices"]):
                size_index = row_start + local_region_index
                height, width = sizes[size_index]
                x1, y1, x2, y2 = box
                bbox_slot_targets.append([x1 / width, y1 / height, x2 / width, y2 / height])

            for center, local_region_index in zip(example["center_slot_targets"], example["center_slot_region_indices"]):
                if center is None:
                    continue
                size_index = row_start + local_region_index
                height, width = sizes[size_index]
                cx, cy = center
                center_slot_targets.append([cx / width, cy / height])

        return {
            "images": images,
            "pixel_values": pixel_values,
            "mask_values": mask_values,
            "tiles": tiles,
            "boxes": boxes,
            "label": torch.tensor(labels_per_region, dtype=torch.long),
            "numeric_targets": torch.tensor(numeric_targets, dtype=torch.float32) / GENERIC_NUM_SCALE,
            "class_slot_targets": torch.tensor(class_slot_targets, dtype=torch.long),
            "bbox_slot_targets": torch.tensor(bbox_slot_targets, dtype=torch.float32),
            "center_slot_targets": torch.tensor(center_slot_targets, dtype=torch.float32),
            "row_image_counts": row_image_counts,
            "slot_token_ids": slot_token_ids,
            "sizes": sizes,
            "input_ids": text_batch["input_ids"],
            "attention_mask": text_batch["attention_mask"],
            "prompt_input_ids": prompt_batch["input_ids"],
            "prompt_attention_mask": prompt_batch["attention_mask"],
            "labels": labels,
        }
