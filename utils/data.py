import json
import re
from PIL import ImageOps
import torch
from torchvision.transforms.functional import pil_to_tensor

SPECIAL_TOKENS = ["<OBJ>", "<BBOX>", "<NUM>", "<REG>", "<SEG>"]


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


def strip_region_tags(text):
    text = re.sub(r"</?region>", "", text or "", flags=re.DOTALL)
    text = re.sub(r"<SEG>", "", text, flags=re.DOTALL)
    text = re.sub(r"<bbox>.*?</bbox>", "", text, flags=re.DOTALL)
    text = re.sub(r"<center>.*?</center>", "<nums><NUM></nums>", text, flags=re.DOTALL)
    text = text.strip()
    return text


def replace_structured_values(text):
    text = re.sub(r"<bbox>.*?</bbox>", "<bbox><BBOX></bbox>", text or "", flags=re.DOTALL)
    text = re.sub(r"<center>.*?</center>", "<nums><NUM></nums>", text, flags=re.DOTALL)
    return text


def build_region_target(region, evidence_text, answer_text):
    evidence = strip_region_tags(evidence_text)
    if not evidence:
        evidence = f"The region contains a visible {region.get('object_type', 'geological')} feature."

    answer = replace_structured_values(answer_text).strip()
    return (
        "<region>\n"
        "<object><OBJ></object>\n"
        "<class_id><OBJ></class_id>\n"
        f"<evidence>{evidence}</evidence>\n"
        "<bbox><BBOX></bbox>\n"
        "<SEG>\n"
        "</region>\n"
        f"{answer}"
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

    stage1_instruction = (
        "Identify the seismic feature in the image and output one grounded "
        "<region>...</region> block with object, class_id, evidence, bbox, and <SEG>, "
        "then output the final answer. Use <OBJ>, <BBOX>, and <NUM> placeholders "
        "for structured values predicted by the grounding heads."
    )
    # map image and region together
    info = []

    for data in regions:
        conversations = []
        user = {
            "role": "user",
            "content": [],
        }
        assistant = {
            "role": "assistant",
            "content": [],
        }
        # map index individual
        image_idx = data['image_idx']
        mask_idx = data['mask_idx']
        assert(image_idx == mask_idx) # check if it same indexes from original dataset
        region_idx = data['region_idx']
        evidence = extract_regions(evidence_str)
        if evidence and len(evidence) > region_idx:
            evidence_per_region = build_region_target(data, evidence[region_idx], answer_str)
        else:
            evidence_per_region = build_region_target(data, "", answer_str)

        # this will be broad low level evidence-image mapping since question and answer can mislead 1-Many problem.

        user['content'] = [
            # {"type": "image"},
            {"type": "text","text": stage1_instruction}
        ]
        # evidence as answer
        assistant['content'] = [{"type": "text", "text": evidence_per_region}]

        conversations.append(user)
        conversations.append(assistant)

        bbox = data['bbox']

        label = data['class_id']

        W, H = images[image_idx].size

        info.append({
            "i":images[image_idx],
            "m":masks[mask_idx],
            "label": label,
            "bbox": bbox,
            "H": H,
            "W": W,
            "message":conversations,
        })
    return info


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

        return {
            "images": [ex['i'] for ex in examples],
            "pixel_values": [pil_to_tensor(ex["i"].convert("RGB")).float().unsqueeze(0) / 255.0 for ex in examples],
            "mask_values": [pil_to_tensor(ex["m"].convert("L")).float().unsqueeze(0) / 255.0 for ex in examples],
            "tiles": [simple_tiling(ex["i"], ex["H"], ex["W"], 224, 112) for ex in examples],
            "boxes": [ex["bbox"] for ex in examples],
            "label": torch.tensor([ex["label"] for ex in examples], dtype=torch.long),
            "sizes": [(ex["H"], ex["W"]) for ex in examples],
            "input_ids": text_batch["input_ids"],
            "attention_mask": text_batch["attention_mask"],
            "prompt_input_ids": prompt_batch["input_ids"],
            "prompt_attention_mask": prompt_batch["attention_mask"],
            "labels": labels,
        }
