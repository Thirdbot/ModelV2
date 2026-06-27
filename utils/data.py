import json
import re
from PIL import ImageOps
import torch
from torchvision.transforms.functional import pil_to_tensor


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
    return re.findall(r"<region>.*?</region>", text)

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



