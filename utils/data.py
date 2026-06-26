import json
import re
from torchvision.transforms.functional import pil_to_tensor
import torch

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
    evidence_str = example['evidence']

    # map image and region together
    info = []

    for data in regions:

        # map index individual
        image_idx = data['image_idx']
        mask_idx = data['mask_idx']
        assert(image_idx == mask_idx) # check if it same indexes from original dataset
        bbox = data['bbox']

        object = data['object_id']

        # this will be broad low level evidence-image mapping since question and answer can mislead 1-Many problem.


        info.append({
            "i":images[image_idx],
            "m":masks[mask_idx],
            "object":object,
            "bbox":bbox,
            "H":images[image_idx].shape[0],
            "W":images[image_idx].shape[1],
        })
    return info



