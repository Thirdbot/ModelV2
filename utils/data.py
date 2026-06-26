from datasets import load_dataset,Dataset
import json
import re

def extract_regions(text):
    return re.findall(r"<region>.*?</region>", text)

def ie_process(example):
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

    stage1_instruction = (
        "Identify the seismic feature in the image and output one grounded "
        "<region>...</region> block with object, class_id, color, evidence, bbox, and <SEG>."
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
        bbox = data['bbox']
        evidence = extract_regions(evidence_str)
        if evidence and len(evidence) > region_idx:
            evidence_per_region = evidence[region_idx]
        else:
            evidence_per_region = ""

        # this will be broad low level evidence-image mapping since question and answer can mislead 1-Many problem.

        user['content'] = [
            {"type": "image"},{"type": "text","text": stage1_instruction}
        ]
        # evidence as answer
        assistant['content'] = [{"type": "text", "text": evidence_per_region}]

        conversations.append(user)
        conversations.append(assistant)

        info.append({
            "i":images[image_idx],
            "m":masks[mask_idx],
            "bbox":bbox,
            "message":conversations,
        })
    return info

if __name__ == "__main__":
    from utils.region_encoder import RegionEncoder,NcsEncoder
    # inspecting
    dataset = load_dataset("thirdExec/synthetic-seismic-vlm")
    dataset = dataset['train']

    rows = []
    for example in dataset:
        processed = ie_process(example)
        if isinstance(processed, list):
            rows.extend(processed)
        else:
            rows.append(processed)
    temped_dataset = Dataset.from_list(rows)
    print(temped_dataset)

    re = RegionEncoder(output_size=224,spatial_scale=1,sampling_ratio=2)
    ncs_enc = NcsEncoder()
    for example in temped_dataset:
        # print(example['i'],example['bbox'])
        cropped_image_feature,bbox_norm = re(example['i'],example['bbox'])
        ncs_feature = ncs_enc(cropped_image_feature)
        print(ncs_feature.shape)