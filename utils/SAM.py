from transformers import SamModel,SamConfig,SamProcessor
import json
import torch

def preprocessor_fn(example):
    images = example['images']
    masks = example['masks']
    region_str = example['regions']
    regions = json.loads(region_str)

    bboxes = []
    for region in regions:
        bbox = region['bbox']
        bboxes.append(bbox)
    mapped = {
        "i":list(images) if isinstance(images, list) else images,
        "m":list(masks) if isinstance(masks, list) else masks,
        "bboxes": list(bboxes),
    }
    return mapped

if __name__ == "__main__":
    from datasets import load_dataset
    samModel = SamModel.from_pretrained("facebook/sam-vit-base")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    dataset = load_dataset("thirdExec/synthetic-seismic-vlm")
    dataset = dataset["train"]
    dataset = dataset.map(preprocessor_fn,remove_columns=dataset.column_names)
    print(dataset[0])
    for data in dataset:
        image = data['i']
        mask = data['m']
        boxes = data['bboxes']
        inputs = processor(images=image, input_boxes=[boxes],return_tensors="pt")
        with torch.no_grad():
            outputs = samModel(**inputs)

        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs.original_sizes.cpu(), inputs.reshaped_input_sizes.cpu()
        )
        object_mask = masks[0][0][0]
        print(object_mask)
