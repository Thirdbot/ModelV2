"""
use clip with tiles image for global encoder
"""
import torch
from transformers import AutoProcessor, CLIPVisionModel
import torch.nn as nn

class GlobalEncoder(nn.Module):
    """
    2 jobs
    1: receiving and arbitrary size image with tiles
    2: output feature and bbox prediction
    """
    def __init__(self,output_size=224,overlap=112):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.output_size = output_size
        self.overlap = overlap

    def forward(self, tiles,H,W):
        device = next(self.model.parameters()).device
        output_tiles = []
        for tile in tiles:

            inputs = self.processor(
                    images=tile["image"],
                    return_tensors="pt",
                )
            inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.no_grad():
                output = self.model(**inputs)

            cls_features = output.last_hidden_state[:, 0, :]
            bbox = tile['bbox_abs']
            bbox_norm = tile['bbox_norm']

            output_tiles.append({
                "image": tile['image'],
                "feature":cls_features,
                "bbox_abs": bbox,
                "bbox_norm": bbox_norm,
            })
        return output_tiles
