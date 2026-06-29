import torch
from torchvision.transforms.functional import pil_to_tensor
from transformers import ViTModel
import torch.nn as nn

class GlobalEncoder(nn.Module):
    """
    2 jobs
    1: receiving and arbitrary size image with tiles
    2: output feature and bbox prediction
    """
    def __init__(self,output_size=224,overlap=112):
        super().__init__()
        self.model = ViTModel.from_pretrained(
            "NorskRegnesentralSTI/NCS-v1-2d-base",
            add_pooling_layer=False,
        )
        mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.output_size = output_size
        self.overlap = overlap

    def forward(self, tiles,H,W):
        device = next(self.model.parameters()).device
        output_tiles = []
        for tile in tiles:

            pixel_values = pil_to_tensor(tile["image"].convert("RGB")).float().unsqueeze(0).to(device) / 255.0
            pixel_values = (pixel_values - self.image_mean.to(device)) / self.image_std.to(device)

            with torch.no_grad():
                output = self.model(pixel_values=pixel_values)

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
