"""
use clip with tiles image for global encoder
"""
import torch
from transformers import AutoProcessor, CLIPVisionModel
from PIL import ImageOps
import torch.nn as nn

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
        self.output_size = output_size
        self.overlap = overlap

    def forward(self, img,H,W):
        tiles = simple_tiling(img,H,W,self.output_size, self.overlap)
        output_tiles = []
        for tile in tiles:

            inputs = self.processor(
                    images=tile["image"],
                    return_tensors="pt",
                )

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
