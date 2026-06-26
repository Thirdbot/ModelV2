from torchvision.ops import roi_align
import torch
import torch.nn as nn
from transformers import ViTModel
from torchvision.transforms.functional import pil_to_tensor

class RegionEncoder:
    def __init__(self,output_size,spatial_scale,sampling_ratio):
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
    def __call__(self, img,bbox,H,W):

        img = img.convert("RGB")
        tensor_image = pil_to_tensor(img).float().unsqueeze(0) / 255.0

        boxes = torch.as_tensor(bbox, dtype=torch.float32)
        if boxes.ndim == 1:
            boxes = boxes.unsqueeze(0)

        batch_index = torch.zeros((boxes.shape[0], 1), dtype=boxes.dtype, device=boxes.device)
        tensor_bbox = torch.cat([batch_index, boxes], dim=-1)

        x1, y1, x2, y2 = boxes.unbind(dim=-1)
        bbox_norm = torch.stack(
            [
                x1 / W,
                y1 / H,
                x2 / W,
                y2 / H,
                ((x1 + x2) / 2) / W,
                ((y1 + y2) / 2) / H,
                ((x2 - x1) * (y2 - y1)) / (W * H),
            ],
            dim=-1,
        )


        return (roi_align(input=tensor_image,boxes=tensor_bbox,output_size=self.output_size,
                         spatial_scale=self.spatial_scale,sampling_ratio=self.sampling_ratio),
                bbox_norm)

class NcsEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ViTModel.from_pretrained("NorskRegnesentralSTI/NCS-v1-2d-base",
                                              add_pooling_layer=False)
    def forward(self, img):
        with torch.no_grad():
            output = self.model(pixel_values=img)
            cls_features = output.last_hidden_state[:, 0, :]  # shape: (B, 768)
            return cls_features
