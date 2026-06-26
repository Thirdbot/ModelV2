from torch._dynamo.variables import torch
from torchvision.ops import roi_align
import torch
from torchvision.transforms.functional import pil_to_tensor
from transformers import ViTModel


class RegionEncoder:
    def __init__(self,output_size,spatial_scale,sampling_ratio):
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
    def __call__(self, img,bbox):
        img = img.convert("RGB")
        W,H = img.size
        tensor_image = pil_to_tensor(img).float().unsqueeze(0) / 255.0
        x1, y1, x2, y2 = bbox
        tensor_bbox = torch.tensor(
            [[0, x1, y1, x2, y2]],
            dtype=torch.float32,
        )

        bbox_norm = [
            x1 / W,
            y1 / H,
            x2 / W,
            y2 / H,
            ((x1 + x2) / 2) / W,
            ((y1 + y2) / 2) / H,
            ((x2 - x1) * (y2 - y1)) / (W * H),
        ]

        return (roi_align(input=tensor_image,boxes=tensor_bbox,output_size=self.output_size,
                         spatial_scale=self.spatial_scale,sampling_ratio=self.sampling_ratio),
                bbox_norm)


class NcsEncoder:
    def __init__(self):
        self.model = ViTModel.from_pretrained("NorskRegnesentralSTI/NCS-v1-2d-base",
                                              add_pooling_layer=False)
    def __call__(self, img):
        with torch.no_grad():
            output = self.model(pixel_values=img)
            cls_features = output.last_hidden_state[:, 0, :]  # shape: (B, 768)
            return cls_features
