import torch
from utils.region_encoder import RegionEncoder,NcsEncoder
from utils.global_encoder import GlobalEncoder
import torch.nn as nn


def norm_to_abs(boxes, W, H):
    out = boxes.clone()
    out[..., [0, 2]] *= W
    out[..., [1, 3]] *= H
    return out

class BBoxHead(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, 4),
        )

    def forward(self, x):
        pred = self.net(x).sigmoid()

        cx = pred[..., 0]
        cy = pred[..., 1]
        w = pred[..., 2]
        h = pred[..., 3]

        x1 = (cx - 0.5 * w).clamp(0, 1)
        y1 = (cy - 0.5 * h).clamp(0, 1)
        x2 = (cx + 0.5 * w).clamp(0, 1)
        y2 = (cy + 0.5 * h).clamp(0, 1)

        return torch.stack([x1, y1, x2, y2], dim=-1)


class ProposalHead(nn.Module):
    def __init__(self, hidden_size=768, num_classes=4):
        super().__init__()
        self.class_head = nn.Linear(hidden_size, num_classes + 1)
        self.objectness_head = nn.Linear(hidden_size, 1)
        self.bbox_head = BBoxHead(hidden_size)

    def forward(self, tile_tokens):
        return {
            "class_logits": self.class_head(tile_tokens),
            "objectness_logits": self.objectness_head(tile_tokens).squeeze(-1),
            "boxes": self.bbox_head(tile_tokens),
        }

class DualEncoder(nn.Module):
    def __init__(self,is_train=False):
        super().__init__()
        self.bbox_mlp = nn.Sequential(
            nn.Linear(4, 256),
            nn.GELU(),
            nn.Linear(256, 768),
        )
        self.is_train = is_train
        self.re = RegionEncoder(output_size=224, spatial_scale=1, sampling_ratio=2)  # no scaling from exact image
        self.ge = GlobalEncoder(output_size=224, overlap=112)
        self.ncs_enc = NcsEncoder()
        self.proposal_head = ProposalHead(768)

    def forward(self,pixel_values,bbox=None,H=None,W=None):
        global_output = self.ge(pixel_values,H,W) # tiling
        global_feature = torch.cat([tile['feature'] for tile in global_output],dim=0)
        bbox_global_tiles = torch.tensor(
            [tile['bbox_norm'] for tile in global_output],
            dtype=global_feature.dtype,
            device=global_feature.device,
        )
        global_tile_pos = self.bbox_mlp(bbox_global_tiles)

        global_tiles = global_feature + global_tile_pos
        proposal = self.proposal_head(global_tiles)

        global_cls_pred = proposal['class_logits']
        roi_bbox = norm_to_abs(proposal['boxes'],W,H)

        if self.is_train:
            bboxes = bbox
        else:
            bboxes = roi_bbox


        cropped_image,bbox_norm = self.re(pixel_values,bboxes,H,W)
        ncs_feature = self.ncs_enc(cropped_image) # roi_align for precision cropping and pass to ncs
        return {
            "global_tiles": global_tiles,
            "proposal": proposal,
            "roi_bbox": roi_bbox,
            "region_bbox": bboxes,
            "cropped_image": cropped_image,
            "region_bbox_norm": bbox_norm,
            "region_feature": ncs_feature,
        }

if __name__ == "__main__":
    from data import bcx_process
    from datasets import load_dataset, Dataset
    # inspecting
    dataset = load_dataset("thirdExec/synthetic-seismic-vlm")
    dataset = dataset['train']

    rows = []
    for example in dataset:
        processed = bcx_process(example)
        if isinstance(processed, list):
            rows.extend(processed)
        else:
            rows.append(processed)
    temped_dataset = Dataset.from_list(rows)
    print(temped_dataset)


    for idx,example in enumerate(temped_dataset):

        # print("="*100)
        # print(f"image at index{idx}")
        # print(f"total cropped image:{len(cropped_image)}")
        # for idx,data in enumerate(global_feature):
        #     print(f"\t global feature at index {idx} at bbox {data['bbox_abs']} to bbox_norm: {data['bbox_norm']} size:{data['feature'].shape}")
        # print(f"\t\t local feature at bbox {example['bbox']} to  bbox_norm {bbox_norm} size: {ncs_feature.shape}")

        tiles_token = global_feature
