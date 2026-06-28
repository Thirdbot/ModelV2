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
    def __init__(self, hidden_size=768, num_classes=6):
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


class MaskHead(nn.Module):
    def __init__(self, hidden_size=768, base_channels=128, start_size=14):
        super().__init__()
        self.start_size = start_size
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, base_channels * start_size * start_size),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels, 64, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
        )

    def forward(self, region_features):
        x = self.fc(region_features)
        x = x.view(region_features.shape[0], -1, self.start_size, self.start_size)
        return self.decoder(x)


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
        self.mask_head = MaskHead(768)

    def forward_one(self,pixel_values,tiles,bbox=None,class_ids=None,H=None,W=None):
        device = next(self.parameters()).device
        pixel_values = pixel_values.to(device)

        global_output = self.ge(tiles,H,W) # tiling
        global_feature = torch.cat([tile['feature'] for tile in global_output],dim=0)
        bbox_global_tiles = torch.tensor(
            [tile['bbox_norm'] for tile in global_output],
            dtype=global_feature.dtype,
            device=global_feature.device,
        )
        global_tile_pos = self.bbox_mlp(bbox_global_tiles)

        global_tiles = global_feature + global_tile_pos
        proposal = self.proposal_head(global_tiles)

        roi_bbox = norm_to_abs(proposal['boxes'],W,H)

        if self.is_train:
            bboxes = bbox
            region_class_ids = torch.as_tensor(class_ids, dtype=torch.long, device=device)
            if region_class_ids.ndim == 0:
                region_class_ids = region_class_ids.unsqueeze(0)
        else:
            bboxes = roi_bbox
            region_class_ids = proposal["class_logits"].argmax(dim=-1)


        cropped_image,bbox_norm = self.re(pixel_values,bboxes,H,W)
        ncs_feature = self.ncs_enc(cropped_image) # roi_align for precision cropping and pass to ncs
        mask_logits = self.mask_head(ncs_feature)

        return {
            "global_tiles": global_tiles,
            "tile_bbox_abs": torch.tensor(
                [tile["bbox_abs"] for tile in global_output],
                dtype=global_feature.dtype,
                device=global_feature.device,
            ),
            "tile_bbox_norm": bbox_global_tiles,
            "proposal": proposal,
            "roi_bbox": roi_bbox,
            "region_bbox": bboxes,
            "cropped_image": cropped_image,
            "region_bbox_norm": bbox_norm,
            "region_class_ids": region_class_ids,
            "region_feature": ncs_feature,
            "mask_logits": mask_logits,
        }

    def forward(self,pixel_values,tiles,bbox=None,class_ids=None,H=None,W=None):
        if isinstance(pixel_values, list):
            outputs = []
            for idx, pixel_value in enumerate(pixel_values):
                sample_bbox = bbox[idx] if bbox is not None else None
                sample_class_ids = class_ids[idx] if class_ids is not None else None
                sample_h, sample_w = H[idx], W[idx]
                outputs.append(
                    self.forward_one(
                        pixel_values=pixel_value,
                        tiles=tiles[idx],
                        bbox=sample_bbox,
                        class_ids=sample_class_ids,
                        H=sample_h,
                        W=sample_w,
                    )
                )
            return outputs

        return self.forward_one(
            pixel_values=pixel_values,
            tiles=tiles,
            bbox=bbox,
            class_ids=class_ids,
            H=H,
            W=W,
        )

if __name__ == "__main__":
    from utils.data import bcx_process
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
    saved_use_dataset = temped_dataset.train_test_split(0.2)
    saved_test_dataset = saved_use_dataset['test'].train_test_split(0.5)

    train_dataset = saved_use_dataset['train']
    eval_dataset = saved_test_dataset['test']
    test_dataset = saved_test_dataset['test']


    print(f"train: {len(train_dataset)}\n"
          f"test: {len(test_dataset)}\n"
          f"eval: {len(eval_dataset)}\n"
          f"example: {train_dataset[0]}")
