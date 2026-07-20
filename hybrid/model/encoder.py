"""Frozen NCS vision encoder + tile stitching.

- `NcsEncoder` — the domain-pretrained NCS-v1 seismic ViT, used frozen (optional
  last-N-block unfreeze). Returns CLS features and/or the patch-grid spatial map.
- `stitch`     — run the encoder over overlapping tiles of a full image and
  stitch the per-tile spatial features into one full-resolution feature map.
  This is the dense feature map the detector reads its facts from.
"""
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from transformers import ViTModel

from hybrid.data.config import PATCH, TILE_GRID, TILE_SIZE, TILE_STRIDE
from hybrid.data.dataset import simple_tiling

device = torch.device("cuda")


class NcsEncoder(nn.Module):
    def __init__(self, trainable_blocks=0):
        super().__init__()
        self.model = ViTModel.from_pretrained("NorskRegnesentralSTI/NCS-v1-2d-base",
                                              add_pooling_layer=False)
        mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        # Partial unfreeze: make the LAST `trainable_blocks` transformer layers
        # (+ the final layernorm) trainable. Masked-pretrained ViT features are
        # weak when frozen, but full fine-tuning overfits on small data — the last
        # few blocks are the low-cost middle ground.
        self.trainable_blocks = int(trainable_blocks)
        if self.trainable_blocks > 0:
            for blk in self.model.encoder.layer[-self.trainable_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True
            if getattr(self.model, "layernorm", None) is not None:
                for p in self.model.layernorm.parameters():
                    p.requires_grad = True
            # Gradient checkpointing: recompute ViT activations in backward rather
            # than storing them per tile — the encoder-unfreeze memory would
            # otherwise OOM a ~6GB GPU on longer sequences. use_reentrant=False
            # tolerates the (grad-free) pixel input to the frozen early blocks.
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})

    def normalize(self, img):
        img = img.float()
        if img.max() > 2:
            img = img / 255.0
        return (img - self.image_mean.to(img.device)) / self.image_std.to(img.device)

    def forward(self, img, return_spatial=False):
        # Track gradients only when some block is unfrozen; otherwise no_grad to
        # save memory (frozen encoder needs no activations retained).
        grad_ctx = torch.enable_grad() if self.trainable_blocks > 0 else torch.no_grad()
        with grad_ctx:
            img = self.normalize(img)
            output = self.model(pixel_values=img)
            cls_features = output.last_hidden_state[:, 0, :]  # shape: (B, 768)
            if not return_spatial:
                return cls_features

            patch_tokens = output.last_hidden_state[:, 1:, :]
            grid_size = int(patch_tokens.shape[1] ** 0.5)
            if grid_size * grid_size != patch_tokens.shape[1]:
                return cls_features, None

            spatial_features = patch_tokens.transpose(1, 2).reshape(
                patch_tokens.shape[0],
                patch_tokens.shape[2],
                grid_size,
                grid_size,
            )
            return cls_features, spatial_features


def stitch(enc, path, black=False):
    """Run `enc` over overlapping tiles of the image at `path` and stitch the
    per-tile spatial features into one (768, fH, fW) map. Returns (map, (H, W))."""
    im = Image.open(path).convert("RGB")
    W, H = im.size
    if black:
        im = Image.new("RGB", (W, H), 0)
    fH, fW = max(1, H // PATCH), max(1, W // PATCH)
    accum = torch.zeros(768, fH, fW, device=device)
    cnt = torch.zeros(1, fH, fW, device=device)
    for t in simple_tiling(im, H, W, TILE_SIZE, TILE_STRIDE):
        x = pil_to_tensor(t["image"].convert("RGB")).float().unsqueeze(0).to(device) / 255.0
        _, sp = enc(x, return_spatial=True)
        sp = sp.squeeze(0)
        x1, y1, _, _ = t["bbox_abs"]
        fy, fx = int(round(y1 / PATCH)), int(round(x1 / PATCH))
        gy, gx = min(TILE_GRID, fH - fy), min(TILE_GRID, fW - fx)
        if gy <= 0 or gx <= 0:
            continue
        accum[:, fy:fy + gy, fx:fx + gx] += sp[:, :gy, :gx]
        cnt[:, fy:fy + gy, fx:fx + gx] += 1
    return accum / cnt.clamp_min(1.0), (H, W)
