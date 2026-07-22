"""<SEG> mask decoder — content-prompted masks from the LM hidden state.

The LM emits <SEG>; its hidden state at that position is the CONTENT PROMPT (which
object to segment). The decoder dots that prompt with per-pixel vision features →
mask, then upsamples to (H, W). This is the LISA-style path that hits ~0.90 because
the prompt is rich (the language says WHAT to segment) — vs a blind dense segmenter.
Masks are POST-LM and on-demand; measurement stays on the reader, pre-LM.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda")


class SegMaskDecoder(nn.Module):
    """Content-prompted mask: LM <SEG> hidden → query; a learned pixel-decoder UPSAMPLES
    the coarse NCS features (thin faults need resolution the 6-col grid can't give) →
    per-pixel features; query · features → mask, then bilinear to (H, W)."""
    def __init__(self, lm_dim=1536, vdim=768, d=128, up=3):
        super().__init__()
        self.q_proj = nn.Sequential(nn.Linear(lm_dim, d), nn.GELU(), nn.Linear(d, d))
        layers = [nn.Conv2d(vdim, d, 3, padding=1), nn.GroupNorm(8, d), nn.GELU()]
        for _ in range(up):                                      # 2^up upsampling (6→48 cols)
            layers += [nn.ConvTranspose2d(d, d, 4, stride=2, padding=1),
                       nn.GroupNorm(8, d), nn.GELU()]
        layers += [nn.Conv2d(d, d, 1)]
        self.up = nn.Sequential(*layers)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, seg_hidden, smap, hw):
        """seg_hidden (n_seg, lm_dim), smap (vdim, fH, fW) -> mask logits (n_seg, H, W)."""
        q = self.q_proj(seg_hidden)                              # (n, d)
        f = self.up(smap.unsqueeze(0))                           # (1, d, H', W')  upsampled
        m = torch.einsum("nd,dhw->nhw", q, f[0]) + self.bias     # (n, H', W')
        return F.interpolate(m.unsqueeze(0), size=hw, mode="bilinear", align_corners=False)[0]


def mask_loss(logits, gt):
    """BCE(pos-weighted) + soft-dice for thin masks. logits,gt: (n, H, W)."""
    pw = torch.tensor([40.0], device=device)
    bce = F.binary_cross_entropy_with_logits(logits, gt, pos_weight=pw)
    p = logits.sigmoid()
    dice = 1 - (2 * (p * gt).sum() + 1) / (p.sum() + gt.sum() + 1)
    return bce + dice


def seg_positions(tok, target_ids):
    """Token index that COMPLETES each <SEG> in a 1-D target id tensor (for grabbing
    the hidden state there). Robust to BPE context: decode prefixes and detect when a
    new '<SEG>' appears, rather than matching a standalone tokenization."""
    ids = target_ids.tolist()
    out, seen = [], 0
    for i in range(len(ids)):
        c = tok.decode(ids[: i + 1]).count("<SEG>")
        if c > seen:
            out.append(i); seen = c
    return out
