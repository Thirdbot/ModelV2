"""Dense segmentation front-end — replaces the N-query DETR detector.

ONE dense fault/closure segmentation over the stitched NCS features (no object
queries, no N cap). At inference, faults are SEPARATED + COUNTED by SEQUENTIAL
RANSAC: fit the dominant line, remove its inliers, repeat — each line = one
fault (robust to crossing → separates; robust to fragmentation → bridges gaps),
and each line's angle = its apparent dip. Closures come from connected
components. Output feeds the digit-token bridge exactly as before (count +
per-fault dip), so the narration half is unchanged.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda")


class DenseSegmenter(nn.Module):
    """smap (768, fH, fW) -> shared TRUNK features -> seg_head -> (2, H, W)
    [fault, closure] segmentation. `forward` returns seg logits, so measure / count /
    RANSAC dip read off it unchanged. Trained in Stage 2a, frozen thereafter."""

    def __init__(self, vdim=768, hidden=128):
        super().__init__()
        self.hidden = hidden
        self.trunk = nn.Sequential(
            nn.Conv2d(vdim, hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.GELU(),
        )
        self.seg_head = nn.Conv2d(hidden, 2, 1)

    def _up(self, x, hw):
        return F.interpolate(x, size=hw, mode="bilinear", align_corners=False)[0]

    def forward(self, smap, hw):
        f = self.trunk(smap.unsqueeze(0))                    # (1, hidden, fH, fW)
        return self._up(self.seg_head(f), hw)                # (2, H, W) seg logits


def _dice(logit, gt):
    p = logit.sigmoid()
    return 1 - (2 * (p * gt).sum() + 1) / (p.sum() + gt.sum() + 1)


# --- loss config (overridable by experiments/) ---
# Main loss = pos-weighted BCE (focal-modulated) + soft-dice, per class channel.
POS_WEIGHT = 50.0     # BCE positive weight (thin-fault imbalance)
FOCAL_GAMMA = 2.0     # 0 = plain pos-weighted BCE; >0 = focal hard-pixel modulation

# --- clDice (centerline dice) — ABLATED OUT, kept commented for a possible
# text-rich/mask-poor revisit. It cost held-out dice for the same over-detection
# gain focal already gives (see experiments/results/ablation_summary.json).
# def _min_pool(x):
#     return -F.max_pool2d(-x, 3, 1, 1)             # soft erosion
# def _soft_open(x):
#     return F.max_pool2d(_min_pool(x), 3, 1, 1)    # erode then dilate
# def _soft_skel(x, iters=8):
#     sk = F.relu(x - _soft_open(x))
#     for _ in range(iters):
#         x = _min_pool(x)
#         delta = F.relu(x - _soft_open(x))
#         sk = sk + F.relu(delta - sk * delta)
#     return sk
# def cldice(logit, gt, iters=8, smooth=1e-3):
#     """1 - harmonic mean of (pred-skeleton in GT) and (GT-skeleton in pred)."""
#     p = logit.sigmoid()[None, None]; g = gt[None, None]
#     sp, sg = _soft_skel(p, iters), _soft_skel(g, iters)
#     tprec = (sp * g).sum() / (sp.sum() + smooth)   # pred centerline inside GT
#     tsens = (sg * p).sum() / (sg.sum() + smooth)   # GT centerline inside pred
#     return 1 - 2 * tprec * tsens / (tprec + tsens + smooth)


def _bce_term(logit, gt):
    """Pos-weighted BCE, optionally focal-modulated (down-weights easy pixels)."""
    pw = torch.tensor([POS_WEIGHT], device=device)
    if FOCAL_GAMMA > 0:
        ce = F.binary_cross_entropy_with_logits(logit, gt, pos_weight=pw, reduction="none")
        p = logit.sigmoid()
        pt = torch.where(gt > 0.5, p, 1 - p)          # prob of the true class
        return ((1 - pt) ** FOCAL_GAMMA * ce).mean()
    return F.binary_cross_entropy_with_logits(logit, gt, pos_weight=pw)


def seg_loss(seg, fault_field, closure_field):
    """Dense per-channel loss = pos-weighted BCE (focal-modulated) + soft-dice,
    per class channel. Config-driven (POS_WEIGHT / FOCAL_GAMMA) so experiments/
    can ablate without editing this file."""
    lf = _bce_term(seg[0], fault_field) + _dice(seg[0], fault_field)
    lc = _bce_term(seg[1], closure_field) + _dice(seg[1], closure_field)
    return lf + lc


def field_dice(logit, gt, thresh=0.5):
    p = (torch.sigmoid(logit) > thresh).float()
    return float(2 * (p * gt).sum() / (p.sum() + gt.sum()).clamp_min(1e-6))


def pool_instance(smap, comp_mask_hw):
    """Masked-average smap (vdim,fH,fW) over an instance mask (H,W) -> (vdim,)."""
    fH, fW = smap.shape[1], smap.shape[2]
    m = F.adaptive_max_pool2d(comp_mask_hw[None, None], (fH, fW))[0, 0]
    return (smap * m[None]).sum(dim=(1, 2)) / m.sum().clamp_min(1e-6)


class ThrowHead(nn.Module):
    """Throw is a MAGNITUDE, so a small pooled-feature head reads it (magnitudes
    survive pooling; only the angle ever needed geometry). Predicts throw/scale."""
    def __init__(self, vdim=768, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(vdim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, feat):
        return self.net(feat).squeeze(-1)


class VisionModel(nn.Module):
    """Vision front-end = dense segmenter + throw head. `forward` -> seg logits;
    `measure` -> class-driven facts (count/dip/area + throw + bbox per instance).
    Trained in Stage 2a, frozen thereafter."""
    def __init__(self):
        super().__init__()
        self.seg = DenseSegmenter()
        self.throw = ThrowHead()                              # raw-smap, used at inference

    def forward(self, smap, hw):
        return self.seg(smap, hw)

    def measure(self, smap, hw):
        return measure_instances(self.seg(smap, hw), smap, self.throw)


def throw_loss(net, scene, throw_scale=500.0):
    """Supervise the throw head on GT fault objects carrying a throw label."""
    smap = scene["smap"]
    losses = [F.smooth_l1_loss(net.throw(pool_instance(smap, o["mask"])),
                               o["meas"][1] / throw_scale)
              for o in scene["objs"]
              if int(o["cls"]) == 1 and float(o["mmask"][1]) > 0]
    return torch.stack(losses).mean() if losses else torch.zeros((), device=smap.device)


def vision_loss(net, scene, throw_scale=500.0):
    """Stage-2 vision objective: dense fault/closure segmentation + the throw head.
    (Aux bbox/class/dip/area heads were removed — they lifted train fit but not
    held-out dice; the mask instead gets refined by language in Stage-3 co-refine.)"""
    seg = net(scene["smap"], scene["hw"])
    return seg_loss(seg, scene["fault_field"], scene["closure_field"]) \
        + throw_loss(net, scene, throw_scale)


def _line_dip(pts, inlier_dist=3.0, iters=300):
    """Single-line RANSAC on one instance's pixels -> its apparent dip (deg).
    RANSAC ONLY reads the angle here; it does not count."""
    if len(pts) < 5:
        return None
    rng = np.random.default_rng(0)
    best_in, best_d, best_a = -1, None, 0
    ii, jj = rng.integers(0, len(pts), iters), rng.integers(0, len(pts), iters)
    for a, b in zip(ii.tolist(), jj.tolist()):
        if a == b:
            continue
        d = pts[b] - pts[a]
        nr = math.hypot(float(d[0]), float(d[1]))
        if nr < 1e-6:
            continue
        d = d / nr
        n = np.array([-d[1], d[0]])
        k = int((np.abs((pts - pts[a]) @ n) < inlier_dist).sum())
        if k > best_in:
            best_in, best_d, best_a = k, d, a
    n = np.array([-best_d[1], best_d[0]])
    inl = pts[np.abs((pts - pts[best_a]) @ n) < inlier_dist]
    if len(inl) >= 2:
        c = inl - inl.mean(0)
        _, v = np.linalg.eigh(c.T @ c)
        best_d = v[:, -1]
    ang = abs(math.degrees(math.atan2(float(best_d[1]), float(best_d[0]))))
    return ang if ang <= 90 else 180 - ang


def _components(binary):
    """Connected components (fault INSTANCES), 8-CONNECTED so diagonal fault
    lines stay whole. scipy if present, else BFS."""
    try:
        from scipy.ndimage import label
        lab, n = label(binary, structure=np.ones((3, 3), dtype=int))
        return lab, n
    except Exception:
        lab = np.zeros_like(binary, dtype=np.int32)
        H, W = binary.shape
        nxt = 0
        nbrs = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
        for sy in range(H):
            for sx in range(W):
                if binary[sy, sx] and lab[sy, sx] == 0:
                    nxt += 1
                    stack = [(sy, sx)]
                    while stack:
                        y, x = stack.pop()
                        if 0 <= y < H and 0 <= x < W and binary[y, x] and lab[y, x] == 0:
                            lab[y, x] = nxt
                            stack += [(y + dy, x + dx) for dy, dx in nbrs]
        return lab, nxt


def instance_dips(mask, is_logits=True, thresh=0.5, min_size=40, inlier_dist=3.0):
    """Fault-channel dips: COUNT from dense connected components (size-filtered),
    DIP from single-line RANSAC per component. len(result) = fault count."""
    p = torch.sigmoid(mask) if is_logits else mask
    b = (p > thresh).cpu().numpy()
    lab, n = _components(b)
    dips = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if len(xs) < min_size:                      # dense noise blob -> not a fault
            continue
        d = _line_dip(np.stack([xs, ys], 1).astype(float), inlier_dist)
        if d is not None:
            dips.append(d)
    return dips


# per-class attribute schema — the CLASS (which channel) routes what is measured.
# Adding a class or attribute is a new entry here, not new plumbing.
CLASS_CHANNELS = {"fault": 0, "closure": 1}


def measure_instances(seg, smap=None, throw_head=None, thresh=0.5, min_size=40,
                      inlier_dist=3.0, throw_scale=500.0):
    """The class-driven measurement (one mechanism, class routes the attribute).
    For each dense instance, its CLASS = which channel it came from, and the class
    selects the attribute + the reader:
        fault   -> dip   (angle     → per-instance RANSAC line-fit)
                +  throw (magnitude → pooled-feature head, if smap+throw_head given)
        closure -> area% (size      → pixel fraction of the section)
    Every instance also carries its bbox [x1,y1,x2,y2] (component extent).
    Returns {'faults':[{'dip','throw','bbox'}], 'closures':[{'area_pct','bbox'}]};
    per-class count = len(list). New class = new channel + one branch here."""
    _, H, W = seg.shape
    area_img = float(H * W)
    out = {"faults": [], "closures": []}
    for cls, ch in CLASS_CHANNELS.items():
        b = (torch.sigmoid(seg[ch]) > thresh).cpu().numpy()
        lab, n = _components(b)
        for i in range(1, n + 1):
            ys, xs = np.where(lab == i)
            if len(xs) < min_size:
                continue
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            if cls == "fault":                        # angle attribute → RANSAC
                d = _line_dip(np.stack([xs, ys], 1).astype(float), inlier_dist)
                if d is None:
                    continue
                fact = {"dip": d, "bbox": bbox}
                if smap is not None and throw_head is not None:   # magnitude → head
                    cm = torch.from_numpy((lab == i).astype("float32")).to(smap.device)
                    fact["throw"] = float(throw_head(pool_instance(smap, cm)).detach()) * throw_scale
                out["faults"].append(fact)
            elif cls == "closure":                    # size attribute → pixel count
                out["closures"].append({"area_pct": 100.0 * len(xs) / area_img, "bbox": bbox})
    return out
