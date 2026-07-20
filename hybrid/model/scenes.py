"""Scene building — dataset -> per-image training scenes for the vision stage.

Reads the CSV, groups rows by image (a region's dip may live in ANY of that
image's rows -> aggregate the evidence), encodes each image to a stitched NCS
feature map, and builds the DENSE targets (fault/closure fields, union of masks
by class) plus per-object GT (class, dip/throw/pct, dilated mask). One scene per
unique image (image-level split, no leakage). Feeds the dense segmenter
(`hybrid.model.segmenter`). The DETR detector this file once held is gone —
detection is now the dense segmenter + class-driven `measure_instances`.
"""
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from hybrid.model.encoder import NcsEncoder, stitch
from hybrid.data.dataset import load_local_csv

CSV = "/home/third/Desktop/simulationv2/Dataset/multimodal_multi_image_dataset.csv"
device = torch.device("cuda")
MEAS_SCALE = torch.tensor([90.0, 500.0, 100.0])       # dip(deg) · throw(ms) · percent
MEAS_MASK = {1: [1.0, 1.0, 0.0], 2: [0.0, 0.0, 1.0]}   # per-class attribute schema
BLOCK = re.compile(r"<region>(.*?)</region>", re.S)
DIP = re.compile(r"dips?.*?<nums>([-\d.]+)</nums>\s*degrees", re.I | re.S)
THROW = re.compile(r"throw.*?<nums>([-\d.]+)</nums>\s*ms", re.I | re.S)
PCT = re.compile(r"<nums>([-\d.]+)</nums>\s*percent", re.I)
MAX_SCENES = 320      # one scene per UNIQUE image; the dataset has ~282 fault images
DILATE_R = 3          # fatten the thin fault line to ~1 feature-cell wide


def load_mask_hw(pil, hw):
    H, W = hw
    a = np.array(pil.convert("L").resize((W, H)), dtype=np.float32)
    return torch.from_numpy((a > 40).astype("float32")).to(device)


def dilate(m, r=DILATE_R):
    """Symmetric dilation (keeps the principal axis, so dip is unchanged)."""
    if r <= 0:
        return m
    return F.max_pool2d(m[None, None], 2 * r + 1, stride=1, padding=r)[0, 0]


def build_scenes():
    rows = load_local_csv(csv_path=CSV)
    enc = NcsEncoder().to(device).eval()
    # Each image recurs across many rows with different Q&A; a region's dip may
    # live in ANY of them -> group by image and aggregate the evidence, one
    # scene per image (a true image-level unit, no train/test leakage).
    by_img = {}
    for r in rows:
        ips = r.get("image_paths") or []
        if ips and Path(ips[0]).exists():
            by_img.setdefault(ips[0], []).append(r)
    scenes = []
    for img, rr in by_img.items():
        regs = rr[0].get("regions") or []
        mps = rr[0].get("mask_paths") or []
        if not regs:
            continue
        all_blocks = [BLOCK.findall(r.get("evidence") or "") for r in rr]
        W, H = Image.open(img).size
        hw = (H, W)
        objs = []
        for i, reg in enumerate(regs):
            cid = int(reg.get("class_id", 0))
            if cid not in MEAS_MASK:
                continue
            mi = reg.get("mask_idx", i)
            if not (isinstance(mi, int) and 0 <= mi < len(mps) and Path(mps[mi]).exists()):
                continue
            x1, y1, x2, y2 = reg["bbox"]
            # a region's dip/throw/pct may appear in only some rows -> first hit
            d = t = p = None
            for blocks in all_blocks:
                blk = blocks[i] if i < len(blocks) else ""
                d = d or DIP.search(blk)
                t = t or THROW.search(blk)
                p = p or PCT.search(blk)
            meas = [0.0, 0.0, 0.0]
            mm = [0.0, 0.0, 0.0]   # supervise ONLY measurements actually present
            if cid == 1 and d:
                meas[0] = float(d.group(1)); mm[0] = 1.0
                if t:
                    meas[1] = float(t.group(1)); mm[1] = 1.0
            if cid == 2 and p:
                meas[2] = float(p.group(1)); mm[2] = 1.0
            objs.append(dict(cls=cid, bbox=[x1 / W, y1 / H, x2 / W, y2 / H],
                             mask=dilate(load_mask_hw(Image.open(mps[mi]), hw)),
                             meas=torch.tensor(meas, device=device),
                             mmask=torch.tensor(mm, device=device)))
            # no object cap: the dense segmenter has no N-query limit, and count
            # comes from connected components over the whole field.
        # encode only images that actually have a measured fault (saves NCS compute)
        if not any(int(o["cls"]) == 1 and float(o["mmask"][0]) > 0 for o in objs):
            continue
        smap, _ = stitch(enc, img)
        ff = torch.zeros(hw, device=device)   # dense targets: union of masks by class
        cf = torch.zeros(hw, device=device)
        for o in objs:
            if int(o["cls"]) == 1:
                ff = torch.maximum(ff, o["mask"])
            elif int(o["cls"]) == 2:
                cf = torch.maximum(cf, o["mask"])
        scenes.append(dict(smap=smap, hw=hw, objs=objs, img=img,
                           fault_field=ff, closure_field=cf))
        if len(scenes) >= MAX_SCENES:
            break
    return scenes
