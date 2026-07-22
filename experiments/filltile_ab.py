"""Fill-the-tile A/B: pad (fW≈6) vs proportional upsample-to-fill (fW≈14). Re-encodes the
SAME scenes both ways, trains the reader on each, compares held-out dip + per-fault mask dice.
Tests whether the 6→14 feature-column resolution gain beats the OOD cost of interpolation."""
import random
import numpy as np
import torch

import hybrid.model.scenes as sc
sc.MAX_SCENES = 100
import hybrid.model.encoder as enc_mod

from hybrid.model.scenes import build_scenes
from hybrid.model.narrator import faults_of
from hybrid.model.reader import InstanceReader, scene_to_gt, FAULT
from hybrid.model.segmenter import field_dice

device = torch.device("cuda")


def run(fill):
    enc_mod.FILL_TILE = fill
    scenes = [s for s in build_scenes() if faults_of(s["objs"])]
    fW = scenes[0]["smap"].shape[2] if scenes else 0
    rng = random.Random(0); idx = list(range(len(scenes))); rng.shuffle(idx)
    cut = int(len(idx) * 0.75); tr = [scenes[i] for i in idx[:cut]]; te = [scenes[i] for i in idx[cut:]]
    torch.manual_seed(0)
    net = InstanceReader().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    data = [(s["smap"], scene_to_gt(s)) for s in tr]; data = [(sm, g) for sm, g in data if g]
    net.train()
    for ep in range(150):
        for sm, g in data:
            opt.zero_grad(); l, _ = net(sm, g); l.backward(); opt.step()
    net.eval()
    dip, dice, cnt = [], [], []
    with torch.no_grad():
        for s in te:
            gt = scene_to_gt(s)
            if not gt:
                continue
            pred = net.detect(s["smap"]); cnt.append(abs(len(pred) - len(gt)))
            gd = sorted(o["dip"] for o in gt if o["cls"] == FAULT and o.get("dip") is not None)
            pd = sorted(o["dip"] for o in pred if o["cls"] == FAULT)
            for i, x in enumerate(gd):
                if i < len(pd):
                    dip.append(abs(pd[i] - x))
            ml = net.tf_masks(s["smap"], gt)
            for i, o in enumerate(gt):
                if o["cls"] == FAULT:
                    dice.append(field_dice(ml[i], o["mask_full"].to(device)))
    return fW, np.mean(cnt), np.mean(dip), np.mean(dice), len(dice)


for name, f in (("pad (fW≈6)", False), ("fill-tile (fW≈14)", True)):
    fW, c, d, dc, n = run(f)
    print(f"[filltile {name:>18}] fW={fW} · count MAE {c:.2f} · dip MAE {d:.1f}deg · mask dice {dc:.3f} (n{n})", flush=True)
print("FILLTILE_AB_DONE", flush=True)
