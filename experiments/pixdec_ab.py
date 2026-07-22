"""A/B: does the pixel decoder (global attn over the stitched map) improve held-out
dip? Same synthetic split, train reader with vs without it, compare held-out dip/count.
Dip is the metric most sensitive to cross-tile fragmentation of tall faults."""
import random
import numpy as np
import torch

import hybrid.model.scenes as sc
sc.MAX_SCENES = 100

from hybrid.model.scenes import build_scenes
from hybrid.model.narrator import faults_of
from hybrid.model.reader import InstanceReader, scene_to_gt, FAULT

device = torch.device("cuda")
scenes = [s for s in build_scenes() if faults_of(s["objs"])]
rng = random.Random(0); idx = list(range(len(scenes))); rng.shuffle(idx)
cut = int(len(idx) * 0.75); tr = [scenes[i] for i in idx[:cut]]; te = [scenes[i] for i in idx[cut:]]
print(f"[ab] scenes {len(scenes)} · train {len(tr)} · test {len(te)}", flush=True)
train_data = [(s["smap"], scene_to_gt(s)) for s in tr]
train_data = [(sm, g) for sm, g in train_data if g]


def run(pixdec):
    torch.manual_seed(0)
    net = InstanceReader(pixel_decoder=pixdec).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    net.train()
    for ep in range(150):
        for sm, g in train_data:
            opt.zero_grad(); l, _ = net(sm, g); l.backward(); opt.step()
    net.eval()
    err, cnt = [], []
    with torch.no_grad():
        for s in te:
            pred = net.detect(s["smap"]); gt = scene_to_gt(s)
            cnt.append(abs(len(pred) - len(gt)))
            gf = sorted(o["dip"] for o in gt if o["cls"] == FAULT and o.get("dip") is not None)
            pf = sorted(o["dip"] for o in pred if o["cls"] == FAULT)
            for i, gd in enumerate(gf):
                if i < len(pf):
                    err.append(abs(pf[i] - gd))
    return np.mean(cnt), np.mean(err), len(err)


for name, flag in (("no-pixdec", False), ("pixdec", True)):
    c, d, n = run(flag)
    print(f"[ab {name:>10}] held-out count MAE {c:.2f} · dip MAE {d:.1f}deg (n{n})", flush=True)
print("AB_DONE", flush=True)
