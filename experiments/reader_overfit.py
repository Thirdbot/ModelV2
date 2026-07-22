"""Quick overfit for the InstanceReader (step 3, part 1).

Overfits the autoregressive reader on a few scenes, then greedy-decodes and checks
count / class / dip / throw. The key thing to validate: DIP learns (from the spatial
footprint stats) rather than collapsing to a constant like the old pooled head did.
"""
import random
import torch
import torch.nn.functional as F

import hybrid.model.scenes as sc
sc.MAX_SCENES = 8

from hybrid.model.scenes import build_scenes
from hybrid.model.narrator import faults_of
from hybrid.model.reader import InstanceReader, FAULT, CLOSURE

device = torch.device("cuda")


def prep(scene):
    smap = scene["smap"]; fH, fW = smap.shape[1], smap.shape[2]
    objs = []
    for o in scene["objs"]:
        c = int(o["cls"])
        if c not in (1, 2):
            continue
        x1, y1, x2, y2 = o["bbox"]
        ctr = torch.tensor([(y1 + y2) / 2, (x1 + x2) / 2], device=device, dtype=torch.float32)
        mfw = F.adaptive_avg_pool2d(o["mask"][None, None].float(), (fH, fW))[0, 0].clamp(0, 1)
        g = dict(cls=c, ctr=ctr, mask_fW=mfw)
        g["dip"] = float(o["meas"][0]) if (c == 1 and float(o["mmask"][0]) > 0) else None
        g["throw"] = float(o["meas"][1]) if (c == 1 and float(o["mmask"][1]) > 0) else None
        g["area"] = float(o["meas"][2]) if (c == 2 and float(o["mmask"][2]) > 0) else None
        objs.append(((x1 + x2) / 2, g))
    objs.sort(key=lambda t: t[0])
    return [g for _, g in objs]


scenes = [s for s in build_scenes() if faults_of(s["objs"])][:6]
data = [(s["smap"], prep(s)) for s in scenes]
data = [(sm, g) for sm, g in data if g]
print(f"[reader] overfit on {len(data)} scenes", flush=True)

net = InstanceReader().to(device)
opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
for ep in range(300):
    tot = 0.0; last = {}
    for smap, gt in data:
        opt.zero_grad()
        loss, parts = net(smap, gt)
        loss.backward(); opt.step(); tot += loss.item(); last = parts
    if ep % 50 == 0 or ep == 299:
        print(f"[reader] ep {ep} loss {tot/len(data):.3f} · {last}", flush=True)

net.eval()
cnt_err, dip_err, thr_err, cls_hit, cls_tot = [], [], [], 0, 0
for smap, gt in data:
    pred = net.detect(smap)
    cnt_err.append(abs(len(pred) - len(gt)))
    gf = sorted(o["dip"] for o in gt if o["cls"] == FAULT and o.get("dip") is not None)
    pf = sorted(o["dip"] for o in pred if o["cls"] == FAULT)
    for i, gd in enumerate(gf):
        if i < len(pf):
            dip_err.append(abs(pf[i] - gd))
    gt_thr = sorted(o["throw"] for o in gt if o.get("throw") is not None)
    pt = sorted(o["throw"] for o in pred if o["cls"] == FAULT)
    for i, gth in enumerate(gt_thr):
        if i < len(pt):
            thr_err.append(abs(pt[i] - gth))
    for i in range(min(len(pred), len(gt))):
        cls_tot += 1; cls_hit += int(pred[i]["cls"] == gt[i]["cls"])


def m(x): return f"{sum(x)/len(x):.2f}(n{len(x)})" if x else "n0"
print(f"[reader RESULT] count MAE {m(cnt_err)} · class acc {cls_hit}/{cls_tot} · "
      f"dip MAE {m(dip_err)}deg · throw MAE {m(thr_err)}ms", flush=True)
# show one scene's predicted dips vs GT to see the collapse-or-not
smap, gt = data[0]
pd = [round(o["dip"], 1) for o in net.detect(smap) if o["cls"] == FAULT]
gd = [round(o["dip"], 1) for o in gt if o["cls"] == FAULT and o.get("dip") is not None]
print(f"[reader sample] pred dips {pd} · GT dips {gd}", flush=True)
print("READER_OVERFIT_DONE", flush=True)
