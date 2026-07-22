"""Loosened-gate Stage-4 test: 300m/4pts (~2x windows) → does more (noisier) real
data help the fine-tune vs the 72-window overfit (23.8→29.2)?"""
import random
import numpy as np
import torch

import hybrid.data.real as R
R.MATCH_THRESH_M = 300.0; R.MIN_FAULT_PTS = 4
from hybrid.model.reader import InstanceReader, scene_to_gt, FAULT

device = torch.device("cuda")
scenes = R.build_real_windows()
print(f"[loose] {len(scenes)} windows @300m/4pts (was 72 @150m/6)", flush=True)
rng = random.Random(42); idx = list(range(len(scenes))); rng.shuffle(idx)
cut = int(len(idx) * 0.75); tr = [scenes[i] for i in idx[:cut]]; te = [scenes[i] for i in idx[cut:]]


def dip_mae(net, sp):
    net.eval(); e = []
    with torch.no_grad():
        for s in sp:
            sm = s["smap"].to(device); pred = net.detect(sm); gt = scene_to_gt(s)
            gf = sorted(o["dip"] for o in gt if o["cls"] == FAULT and o.get("dip") is not None)
            pf = sorted(o["dip"] for o in pred if o["cls"] == FAULT)
            for i, gd in enumerate(gf):
                if i < len(pf):
                    e.append(abs(pf[i] - gd))
            del sm; torch.cuda.empty_cache()
    return (np.mean(e), len(e)) if e else (float("nan"), 0)


net = InstanceReader().to(device)
net.load_state_dict(torch.load("hybrid/checkpoints/reader.pt", map_location=device))
z = dip_mae(net, te)
print(f"[loose] zero-shot held-out dip {z[0]:.1f}deg (n{z[1]}) · train {len(tr)}", flush=True)
opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4); net.train()
for ep in range(120):
    tot = 0.0
    for s in tr:
        sm = s["smap"].to(device); gt = scene_to_gt(s)
        if gt:
            opt.zero_grad(); l, _ = net(sm, gt); l.backward(); opt.step(); tot += l.item()
        del sm; torch.cuda.empty_cache()
    if ep % 30 == 0 or ep == 119:
        print(f"[loose] ft ep {ep} loss {tot/len(tr):.3f}", flush=True)
a = dip_mae(net, te)
print(f"[loose] fine-tuned held-out dip {a[0]:.1f}deg (n{a[1]})", flush=True)
print("LOOSE_DONE", flush=True)
