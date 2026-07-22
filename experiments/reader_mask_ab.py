"""Stage-2 joint mask (reader query · shared pixel-decoder features) held-out dice —
vs the decoupled LM-<SEG> decoder baseline (0.18). Also checks dip/count didn't regress
(the mask loss co-trains the shared trunk)."""
import random
import numpy as np
import torch
import torch.nn.functional as F

import hybrid.model.scenes as sc
sc.MAX_SCENES = 100

from hybrid.model.scenes import build_scenes
from hybrid.model.narrator import faults_of
from hybrid.model.reader import InstanceReader, scene_to_gt, FAULT
from hybrid.train.stage_reader_mask import train_reader
from hybrid.model.segmenter import field_dice

device = torch.device("cuda")
scenes = [s for s in build_scenes() if faults_of(s["objs"])]
rng = random.Random(0); idx = list(range(len(scenes))); rng.shuffle(idx)
cut = int(len(idx) * 0.75); tr = [scenes[i] for i in idx[:cut]]; te = [scenes[i] for i in idx[cut:]]
print(f"[mask-ab] scenes {len(scenes)} · train {len(tr)} · test {len(te)}", flush=True)

reader = train_reader(tr, epochs=150)          # jointly trains facts + mask head now

dices, dip_err, cnt = [], [], []
with torch.no_grad():
    for s in te:
        objs, masks = reader.detect(s["smap"], want_masks=True)
        gt = scene_to_gt(s)
        cnt.append(abs(len(objs) - len(gt)))
        gd = sorted(o["dip"] for o in gt if o["cls"] == FAULT and o.get("dip") is not None)
        pd = sorted(o["dip"] for o in objs if o["cls"] == FAULT)
        for i, x in enumerate(gd):
            if i < len(pd):
                dip_err.append(abs(pd[i] - x))
        pf = [(objs[i], masks[i]) for i in range(len(objs)) if objs[i]["cls"] == FAULT]
        gm = [o["mask_full"] for o in gt if o["cls"] == FAULT]
        for i in range(min(len(pf), len(gm))):
            H, W = gm[i].shape
            ml = F.interpolate(pf[i][1][None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
            dices.append(field_dice(ml, gm[i].to(device)))


def m(x): return f"{np.mean(x):.2f}(n{len(x)})" if x else "n0"
print(f"[mask-ab RESULT] Stage-2 reader-query mask dice {m(dices)} · dip MAE {m(dip_err)}deg "
      f"· count MAE {m(cnt)}  (vs decoupled LM-<SEG> 0.18)", flush=True)
print("MASK_AB_DONE", flush=True)
