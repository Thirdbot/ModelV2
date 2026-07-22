"""Integrated step-3 overfit: reader (foot-fixed) + per-object <SEG> mask decoder + LM.

Validates the three pieces wired together on a few scenes:
  1. InstanceReader produces the facts (count/class/dip/throw) — replaces dense+RANSAC.
  3. foot occupancy now has its own head + pos-BCE/dice (should DROP, not stall at 0.87).
  2. per-object <SEG>: the LM emits one <SEG> per fault; each <SEG> hidden drives THAT
     fault's mask via the decoder (referring segmentation), supervised per-instance.
LM frozen for the mask branch (joint fine-tune deferred). Copy is unchanged (val 1.00).
"""
import torch
import torch.nn.functional as F

import hybrid.model.scenes as sc
sc.MAX_SCENES = 8

from hybrid.model.scenes import build_scenes
from hybrid.model.narrator import Narrator, faults_of, scene_facts, facts_to_kv
from hybrid.model.reader import InstanceReader, FAULT
from hybrid.model.mask_decoder import SegMaskDecoder, mask_loss
from hybrid.model.segmenter import field_dice

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


def fault_masks(scene):                     # per-fault masks, SAME order as scene_facts
    return [o["mask"] for o in scene["objs"] if int(o["cls"]) == 1 and float(o["mmask"][0]) > 0]


def per_obj_target(facts):                  # one <SEG> per fault (referring)
    fs = facts["faults"]
    parts = ["<evidence>", f"There are {len(fs)} faults."]
    for i, f in enumerate(fs):
        parts.append(f"Fault {i + 1} dips at {round(float(f['dip']), 1):g} degrees <SEG>.")
    parts.append("</evidence>")
    return " ".join(parts)


scenes = [s for s in build_scenes() if faults_of(s["objs"])][:6]

# ---- 1 + 3: reader (foot-fixed) ----
rdata = [(s["smap"], prep(s)) for s in scenes]
rdata = [(sm, g) for sm, g in rdata if g]
reader = InstanceReader().to(device)
opt = torch.optim.AdamW(reader.parameters(), lr=3e-4, weight_decay=1e-4)
for ep in range(300):
    tot = 0.0; last = {}
    for sm, g in rdata:
        opt.zero_grad(); l, p = reader(sm, g); l.backward(); opt.step(); tot += l.item(); last = p
    if ep % 60 == 0 or ep == 299:
        print(f"[reader] ep {ep} loss {tot/len(rdata):.3f} · {last}", flush=True)
reader.eval()
cnt_err, dip_err, thr_err, hit, totc = [], [], [], 0, 0
for sm, g in rdata:
    pred = reader.detect(sm)
    cnt_err.append(abs(len(pred) - len(g)))
    gf = sorted(o["dip"] for o in g if o["cls"] == FAULT and o.get("dip") is not None)
    pf = sorted(o["dip"] for o in pred if o["cls"] == FAULT)
    for i, gd in enumerate(gf):
        if i < len(pf):
            dip_err.append(abs(pf[i] - gd))
    for i in range(min(len(pred), len(g))):
        totc += 1; hit += int(pred[i]["cls"] == g[i]["cls"])


def m(x): return f"{sum(x)/len(x):.2f}(n{len(x)})" if x else "n0"
print(f"[reader RESULT] count MAE {m(cnt_err)} · class {hit}/{totc} · dip MAE {m(dip_err)}deg "
      f"· foot_loss {last.get('foot'):.3f}", flush=True)

# ---- 2: per-object <SEG> mask decoder (LM frozen) ----
nar = Narrator(); nar.set_stage("s3"); nar.eval_mode()
mdec = SegMaskDecoder().to(device)
mdata = []
for s in scenes:
    facts = scene_facts(s); fm = fault_masks(s)
    if not fm:
        continue
    with torch.no_grad():
        segh, nseg = nar.seg_hidden(facts_to_kv(facts), per_obj_target(facts), question=None)
        segh = segh.float()
    k = min(nseg, len(fm))
    if k == 0:
        continue
    mdata.append((segh[:k], s["smap"], torch.stack(fm[:k]).to(device), s["hw"]))
print(f"[mask] per-object <SEG> on {len(mdata)} scenes "
      f"({sum(d[0].shape[0] for d in mdata)} instances)", flush=True)
opt = torch.optim.AdamW(mdec.parameters(), lr=1e-3)
for ep in range(300):
    tot = 0.0
    for segh, smap, gt, hw in mdata:
        opt.zero_grad(); l = mask_loss(mdec(segh, smap, hw), gt); l.backward(); opt.step(); tot += l.item()
    if ep % 60 == 0 or ep == 299:
        print(f"[mask] ep {ep} loss {tot/len(mdata):.3f}", flush=True)
mdec.eval()
dices = []
with torch.no_grad():
    for segh, smap, gt, hw in mdata:
        mk = mdec(segh, smap, hw)
        dices += [field_dice(mk[i], gt[i]) for i in range(mk.shape[0])]
print(f"[PIPELINE RESULT] per-object mask dice {m(dices)}", flush=True)
print("PIPELINE_DONE", flush=True)
