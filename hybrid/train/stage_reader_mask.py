"""Stage 2 (new) — the instance reader + the <SEG> mask decoder.

Replaces the dense-seg + RANSAC front-end. `train_reader` fits the autoregressive
reader (facts = count/class/dip/throw); `reader_facts` adapts its detections to the
digit-bridge fact dict; `train_mask_decoder` fits the per-object <SEG> mask decoder
on the (frozen) narrator's <SEG> hidden states. Accuracy helpers report held-out.
"""
import torch

from hybrid.model.reader import InstanceReader, scene_to_gt, FAULT, CLOSURE
from hybrid.model.mask_decoder import SegMaskDecoder, mask_loss
from hybrid.model.segmenter import field_dice
from hybrid.model.narrator import scene_facts, facts_to_kv

device = torch.device("cuda")


def train_reader(scenes, epochs=150, lr=3e-4):
    net = InstanceReader().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    data = [(s["smap"], scene_to_gt(s)) for s in scenes]
    data = [(sm, g) for sm, g in data if g]
    net.train()
    for ep in range(epochs):
        tot = 0.0
        for smap, gt in data:
            opt.zero_grad(); loss, _ = net(smap, gt); loss.backward(); opt.step(); tot += loss.item()
        if ep % max(1, epochs // 5) == 0 or ep == epochs - 1:
            print(f"[reader] ep {ep} loss {tot/len(data):.3f}", flush=True)
    net.eval()
    return net


@torch.no_grad()
def reader_facts(net, scene):
    """reader.detect -> the fact dict the digit bridge consumes (bbox unused → 0s)."""
    faults, closures = [], []
    for o in net.detect(scene["smap"]):
        if o["cls"] == FAULT:
            faults.append({"dip": o["dip"], "bbox": [0, 0, 0, 0], "throw": o["throw"]})
        elif o["cls"] == CLOSURE:
            closures.append({"area_pct": o["area"], "bbox": [0, 0, 0, 0]})
    return {"faults": faults, "closures": closures}


@torch.no_grad()
def reader_accuracy(net, scenes):
    cnt, dip, hit, tot = [], [], 0, 0
    for s in scenes:
        gt = scene_to_gt(s); pred = net.detect(s["smap"])
        cnt.append(abs(len(pred) - len(gt)))
        gf = sorted(o["dip"] for o in gt if o["cls"] == FAULT and o.get("dip") is not None)
        pf = sorted(o["dip"] for o in pred if o["cls"] == FAULT)
        for i, gd in enumerate(gf):
            if i < len(pf):
                dip.append(abs(pf[i] - gd))
        for i in range(min(len(pred), len(gt))):
            tot += 1; hit += int(pred[i]["cls"] == gt[i]["cls"])

    def m(x): return (sum(x) / len(x), len(x)) if x else (None, 0)
    return dict(count=m(cnt), dip=m(dip), cls=(hit, tot))


def per_obj_target(facts):
    fs = facts["faults"]
    parts = ["<evidence>", f"There are {len(fs)} faults."]
    for i, f in enumerate(fs):
        parts.append(f"Fault {i + 1} dips at {round(float(f['dip']), 1):g} degrees <SEG>.")
    parts.append("</evidence>")
    return " ".join(parts)


def fault_masks(scene):
    return [o["mask"] for o in scene["objs"] if int(o["cls"]) == 1 and float(o["mmask"][0]) > 0]


def _mask_rows(nar, scenes):
    rows = []
    for s in scenes:
        facts = scene_facts(s); fm = fault_masks(s)
        if not fm:
            continue
        with torch.no_grad():
            segh, nseg = nar.seg_hidden(facts_to_kv(facts), per_obj_target(facts))
        k = min(nseg, len(fm))
        if k == 0:
            continue
        rows.append((segh[:k].float(), s["smap"], torch.stack(fm[:k]).to(device), s["hw"]))
    return rows


def train_mask_decoder(nar, scenes, epochs=150, lr=1e-3):
    mdec = SegMaskDecoder().to(device)
    data = _mask_rows(nar, scenes)
    opt = torch.optim.AdamW(mdec.parameters(), lr=lr)
    mdec.train()
    for ep in range(epochs):
        tot = 0.0
        for segh, smap, gt, hw in data:
            opt.zero_grad(); loss = mask_loss(mdec(segh, smap, hw), gt); loss.backward(); opt.step()
            tot += loss.item()
        if ep % max(1, epochs // 5) == 0 or ep == epochs - 1:
            print(f"[mask] ep {ep} loss {tot/max(1,len(data)):.3f}", flush=True)
    mdec.eval()
    return mdec


@torch.no_grad()
def mask_accuracy(nar, mdec, scenes):
    dices = []
    for segh, smap, gt, hw in _mask_rows(nar, scenes):
        mk = mdec(segh, smap, hw)
        dices += [field_dice(mk[i], gt[i]) for i in range(mk.shape[0])]
    return (sum(dices) / len(dices), len(dices)) if dices else (None, 0)
