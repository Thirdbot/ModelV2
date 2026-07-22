"""Stage 2 (new) — the instance reader (facts + per-instance masks, one shared trunk).

Replaces the dense-seg + RANSAC front-end. `train_reader` fits the autoregressive reader
(count/class/dip/throw AND the joint mask head); `reader_facts` adapts its detections to the
digit-bridge fact dict; `reader_accuracy` reports held-out count/dip/class.
(The old LM-<SEG> mask decoder path was retired — masks come from the reader now.)
"""
import torch

from hybrid.model.reader import InstanceReader, scene_to_gt, FAULT, CLOSURE

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
