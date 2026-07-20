"""Stage 2 (vision) — train the DENSE segmentation front-end, then read facts.

image -> ONE fault/closure segmentation (no queries, no N cap). Faults are
counted + separated by SEQUENTIAL RANSAC (each dominant line = one fault); each
line's angle = its apparent dip. These (count + per-fault dip) are the facts the
narrator copies in Stage 3.
"""
import torch

from hybrid.model.segmenter import (
    VisionModel, vision_loss, field_dice, instance_dips, measure_instances,
)
from hybrid.model.narrator import faults_of

device = torch.device("cuda")
DET_EPOCHS = 250      # high epochs for the full-data generalization run
DIP_STRATA = [(0, 5), (5, 20), (20, 60), (60, 90)]


def train_detector(train_scenes, rng, scale, epochs=DET_EPOCHS):
    net = VisionModel().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=5e-4, weight_decay=1e-4)
    for ep in range(epochs):
        rng.shuffle(train_scenes)
        tot = 0.0
        for s in train_scenes:
            opt.zero_grad()
            loss = vision_loss(net, s)          # seg + bbox-localization + attr + throw (one pass)
            loss.backward(); opt.step(); tot += loss.item()
        if ep % max(1, epochs // 5) == 0 or ep == epochs - 1:
            print(f"[segmenter] ep {ep} loss {tot/len(train_scenes):.3f}", flush=True)
    net.eval()
    return net


@torch.no_grad()
def detected_facts(net, s):
    """Class-driven facts for a scene: fault dips + throws, closure areas."""
    return net.measure(s["smap"], s["hw"])


@torch.no_grad()
def detected_dips(net, s, scale):
    return [f["dip"] for f in net.measure(s["smap"], s["hw"])["faults"]]


def _agg(vs):
    return (sum(vs) / len(vs), len(vs)) if vs else (None, 0)


@torch.no_grad()
def measure_accuracy(net, scenes, scale):
    """Dense front-end accuracy: COUNT error, fault-field dice, and dip via
    sequential RANSAC (predicted seg vs GT-field ceiling), stratified by GT dip.
    Dips matched sorted (smallest→smallest) to GT dips."""
    def binof(d):
        for b in DIP_STRATA:
            if b[0] <= d < (91 if b[1] == 90 else b[1]):
                return b
        return None

    count_err, dices = [], []
    perr = {b: [] for b in DIP_STRATA}   # dip via predicted seg
    gerr = {b: [] for b in DIP_STRATA}   # dip via GT field (ceiling)
    for s in scenes:
        gt = sorted(faults_of(s["objs"]))
        if not gt:
            continue
        seg = net(s["smap"], s["hw"])
        pred = sorted(instance_dips(seg[0]))
        gtf = sorted(instance_dips(s["fault_field"], is_logits=False))
        count_err.append(abs(len(pred) - len(gt)))
        dices.append(field_dice(seg[0], s["fault_field"]))
        for i, gd in enumerate(gt):
            b = binof(gd)
            if b is None:
                continue
            if i < len(pred):
                perr[b].append(abs(pred[i] - gd))
            if i < len(gtf):
                gerr[b].append(abs(gtf[i] - gd))
    allp = [v for b in DIP_STRATA for v in perr[b]]
    allg = [v for b in DIP_STRATA for v in gerr[b]]
    return dict(
        count=_agg(count_err), dice=_agg(dices),
        dip_pred_all=_agg(allp), dip_gt_all=_agg(allg),
        dip_pred={b: _agg(perr[b]) for b in DIP_STRATA},
        dip_gt={b: _agg(gerr[b]) for b in DIP_STRATA},
    )
