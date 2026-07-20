"""Benchmark — the VISION part on real-field held-out (Smeaheia).

The dedicated real-field test. Reports the vision values the model must get right on
real data — narration is NOT involved (it's frozen; this measures syn->real detection):
  fault mask dice · dip MAE (RANSAC on predicted mask vs GT apparent dip) ·
  throw MAE (head vs GT) · bbox IoU.
count / closure are ignored (not on 2D lines).

Run on either checkpoint to compare synthetic-only vs real-refined:
  synthetic:    benchmark("stage3_vision.pt")
  real-refined: benchmark("stage4_vision_real.pt")
"""
import numpy as np
import torch

from hybrid.data.real import load_real_split
from hybrid.model.segmenter import VisionModel, field_dice, instance_dips
from hybrid.checkpoints import load_vision

device = torch.device("cuda")


def _bbox_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@torch.no_grad()
def benchmark(ckpt="stage4_vision_real.pt"):
    scenes, tr, te = load_real_split()
    net = load_vision(VisionModel().to(device), ckpt)
    dices, dip_err, thr_err, ious = [], [], [], []
    for s in te:
        seg = net(s["smap"], s["hw"])
        dices.append(field_dice(seg[0], s["fault_field"]))
        pred = sorted(instance_dips(seg[0]))
        gt = sorted(float(o["meas"][0]) for o in s["objs"])
        for i, g in enumerate(gt):
            if i < len(pred):
                dip_err.append(abs(pred[i] - g))
        facts = net.measure(s["smap"], s["hw"])
        H, W = s["hw"]
        for i, f in enumerate(facts["faults"][:len(s["objs"])]):
            o = s["objs"][i]
            gb = [o["bbox"][0] * W, o["bbox"][1] * H, o["bbox"][2] * W, o["bbox"][3] * H]
            ious.append(_bbox_iou(f["bbox"], gb))
            if float(o["mmask"][1]) > 0 and f.get("throw") is not None:
                thr_err.append(abs(f["throw"] - float(o["meas"][1])))

    def m(x):
        return f"{np.mean(x):.2f}(n{len(x)})" if x else "n0"
    print(f"[real-bench {ckpt}] fault dice {m(dices)} · dip MAE {m(dip_err)}deg · "
          f"throw MAE {m(thr_err)}ms · bbox IoU {m(ious)}", flush=True)
    print("REAL_BENCH_DONE", flush=True)
    return dict(dice=dices, dip=dip_err, throw=thr_err, iou=ious)


if __name__ == "__main__":
    benchmark("stage3_vision.pt")     # synthetic-only baseline on real held-out
