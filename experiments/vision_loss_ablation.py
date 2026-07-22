"""Vision loss ablation — does any LOSS-side change beat the resolution ceiling?

Compares, on ONE fast held-out split, the fault-channel loss:
  baseline (BCE+dice) vs +focal (BCE-focal + dice).
Focal is the imbalance lever (fewer false positives / less over-detection), so it
shows on HELD-OUT over-detection, not the overfit ceiling (~0.72 dice / ~8.5 deg
dip, which is resolution-bound and loss-invariant). clDice was ablated out (it
cost held-out dice for the same over-detection gain) — see the earlier
ablation_summary.json / cldice.json receipts.

Loss knobs are the main-model globals (segmenter.POS_WEIGHT / FOCAL_GAMMA /
CLDICE_W) — set here, not forked. Weights -> experiments/checkpoints/<name>.pt,
metrics -> experiments/results/<name>.json (+ ablation_summary.json).

Run:  PYTHONPATH=. .venv/bin/python experiments/vision_loss_ablation.py
"""
import json
import random
from pathlib import Path

import torch

import hybrid.model.scenes as sc
import hybrid.model.segmenter as seg
from hybrid.model.scenes import build_scenes, MEAS_SCALE
from hybrid.model.narrator import faults_of
from hybrid.model.segmenter import instance_dips
from hybrid.train.stage2_detector import train_detector, measure_accuracy

device = torch.device("cuda")
ROOT = Path("experiments")
(ROOT / "results").mkdir(parents=True, exist_ok=True)
(ROOT / "checkpoints").mkdir(parents=True, exist_ok=True)

# --- fast held-out config (small + safe on a 15GB laptop; relative comparison) ---
N_SCENES = 50
EPOCHS = 80
SEED = 0

CONFIGS = [
    dict(name="baseline_bce_dice", pos_weight=50.0, focal_gamma=0.0),
    dict(name="focal",             pos_weight=50.0, focal_gamma=2.0),
]


def _fmt(mn):
    return f"{mn[0]:.3f}(n{mn[1]})" if mn and mn[0] is not None else "n0"


@torch.no_grad()
def _count_bias(net, split):
    """Signed pred-minus-GT fault count (positive = OVER-detection). Focal's target."""
    bias = []
    for s in split:
        gt = faults_of(s["objs"])
        if not gt:
            continue
        pred = instance_dips(net(s["smap"], s["hw"])[0])
        bias.append(len(pred) - len(gt))
    return (sum(bias) / len(bias) if bias else None, len(bias))


def _summ(net, split, scale):
    a = measure_accuracy(net, split, scale)
    return dict(
        dice=list(a["dice"]),
        dip_pred=list(a["dip_pred_all"]),
        dip_gt=list(a["dip_gt_all"]),
        count_mae=list(a["count"]),
        count_bias=list(_count_bias(net, split)),
    )


def run():
    sc.MAX_SCENES = N_SCENES
    scale = MEAS_SCALE.to(device)
    scenes = [s for s in build_scenes() if faults_of(s["objs"])]
    rng = random.Random(SEED)
    idx = list(range(len(scenes)))
    rng.shuffle(idx)
    cut = int(len(idx) * 0.75)
    tr = [scenes[i] for i in idx[:cut]]
    te = [scenes[i] for i in idx[cut:]]
    print(f"[ablation] scenes {len(scenes)} · train {len(tr)} · test {len(te)}", flush=True)

    summary = []
    for cfg in CONFIGS:
        seg.POS_WEIGHT = cfg["pos_weight"]
        seg.FOCAL_GAMMA = cfg["focal_gamma"]
        torch.manual_seed(SEED)
        net = train_detector(list(tr), random.Random(SEED), scale, epochs=EPOCHS)

        res = dict(cfg=cfg, train=_summ(net, tr, scale), test=_summ(net, te, scale))
        summary.append(res)
        torch.save(net.state_dict(), ROOT / "checkpoints" / f"{cfg['name']}.pt")
        json.dump(res, open(ROOT / "results" / f"{cfg['name']}.json", "w"), indent=2)
        t = res["test"]
        print(f"[{cfg['name']:>18}] TEST dice {_fmt(t['dice'])} · dip(pred) {_fmt(t['dip_pred'])} "
              f"· count_bias {_fmt(t['count_bias'])} · dip(GT) {_fmt(t['dip_gt'])} "
              f"|| TRAIN dice {_fmt(res['train']['dice'])}", flush=True)

    json.dump(summary, open(ROOT / "results" / "ablation_summary.json", "w"), indent=2)
    print("ABLATION_DONE", flush=True)


if __name__ == "__main__":
    run()
