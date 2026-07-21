"""Train the main model end to end.

Stage 2 : vision front-end (detector) -> per-fault dips.
Stage 3 : facts-bridge narrator -> copies the numbers into grounded language.
Then evaluate: held-out copy score + faithfulness swap.

Run:  python -m hybrid.train.train

Stage 1 (the geology adapter) is built once, out of band, via
`python -m hybrid.train.stage1_geology`, and loaded frozen here.
"""
import random
from pathlib import Path

import torch

from hybrid.model.scenes import build_scenes, MEAS_SCALE
from hybrid.model.narrator import Narrator, faults_of, scene_facts
from hybrid.train.stage2_detector import (
    train_detector, detected_facts, measure_accuracy, DIP_STRATA,
)
from hybrid.train.stage2_grounding import train_grounding
from hybrid.train.stage3_narrator import train_narrator
from hybrid.test.evaluate import evaluate
from hybrid.inference.infer import collect_demo, save_overlays
from hybrid.checkpoints import save_vision, save_narrator

device = torch.device("cuda")
SEED = 42
# ---- fast overfit config: cap the dataset so the whole model trains in ~15 min.
# Raise these (or set OVERFIT=False) for a full generalization run.
OVERFIT = True        # quick overfit (verify pipeline before real data)
SCENE_CAP = 24        # (only used when OVERFIT)
VIS_EPOCHS = 120
LM_EPOCHS = 80       # grounded-narration epochs (Stage 3, fuse only)
EVAL_N = 5            # scenes per split in eval (greedy gen is slow)


def load_split():
    rng = random.Random(SEED)
    scenes = [s for s in build_scenes() if faults_of(s["objs"])]
    idx = list(range(len(scenes))); rng.shuffle(idx)
    cut = int(len(idx) * 0.75)
    return scenes, [scenes[i] for i in idx[:cut]], [scenes[i] for i in idx[cut:]], rng


def main():
    scale = MEAS_SCALE.to(device)
    scenes, tr, te, rng = load_split()
    tr_v = tr[:SCENE_CAP] if OVERFIT else tr
    print(f"[train] scenes {len(scenes)} · train {len(tr)} (vision {len(tr_v)}) · test {len(te)}", flush=True)

    def report_acc(tag):
        """Mask/dip accuracy. PRED = dip via predicted mask (real); GT = via GT mask (ceiling)."""
        def fmt(mn):
            return f"{mn[0]:.1f}(n{mn[1]})" if mn[0] is not None else "n0"
        for name, split in (("train", tr_v), ("test(held-out)", te[:SCENE_CAP])):
            a = measure_accuracy(net, split, scale)
            print(f"[acc {tag}] {name}: count MAE {fmt(a['count'])} · dice {fmt(a['dice'])} · "
                  f"dip(pred) {fmt(a['dip_pred_all'])} · dip(GT) {fmt(a['dip_gt_all'])}", flush=True)

    # ---- Stage 2a vision: dense segmenter + throw (frozen from here on) ----
    net = train_detector(tr_v, rng, scale, epochs=VIS_EPOCHS)
    report_acc("vision")
    save_vision(net, "stage3_vision.pt")

    # ---- Stage 2b grounding: fact-preamble copy grounds the shared latent ----
    # image -> GT facts: the target's fact preamble is built from these (correspondence), and
    # facts_to_kv injects the SAME structure inference uses — so every marker has a home.
    facts_by_img = {s["img"]: scene_facts(s) for s in tr_v}
    nar = Narrator()
    train_grounding(nar, facts_by_img)

    # ---- Stage 3: fuse aligns the grounded narration (geology+grounding frozen).
    # Target = fact preamble (copy zone, from measured facts) + grounded answer (free reasoning).
    train_narrator(nar, facts_by_img, epochs=LM_EPOCHS)
    save_narrator(nar, "stage3_narrator.pt")

    # ---- end-to-end facts + overlays ----
    te_f = sorted([s for s in te if faults_of(s["objs"])],
                  key=lambda s: faults_of(s["objs"])[0])
    picks = [te_f[i] for i in sorted(set(
        [int(round(q * (len(te_f) - 1))) for q in (0.0, 0.25, 0.5, 0.75, 1.0)]))] if te_f else []
    det_facts = {id(s): detected_facts(net, s) for s in te[:5] + picks}
    demo = collect_demo(net, nar, picks, scale)

    # ---- evaluate + visual overlays ----
    evaluate(nar, tr_v, te[:EVAL_N], det_facts)
    paths = save_overlays(demo, Path("hybrid/inference_outputs"))
    print(f"[demo] saved {len(paths)} overlays -> hybrid/inference_outputs", flush=True)


if __name__ == "__main__":
    main()
