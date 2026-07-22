"""Inference capability test — probe the trained model with three question types on
held-out scenes, end-to-end from the detector's measured facts, and write overlays.

- grounding : the model must COPY the measured facts (counts, dips).
- geology   : domain reasoning (prospectivity, trap type) from the grounded latent.
- mixed     : both — reason about the facts.
"""
import random
from pathlib import Path

import torch

from hybrid.model.scenes import build_scenes, MEAS_SCALE
from hybrid.model.narrator import faults_of, scene_facts, facts_to_kv
from hybrid.checkpoints import load_full
from hybrid.train.stage2_detector import detected_facts
from hybrid.inference.infer import collect_demo, save_overlays

device = torch.device("cuda")

GROUNDING_Q = "How many faults are in this section and what is each fault's dip?"
GEOLOGY_Q = "What kind of hydrocarbon trap could these faults form?"
MIXED_Q = "Given these faults and their dips, is the structure prospective for hydrocarbons? Explain."


def held_out(n=4):
    rng = random.Random(42)
    scenes = [s for s in build_scenes() if faults_of(s["objs"])]
    idx = list(range(len(scenes))); rng.shuffle(idx)
    te = [scenes[idx[i]] for i in range(int(len(idx) * 0.75), len(idx))]
    return te[:n]


def _mem(tag):
    a = torch.cuda.memory_allocated() / 1e9
    r = torch.cuda.memory_reserved() / 1e9
    print(f"[mem {tag}] alloc {a:.2f}G reserved {r:.2f}G", flush=True)


def main():
    te = held_out(4)                                           # build scenes, keep only 4 held-out
    import gc
    gc.collect(); torch.cuda.empty_cache()
    _mem("after scenes")
    net, nar = load_full()
    _mem("after load_full")
    lines = ["# Inference capability test — grounding / geology / mixed\n"]
    def gen(kv, q=None):
        try:
            out = nar.narrate(kv, question=q, max_new_tokens=100)
        except Exception as e:
            out = f"(gen error: {e})"
        torch.cuda.empty_cache()
        return " ".join(out.split())[:320]

    for k, s in enumerate(te):
        _mem(f"scene {k+1} start")
        facts = detected_facts(net, s)                         # real end-to-end (detector facts)
        kv = facts_to_kv(facts)
        gt = [round(f["dip"], 1) for f in scene_facts(s)["faults"]]
        det = [round(f["dip"], 1) for f in facts["faults"]]
        lines += [f"## Scene {k+1} — GT dips {gt} · detected {det}",
                  f"**[base narration]** {gen(kv)}",
                  f"**[grounding Q]** {GROUNDING_Q}\n  {gen(kv, GROUNDING_Q)}",
                  f"**[geology Q]** {GEOLOGY_Q}\n  {gen(kv, GEOLOGY_Q)}",
                  f"**[mixed Q]** {MIXED_Q}\n  {gen(kv, MIXED_Q)}\n"]
        torch.cuda.empty_cache()
        print(f"[infer] scene {k+1}/{len(te)} done", flush=True)

    demo = collect_demo(net, nar, te, MEAS_SCALE.to(device))
    paths = save_overlays(demo, Path("hybrid/inference_outputs"))
    out = Path("hybrid/experiments/main_model/INFER_TEST.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"[infer] report -> {out}", flush=True)
    print(f"[infer] overlays -> {paths}", flush=True)
    print("INFER_DONE", flush=True)


if __name__ == "__main__":
    main()
