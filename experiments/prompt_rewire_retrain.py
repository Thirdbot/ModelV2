"""Confirmation retrain for the chatml prompt rewire (facts→system, unified skeleton).

Trains S2 grounding + S3 narrator under the NEW build_prompt, then reports held-out
copy score (do injected dips survive into the output?) + sample chains (so we can see
the evidence/think/answer structure and whether <think> is empty). LM-only: uses GT
facts (scene_facts), so no vision reload. This is the "retrain + summarize" gate before
deciding architecture fixes.
"""
import random
from pathlib import Path

import torch

import hybrid.model.scenes as sc
sc.MAX_SCENES = 30

from hybrid.model.scenes import build_scenes, MEAS_SCALE
from hybrid.model.narrator import Narrator, faults_of, scene_facts, facts_to_kv, K_DIP
from hybrid.train.stage2_grounding import train_grounding
import hybrid.train.stage3_narrator as s3

s3.MAX_ROWS = 150
S3_EPOCHS = 40
device = torch.device("cuda")

scenes = [s for s in build_scenes() if faults_of(s["objs"])]
rng = random.Random(0)
idx = list(range(len(scenes))); rng.shuffle(idx)
cut = int(len(idx) * 0.8)
tr = [scenes[i] for i in idx[:cut]]; te = [scenes[i] for i in idx[cut:]]
print(f"[rewire] scenes {len(scenes)} · train {len(tr)} · test {len(te)}", flush=True)

facts_by_img = {s["img"]: scene_facts(s) for s in tr}
nar = Narrator()

print("[rewire] === Stage 2 grounding (chatml, facts→system) ===", flush=True)
train_grounding(nar, facts_by_img)
print("[rewire] === Stage 3 narrator (chatml, dataset instruction + question) ===", flush=True)
s3.train_narrator(nar, facts_by_img, epochs=S3_EPOCHS)
nar.eval_mode()

Q = "How many faults are present and what is each fault's dip?"
hit = tot = 0
samples = []
for s in te:
    facts = scene_facts(s)
    if not facts["faults"]:
        continue
    dips = [v for k, v in facts_to_kv(facts) if k == K_DIP]
    out = nar.generate(facts, question=Q, max_new_tokens=140)
    for d in dips:
        tot += 1; hit += (d in out)
    if len(samples) < 3:
        samples.append((dips, out))

print(f"[rewire COPY] held-out copy {hit/tot if tot else 0:.2f} ({hit}/{tot})", flush=True)
for i, (dips, out) in enumerate(samples):
    print(f"--- sample {i} · injected dips {dips} ---", flush=True)
    print(out[:600], flush=True)
print("REWIRE_DONE", flush=True)

Path("experiments/checkpoints").mkdir(parents=True, exist_ok=True)
torch.save({"lora": {k: v.detach().cpu() for k, v in nar.dec.state_dict().items() if "lora_" in k},
            "facts": nar.facts_mod.state_dict()}, "experiments/checkpoints/narrator_chatml.pt")
print("saved -> experiments/checkpoints/narrator_chatml.pt", flush=True)
