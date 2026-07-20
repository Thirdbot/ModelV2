"""Stage 3 — grounded narration (fuse alignment; segmentor frozen).

Trains ONLY the fuse combiner (geology + grounding frozen) to produce the dataset's
grounded narration — the tagged <evidence>...</evidence> <think></think> <answer>...
</answer> chain — with every number COPIED from the injected role-tagged digit facts.
The segmentor is NOT touched here (co-refine removed: its held-out benefit was unproven
and, like the aux heads, it hit the generalization gap; mask generalization is the
real-field stage's job). The LM narrates from the injected numbers only — decoupled.

`reason` column is empty (no reason data) — the <think> slot stays empty, to be filled
later by stage-4 RL / a tiny reason set. Reasoning still runs through the grounded latent.
Output is the LM's; nothing is templated.
"""
import re

import torch

from hybrid.data.dataset import load_local_csv
from hybrid.model.scenes import CSV
from hybrid.model.narrator import evidence_kv, structured_narration

NUMS = re.compile(r"<nums>([-\d.]+)</nums>")
LM_EPOCHS = 150
MAX_ROWS = 40


def narration_rows():
    """(role-tagged facts kv, grounded narration target). Tags cleaned to
    <evidence>/<think>/<answer>/<SEG>, numbers plain; role-tagged numbers injected —
    the LM copies them into the chain."""
    out = []
    for r in load_local_csv(csv_path=CSV):
        ev, an = r.get("evidence") or "", r.get("answer") or ""
        kv = evidence_kv(ev)
        if not kv:
            continue
        out.append((kv, structured_narration(ev, an)))
        if len(out) >= MAX_ROWS:
            break
    return out


def train_narrator(nar, epochs=LM_EPOCHS):
    """Stage 3: train the fuse (geology+grounding frozen) on the grounded narration."""
    nar.set_stage("s3")
    data = narration_rows()
    opt = torch.optim.AdamW(nar.trainable_params(), lr=1e-4)
    nar.train_mode()
    print(f"[narrator] grounded narration on {len(data)} rows", flush=True)
    for ep in range(epochs):
        tot = 0.0
        for kv, target in data:
            opt.zero_grad()
            loss = nar.ground_loss(kv, target)
            loss.backward(); opt.step(); tot += loss.item()
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"[narrator] ep {ep} loss {tot/max(1, len(data)):.3f}", flush=True)
