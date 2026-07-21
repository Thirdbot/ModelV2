"""Stage 2 (language half) — evidence-copy grounding.

Trains the grounding adapter to COPY injected fact values into the dataset's
evidence text, grounding the shared linguistic latent (per the staging + fuse
design). geology frozen, fuse inactive. Grounding is frozen afterwards, when
stage 3 switches the decoder to train the fuse combiner.
"""
import re

import torch

from hybrid.model.narrator import facts_to_kv, grounding_target

GROUND_EPOCHS = 15
MAX_ROWS = 40


def evidence_rows(facts_by_img):
    """(injected facts, fact-preamble target) — one per scene. Inject facts_to_kv (the SAME
    structure inference injects) and target = the fact preamble, so every marker has a home
    (correspondence). Built from the measured facts at train time — scalable, no data change."""
    out = []
    for img, facts in facts_by_img.items():
        if not facts["faults"]:
            continue
        out.append((facts_to_kv(facts), grounding_target(facts)))
        if len(out) >= MAX_ROWS:
            break
    return out


def train_grounding(nar, facts_by_img, epochs=GROUND_EPOCHS):
    nar.set_stage("s2")
    data = evidence_rows(facts_by_img)
    opt = torch.optim.AdamW(nar.trainable_params(), lr=1e-4)
    nar.train_mode()
    for ep in range(epochs):
        tot = 0.0
        for kv, target in data:
            opt.zero_grad()
            loss = nar.ground_loss(kv, target)
            loss.backward(); opt.step(); tot += loss.item()
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"[grounding] ep {ep} loss {tot/max(1, len(data)):.3f}", flush=True)
