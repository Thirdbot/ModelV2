"""Stage 2 (language half) — evidence-copy grounding.

Trains the grounding adapter to COPY injected fact values into the dataset's
evidence text, grounding the shared linguistic latent (per the staging + fuse
design). geology frozen, fuse inactive. Grounding is frozen afterwards, when
stage 3 switches the decoder to train the fuse combiner.
"""
import re

import torch

from hybrid.data.dataset import load_local_csv
from hybrid.model.scenes import CSV
from hybrid.model.narrator import evidence_kv, structured_evidence

GROUND_EPOCHS = 15
MAX_ROWS = 40


def evidence_rows():
    """(role-tagged facts, cleaned evidence narration) for rows carrying values.
    Each value is role-tagged (dip/throw/area/count) by evidence_kv."""
    out = []
    for r in load_local_csv(csv_path=CSV):
        ev = r.get("evidence") or ""
        kv = evidence_kv(ev)                            # role-tagged, from the raw evidence
        if not kv:
            continue
        out.append((kv, structured_evidence(ev)))
        if len(out) >= MAX_ROWS:
            break
    return out


def train_grounding(nar, epochs=GROUND_EPOCHS):
    nar.set_stage("s2")
    data = evidence_rows()
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
