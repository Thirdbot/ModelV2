"""Stage 3 — grounded narration (fuse alignment; segmentor frozen).

Trains ONLY the fuse combiner (geology + grounding frozen) to produce the dataset's
grounded narration — the tagged <evidence>...</evidence> <think></think> <answer>...
</answer> chain — with every number COPIED from the injected role-tagged digit facts.
The segmentor is NOT touched here (co-refine removed: its held-out benefit was unproven
and, like the aux heads, it hit the generalization gap; mask generalization is the
real-field stage's job). The LM narrates from the injected numbers only — decoupled.

`reason` column is empty (no reason data) — the <think> slot stays empty, to be filled
later by a tiny reason set. Reasoning still runs through the grounded latent. The fact
preamble (copy zone) is built from the measured facts; the answer is the LM's grounded
reasoning — nothing about the reasoning is templated.
"""
import torch

from hybrid.data.dataset import load_local_csv
from hybrid.model.scenes import CSV
from hybrid.model.narrator import facts_to_kv, narration_target

LM_EPOCHS = 150
MAX_ROWS = 40


def narration_rows(facts_by_img):
    """(injected facts, preamble+answer target). Inject facts_to_kv (same as inference);
    target = the fact preamble (copy) + the row's grounded answer (reasoning). One row per
    (image, answer) so the reasoning varies while the fact preamble stays correspondence-exact."""
    out = []
    for r in load_local_csv(csv_path=CSV):
        facts = facts_by_img.get((r.get("image_paths") or [None])[0])
        if facts is None or not facts["faults"]:
            continue
        out.append((facts_to_kv(facts), narration_target(facts, r.get("answer") or "")))
        if len(out) >= MAX_ROWS:
            break
    return out


def train_narrator(nar, facts_by_img, epochs=LM_EPOCHS):
    """Stage 3: train the fuse (geology+grounding frozen) on the grounded narration."""
    nar.set_stage("s3")
    data = narration_rows(facts_by_img)
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
