"""Stage 4 — syn -> real VISION refinement (the LAST stage).

Loads the synthetic-refined vision front-end and fine-tunes DETECTION / MASKING on
real Smeaheia GT (real fault masks + throw). This is a syn->real transfer for the
VISION part only:
  - NCS ViT: frozen (smap is precomputed).
  - dense segmenter + throw head: TRAINED on real GT (seg_loss + throw_loss).
  - narration (LM / grounding / fuse / bridge): FROZEN — not even loaded here. The
    narrator is already strong from the synthetic stages; this stage only makes the
    real masks/values good so the (unchanged) measure_instances reads real facts.

Reproducible: seeded split (hybrid/data/real.load_real_split). Start from
stage3_vision.pt (synthetic), save stage4_vision_real.pt (retrainable).
"""
import random

import torch

from hybrid.data.real import load_real_split
from hybrid.model.segmenter import VisionModel, vision_loss
from hybrid.checkpoints import load_vision, save_vision

device = torch.device("cuda")
REAL_EPOCHS = 120
LR = 3e-4
SEED = 0


def train_realfield(epochs=REAL_EPOCHS):
    scenes, tr, te = load_real_split()
    print(f"[real] scenes {len(scenes)} · train {len(tr)} · test {len(te)}", flush=True)
    net = VisionModel().to(device)
    load_vision(net, "stage3_vision.pt")           # start from the synthetic-refined vision
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-4)
    net.train()
    rng = random.Random(SEED)
    for ep in range(epochs):
        rng.shuffle(tr)
        tot = 0.0
        for s in tr:
            opt.zero_grad()
            loss = vision_loss(net, s)             # real fault masks + throw; narration untouched
            loss.backward(); opt.step(); tot += loss.item()
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"[real] ep {ep} loss {tot/max(1, len(tr)):.3f}", flush=True)
    net.eval()
    save_vision(net, "stage4_vision_real.pt")      # real-refined vision (retrainable / loadable)
    return net, tr, te


if __name__ == "__main__":
    train_realfield()
