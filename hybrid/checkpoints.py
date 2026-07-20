"""Explicit per-stage checkpoints.

Saves the TRAINABLE weights of each stage so the model is LOADABLE (standalone
inference) and RETRAINABLE (resume a stage) without redoing earlier stages.
Stage 1 (geology) is the cached adapter dir; stages 2-3 save here. Only trainable
params are saved (dense segmenter + throw head; grounding+fuse LoRA + FactTokens)
and loaded with strict=False, so the frozen 4-bit base reloads fresh each time.
"""
import torch
from pathlib import Path

CKPT = Path("hybrid/checkpoints")
device = torch.device("cuda")


def save_vision(net, name="stage2_vision.pt"):
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), CKPT / name)
    print(f"[ckpt] vision -> {CKPT / name}", flush=True)


def load_vision(net, name="stage2_vision.pt"):
    # strict=False: a checkpoint from before the multi-task heads (box/attr) loads
    # its seg+throw weights and leaves the new heads fresh; a full one loads whole.
    net.load_state_dict(torch.load(CKPT / name, map_location=device), strict=False)
    return net.eval()


def _lora_state(dec, names=("grounding", "fuse")):
    return {k: v.detach().cpu() for k, v in dec.state_dict().items()
            if "lora_" in k and any(nm in k for nm in names)}


def save_narrator(nar, name="stage3_narrator.pt"):
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save({"lora": _lora_state(nar.dec), "facts": nar.facts_mod.state_dict()},
               CKPT / name)
    print(f"[ckpt] narrator -> {CKPT / name}", flush=True)


def load_narrator(nar, name="stage3_narrator.pt"):
    d = torch.load(CKPT / name, map_location=device)
    nar.dec.load_state_dict(d["lora"], strict=False)   # grounding+fuse LoRA only
    nar.facts_mod.load_state_dict(d["facts"])
    return nar


def load_full():
    """Reconstruct the trained model for inference: the segmentor + narrator.
    Returns (vision_net, narrator)."""
    from hybrid.model.segmenter import VisionModel
    from hybrid.model.narrator import Narrator
    net = load_vision(VisionModel().to(device), "stage3_vision.pt")
    nar = Narrator(); load_narrator(nar); nar.set_stage("s3"); nar.eval_mode()
    return net, nar
