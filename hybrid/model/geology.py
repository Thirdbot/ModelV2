"""Stage-1 geology adapter — cached loader for the main model.

`load_geology_adapter` returns the cached Stage-1 LoRA adapter dir (built once by
`hybrid.train.stage1_geology`, out of band because Unsloth must be imported before
transformers/peft). The decoder loads this frozen geology adapter, then adds the
trainable grounding adapter on top.
"""
import hashlib
import json
from pathlib import Path

CKPT = Path("hybrid/experiments/ckpt")

# Stage-1 config — the hash of this dict names the cached adapter dir, so it must
# stay identical to whatever produced the cache. Consumed by both this loader and
# the `stage1_geology` training script.
GEOLOGY_CFG = dict(
    base_model="Qwen/Qwen2.5-1.5B-Instruct",
    max_train_samples=5000, max_eval_samples=64, max_seq_length=1024,
    lora_r=16, lora_alpha=16, num_epochs=1, learning_rate=2e-5,
    batch_size=1, grad_accum=8, save_mode="adapter",
    evidence_placeholder=True,
)


def _hash(d: dict) -> str:
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:10]


def adapter_dir(geology_cfg: dict = GEOLOGY_CFG) -> Path:
    """Cache dir this config hashes to (built or not)."""
    return CKPT / f"stage1_{_hash(geology_cfg)}"


def load_geology_adapter(geology_cfg: dict = GEOLOGY_CFG) -> str:
    """Return the cached geology adapter dir (raise if it hasn't been built)."""
    merged = adapter_dir(geology_cfg)
    if (merged / "stage1_config.json").exists():
        print(f"[geology] cache hit -> {merged}", flush=True)
        return str(merged)
    raise RuntimeError(
        f"Geology adapter not cached at {merged}; build it first: "
        f"python -m hybrid.train.stage1_geology")
