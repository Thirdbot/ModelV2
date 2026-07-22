# experiments/

Self-contained experiment drivers, kept out of the main training path.

- One experiment = one file here (`*.py`), run with the repo venv:
  `PYTHONPATH=. .venv/bin/python experiments/<name>.py`
- **Loss/knobs live in the main model files** (`hybrid/model/segmenter.py`:
  `POS_WEIGHT`, `FOCAL_GAMMA`, `CLDICE_W`). Experiments just *set* these globals
  before training — they don't fork the model code.
- **Weights → `experiments/checkpoints/`** (git-ignored). Main-model run
  checkpoints stay in the main location (`hybrid/checkpoints`) and are never
  written here.
- **Metrics/logs → `experiments/results/`** (small JSON, tracked).

## vision_loss_ablation.py
Does any LOSS-side change lift the fault mask past the resolution ceiling?
Compares BCE+dice (baseline) · +clDice · +focal · +focal+clDice on a fast
held-out split. Reference: the 8-scene overfit ceiling is ~0.72 dice / ~8.5° dip
(both clDice-on and -off), i.e. the wall is feature resolution, not the loss —
this ablation checks whether focal helps the *generalization* axis (held-out
over-detection) even though it can't move the overfit ceiling.
