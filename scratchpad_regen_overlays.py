import random
from pathlib import Path
import torch
from hybrid.model.scenes import build_scenes, MEAS_SCALE
from hybrid.model.narrator import faults_of
from hybrid.checkpoints import load_full
from hybrid.inference.infer import collect_demo, save_overlays

rng = random.Random(42)
scenes = [s for s in build_scenes() if faults_of(s["objs"])]
idx = list(range(len(scenes))); rng.shuffle(idx)
te = [scenes[idx[i]] for i in range(int(len(idx)*0.75), len(idx))][:4]
net, nar = load_full()
demo = collect_demo(net, nar, te, MEAS_SCALE.to(torch.device("cuda")))
paths = save_overlays(demo, Path("hybrid/inference_outputs"))
print("OVERLAY_DONE", len(paths), flush=True)
