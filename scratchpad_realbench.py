"""Build real Smeaheia scenes ONCE (cached, CPU-offloaded), then benchmark synthetic
vision on the real held-out. Real lines are huge (up to 2301x11657) and the GPU is 5.67GB,
so every scene is built and stored on CPU; only one scene at a time goes to GPU. No repo
edits — reuses the loader's geometry/stitch, just offloads memory.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import hybrid.data.real as real
from hybrid.model.segmenter import VisionModel, field_dice, instance_dips
from hybrid.checkpoints import load_vision

device = torch.device("cuda")
CACHE = Path("data/real_data/real_scenes.pt")


def build_cpu_scenes():
    """Loader's build loop, but each scene lands on CPU (smap.cpu()) and GPU is freed
    between lines so the 63 big real sections fit in 5.67GB."""
    faults = real.load_fault_sticks()
    horizons = real.load_horizons()
    print(f"[real] {len(faults)} faults · {len(horizons)} horizons", flush=True)
    enc = real.NcsEncoder().to(device).eval()
    real.RENDER_DIR.mkdir(parents=True, exist_ok=True)
    scenes = []
    for segy in sorted(real.SEGY_DIR.rglob("*")):
        if segy.is_dir() or segy.suffix.lower() in (".xml", ".txt", ".pdf"):
            continue
        try:
            geo = real.line_geometry(segy, faults, horizons)
        except Exception as e:
            print(f"[real] skip {segy.name}: {e}", flush=True); continue
        if geo is None:
            continue
        png = real.RENDER_DIR / f"{segy.parent.name}__{segy.name}.png"
        Image.fromarray(geo["img_arr"]).save(png)
        try:
            smap, _ = real.stitch(enc, str(png))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[real] OOM skip {png.name} hw={geo['hw']}", flush=True); continue
        objs = [{**o, "mask": o["mask"].cpu(), "meas": o["meas"].cpu(), "mmask": o["mmask"].cpu()}
                for o in geo["objs"]]
        scenes.append(dict(smap=smap.cpu(), hw=geo["hw"], objs=objs, img=str(png),
                           fault_field=geo["fault_field"].cpu(),
                           closure_field=torch.zeros(geo["hw"])))
        del smap; torch.cuda.empty_cache()
        print(f"[real] {png.name}: {len(objs)} faults · hw={geo['hw']}", flush=True)
    return scenes


def to_gpu(s):
    return dict(smap=s["smap"].to(device), hw=s["hw"], img=s["img"],
                fault_field=s["fault_field"].to(device),
                objs=[{**o, "mask": o["mask"].to(device), "meas": o["meas"].to(device),
                       "mmask": o["mmask"].to(device)} for o in s["objs"]])


def _iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    if CACHE.exists():
        cpu_scenes = torch.load(CACHE, weights_only=False)
        print(f"[cache] loaded {len(cpu_scenes)} scenes", flush=True)
    else:
        cpu_scenes = build_cpu_scenes()
        torch.save(cpu_scenes, CACHE)
        print(f"[cache] saved {len(cpu_scenes)} scenes -> {CACHE}", flush=True)

    import random
    rng = random.Random(real.SEED)
    idx = list(range(len(cpu_scenes))); rng.shuffle(idx)
    cut = int(len(idx) * 0.75)
    te = [cpu_scenes[i] for i in idx[cut:]]
    print(f"[split] {len(cpu_scenes)} scenes · held-out {len(te)}", flush=True)

    net = load_vision(VisionModel().to(device), "stage3_vision.pt")
    dices, dip_err, thr_err, ious = [], [], [], []
    with torch.no_grad():
        for cs in te:
            s = to_gpu(cs)
            seg = net(s["smap"], s["hw"])
            dices.append(field_dice(seg[0], s["fault_field"]))
            pred = sorted(instance_dips(seg[0]))
            gt = sorted(float(o["meas"][0]) for o in s["objs"])
            for i, g in enumerate(gt):
                if i < len(pred):
                    dip_err.append(abs(pred[i] - g))
            facts = net.measure(s["smap"], s["hw"])
            H, W = s["hw"]
            for i, f in enumerate(facts["faults"][:len(s["objs"])]):
                o = s["objs"][i]
                gb = [o["bbox"][0]*W, o["bbox"][1]*H, o["bbox"][2]*W, o["bbox"][3]*H]
                ious.append(_iou(f["bbox"], gb))
                if float(o["mmask"][1]) > 0 and f.get("throw") is not None:
                    thr_err.append(abs(f["throw"] - float(o["meas"][1])))
            del s, seg; torch.cuda.empty_cache()

    def m(x):
        return f"{np.mean(x):.2f}(n{len(x)})" if x else "n0"
    print(f"[real-bench stage3_vision.pt] fault dice {m(dices)} · dip MAE {m(dip_err)}deg · "
          f"throw MAE {m(thr_err)}ms · bbox IoU {m(ious)}", flush=True)
    print("REAL_BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
