"""Real-field scene builder — Smeaheia 2D lines -> the SAME scene structure the
synthetic pipeline uses, so the vision code (dense segmenter + measure_instances)
runs unchanged.

Source: Smeaheia Dataset (c) Equinor & Gassnova, CO2DataShare — modified CC BY 4.0,
ATTRIBUTION REQUIRED (see data/real_data/README.md).

Per 2D SEG-Y sub-line:
  SEG-Y            -> amplitude image (samples x traces) + per-trace (X,Y) + TWT axis
  3D fault sticks  -> PROJECT onto the line (world-XY match) -> per-fault polyline
                      (trace col, TWT row) -> rasterized fault mask + bbox
  apparent dip     -> read off the mask with the SAME reader the model uses (RANSAC)
  throw            -> horizon offset across the fault (see `_throw`, first-pass)
count / closure / class are NOT on 2D lines -> ignored (real = fault channel only).
true dip needs 3D + view -> NOT an ML target; kept only as an eval-extra (o["true_dip"]).

Scene contract (identical to hybrid/model/scenes.build_scenes):
  scene = dict(smap, hw, objs, img, fault_field, closure_field)
  obj   = dict(cls=1, bbox=[x1,y1,x2,y2](norm), mask=(H,W), meas=[dip,throw,0],
               mmask=[1, throw?1:0, 0], true_dip=None)
"""
import math
import random
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from hybrid.model.encoder import NcsEncoder, stitch
from hybrid.model.scenes import dilate
from hybrid.model.segmenter import _line_dip

try:
    import segyio
except Exception:
    segyio = None

device = torch.device("cuda")

REAL_ROOT = Path("data/real_data")
SEGY_DIR = REAL_ROOT / "segy"                 # extracted SEG-Y: <survey>/<subline> (no ext)
FAULT_ZIP = REAL_ROOT / "raw/fault_sticks.zip"
HORIZON_DIR = REAL_ROOT / "horizons"          # POINTZ horizon shapefiles (.shp only needed)
RENDER_DIR = REAL_ROOT / "render"             # cached PNGs for encoder + overlays
MATCH_THRESH_M = 150.0                         # fault point within this of a trace -> on the line
MIN_FAULT_PTS = 6                              # min projected points to accept a crossing
HZ_MATCH_M = 90.0                              # horizon point within this of a trace to sample it
THROW_WIN = 45                                 # traces each side of the fault for the throw trend fit
DILATE_R = 3
SEED = 42


# --------------------------- fault sticks (3D world) ---------------------------

def load_fault_sticks():
    """Both fault-stick files -> {namespaced_fault: np.array([[X,Y,Z], ...])}.
    Namespaced by source file so identically-named faults don't merge."""
    faults = {}
    with zipfile.ZipFile(FAULT_ZIP) as z:
        for n in z.namelist():
            base = Path(n).name
            if base not in ("fault_sticks_2010", "fault_Sticks_GN1101_2012"):
                continue
            src = "2010" if "2010" in base else "GN1101"
            for ln in z.read(n).decode("latin-1").splitlines():
                p = ln.split()
                if len(p) < 7:
                    continue
                try:
                    x, y, zz = float(p[3]), float(p[4]), float(p[5])
                except ValueError:
                    continue
                name = f"{src}:{' '.join(p[6:-1])}"
                faults.setdefault(name, []).append((x, y, zz))
    return {k: np.array(v) for k, v in faults.items()}


# --------------------------- SEG-Y -> image + geometry ---------------------------

def _coord_scale(hdr):
    sc = hdr[segyio.TraceField.SourceGroupScalar] or 1
    return (1.0 / abs(sc)) if sc < 0 else (float(sc) if sc > 0 else 1.0)


def read_segy(path):
    """SEG-Y sub-line -> (amplitude (samples,traces), trace X (traces,), trace Y,
    TWT samples (ms)). CDP X/Y from headers (fallback Source X/Y), scaled."""
    with segyio.open(str(path), ignore_geometry=True) as f:
        n = f.tracecount
        data = np.stack([f.trace[i] for i in range(n)], axis=1).astype(np.float32)
        samples = np.array(f.samples, dtype=np.float32)     # TWT ms
        s = _coord_scale(f.header[0])
        cx = np.array([f.header[i][segyio.TraceField.CDP_X] for i in range(n)], float)
        cy = np.array([f.header[i][segyio.TraceField.CDP_Y] for i in range(n)], float)
        if not cx.any():                                    # fall back to source coords
            cx = np.array([f.header[i][segyio.TraceField.SourceX] for i in range(n)], float)
            cy = np.array([f.header[i][segyio.TraceField.SourceY] for i in range(n)], float)
    return data, cx * s, cy * s, samples


def _to_image(data):
    """Amplitude (samples,traces) -> uint8 grayscale, per-trace balanced (AGC-ish)."""
    a = data.copy()
    a -= a.mean(axis=0, keepdims=True)
    scale = np.percentile(np.abs(a), 99) + 1e-6
    a = np.clip(a / scale, -1, 1)
    return ((a + 1) * 127.5).astype(np.uint8)               # (H=samples, W=traces)


# --------------------------- project faults onto the line ---------------------------

def _rasterize(poly, hw):
    H, W = hw
    im = Image.new("L", (W, H), 0)
    ImageDraw.Draw(im).line([(int(c), int(r)) for c, r in poly], fill=1, width=2 * DILATE_R + 1)
    return torch.from_numpy(np.array(im, dtype=np.float32))


def project_faults(faults, cx, cy, samples, hw):
    """For each fault, keep its 3D points within MATCH_THRESH_M of a trace, project to
    (trace col, TWT row), and return [(fault_name, polyline (col,row))] crossings."""
    H, W = hw
    tree = cKDTree(np.c_[cx, cy])
    t0, dt = float(samples[0]), float(samples[1] - samples[0])
    out = []
    for name, P in faults.items():
        d, idx = tree.query(P[:, :2])
        keep = d < MATCH_THRESH_M
        if int(keep.sum()) < MIN_FAULT_PTS:
            continue
        cols = idx[keep].astype(float)
        rows = (P[keep, 2] - t0) / dt                       # TWT(ms) -> sample row
        inb = (rows >= 0) & (rows < H)
        if int(inb.sum()) < MIN_FAULT_PTS:
            continue
        poly = np.c_[cols[inb], rows[inb]]
        poly = poly[np.argsort(poly[:, 1])]                 # order top->bottom for a clean stick
        out.append((name, poly))
    return out


def load_horizons():
    """Key horizons (POINTZ shapefiles, .shp only) -> {name: (Z_abs, cKDTree(XY))}.
    Z is TWT ms (confirmed empirically: Draupne aligns with fault TWT), down-positive."""
    import shapefile
    hz = {}
    for shp in sorted(HORIZON_DIR.glob("*.shp")):
        r = shapefile.Reader(shp=str(shp))
        XY, Z = [], []
        for s in r.iterShapes():
            if s.points and getattr(s, "z", None):
                XY.append(s.points[0]); Z.append(s.z[0])
        if XY:
            hz[shp.stem] = (np.abs(np.array(Z)), cKDTree(np.array(XY)))
    return hz


def _throw(poly, cx, cy, horizons):
    """Throw = TWT offset of a horizon across the fault via a TWO-SIDED trend fit:
    fit the horizon along the line on each side of the fault, extrapolate both to the
    fault column, difference = throw (removes the horizon's regional dip). Max over
    horizons present on both sides; None if none straddles the fault. TWT ms."""
    if not horizons:
        return None
    fc = int(np.median(poly[:, 0]))

    def side(rng, tree, Z):
        ts, zs = [], []
        for t in rng:
            if 0 <= t < len(cx):
                d, i = tree.query([cx[t], cy[t]])
                if d < HZ_MATCH_M:
                    ts.append(t); zs.append(Z[i])
        return np.array(ts, float), np.array(zs, float)

    best = None
    for Z, tree in horizons.values():
        tL, zL = side(range(fc - THROW_WIN, fc - 4), tree, Z)
        tR, zR = side(range(fc + 4, fc + THROW_WIN), tree, Z)
        if len(zL) >= 6 and len(zR) >= 6:
            jump = abs(np.polyval(np.polyfit(tL, zL, 1), fc)
                       - np.polyval(np.polyfit(tR, zR, 1), fc))
            best = jump if best is None else max(best, jump)
    return best


# --------------------------- scene assembly ---------------------------

def line_geometry(segy_path, faults, horizons=None):
    """SEG-Y sub-line -> image array + objs + fault_field, WITHOUT smap (CPU-testable).
    None if no fault crosses this line."""
    data, cx, cy, samples = read_segy(segy_path)
    img = _to_image(data)
    H, W = img.shape
    hw = (H, W)
    crossings = project_faults(faults, cx, cy, samples, hw)
    if not crossings:
        return None
    ff = torch.zeros(hw)
    objs = []
    for name, poly in crossings:
        m = dilate(_rasterize(poly, hw).to(device)).cpu()
        ff = torch.maximum(ff, m)
        dip = _line_dip(poly.astype(float))                 # apparent dip = SAME reader the model uses
        if dip is None:
            continue
        thr = _throw(poly, cx, cy, horizons)                # horizon offset across the fault (TWT ms)
        xs, ys = poly[:, 0], poly[:, 1]
        bbox = [xs.min() / W, ys.min() / H, xs.max() / W, ys.max() / H]
        objs.append(dict(cls=1, bbox=bbox, mask=m,
                         meas=torch.tensor([dip, thr or 0.0, 0.0]),
                         mmask=torch.tensor([1.0, 1.0 if thr is not None else 0.0, 0.0]),
                         true_dip=None, name=name))
    if not objs:
        return None
    return dict(img_arr=img, hw=hw, objs=objs, fault_field=ff)


def build_real_scenes():
    """One scene per SEG-Y sub-line that a fault crosses (synthetic scene structure)."""
    assert segyio is not None, "pip install segyio"
    faults = load_fault_sticks()
    horizons = load_horizons()
    print(f"[real] {len(faults)} faults · {len(horizons)} horizons loaded", flush=True)
    enc = NcsEncoder().to(device).eval()
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    scenes = []
    for segy in sorted(SEGY_DIR.rglob("*")):
        if segy.is_dir() or segy.suffix.lower() in (".xml", ".txt", ".pdf"):
            continue
        try:
            geo = line_geometry(segy, faults, horizons)
        except Exception as e:
            print(f"[real] skip {segy.name}: {e}", flush=True); continue
        if geo is None:
            continue
        png = RENDER_DIR / f"{segy.parent.name}__{segy.name}.png"
        Image.fromarray(geo["img_arr"]).save(png)
        try:
            smap, _ = stitch(enc, str(png))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[real] OOM skip {png.name} hw={geo['hw']}", flush=True); continue
        # OFFLOAD each scene to CPU: real lines are huge (up to 2301x11657) and the GPU is
        # 5.67GB — keeping every scene's smap on GPU OOMs. Move to GPU one at a time downstream.
        objs = [{**o, "mask": o["mask"].cpu(),
                 "meas": o["meas"].cpu(), "mmask": o["mmask"].cpu()} for o in geo["objs"]]
        scenes.append(dict(smap=smap.cpu(), hw=geo["hw"], objs=objs, img=str(png),
                           fault_field=geo["fault_field"].cpu(),
                           closure_field=torch.zeros(geo["hw"])))
        del smap; torch.cuda.empty_cache()
        print(f"[real] {png.name}: {len(objs)} faults · hw={geo['hw']}", flush=True)
    return scenes


def build_real_windows(w_win=160, h_win=512, limit_lines=None):
    """WINDOWED build (memory-safe): one fault-centred window per crossing, cropped to
    ~synthetic scale (h_win x w_win). Each scene's smap is ~1MB (vs ~320MB full-line), so
    peak RAM stays ~1 line (~130MB) + the small accumulated scenes — no freeze. Dip is read
    from the windowed segment; throw from the FULL line (more horizon support)."""
    assert segyio is not None, "pip install segyio"
    faults = load_fault_sticks(); horizons = load_horizons()
    print(f"[real-win] {len(faults)} faults · {len(horizons)} horizons · window {h_win}x{w_win}", flush=True)
    enc = NcsEncoder().to(device).eval()
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    scenes = []
    lines = [p for p in sorted(SEGY_DIR.rglob("*"))
             if not p.is_dir() and p.suffix.lower() not in (".xml", ".txt", ".pdf")]
    for li, segy in enumerate(lines[:limit_lines] if limit_lines else lines):
        try:
            data, cx, cy, samples = read_segy(segy)
        except Exception as e:
            print(f"[real-win] skip {segy.name}: {e}", flush=True); continue
        img_full = _to_image(data); H, W = img_full.shape
        crossings = project_faults(faults, cx, cy, samples, (H, W))
        del data
        for name, poly in crossings:
            thr = _throw(poly, cx, cy, horizons)                 # full-line throw
            fc, fr = int(np.median(poly[:, 0])), int(np.median(poly[:, 1]))
            c0 = int(np.clip(fc - w_win // 2, 0, max(0, W - w_win))); c1 = min(W, c0 + w_win)
            r0 = int(np.clip(fr - h_win // 2, 0, max(0, H - h_win))); r1 = min(H, r0 + h_win)
            wp = poly.astype(float).copy(); wp[:, 0] -= c0; wp[:, 1] -= r0
            inb = (wp[:, 0] >= 0) & (wp[:, 0] < c1 - c0) & (wp[:, 1] >= 0) & (wp[:, 1] < r1 - r0)
            wp = wp[inb]
            if len(wp) < MIN_FAULT_PTS:
                continue
            dip = _line_dip(wp)
            if dip is None:
                continue
            hw = (r1 - r0, c1 - c0)
            m = dilate(_rasterize(wp, hw).to(device)).cpu()
            xs, ys = wp[:, 0], wp[:, 1]
            obj = dict(cls=1, bbox=[xs.min() / hw[1], ys.min() / hw[0], xs.max() / hw[1], ys.max() / hw[0]],
                       mask=m, meas=torch.tensor([dip, thr or 0.0, 0.0]),
                       mmask=torch.tensor([1.0, 1.0 if thr is not None else 0.0, 0.0]),
                       true_dip=None, name=name)
            png = RENDER_DIR / f"{segy.parent.name}__{segy.name}__{name}_{c0}_{r0}.png"
            Image.fromarray(img_full[r0:r1, c0:c1]).save(png)
            try:
                smap, _ = stitch(enc, str(png))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); continue
            scenes.append(dict(smap=smap.cpu(), hw=hw, objs=[obj], img=str(png),
                               fault_field=m, closure_field=torch.zeros(hw)))
            del smap; torch.cuda.empty_cache()
        del img_full
        print(f"[real-win] line {li+1}/{len(lines)} {segy.name}: {len(scenes)} windows total", flush=True)
    return scenes


WINDOW_CACHE = REAL_ROOT / "real_windows.pt"


def load_real_windows(test_frac=0.25, rebuild=False):
    if WINDOW_CACHE.exists() and not rebuild:
        scenes = torch.load(WINDOW_CACHE, weights_only=False)
    else:
        scenes = build_real_windows()
        torch.save(scenes, WINDOW_CACHE)
        print(f"[real-win] saved {len(scenes)} windows -> {WINDOW_CACHE}", flush=True)
    rng = random.Random(SEED); idx = list(range(len(scenes))); rng.shuffle(idx)
    cut = int(len(idx) * (1 - test_frac))
    return scenes, [scenes[i] for i in idx[:cut]], [scenes[i] for i in idx[cut:]]


SCENE_CACHE = REAL_ROOT / "real_scenes.pt"


def load_real_split(test_frac=0.25):
    if SCENE_CACHE.exists():
        scenes = torch.load(SCENE_CACHE, weights_only=False)
        print(f"[real] cache -> {len(scenes)} scenes", flush=True)
    else:
        scenes = build_real_scenes()
        torch.save(scenes, SCENE_CACHE)
        print(f"[real] saved {len(scenes)} scenes -> {SCENE_CACHE}", flush=True)
    rng = random.Random(SEED)
    idx = list(range(len(scenes))); rng.shuffle(idx)
    cut = int(len(idx) * (1 - test_frac))
    return scenes, [scenes[i] for i in idx[:cut]], [scenes[i] for i in idx[cut:]]


if __name__ == "__main__":
    sc = build_real_scenes()
    print(f"[real] built {len(sc)} real-field scenes", flush=True)
