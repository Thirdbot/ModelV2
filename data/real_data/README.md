# Real-field data — Smeaheia (syn→real vision stage)

Real 2D seismic + interpretation used to **benchmark and fine-tune the VISION part**
of the model (NCS ViT → dense segmenter → connected components → `measure_instances`)
after the synthetic stages bind visual + reasoning. Narration is **frozen** in this
stage — it's a syn→real transfer for detection/masking only.

## Source & license (attribution REQUIRED)

Smeaheia Dataset — **© Equinor & Gassnova**, via CO2DataShare.
- Dataset: https://co2datashare.org/dataset/smeaheia-dataset
- License: https://co2datashare.org/view/license/26af9426-203f-4993-9d41-2e1bf191ceaf
  (modified CC BY 4.0 — research/ML/derivatives/redistribution allowed **with
  attribution to Equinor & Gassnova**; **may not sell** the material.)

Any published result or redistributed derivative MUST credit **Equinor and Gassnova**
and link the license above.

## What we use (2D versions — no 3D→2D projection needed)

| resource | → GT for |
|---|---|
| Seismic 2D lines (BPN88, GSB-85R97) | the images (SEG-Y → normalized) |
| Fault sticks (2D, Triassic–Jurassic) | fault masks + bbox + dip |
| Horizons (2D) | throw (horizon offset across a fault) |
| Interval velocity maps | true dip / time→depth |

Not on 2D lines → **ignored**: count, closure/area, class (real = fault channel only).

## Expected layout (place downloaded + converted files here)

```
data/real_data/
  raw/                     # downloaded Smeaheia files (untracked; big)
  lines/     <line_id>.sgy         # 2D SEG-Y lines  (or <line_id>.png if pre-rendered)
  faults/    <line_id>.dat         # 2D fault sticks on that line (trace, twt) polylines
  horizons/  <line_id>.dat         # 2D horizon picks on that line
  velocity/  interval_velocity.*   # for true dip / depth
  scenes/                  # built scene cache (produced by hybrid/data/real.py)
```

`<line_id>` is the survey line name; `faults/`, `horizons/` are keyed to it so each 2D
line assembles into one scene.

## Reproduce

1. Download the 2D lines, fault sticks, horizons, velocity from the dataset page
   (accept the license). Put originals in `raw/`, arrange per the layout above.
2. `pip install segyio` (SEG-Y reader; add to pyproject).
3. Build scenes:  `python -m hybrid.data.real`   (writes `scenes/` cache + a manifest)
4. Benchmark the synthetic model on real:  `python -m hybrid.test.benchmark_real`
5. Fine-tune the vision part (narration frozen):  `python -m hybrid.train.stage4_realfield`

Splits are seeded (see `hybrid/data/real.py`) so train/test is deterministic.
