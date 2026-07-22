"""SAFE test of the windowed real builder — 8 lines only, prints size + memory.
Confirms peak stays bounded (no 30GB freeze) before any full build."""
import torch
from hybrid.data.real import build_real_windows

scenes = build_real_windows(limit_lines=8)
print(f"[test] built {len(scenes)} windows from 8 lines", flush=True)
if scenes:
    s = scenes[0]
    print(f"[test] scene0 smap {tuple(s['smap'].shape)} · hw {s['hw']} · "
          f"dip {float(s['objs'][0]['meas'][0]):.1f}deg · "
          f"throw {float(s['objs'][0]['meas'][1]):.0f}ms", flush=True)
    mb = sum(x['smap'].numel() * 4 + x['fault_field'].numel() * 4 * 2 for x in scenes) / 1e6
    print(f"[test] {len(scenes)} scenes ~{mb:.0f}MB total ({mb/len(scenes):.1f}MB each)", flush=True)
print("REAL_WIN_TEST_DONE", flush=True)
