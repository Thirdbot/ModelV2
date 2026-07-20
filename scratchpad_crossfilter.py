"""Find which SEG-Y lines cross faults, from ASC nav + fault sticks only (NO segy
extraction, NO encoder). Uses the loader's own rule: a line crosses if some fault has
>= MIN_FAULT_PTS of its points within MATCH_THRESH_M of the line's traces. Prints the
crossing lines that also have a SEG-Y file in the raw zip, and writes the extract list.
"""
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from hybrid.data.real import load_fault_sticks, MATCH_THRESH_M, MIN_FAULT_PTS

ROOT = Path("data/real_data")
ASC_DIR = ROOT / "asc"
RAW_ZIP = ROOT / "raw/seismic_2d_lines.zip"
OUT_LIST = Path("scratchpad_crossing_lines.txt")

# two floats = X, Y (Y terminates at the '-' of the glued negative SP field)
COORD = re.compile(r"(\d{4,7}\.\d+)\s+(\d{5,8}\.\d+)")


def asc_lines():
    """{stem: np.array([[X,Y],...])} over every ASC survey (stem = first token)."""
    lines = defaultdict(list)
    for asc in sorted(ASC_DIR.glob("*.asc")):
        for ln in asc.read_text("latin-1").splitlines():
            if ln.startswith("#") or not ln.strip():
                continue
            m = COORD.search(ln)
            if not m:
                continue
            stem = ln[:m.start()].split()[0].strip()
            lines[stem].append((float(m.group(1)), float(m.group(2))))
    return {k: np.array(v) for k, v in lines.items() if len(v) > 1}


def segy_in_zip():
    """{stem: zip-entry-name} for every real SEG-Y line file in the raw archive."""
    out = {}
    with zipfile.ZipFile(RAW_ZIP) as z:
        for n in z.namelist():
            if "/segy/" in n and not n.endswith(("/", ".xml", ".pdf")):
                out[Path(n).name] = n
    return out


def main():
    faults = load_fault_sticks()
    fault_xy = [P[:, :2] for P in faults.values()]
    nav = asc_lines()
    zsegy = segy_in_zip()
    print(f"[filter] {len(faults)} faults · {len(nav)} nav lines · {len(zsegy)} segy in zip", flush=True)

    crossing = []
    for stem, xy in nav.items():
        if stem not in zsegy:                       # nav line with no SEG-Y file
            continue
        tree = cKDTree(xy)
        hit = False
        for P in fault_xy:
            d, _ = tree.query(P)
            if int((d < MATCH_THRESH_M).sum()) >= MIN_FAULT_PTS:
                hit = True
                break
        if hit:
            crossing.append((stem, zsegy[stem]))

    crossing.sort()
    OUT_LIST.write_text("\n".join(f"{s}\t{z}" for s, z in crossing) + "\n")
    bysurv = defaultdict(int)
    for s, z in crossing:
        bysurv[z.split("/segy/")[1].split("/")[0]] += 1
    print(f"[filter] CROSSING lines with SEG-Y: {len(crossing)}", flush=True)
    for k, v in sorted(bysurv.items()):
        print(f"    {k}: {v}", flush=True)


if __name__ == "__main__":
    main()
