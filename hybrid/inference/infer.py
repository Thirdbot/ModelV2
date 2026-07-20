"""Inference — grounded narration from the detector's measured facts.

The segmentor produces the mask → measured facts; the LM narrates from those facts
(role-tagged digit injection), copying the numbers into the grounded chain. `<SEG>`
anchors map to mask instances by order. Decoupled: the LM sees numbers, not features.
"""
import torch

from hybrid.model.narrator import Narrator, faults_of

device = torch.device("cuda")


def infer_detected(nar, scenes, det_facts):
    """Narrate held-out scenes end-to-end from the DETECTOR's measured facts."""
    out = []
    for s in scenes:
        facts = det_facts.get(id(s))
        dd = [f["dip"] for f in facts["faults"]] if facts else []
        txt = (nar.generate(facts) if facts and (facts["faults"] or facts["closures"])
               else "(nothing detected)")
        out.append((dd, txt))
    return out


def render_grounded(narration, facts):
    """Associate each <SEG> anchor in the narration with a dense instance BY ORDER
    (faults then closures, matching the injected fact order in facts_to_kv).
    Returns [(seg_index, instance), ...] linking each text anchor to its mask/bbox."""
    instances = facts["faults"] + facts.get("closures", [])
    n = narration.count("<SEG>")
    return [(i, instances[i]) for i in range(min(n, len(instances)))]


@torch.no_grad()
def collect_demo(net, nar, scenes, scale):
    """Per scene, gather what an overlay needs: predicted/GT fault fields, the class-
    driven facts, and the grounded narration (from the measured facts)."""
    out = []
    for s in scenes:
        seg = net(s["smap"], s["hw"])
        facts = net.measure(s["smap"], s["hw"])
        narr = (nar.generate(facts) if (facts["faults"] or facts["closures"])
                else "(nothing detected)")
        out.append(dict(img=s["img"], hw=s["hw"],
                        pmask=torch.sigmoid(seg[0]).cpu(),
                        gmask=s["fault_field"].cpu(),
                        faults=facts["faults"], closures=facts["closures"],
                        gt_dips=faults_of(s["objs"]), narr=narr))
    return out


def save_overlays(demo, out_dir):
    """Composite per scene: seismic image + pred(red)/GT(green) masks + per-instance
    BOUNDING BOXES (fault=red, closure=blue) + detected facts + the grounded
    narration (already generated in collect_demo). Returns the saved paths."""
    import textwrap
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    try:
        fnt = ImageFont.truetype("DejaVuSans.ttf", 13)
        fntb = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
    except Exception:
        fnt = fntb = ImageFont.load_default()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for k, d in enumerate(demo):
        faults, closures = d["faults"], d["closures"]
        facts = {"faults": faults, "closures": closures}
        narr = d["narr"]
        n_seg = len(render_grounded(narr, facts))            # <SEG> → mask links
        narr = ". ".join(" ".join(narr.split()).split(". ")[:4]).strip()
        H, W = d["hw"]
        base = np.array(Image.open(d["img"]).convert("RGB").resize((W, H)), dtype=np.float32)
        pm = (d["pmask"].numpy() > 0.5)[..., None]
        gm = (d["gmask"].numpy() > 0.5)[..., None]
        red = np.array([230, 40, 40], np.float32); green = np.array([30, 200, 60], np.float32)
        base = base * (1 - 0.40 * gm) + green * (0.40 * gm)
        base = base * (1 - 0.45 * pm) + red * (0.45 * pm)
        img = Image.fromarray(base.astype(np.uint8))
        d0 = ImageDraw.Draw(img)
        for f in faults:
            d0.rectangle(f["bbox"], outline=(255, 90, 90), width=1)     # fault box
        for c in closures:
            d0.rectangle(c["bbox"], outline=(90, 150, 255), width=1)    # closure box
        pw = 480
        canvas = Image.new("RGB", (W + pw, max(H, 340)), (250, 250, 248))
        canvas.paste(img, (0, 0))
        dr = ImageDraw.Draw(canvas)
        x = W + 14; y = 10
        gt = [round(v, 1) for v in d["gt_dips"]]
        dr.text((x, y), f"Held-out scene #{k+1} — end-to-end", font=fntb, fill=(20, 20, 20)); y += 26
        dr.text((x, y), f"GT faults: {len(gt)} · dips {gt}", font=fnt, fill=(20, 110, 40)); y += 22
        dr.text((x, y), f"Detected faults: {len(faults)}", font=fntb, fill=(160, 30, 30)); y += 18
        for j, f in enumerate(faults[:6]):
            thr = f.get("throw")
            line = f"  fault {j+1}: dip {f['dip']:.1f}°" + (f" · throw {thr:.0f}ms" if thr is not None else "")
            dr.text((x, y), line, font=fnt, fill=(160, 30, 30)); y += 16
        dr.text((x, y), f"Detected closures: {len(closures)}", font=fntb, fill=(40, 90, 180)); y += 18
        for j, c in enumerate(closures[:4]):
            dr.text((x, y), f"  closure {j+1}: area {c['area_pct']:.1f}%", font=fnt, fill=(40, 90, 180)); y += 16
        y += 6
        dr.text((x, y), "Narration:", font=fntb, fill=(20, 20, 20)); y += 20
        for ln in textwrap.wrap(narr, 56):
            dr.text((x, y), ln, font=fnt, fill=(40, 40, 40)); y += 18
        dr.text((x, y + 4), f"{n_seg} <SEG> anchors → masks (order-mapped)", font=fnt, fill=(90, 90, 90))
        dr.text((x, canvas.height - 22), "red = pred mask/box · green = GT mask · blue = closure box",
                font=fnt, fill=(120, 120, 120))
        p = out_dir / f"demo_{k+1}.png"
        canvas.save(p); paths.append(str(p))
    return paths


def main():
    """Standalone inference from the saved checkpoints (no retrain)."""
    from pathlib import Path
    from hybrid.checkpoints import load_full
    from hybrid.model.scenes import build_scenes, MEAS_SCALE
    net, nar = load_full()
    scenes = [s for s in build_scenes() if faults_of(s["objs"])][:5]
    demo = collect_demo(net, nar, scenes, MEAS_SCALE.to(device))
    paths = save_overlays(demo, Path("hybrid/inference_outputs"))
    print(f"saved {len(paths)} overlays -> hybrid/inference_outputs", flush=True)


if __name__ == "__main__":
    main()
