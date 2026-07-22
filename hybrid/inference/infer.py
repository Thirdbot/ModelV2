"""Inference — grounded narration from the detector's measured facts.

The segmentor produces the mask → measured facts; the LM narrates from those facts
(role-tagged digit injection), copying the numbers into the grounded chain. `<SEG>`
anchors map to mask instances by order. Decoupled: the LM sees numbers, not features.
"""
import torch

from hybrid.model.narrator import Narrator, faults_of, facts_to_kv

device = torch.device("cuda")

# question types shown on the overlay (grounding / geology / mixed)
QA_QUESTIONS = [
    ("Grounding", "How many faults are present and what is each fault's dip?"),
    ("Geology", "What hydrocarbon trap could these faults form?"),
    ("Mixed", "Given these faults and dips, is the structure prospective? Explain."),
]


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


def _answer_text(narr):
    """Pull the <answer>…</answer> content out of the full chain (evidence/think/answer)
    so the overlay shows the actual answer, not the leading evidence. Handles an unclosed
    <answer> (truncated generation) and strips any stray tags."""
    import re
    i = narr.find("<answer>")
    tail = narr[i + len("<answer>"):] if i >= 0 else narr
    tail = tail.split("</answer>")[0]
    return " ".join(re.sub(r"<[^>]+>", " ", tail).split()) or "(no answer)"


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
        has = facts["faults"] or facts["closures"]
        narr = nar.generate(facts) if has else "(nothing detected)"
        qa = []                                            # per-question answers (grounding/geology/mixed)
        if has:
            kv = facts_to_kv(facts)
            for label, q in QA_QUESTIONS:
                try:
                    ans = nar.narrate(kv, question=q, max_new_tokens=130)
                except Exception as e:
                    ans = f"(error: {e})"
                torch.cuda.empty_cache()
                qa.append((label, q, ans))
        out.append(dict(img=s["img"], hw=s["hw"],
                        pmask=torch.sigmoid(seg[0]).cpu(),
                        gmask=s["fault_field"].cpu(),
                        faults=facts["faults"], closures=facts["closures"],
                        gt_dips=faults_of(s["objs"]), narr=narr, qa=qa))
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
        pw = 510
        canvas = Image.new("RGB", (W + pw, max(H, 680)), (250, 250, 248))
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
        dr.text((x, y), "Base narration:", font=fntb, fill=(20, 20, 20)); y += 19
        for ln in textwrap.wrap(" ".join(narr.split()), 60)[:2]:
            dr.text((x, y), ln, font=fnt, fill=(40, 40, 40)); y += 16
        y += 8
        dr.text((x, y), "Questions → answers:", font=fntb, fill=(20, 20, 20)); y += 20
        for label, q, ans in d.get("qa", []):
            for ln in textwrap.wrap(f"[{label}] {q}", 62):
                dr.text((x, y), ln, font=fntb, fill=(30, 70, 140)); y += 16
            for ln in textwrap.wrap("A: " + _answer_text(ans), 62)[:3]:
                dr.text((x, y), ln, font=fnt, fill=(45, 45, 45)); y += 16
            y += 5
        dr.text((x, canvas.height - 20),
                f"red = pred mask/box · green = GT · blue = closure · {n_seg} <SEG>",
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
