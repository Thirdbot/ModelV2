"""Test — held-out copy score + swap faithfulness + grounded-reasoning check.

copy score   : fraction of faults whose GENERATED dip == the injected fact.
swap follows : perturb an injected dip, regenerate, check the output follows it.
reasoning    : generate emergent grounded reasoning from the facts and count
               INVENTED numbers (stated ∉ facts) — the verifiable faithfulness of
               the grounded reasoning (absorbed from the evidence-copy transfer test).
"""
import copy
import re
from pathlib import Path

from hybrid.model.narrator import faults_of, scene_facts, FAULT_LINE
from hybrid.inference.infer import infer_detected

OUT = Path("hybrid/experiments/main_model")

PROBES = ["Interpret the structural setting.",
          "Is this structure prospective for hydrocarbons? Explain.",
          "What kind of trap could these faults form?"]
# any number that quantifies a (lowercase) word — catches bare counts like "6 faults"
# / "8 clumps", not just unit-suffixed. Lowercase-next excludes sentence list markers
# ("1. The..."); the lookbehinds exclude instance INDICES ("Fault 4", "Closure 2")
# which are labels, not claims. This is what makes the invented-number count honest.
NUM_RE = re.compile(r"(?<![Ff]ault )(?<![Cc]losure )(-?\d+\.?\d*)\s+[a-z]")


def reasoning_report(nar, split, n=4):
    """Grounded reasoning on held-out facts (emergent, no reason data); count
    numbers stated that are NOT in the facts (invented)."""
    inv_tot, shown, lines = 0, 0, []
    for s in split[:n]:
        vals = faults_of(s["objs"])
        if not vals:
            continue
        for q in PROBES:
            txt = nar.generate_reasoning(vals, q)
            nums = [float(x) for x in NUM_RE.findall(txt)]
            invented = [x for x in nums if all(abs(x - v) > 2 for v in vals)]
            inv_tot += len(invented); shown += 1
            lines.append(f"- facts={[round(v, 1) for v in vals]} q='{q[:24]}' "
                         f"invented={len(invented)} | {' '.join(txt.split())[:150]}")
    return inv_tot, shown, lines


def copy_score(gen_text, dips):
    got = [float(x) for x in FAULT_LINE.findall(gen_text)]
    if not dips:
        return 0.0
    hit = sum(1 for i, d in enumerate(dips) if i < len(got) and abs(got[i] - d) < 1.0)
    return hit / len(dips)


def report_split(nar, split):
    cs, follow, n_sw, ex = 0.0, 0, 0, []
    for s in split:
        facts = scene_facts(s)
        dips = [f["dip"] for f in facts["faults"]]
        txt = nar.generate(facts)
        cs += copy_score(txt, dips)
        if dips:
            alt = copy.deepcopy(facts); base = alt["faults"][0]["dip"]
            alt["faults"][0]["dip"] = min(89.0, base + 15.0)
            txt2 = nar.generate(alt)                      # swap the injected dip, regenerate
            got = FAULT_LINE.findall(txt2)
            follow += bool(got and abs(float(got[0]) - alt["faults"][0]["dip"]) < 2.0); n_sw += 1
            if len(ex) < 4:
                ex.append((dips, txt, base, alt["faults"][0]["dip"], txt2))
    return cs / max(1, len(split)), follow, n_sw, ex


EVAL_CAP = 5   # cap generated scenes per split — greedy gen on 4-bit is slow


def evaluate(nar, tr, te, det_facts):
    nar.eval_mode()
    tr_cs, tr_f, tr_n, _ = report_split(nar, tr[:EVAL_CAP])
    te_cs, te_f, te_n, te_ex = report_split(nar, te[:EVAL_CAP])
    det = infer_detected(nar, te[:5], det_facts)
    inv_tot, inv_n, reason_lines = reasoning_report(nar, te, n=2)

    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# Main model — grounded narrator (copy + grounded reasoning)\n",
             f"**copy score — train {tr_cs:.2f} · test(held-out) {te_cs:.2f}**\n",
             f"**swap follows — train {tr_f}/{tr_n} · test {te_f}/{te_n}**\n",
             f"**reasoning — invented numbers {inv_tot}/{inv_n} probes** _(stated ∉ facts; lower = more grounded)_\n",
             "\n## held-out examples (GT facts)\n"]
    for dips, txt, base, alt0, txt2 in te_ex:
        lines += [f"### {len(dips)} faults {[round(d, 1) for d in dips]}",
                  f"- generated: {txt.strip()[:150]}",
                  f"- swap {base:.1f}->{alt0:.1f}: {txt2.strip()[:150]}", ""]
    lines += ["## held-out with DETECTED facts (real end-to-end)\n"]
    lines += [f"- detected {[round(x, 1) for x in dd]} -> {t[:120]}" for dd, t in det]
    lines += ["\n## grounded reasoning (held-out, emergent — no reason data)\n"]
    lines += reason_lines
    (OUT / "RESULT.md").write_text("\n".join(lines) + "\n")
    print(f"[test] copy train {tr_cs:.2f} test {te_cs:.2f} · swap {te_f}/{te_n} · reasoning invented {inv_tot}/{inv_n}", flush=True)
    print("MAIN_MODEL_DONE", flush=True)
    return te_cs, te_f, te_n
