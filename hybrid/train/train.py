"""Train the main model end to end — reader + narrator + <SEG> mask decoder.

Stage 2 : instance READER (autoregressive; facts = count/class/dip/throw) — replaces
          the dense-seg + RANSAC front-end.
Stage 3 : chatml narrator (copies the numbers into grounded language) + the <SEG>
          mask decoder (per-object content-prompted masks).
Then evaluate on HELD-OUT: reader count/dip · copy fidelity · per-object mask dice.

Stage 1 (geology adapter) is built once via `python -m hybrid.train.stage1_geology`.
Run:  python -m hybrid.train.train
"""
import random
from pathlib import Path

import torch

import hybrid.model.scenes as sc
SCENE_CAP = 120                # cap for a quick-but-real held-out run (raise for a full run)
sc.MAX_SCENES = SCENE_CAP

from hybrid.model.scenes import build_scenes
from hybrid.model.narrator import Narrator, faults_of, scene_facts, facts_to_kv, K_DIP
from hybrid.train.stage_reader_mask import train_reader, reader_accuracy, reader_facts
from hybrid.model.reader import scene_to_gt
from hybrid.model.segmenter import field_dice
from hybrid.train.stage2_grounding import train_grounding
from hybrid.train.stage3_narrator import train_narrator
from hybrid.checkpoints import save_narrator

device = torch.device("cuda")
SEED = 42
READER_EPOCHS = 150
LM_EPOCHS = 12
CKPT = Path("hybrid/checkpoints")


def load_split():
    rng = random.Random(SEED)
    scenes = [s for s in build_scenes() if faults_of(s["objs"])]
    idx = list(range(len(scenes))); rng.shuffle(idx)
    cut = int(len(idx) * 0.75)
    return scenes, [scenes[i] for i in idx[:cut]], [scenes[i] for i in idx[cut:]]


def _fmt(mn): return f"{mn[0]:.2f}(n{mn[1]})" if mn and mn[0] is not None else "n0"


def main():
    scenes, tr, te = load_split()
    print(f"[train] scenes {len(scenes)} · train {len(tr)} · test {len(te)}", flush=True)

    # ---- Stage 2: instance reader (facts) ----
    reader = train_reader(tr, epochs=READER_EPOCHS)
    for tag, sp in (("train", tr), ("test(held-out)", te)):
        a = reader_accuracy(reader, sp)
        dices = []
        for s in sp:
            gt = scene_to_gt(s)
            if not gt:
                continue
            ml = reader.tf_masks(s["smap"], gt)
            dices += [field_dice(ml[i], o["mask_full"].to(device))
                      for i, o in enumerate(gt) if o["cls"] == 1]
        md = (sum(dices) / len(dices), len(dices)) if dices else (None, 0)
        print(f"[reader {tag}] count MAE {_fmt(a['count'])} · dip MAE {_fmt(a['dip'])}deg · "
              f"class {a['cls'][0]}/{a['cls'][1]} · mask dice {_fmt(md)}", flush=True)
    torch.save(reader.state_dict(), CKPT / "reader.pt")

    # ---- Stage 3a/b: chatml narrator (copies facts into grounded language) ----
    facts_by_img = {s["img"]: scene_facts(s) for s in tr}
    nar = Narrator()
    train_grounding(nar, facts_by_img)
    train_narrator(nar, facts_by_img, epochs=LM_EPOCHS)
    save_narrator(nar, "stage3_narrator.pt")

    # ---- end-to-end held-out copy fidelity (reader-detected facts -> narration) ----
    nar.eval_mode()
    Q = "How many faults are present and what is each fault's dip?"
    hit = tot = 0
    for s in te[:20]:
        facts = reader_facts(reader, s)
        if not facts["faults"]:
            continue
        dips = [v for k, v in facts_to_kv(facts) if k == K_DIP]
        out = nar.generate(facts, question=Q, max_new_tokens=140)
        for d in dips:
            tot += 1; hit += (d in out)
    print(f"[copy held-out] reader-facts copied {hit}/{tot}", flush=True)
    print("MAIN_MODEL_DONE", flush=True)


if __name__ == "__main__":
    main()
