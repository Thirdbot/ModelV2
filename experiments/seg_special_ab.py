"""LISA-style <SEG> as a SPECIAL token vs PLAIN-TEXT <SEG> → mask decoder.

Same single-fault scenes, same frozen LM, same mask decoder. Only difference: how <SEG>
is tokenized and where we read its hidden state.
  Arm A (plain-text): <SEG> = [ '<','SEG','>' ]  → hidden at the '>' that completes it.
  Arm B (special):    <SEG> = one added token (padded row, NO resize) → its single hidden.
Tests whether a dedicated special token is a cleaner mask prompt than the plain-text one.
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import hybrid.model.scenes as sc
sc.MAX_SCENES = 200

from hybrid.model.scenes import build_scenes
from hybrid.model.narrator import (Narrator, faults_of, scene_facts, facts_to_kv,
                                    grounding_target, INSTRUCTION_S3)
from hybrid.model.segmenter import field_dice

device = torch.device("cuda")
PLAIN = [27, 78759, 29]                    # '<','SEG','>'


class MDec(nn.Module):
    def __init__(self, lm=1536, v=768, d=128, up=3):
        super().__init__()
        self.q = nn.Sequential(nn.Linear(lm, d), nn.GELU(), nn.Linear(d, d))
        L = [nn.Conv2d(v, d, 3, padding=1), nn.GroupNorm(8, d), nn.GELU()]
        for _ in range(up):
            L += [nn.ConvTranspose2d(d, d, 4, stride=2, padding=1), nn.GroupNorm(8, d), nn.GELU()]
        self.up = nn.Sequential(*L, nn.Conv2d(d, d, 1)); self.b = nn.Parameter(torch.zeros(1))

    def forward(self, h, smap, hw):
        m = torch.einsum("nd,dhw->nhw", self.q(h), self.up(smap.unsqueeze(0))[0]) + self.b
        return F.interpolate(m.unsqueeze(0), size=hw, mode="bilinear", align_corners=False)[0]


def single_fault(s):
    return len(faults_of(s["objs"])) == 1 and not any(
        int(o["cls"]) == 2 and float(o["mmask"][2]) > 0 for o in s["objs"])


nar = Narrator(); nar.set_stage("s3"); nar.eval_mode()
tok, dec, emb = nar.tok, nar.dec, nar.emb
scenes = [s for s in build_scenes() if single_fault(s)]
print(f"[seg-special] {len(scenes)} single-fault scenes", flush=True)


def hidden_at(target, pos_fn):
    facts_kv = target[1]
    ft = nar.facts_mod(facts_kv)
    prompt = nar.build_prompt(ft, INSTRUCTION_S3, question=None)
    tgt = tok(target[0] + "<|im_end|>", add_special_tokens=False, return_tensors="pt").input_ids.to(device)[0]
    p = pos_fn(tgt.tolist())
    if p is None:
        return None
    inp = torch.cat([prompt, emb(tgt.unsqueeze(0)).squeeze(0)], 0).unsqueeze(0)
    hs = dec(inputs_embeds=inp, output_hidden_states=True).hidden_states[-1][0]
    return hs[prompt.shape[0] + p].detach().float().unsqueeze(0)


def train_eval(pos_fn, tag):
    rows = []
    for s in scenes:
        with torch.no_grad():
            h = hidden_at((grounding_target(scene_facts(s)), facts_to_kv(scene_facts(s))), pos_fn)
        if h is not None:
            rows.append((h, s["smap"], s["fault_field"], s["hw"]))
    rng = random.Random(0); idx = list(range(len(rows))); rng.shuffle(idx)
    cut = int(len(idx) * 0.75); tr = [rows[i] for i in idx[:cut]]; te = [rows[i] for i in idx[cut:]]
    torch.manual_seed(0)
    md = MDec().to(device); opt = torch.optim.AdamW(md.parameters(), lr=1e-3)
    md.train()
    for ep in range(200):
        for h, smap, ff, hw in tr:
            opt.zero_grad()
            m = md(h, smap, hw)
            pw = torch.tensor([40.0], device=device); p = m.sigmoid(); g = ff.unsqueeze(0).to(device)
            loss = F.binary_cross_entropy_with_logits(m, g, pos_weight=pw) + 1 - (2 * (p * g).sum() + 1) / (p.sum() + g.sum() + 1)
            loss.backward(); opt.step()
    md.eval()
    d = [field_dice(md(h, smap, hw)[0], ff.to(device)) for h, smap, ff, hw in te]
    print(f"[seg-special {tag:>12}] held-out mask dice {np.mean(d):.3f} (n{len(d)})", flush=True)


def plain_pos(ids):                        # robust to BPE context: token that completes "<SEG>"
    for i in range(len(ids)):
        if "<SEG>" in tok.decode(ids[: i + 1]):
            return i
    return None


# Arm A: plain-text <SEG> (before adding the special token)
train_eval(plain_pos, "plain-text")

# Arm B: <SEG> as a SINGLE special token (padded row, NO resize)
before = len(tok)
tok.add_special_tokens({"additional_special_tokens": ["<SEG>"]})
seg_id = tok.convert_tokens_to_ids("<SEG>")
assert seg_id < emb.weight.shape[0], "would need resize (NaN risk) — abort"
with torch.no_grad():
    emb.weight.data[seg_id] = emb.weight.data[:before].mean(0)
print(f"[seg-special] added <SEG> id={seg_id} (single token, no resize)", flush=True)
train_eval(lambda ids: (ids.index(seg_id) if seg_id in ids else None), "special-tok")
print("SEG_SPECIAL_DONE", flush=True)
