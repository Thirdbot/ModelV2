"""Quick overfit for the <SEG> mask decoder + LM integration (step 3, part 2).

Runs the (frozen) chatml narrator, extracts the hidden state at the <SEG> token,
feeds it as the CONTENT PROMPT to the mask decoder, and overfits the decoder to
the fault-field mask. Validates that the LM-<SEG>-hidden actually drives a good
mask (the content-prompt claim). Only the mask decoder trains here; joint LM
fine-tune comes later. (Single <SEG> → union mask; per-object <SEG> is a target
format change, noted separately.)
"""
import torch

import hybrid.model.scenes as sc
sc.MAX_SCENES = 8

from hybrid.model.scenes import build_scenes
from hybrid.model.narrator import (Narrator, faults_of, scene_facts, facts_to_kv,
                                    grounding_target, INSTRUCTION_S3)
from hybrid.model.mask_decoder import SegMaskDecoder, mask_loss, seg_positions
from hybrid.model.segmenter import field_dice

device = torch.device("cuda")

nar = Narrator(); nar.set_stage("s3"); nar.eval_mode()
mdec = SegMaskDecoder().to(device)

data = []
for s in [x for x in build_scenes() if faults_of(x["objs"])][:6]:
    facts = scene_facts(s); kv = facts_to_kv(facts)
    target = grounding_target(facts)                         # <evidence> preamble <SEG> </evidence>
    with torch.no_grad():
        ft = nar.facts_mod(kv)
        prompt = nar.build_prompt(ft, INSTRUCTION_S3, question=None)
        tgt_ids = nar.tok(target + "<|im_end|>", add_special_tokens=False,
                          return_tensors="pt").input_ids.to(device)[0]
        inp = torch.cat([prompt, nar.emb(tgt_ids.unsqueeze(0)).squeeze(0)], 0).unsqueeze(0)
        hs = nar.dec(inputs_embeds=inp, output_hidden_states=True).hidden_states[-1][0]
        pos = seg_positions(nar.tok, tgt_ids)
        if not pos:
            continue
        seg_h = hs[[prompt.shape[0] + p for p in pos]].float()   # (n_seg, lm_dim)
    data.append((seg_h, s["smap"], s["fault_field"], s["hw"]))

print(f"[seg] overfit mask decoder on {len(data)} scenes (LM frozen, <SEG> hidden as prompt)", flush=True)
opt = torch.optim.AdamW(mdec.parameters(), lr=1e-3)
for ep in range(300):
    tot = 0.0
    for seg_h, smap, ff, hw in data:
        opt.zero_grad()
        m = mdec(seg_h, smap, hw)                            # (n_seg, H, W)
        gt = ff.unsqueeze(0).expand(m.shape[0], -1, -1)
        loss = mask_loss(m, gt)
        loss.backward(); opt.step(); tot += loss.item()
    if ep % 40 == 0 or ep == 199:
        print(f"[seg] ep {ep} loss {tot/len(data):.3f}", flush=True)

mdec.eval()
dices = []
with torch.no_grad():
    for seg_h, smap, ff, hw in data:
        dices.append(field_dice(mdec(seg_h, smap, hw)[0], ff))
print(f"[seg RESULT] mask dice {sum(dices)/len(dices):.2f} (n{len(dices)})", flush=True)
print("SEG_OVERFIT_DONE", flush=True)
