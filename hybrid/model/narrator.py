"""The facts-bridge narrator — the main model's language half.

Detector facts (count + per-fault dips) become DIGIT-TOKEN embeddings, prepended
to the prompt; the LM copies the exact numbers. The decoder is the stacked-adapter
FUSE: stage 2 trains the grounding adapter to copy facts into evidence text
(grounding the shared latent), then it freezes; stage 3 trains the fuse combiner
to align the detector facts into narration on top of both frozen adapters.

Proven (copy path): held-out copy 1.00, faithfulness swap 16/16.
"""
import re

import torch
import torch.nn as nn

from hybrid.model.geology import load_geology_adapter, GEOLOGY_CFG
from hybrid.model.decoder import GroundedDecoder

device = torch.device("cuda")
K_COUNT, K_DIP, K_EVID, K_NCLOSURE, K_AREA, K_BBOX, K_THROW = 0, 1, 2, 3, 4, 5, 6
FAULT_LINE = re.compile(r"[Ff]ault\s+\d+[^.]*?dips at\s+(?:about\s+)?(?:<nums>)?([\d.]+)")

# Unified chatml prompt: facts live in the SYSTEM turn (vision supplies them — a real
# user never types measurements); the user turn is the question only. One skeleton across
# all stages so geology's <think> (trained under <|im_start|>assistant) fires everywhere.
INSTRUCTION_S2 = ("Report the measured evidence for each region. State values directly; "
                  "end each line with a segmentation marker.")
INSTRUCTION_S3 = ("Answer the question with concise geological evidence. Reference specific "
                  "objects using object tags. Insert one segmentation marker at the end of each "
                  "region-specific evidence line. State the measured values directly. Do not add "
                  "facts unsupported by the image.")


def faults_of(scene_objs):
    """GT per-fault dips (cls==1 with a dip present), in region order."""
    return [float(o["meas"][0]) for o in scene_objs
            if int(o["cls"]) == 1 and float(o["mmask"][0]) > 0]


def scene_facts(scene):
    """GT facts in the SAME structure the detector's measure_instances produces:
    per-fault {dip, bbox(px), throw?} and per-closure {area_pct, bbox(px)}."""
    H, W = scene["hw"]
    faults, closures = [], []
    for o in scene["objs"]:
        x1, y1, x2, y2 = o["bbox"]
        bbox = [int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)]
        if int(o["cls"]) == 1 and float(o["mmask"][0]) > 0:
            f = {"dip": float(o["meas"][0]), "bbox": bbox}
            if float(o["mmask"][1]) > 0:
                f["throw"] = float(o["meas"][1])
            faults.append(f)
        elif int(o["cls"]) == 2 and float(o["mmask"][2]) > 0:
            closures.append({"area_pct": float(o["meas"][2]), "bbox": bbox})
    return {"faults": faults, "closures": closures}


class FactTokens(nn.Module):
    """fact = per-kind marker ++ emb(tokenize(value_string)). Digit tokens only.
    Each number is injected with its ROLE marker (K_DIP/K_COUNT/K_THROW/K_AREA...), so
    a dip can only land in the dip phrase. The marker embedding is where the value's
    'meaning' is learned. Add a role = one keyword in KIND_KW + reuse a marker row."""
    def __init__(self, dim, emb, tok):
        super().__init__()
        self.marker = nn.Embedding(7, dim)
        self.emb, self.tok = emb, tok

    def forward(self, facts):
        segs = []
        for k, vs in facts:
            ids = self.tok(vs, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            mark = self.marker(torch.tensor(k, device=device)).unsqueeze(0)
            segs.append(torch.cat([mark, self.emb(ids).squeeze(0)], 0))
        return torch.cat(segs, 0)


# Role tagging — "make meaning in the value". One keyword routes each number to its
# marker; the marker EMBEDDING (learned, in FactTokens) is where the meaning lives.
# Add an attribute = one keyword here + reuse a marker row. Default = count.
KIND_KW = [("dip", K_DIP), ("throw", K_THROW), ("percent", K_AREA), ("area", K_AREA)]
NUM_CTX = re.compile(r"(.{0,22})<nums>([-\d.]+)</nums>(.{0,12})")
CENTER = re.compile(r"<center>.*?</center>", re.S)
BBOX = re.compile(r"<bbox>.*?</bbox>", re.S)
WRAP_TAGS = re.compile(r"</?(?:region|object)>")   # structural wrappers — dropped, text kept


def _kind(ctx):
    c = ctx.lower()
    for kw, k in KIND_KW:
        if kw in c:
            return k
    return K_COUNT


def evidence_kv(text):
    """Role-tag each <nums> value by a keyword around it -> [(marker, value)]. TRAIN side.
    (bbox dropped from the injection — it mangled the narration + leaked into other slots.)"""
    return [(_kind(pre + " " + post), v) for pre, v, post in NUM_CTX.findall(text)]


def facts_to_kv(facts):
    """Detector facts -> the SAME role-tagged list evidence_kv produces (no bbox):
    count · per-fault dip/throw · per-closure area."""
    faults = facts.get("faults", []); closures = facts.get("closures", [])
    kv = [(K_COUNT, f"{len(faults)}")]
    for f in faults:
        kv.append((K_DIP, f"{round(float(f['dip']), 1):g}"))
        if "throw" in f and f["throw"] is not None:
            kv.append((K_THROW, f"{round(float(f['throw']))}"))
    if closures:
        kv.append((K_NCLOSURE, f"{len(closures)}"))
        for c in closures:
            kv.append((K_AREA, f"{round(float(c['area_pct']))}"))
    return kv


def fact_preamble(facts):
    """Consistent fact statements from the MEASURED facts — states EVERY injected fact
    (count · per-fault dip[/throw] · closure count · per-closure area) in a fixed phrase, in
    the SAME order as facts_to_kv, so each injected role marker has a home. This GUARANTEES
    correspondence → reliable copy, for any scene, built from the dataset's own facts at train
    time (scalable; new fact type = one phrase). Numbers come from vision; the LM copies them.
    It is only the copy scaffold — the grounded reasoning/narration is free after it."""
    faults = facts.get("faults", [])
    parts = [f"There are {len(faults)} faults."]
    for i, f in enumerate(faults):
        parts.append(f"Fault {i + 1} dips at {round(float(f['dip']), 1):g} degrees.")
        if f.get("throw") is not None:
            parts.append(f"Fault {i + 1} has throw of {round(float(f['throw']))} ms.")
    closures = facts.get("closures", [])
    if closures:
        parts.append(f"There are {len(closures)} closures.")
        for j, c in enumerate(closures):
            parts.append(f"Closure {j + 1} covers {round(float(c['area_pct']))} percent.")
    return " ".join(parts)


def qualitative(ev):
    """The evidence's DOMAIN LANGUAGE with numbers + tags removed — the seismic grounding that
    keeps reasoning on-domain (the narrative that made the reasoning transfer work), WITHOUT
    unbacked numbers. Drops all tags (keeping inner text) and keeps only digit-free sentences;
    the preamble owns the facts, this owns the domain."""
    t = re.sub(r"<[^>]+>", " ", ev)                      # drop all tags, keep inner text
    sents = re.split(r"(?<=[.])\s+", t)
    keep = [s.strip() for s in sents if s.strip() and not re.search(r"\d", s)]
    return " ".join(" ".join(keep).split())


def grounding_target(facts, narrative=""):
    """Stage-2 target: fact preamble (copy) + qualitative narrative (domain grounding). EVIDENCE
    ZONE ONLY — stage 2's job is grounding. The <think>/<answer> resolution is stage 3's (the fuse
    composes geology's reasoning-after-evidence + the answer), deferred to the skeleton step."""
    return " ".join(f"<evidence> {fact_preamble(facts)} {narrative} <SEG> </evidence>".split())


def narration_target(facts, narrative, answer):
    """Stage-3 target: preamble (copy) + qualitative narrative (domain grounding) + empty
    <think></think> placeholder + the grounded answer. Numbers live only in the preamble
    (backed); the narrative grounds the seismic domain (the language reasoning transfers from)."""
    return " ".join(
        f"<evidence> {fact_preamble(facts)} {narrative} <SEG> </evidence> "
        f"<think></think> {answer}".split())


def structured_evidence(ev):
    """Target surface: keep <evidence>/<nums>/<SEG> and the text (incl. "Fault N", "dips at");
    drop the structural wrappers <region>/<object>/<center>/<bbox>. The role marker comes from
    the KEYWORD near the number ("dips at"), not these wrappers, so dropping them keeps the
    correspondence intact. <nums> is kept — it preserves the number's integrity."""
    t = BBOX.sub("", CENTER.sub("", ev))
    return " ".join(WRAP_TAGS.sub("", t).split())


def structured_grounding(ev):
    """Stage-2 target — evidence (tags kept) + EMPTY <think>/<answer> placeholders, so the
    <evidence>/<think>/<answer> scaffold is consistent across ALL stages while Stage 2 stays
    evidence-copy focused (no answer content — that is Stage 3's job)."""
    return " ".join(f"{structured_evidence(ev)} <think></think> <answer></answer>".split())


def structured_narration(ev, an):
    """The proven grounded chain, tags kept: <evidence>...</evidence> <think></think>
    <answer>...</answer> (with <nums>/<SEG> inside). <think> is an EMPTY placeholder slot
    (no reason data yet; filled later by a tiny reason set). This is the config that grounded
    both grounding and the reasoning transfer."""
    return " ".join(f"{structured_evidence(ev)} <think></think> {an}".split())


class Narrator:
    """Stacked-adapter decoder (geology + grounding + fuse) + digit-token bridge.

    Stage flow: `set_stage('s2')` + `ground_loss` train the grounding adapter on
    evidence-copy; `set_stage('s3')` + `loss` train the fuse combiner on the
    detector-facts narration with grounding+geology frozen."""
    def __init__(self, lora_r=8, lora_alpha=16, prompt="Describe the faults: "):
        adapter = load_geology_adapter(GEOLOGY_CFG)
        self.model = GroundedDecoder(adapter_dir=adapter, lora_r=lora_r,
                                     lora_alpha=lora_alpha).to(device)
        self.dec, self.tok = self.model.decoder, self.model.tokenizer
        self.emb = self.dec.get_input_embeddings()
        self.facts_mod = FactTokens(self.emb.embedding_dim, self.emb, self.tok).to(device)

    def set_stage(self, stage):
        self.model.set_stage(stage)

    def trainable_params(self):
        return list(self.facts_mod.parameters()) + [q for q in self.dec.parameters() if q.requires_grad]

    def _emb_text(self, s):
        ids = self.tok(s, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        return self.emb(ids).squeeze(0)

    def build_prompt(self, ft, instruction, question=None):
        """Chatml prompt with the measured facts (ft) spliced into the SYSTEM turn — vision
        supplies facts, the user only asks. question=None -> no user turn (S2 grounding). Ends
        at '<|im_start|>assistant\\n' so geology's trained <think> trigger is present."""
        pre = self._emb_text(f"<|im_start|>system\n{instruction}\nMeasured facts: ")
        if question:
            post = self._emb_text(f"<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n"
                                  f"<|im_start|>assistant\n")
        else:
            post = self._emb_text("<|im_end|>\n<|im_start|>assistant\n")
        return torch.cat([pre, ft, post], 0)

    def _lm_loss(self, prompt_emb, target_str):
        tgt = self.tok(target_str + "<|im_end|>", add_special_tokens=False,
                       return_tensors="pt").input_ids.to(device)
        inp = torch.cat([prompt_emb, self.emb(tgt).squeeze(0)], 0).unsqueeze(0)
        labels = torch.cat([torch.full((prompt_emb.shape[0],), -100, device=device),
                            tgt.squeeze(0)], 0).unsqueeze(0)                    # prompt masked
        return self.dec(inputs_embeds=inp, labels=labels).loss

    def ground_loss(self, kv, target, question=None, instruction=None, max_kv=16):
        """Inject role-tagged facts into the SYSTEM turn; supervise the assistant target.
        S2 grounding: instruction=INSTRUCTION_S2, question=None. S3 QA: the dataset
        instruction + the question."""
        ft = self.facts_mod(kv[:max_kv])
        return self._lm_loss(self.build_prompt(ft, instruction or INSTRUCTION_S3, question), target)

    @torch.no_grad()
    def narrate(self, kv, question=None, instruction=None, max_new_tokens=160):
        """Inference: inject the role-tagged (detected/GT) facts into the system turn, ask the
        question in the user turn, generate the grounded chain freely — the LM copies each
        number into its role's phrase. The injected numbers are the only NUMBER source."""
        ft = self.facts_mod(kv[:16])
        prompt = self.build_prompt(ft, instruction or INSTRUCTION_S3, question)
        g = self.dec.generate(inputs_embeds=prompt.unsqueeze(0), max_new_tokens=max_new_tokens,
                              do_sample=False, repetition_penalty=1.3,
                              pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(g[0], skip_special_tokens=True).strip()

    @torch.no_grad()
    def generate(self, facts, max_new_tokens=160, question=None, instruction=None):
        """Grounded narration/answer from structured facts via the role-tagged bridge,
        optionally conditioned on a question (the user turn)."""
        return self.narrate(facts_to_kv(facts), question=question, instruction=instruction,
                            max_new_tokens=max_new_tokens)

    @torch.no_grad()
    def generate_reasoning(self, vals, question, max_new_tokens=120):
        """Grounded reasoning: inject the dip facts (system turn), ask a step-by-step question
        (user turn), let geology's <think> run through the grounded latent."""
        return self.narrate([(K_DIP, f"{v:g}") for v in vals[:6]],
                            question=f"{question} Think step by step.",
                            instruction=INSTRUCTION_S3, max_new_tokens=max_new_tokens)

    def seg_hidden(self, kv, target, instruction=None, question=None, max_kv=16):
        """LM forward on prompt+target; return the hidden state at each <SEG> token
        (n_seg, lm_dim) — the content prompt for the mask decoder — and the <SEG> count.
        Wrap in no_grad for a frozen-LM mask decoder; leave open for joint fine-tune."""
        from hybrid.model.mask_decoder import seg_positions
        ft = self.facts_mod(kv[:max_kv])
        prompt = self.build_prompt(ft, instruction or INSTRUCTION_S3, question)
        tgt = self.tok(target + "<|im_end|>", add_special_tokens=False,
                       return_tensors="pt").input_ids.to(device)[0]
        inp = torch.cat([prompt, self.emb(tgt.unsqueeze(0)).squeeze(0)], 0).unsqueeze(0)
        hs = self.dec(inputs_embeds=inp, output_hidden_states=True).hidden_states[-1][0]
        pos = seg_positions(self.tok, tgt)
        return hs[[prompt.shape[0] + p for p in pos]], len(pos)

    def train_mode(self):
        self.dec.train(); self.facts_mod.train()

    def eval_mode(self):
        self.dec.eval(); self.facts_mod.eval()
