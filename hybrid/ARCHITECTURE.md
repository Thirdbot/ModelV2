# Grounded Seismic VLM — architecture & training spec

Every shape / loss below is what the **code** does (`hybrid/…`). Notation: `d`=Qwen
hidden **1536** (Qwen2.5‑1.5B), image **H×W = 507×100** (tall, narrow), `smap`=stitched
NCS features **(768, 31, 6)** (fH=507//16, fW=100//16).

**Core idea — decoupled facts seam.** The *vision* front‑end **measures** the numbers
(dense mask → connected components → class‑routed readers). The *language* half (Qwen +
stacked LoRA) receives those numbers as **role‑tagged digit tokens** and **copies** them
into grounded narration. The seam between them (`measure_instances`: RANSAC / components /
threshold) is **non‑differentiable** — the LM cannot fabricate a number, and no gradient
crosses back. Faithfulness is structural; accuracy is the detector's job.

---

## 0. Data (`data/dataset.py`, `model/scenes.py::build_scenes`)

One scene per unique image (regions grouped by image):

| field | type | meaning |
|---|---|---|
| image | PNG 507×100 | seismic section |
| regions[i] | dict | `class_id` (1 fault / 2 closure), `bbox`[x1,y1,x2,y2], `mask_idx` |
| mask | 507×100 binary → **dilated r=3** | fault trace, fattened ~1 cell |
| meas | [dip°, throw ms, %] | from evidence `<nums>` (dip = **apparent dip**) |
| mmask | [0/1,0/1,0/1] | which measurements are supervised |
| evidence / answer | tagged text | `<evidence><region><object>…<nums>N</nums>…<SEG></region></evidence>`, `<answer>…</answer>` |

`scene = {smap, hw=(507,100), objs=[{cls,bbox_norm,mask,meas,mmask}], img, fault_field, closure_field}`.
`fault_field`/`closure_field` = union of the class's masks (the dense targets).

---

## 1. Vision front‑end (`model/segmenter.py`)

### 1a. NcsEncoder (`model/encoder.py`) — FROZEN
ViT `NCS-v1-2d-base`, patch 16. `stitch` tiles the image (224/stride 112), runs the ViT
per tile, averages overlaps → **`smap (768, 31, 6)`**. No grad. (NCS was pretrained on
real seismic — it transfers to real field.)

### 1b. DenseSegmenter — TRAINABLE (Stage 2a)
No object queries, no N cap. **Class = which channel.**
| step | op | in → out |
|---|---|---|
| trunk | `Conv2d(768→128,3)·GN·GELU ×2` | `smap` → `(128, 31, 6)` |
| seg_head | `Conv2d(128→2,1)` → bilinear ↑ | → **`seg (2, 507, 100)`** = [fault, closure] logits |

### 1c. Instances + class‑routed measurement (`measure_instances`)
Per channel: `sigmoid>0.5` → **8‑connected components** (size‑filtered) = instances →
`count`. For each instance the **class routes the reader**:
- **fault** → `dip` = **RANSAC line‑fit** on the mask pixels (`_line_dip`, apparent dip 0–90°) · `throw` = `ThrowHead` (pooled raw‑smap feature, magnitude) · `bbox` = component extent.
- **closure** → `area%` = pixel fraction · `bbox`.

Adding a class = +1 seg channel + 1 `CLASS_CHANNELS` entry + 1 reader branch. Nothing else.

### 1d. Vision loss (`vision_loss`)
`seg_loss` = per channel `BCEwithLogits(pos_weight=50) + soft‑dice`, `+ throw_loss`
(smooth‑L1 on GT‑fault throws). *(Aux bbox/class/dip heads were tried and removed — they
lifted train fit but not held‑out dice: a generalization gap, not a signal gap.)*

**Facts produced** = `count`, per‑fault `dip`(RANSAC)/`throw`(head)/`bbox`, per‑closure
`area`/`bbox`. Held‑out mask dice ≈ 0.2 (the open wall → real‑field stage, §4 S4).

---

## 2. Facts bridge — vision → LM (`model/narrator.py`)

Numbers pass as **digit‑token embeddings with a per‑kind role marker** (proven ≫ scalar):
```
fact(kind, "value") = marker[kind] (1,d)  ++  emb(tokenize("value")) (T,d)
```
- `FactTokens.marker = Embedding(7, d)` — K_COUNT/K_DIP/K_EVID/K_NCLOSURE/K_AREA/K_BBOX/K_THROW. TRAINABLE. The **marker is where the value's meaning lives** (a dip can only land in the dip phrase).
- **Role routing** (`KIND_KW`, one keyword each): `evidence_kv(text)` tags each `<nums>` on the TRAIN side; `facts_to_kv(facts)` builds the SAME tagged list on the INFERENCE side — count·dip·throw·closure‑count·area. Identical scheme both sides → the copy transfers.

---

## 3. GroundedDecoder — stacked‑adapter fuse (`model/decoder.py`)

Qwen2.5‑1.5B (4‑bit), `d=1536`. Three additive LoRA adapters (r8/α16, geology r16; on
q,k,v,o,gate,up,down):

| adapter | trained in | inference |
|---|---|---|
| geology | Stage 1 (Unsloth SFT) | active, FROZEN |
| grounding | Stage 2b | active, FROZEN after S2 |
| fuse (combiner) | Stage 3 | active, trainable in S3 |

`set_stage('s2')` → active[geology,grounding], **grounding** trains.
`set_stage('s3')` → active[geology,grounding,fuse], **only fuse** trains.
Active adapters are additive (AdapterFusion / progressive LoRA stacking) over the shared
linguistic latent. Input = `[fact tokens ; prompt ; target]`; loss = next‑token CE on the
target span (fact+prompt positions labelled −100). **The LM sees the injected numbers only —
no vision features** (co‑refine / feature‑bridge removed).

---

## 4. Training stages (each freezes the prior)

**S1 — geology adapter** (`train/stage1_geology.py`, Unsloth, cached, run once).
GeoGPT‑CoT text → SFT → frozen geology LoRA (`experiments/ckpt/stage1_…`).

**S2a — vision** (`train/stage2_detector.py`, `VIS_EPOCHS=150`).
scenes → `vision_loss` (seg + throw). Segmentor frozen thereafter.

**S2b — grounding** (`train/stage2_grounding.py`, 25 ep, 200 rows).
`set_stage('s2')`. `(role‑tagged kv, structured_evidence)` → `ground_loss` (LM CE): copy the
injected numbers into the tagged evidence. Grounds the shared latent, then frozen.

**S3 — fuse narration** (`train/stage3_narrator.py`, `LM_EPOCHS=150`, 200 rows).
`set_stage('s3')`. `(kv, structured_narration)` → `ground_loss`. Target = the dataset's
tagged chain **`<evidence>…</evidence> <think></think> <answer>…</answer>`** (tags kept
verbatim, `<center>`/`<bbox>` dropped, empty `<think>` slot). Only the fuse trains; the
segmentor is untouched.

**S4 — real‑field vision refine** (separate; `train/stage4_realfield.py`).
Load `stage3_vision.pt`, fine‑tune the segmentor + throw on **real Smeaheia** GT
(`data/real.py`: SEG‑Y → image, 3D fault sticks projected → mask + apparent dip, horizons →
throw). **Narration frozen** — this is syn→real for the *vision* only. Benchmark:
`test/benchmark_real.py` (dice · dip MAE · throw MAE · bbox IoU).

*(Reasoning: the empty `<think>` slot is a placeholder for a future faithfulness signal —
stage‑4‑style RLVR or a tiny reason set. Copy‑grounding gets number‑use, not no‑invention.)*

---

## 5. Evaluation (`test/evaluate.py`, held‑out)
- **copy score**: `generate(scene_facts)` → `FAULT_LINE` parses the dip out of the tagged
  narration → fraction == injected (exact within 1°).
- **swap follows**: perturb the injected dip +15°, regenerate, output must follow.
- **reasoning invented** (HONEST metric): `generate_reasoning` → count *every* stated number
  not in the facts (`\d+\s+[a-z]`, excluding "Fault N" indices). Lower = more grounded.
- **detector‑acc** (`train.py`): mask dice; dip via PRED mask vs via GT mask, stratified.
- **overlays** (`inference/infer.py::save_overlays`): image + pred(red)/GT(green) mask +
  bbox + facts + the generated narration.

---

## 6. End‑to‑end inference (`inference/infer.py`)
```
image (507×100)
  → NcsEncoder (frozen) + stitch          → smap (768,31,6)
  → DenseSegmenter                        → seg (2,507,100)  [fault|closure]
  → connected components / channel        → count + per‑instance masks
  → measure_instances (class‑routed)      → fault{dip(RANSAC),throw,bbox} · closure{area,bbox}
  ─────────── non‑differentiable facts seam (LM can't fake numbers) ───────────
  → facts_to_kv → FactTokens (role‑tagged digit tokens)
  → GroundedDecoder [geology+grounding+fuse]  → grounded narration <evidence><think><answer>
                                                (numbers copied by role; reasoning in <think>)
  masks                                    → <SEG> anchors order‑mapped → display overlay
```

**Status.** Copy faithful where it copies (exact); vision accurate on train (dice 0.7,
dip ~7°), the wall is held‑out (dice 0.2) → the real‑field Smeaheia stage. Reasoning
emerges on‑domain + fact‑using but still confabulates in‑domain (honest metric) → needs a
faithfulness signal in the `<think>` slot, not just grounding.
