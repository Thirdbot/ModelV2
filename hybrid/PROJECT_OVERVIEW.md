# Hybrid Grounded Seismic VLM — Project Overview

## 1. Why this needs to exist

Seismic interpretation is a dual task. A geoscientist looking at a seismic section
needs two things at once:

1. **Natural-language geological reasoning** — *"this is a normal fault produced by
   extensional stress; it may seal or leak hydrocarbons depending on juxtaposition."*
2. **Precise, quantitative localization** — *where* the fault is (bounding box, pixel
   mask) and *how much* (dip ≈ 65°, throw ≈ 68 m, closure count).

No existing model class does both:

- **Generic vision-language models** (captioners, chat VLMs) produce fluent text but
  cannot emit calibrated measurements or pixel-accurate masks. Ask one for a fault
  throw and it hallucinates a number as free text.
- **Detection / segmentation models** localize precisely but cannot reason or answer
  open geological questions.
- **Grounded LMMs** (GLaMM, LISA, Kosmos-2) link text phrases to masks/boxes, but they
  are built for *natural images* and have **no mechanism for typed domain
  measurements** — GLaMM structurally cannot output "throw = 68 m."

This project builds a **hybrid grounded VLM** that unifies all of it for the seismic
domain: it emits geological evidence text whose specific claims are anchored to
**typed, learned grounding heads** (numbers, boxes, centers, classes, masks), and it
reasons about the geology of the grounded objects — under a **6 GB consumer-GPU**
budget, which rules out the 7B-parameter + SAM-ViT-H stacks the literature assumes.

## 2. Problem statement

**Input:** a seismic image (2-D section) + a natural-language question.
**Output:** a structured, grounded response:

```
<evidence> object-referenced evidence with grounded slots + <SEG> masks </evidence>
<think>   chain-of-thought geological reasoning </think>
<answer>  final answer, grounded </answer>
```

where each **slot token** is realized by a dedicated regression/classification head:

| slot token      | head            | output                         |
|-----------------|-----------------|--------------------------------|
| `<REG_SLOT>`    | numeric (L1)    | a measurement (log-scaled)     |
| `<BBOX_SLOT>`   | bbox (sigmoid)  | `[x1,y1,x2,y2]` normalized     |
| `<CENTER_SLOT>` | center (sigmoid)| `[cx,cy]` normalized           |
| `<CLASS_SLOT>`  | class (CE)      | fault / closure / …            |
| `<SEG>`         | mask decoder    | full-resolution binary mask    |

## 3. Architecture

Data flow for one sample:

```
 seismic image
     │  tile into overlapping 224px crops (stride 112)
     ▼
 FROZEN NCS encoder  (NorskRegnesentralSTI/NCS-v1-2d-base, ViT, 768-d/tile)
     │                                   │
     │ path A (language)                 │ path B (pixels)
     ▼                                   ▼
 tile_pos + VisualBridge            stitch per-tile patch grids into one
 (MLP 768→1536, +position)          spatial map (768 × H/16 × W/16)
     │  one prefix token per tile        │
     ▼                                   │
 Qwen2.5-1.5B decoder (4-bit QLoRA)      │
   ├─ FROZEN geology adapter            │
   ├─ TRAINABLE grounding adapter       │
   └─ slot_out_head (makes slot         │
      tokens generatable)               │
     │  generates evidence/think/answer │
     │  with slot tokens                 │
     ▼                                   ▼
 hidden state at each slot token → typed head        <SEG> hidden + spatial map
   numeric · bbox · center · class                    → LisaMaskDecoder → mask
     │                                                       │
     └──────────────► grounded output ◄─────────────────────┘
     text · numbers · boxes · classes · masks
```

**Components**
- **Frozen NCS vision encoder** — a domain-pretrained seismic ViT (224 px, 16 px
  patches → 14×14 grid). Frozen: it supplies features, never trains. Two consumers:
  a pooled per-tile vector (language path) and the full patch grid (mask path).
- **Visual bridge + tile-position embedding** — projects each tile's 768-d feature
  (plus a learned embedding of the tile's normalized location) into the decoder's
  1536-d space, so tiles enter the prompt like tokens and the LLM knows the layout.
- **Decoder (Qwen2.5-1.5B, 4-bit QLoRA)** with two LoRA adapters active additively:
  a **frozen geology adapter** (Stage-1 knowledge, never eroded) and a **trainable
  grounding adapter** (learns perception). This is parameter-isolated continual
  learning — it preserves reasoning *and* learns grounding at full learning rate.
- **`slot_out_head`** — a small trainable head that boosts the output logits of the
  five slot tokens. Without it, the newly-added slot tokens (whose tied embeddings are
  frozen for VRAM/stability) can never win at generation, so the grounding pathway is
  dead at inference. This head switches it on.
- **Typed slot heads** — the decoder hidden state at each slot position feeds a small
  head: numeric (log-scaled L1), bbox, center, class. This generalizes LISA's
  "embedding-as-mask" to *arbitrary typed values*, which is the key domain fit.
- **Mask decoder (from-scratch, full-resolution)** — a `<SEG>` hidden state
  cross-attends over the stitched spatial map and produces a full-res mask (BCE +
  Dice). Chosen over a SAM decoder because seismic faults are ~1-px curvilinear
  structures that SAM's low-resolution blob prior cannot trace.

## 4. Training curriculum

| stage | trains | objective |
|-------|--------|-----------|
| **1. Geology CoT** | geology LoRA adapter (Unsloth SFT on GeoGPT-CoT) | learn `<evidence><think><answer>` skeleton + geological reasoning |
| **2. Perception**  | grounding adapter + bridge + heads + mask decoder | image → grounded `<evidence>` (slots + `<SEG>`) |
| **3. Fusion**      | (continues Stage 2) | image → evidence → think → answer; slots live in all three blocks |
| ~~4. RL~~          | — | dropped at current data scale (premature) |

The geology adapter from Stage 1 is **frozen** for Stages 2–3; only the grounding
adapter + heads train. Because all three stages share the same `<evidence><think>
<answer>` skeleton, the model can **compose** grounded evidence with geological
reasoning in one response.

## 5. Evaluation

- **Composite metric** (lower = better), on a **group-wise split by image** (whole
  images held out — no row-level leakage):
  `composite = text + class + bbox + center + numeric + 0.3·mask` (eval losses).
- **Qualitative probes** — grounding / pure-geology / mixed questions, with mask
  overlays vs ground truth.

## 6. Key contributions (for the paper)

1. **Typed slot-token grounding** — extends embedding-as-mask to numeric/box/center/
   class heads, enabling *quantitative* domain grounding a captioner or GLaMM cannot do.
2. **`slot_out_head`** — a minimal fix that makes rare special tokens generatable under
   frozen tied embeddings, without unfreezing the vocabulary.
3. **Parameter-isolated continual learning** (frozen geology adapter + trainable
   grounding adapter) — preserves reasoning knowledge while learning perception at full
   LR, avoiding catastrophic forgetting.
4. **Shared structural skeleton** across curriculum stages → format robustness and
   *emergent* grounding-plus-reasoning composition without hand-authored mixed data.
5. **6 GB-GPU-feasible** grounded VLM — 1.5B QLoRA decoder + frozen domain encoder +
   from-scratch thin-structure mask head.

## 7. Limitations & future work

- **Data scale** — the current synthetic set is image-limited; the honest group-split
  val is small. More distinct scenes is the primary lever.
- **Thin-fault mask sharpness** — the raw mask is faint; a centerline-Dice (clDice)
  loss or GT dilation is the targeted fix.
- **Region reference tokens** — a `<REF>` token carrying pooled region features
  (à la Ferret/GPT4RoI) would bind reasoning to the grounded object at the feature
  level; deferred until data scale justifies the added capacity.
- **Stage-4 RL** — with a completeness + evidence-consistency reward (LLM-judge), to
  reinforce the emergent combination once data grows.
