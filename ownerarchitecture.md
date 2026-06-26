# Owner Architecture

## Goal

Build a seismic grounded VLM that can read one or more full seismic images and produce:

- grounded textual evidence for seismic features
- object type / class information
- approximate position through bbox
- segmentation mask through `<SEG>`
- final reasoning and answer when a question is provided

The model should eventually work at inference without external ground-truth bbox. It should inspect the full image, find relevant seismic regions, describe them, and produce grounded outputs. The dataset already contains images, masks, bboxes, region metadata, evidence, question, reason, and answer, so the architecture should use those relationships instead of treating the task as plain captioning.

## Proposed Architecture

### 1. Global Encoder

The global encoder reads the entire seismic image.

Because the full image can be larger or shaped differently than a fixed-size encoder input, the global encoder may use patching/tiling/patchify logic internally.

Its job is to produce:

- global visual feature map or visual tokens
- candidate object/class predictions
- candidate bbox positions in normalized image coordinates

The global bbox output should be normalized first:

```text
[x1/W, y1/H, x2/W, y2/H]
```

Then it can be converted back into actual full-image coordinates:

```text
[x1, y1, x2, y2]
```

This branch is responsible for finding the rough position of seismic features from full-image context.

### 2. Region Encoder

The region encoder receives a region selected by bbox.

During inference:

```text
global encoder predicted bbox -> region encoder
```

During training:

```text
ground-truth bbox from dataset -> region encoder
```

Using ground-truth bbox during training makes the region encoder stable first. It avoids training the region branch on bad predicted boxes before the global bbox branch has learned useful localization.

The region encoder produces:

- local visual feature
- refined region understanding
- optional refined class / bbox prediction

NCS-v1-2d-base is a strong candidate for the region encoder because it is domain-specific for seismic 2D images. Even if it is not ideal for arbitrary-size full-image dense localization, it can still be useful for region/crop semantic understanding.

### 3. Position / BBox Representation

The bbox should be passed forward as explicit spatial metadata, not only through RoI/crop features.

Recommended position vector:

```text
[x1/W, y1/H, x2/W, y2/H, w/W, h/H, cx/W, cy/H, area/(W*H)]
```

This is passed through a bbox MLP:

```text
bbox_norm -> bbox_position_embedding
```

This gives later layers explicit full-image spatial information.

### 4. Feature Set Passed Forward

After the global and region branches are stable, the next layer receives three main information streams:

```text
1. global feature
2. local / region feature
3. position bbox embedding
```

Optionally:

```text
4. class embedding / class logits
```

These are passed into vision-language layers following the GLaMM-style idea:

```text
global context + region feature + position -> LLM / grounding decoder
```

The LLM produces structured text such as:

```text
<region>
<object>fault</object>
<class_id>1</class_id>
<color>red</color>
<evidence>...</evidence>
<bbox>[x1, y1, x2, y2]</bbox>
<SEG>
</region>
```

Only `<SEG>` needs to be a special token. XML-like tags can remain normal text tokens.

### 5. Mask Decoder

The `<SEG>` token marks the region that should produce a mask.

The hidden state at `<SEG>` can be projected and combined with local/global visual features:

```text
hidden_state(<SEG>) + local feature + global feature -> mask decoder
```

The mask decoder predicts a binary mask. Bbox can later be computed from the predicted mask when needed.

## Training Plan

### Stage A: Region Encoder Warmup

Input:

```text
image + ground-truth bbox
```

The region encoder uses the dataset bbox to crop/RoIAlign the region.

Targets:

- object class
- bbox metadata
- evidence text
- optional mask

Purpose:

```text
teach local seismic region semantics before relying on predicted boxes
```

### Stage B: Global BBox / Proposal Training

Input:

```text
full image
```

Targets:

- object class
- normalized bbox

Losses:

- class loss
- bbox L1 / SmoothL1 loss
- optional GIoU loss

Purpose:

```text
teach global encoder to produce rough feature positions
```

### Stage C: Connect Global Prediction To Region Encoder

Use predicted bboxes from the global branch to feed the region encoder.

To avoid instability, use a schedule:

```text
early: mostly ground-truth bbox
later: mix ground-truth bbox and predicted bbox
final: mostly predicted bbox
```

Purpose:

```text
make inference path match training path
```

### Stage D: Grounded Text Training

Input:

```text
global feature + local feature + bbox position + prompt
```

Target:

```text
<region>...</region>
```

Purpose:

```text
align visual features with structured evidence output
```

### Stage E: Mask Grounding

Input:

```text
same as grounded text training
```

Target:

- text output
- mask for each `<SEG>`

Losses:

```text
text CE loss + BCE mask loss + Dice mask loss
```

Purpose:

```text
connect language grounding token `<SEG>` to pixel mask
```

### Stage F: Full Question Answering

Input:

```text
one or more images + question
```

Target:

```text
regions + reason + answer
```

Purpose:

```text
learn final user-facing seismic interpretation behavior
```

## Debate: What Is Right

### The Architecture Matches The Dataset

The dataset already has:

- image
- bbox
- mask
- region metadata
- evidence
- question
- reason
- answer

So using a global proposal branch, region encoder, bbox embedding, and `<SEG>` mask path is aligned with the data. This is better than forcing a plain VLM to learn all localization through text loss.

### Ground-Truth BBox For Region Training Is Correct

Using dataset bbox during early region-encoder training is the right move. If the region encoder receives bad predicted boxes too early, it learns from corrupted crops/features.

The stable path is:

```text
GT bbox first -> predicted bbox later
```

### Global + Local Feature Is Necessary

Seismic features can be locally ambiguous. A crop can show a texture, but the meaning may depend on full section structure.

So the model should have both:

```text
global context
local region detail
```

### Explicit BBox Embedding Is Necessary

RoIAlign/cropping gives local appearance, but fixed-size RoI output does not clearly preserve absolute full-image position and scale.

Passing normalized bbox metadata solves this.

### NCS Is Valuable As A Region Encoder

NCS is domain-specific for seismic 2D images, so it should be trusted for seismic semantic representation. It is especially useful on region crops where fixed input size is acceptable.

## Debate: What Is Wrong Or Risky

### RoIAlign Does Not Find BBoxes

RoIAlign only does:

```text
bbox + feature map -> region feature
```

It does not do:

```text
feature map -> bbox
```

So the architecture must include a detection/proposal head before RoIAlign if inference should work without external bbox.

### Training With GT BBox Creates Train/Inference Gap

If training always uses ground-truth bbox, but inference uses predicted bbox, the region encoder may fail when predictions are imperfect.

This must be fixed with scheduled mixing:

```text
GT bbox -> mixed bbox -> predicted bbox
```

### NCS May Not Be Enough For Global Dense Localization

NCS is domain-specific, but if it is fixed-input ViT-style, it is awkward for arbitrary-size full-image dense feature maps.

It may be better as:

```text
region semantic encoder
```

while another backbone handles:

```text
full image -> spatial feature map -> bbox proposal
```

### Free-Text BBox Generation Is Weak

The LLM should not be trusted to generate precise coordinates by text alone.

Better:

```text
bbox head predicts bbox
mask decoder predicts mask
final bbox is computed or formatted afterward
```

The XML `<bbox>` field can be filled from model heads, not only generated text.

### Too Many Special Tokens Hurt Training

Only `<SEG>` should be special.

Making all XML tags special requires training embeddings/lm_head. Current LoRA setup does not automatically train those rows, so generation becomes unstable.

## How To Improve

### Start With Detection + Region Classification Before LLM

Before adding text generation, prove the visual part works:

```text
full image -> bbox + class
GT bbox crop -> class
```

If the model cannot localize and classify regions, the VLM layer will not fix it.

### Use Mask-To-BBox When Possible

For final output:

```text
predicted mask -> bbox
```

This is more reliable than generating bbox text directly.

### Add Object Queries For Global Proposals

A DETR-like small query head may fit this task:

```text
global feature map + learned queries -> class + bbox
```

This avoids dense anchor engineering and handles variable object counts.

### Keep Region Order Deterministic

The mapping must be stable:

```text
region 0 -> first <SEG> -> mask 0
region 1 -> second <SEG> -> mask 1
```

This is critical for text-mask alignment.

### Use BBox Normalization Everywhere Internally

Internal training target:

```text
normalized bbox in [0, 1]
```

User-facing output:

```text
absolute bbox in original full-image coordinates
```

### Separate Semantic Numbers From Geometry

Geometry:

```text
bbox, mask, center, area
```

should come from visual heads/mask processing.

Semantic numbers:

```text
throw, count, width, percentile
```

can stay in evidence/reason text unless they are consistently available as structured labels.

## Current Best Direction

The most practical version is:

```text
1. global proposal encoder predicts bbox/class from full image
2. region encoder uses GT bbox first, predicted bbox later
3. bbox MLP preserves full-image position
4. local/global features go into V-L bridge
5. LLM emits evidence text with `<SEG>`
6. mask decoder uses `<SEG>` hidden state
```

This follows the useful parts of GLaMM while adapting to seismic images and the dataset structure.

## Dataset Scope: What To Keep, Move, Or Cut

The dataset generator currently writes many things into XML-like tags. That is useful because the generator knows more metadata than a plain answer dataset. But not every field should be generated by the LLM. Some fields should train visual heads, some should train language, and some should only be used as metadata for rendering/evaluation.

### Keep As Core Visual Supervision

These fields should stay and should be treated as structural supervision:

```text
image
mask
bbox
object / object_type
class_id
region_index
image_index
mask_index
```

Why:

- `object` / `class_id` trains the model to classify seismic feature type.
- `bbox` trains the global proposal head and gives the region encoder its training crop.
- `mask` trains the `<SEG>` mask decoder.
- indexes keep the mapping deterministic:

```text
region -> image -> mask -> <SEG>
```

These should not be removed.

### Move Out Of Generated Text When Possible

These fields are useful, but the LLM should not be the only source of truth for them:

```text
bbox
center
area
position
```

Use them internally as numeric targets:

```text
bbox head target
bbox_norm embedding
mask-derived bbox evaluation
```

The final XML can still show:

```text
<bbox>[x1, y1, x2, y2]</bbox>
```

but ideally that value should be filled from the bbox/mask head, not generated as free text.

Reason:

LLMs are weak at precise coordinates. Visual heads are better for geometry.

### Keep As Text Supervision

These fields should stay in text:

```text
evidence
reason
answer
```

Why:

- `evidence` teaches the model how to describe visual facts.
- `reason` teaches how evidence is combined.
- `answer` teaches the user-facing final response.

But the training stage matters:

```text
Stage 1: image/region -> evidence only
Stage 2: all evidence + question -> reason + answer
Stage 3: images + question -> evidence + reason + answer
```

Do not force one unnested region to produce the full row-level answer unless that region alone is enough to answer.

### Treat Color As Rendering Metadata

`color` is useful, but it is not core visual semantics.

It means:

```text
class -> overlay color
```

not necessarily:

```text
the seismic image visibly contains this color
```

So use it mainly for:

```text
overlay rendering
visualization
class display
postprocessing
```

It can remain in XML for user readability:

```text
<color>red</color>
```

but the model does not need to learn color as a deep visual concept. It can be filled from a class-color map:

```text
fault -> red
onlap -> yellow
lithology -> green
closure -> blue
```

Recommended:

```text
do not make color a primary loss
do not use color as evidence of object identity
derive color from class_id at output time when possible
```

### Cut From Stage 1 Text Target

For the first working model, Stage 1 should not generate too many fields.

Recommended Stage 1 target:

```text
<region>
<object>fault</object>
<class_id>1</class_id>
<evidence>...</evidence>
<SEG>
</region>
```

Optional:

```text
<bbox>...</bbox>
```

but only if it is copied from the bbox head or kept as simple rounded integer text.

Cut or avoid in Stage 1:

```text
final answer
reason
too many decimal measurements
color if it is only overlay metadata
precise angle/throw unless directly visible and consistently labeled
```

Why:

Stage 1 should teach:

```text
visual region -> semantic evidence
```

not full question answering.

### Keep Full Rich Output For Stage 3

The full dataset prompt and richer XML should be used in Stage 3:

```text
images + question -> regions + reason + answer
```

Stage 3 can include:

```text
<region>
<object>...</object>
<class_id>...</class_id>
<color>...</color>
<evidence>...</evidence>
<bbox>...</bbox>
<SEG>
</region>
<reason>...</reason>
<answer>...</answer>
```

At this point the model has already learned lower-level region semantics and evidence aggregation.

### Recommended Dataset Fields For Training

Use this canonical internal format per region:

```python
{
    "source_row": int,
    "region_index": int,
    "image_index": int,
    "mask_index": int,
    "image": PIL.Image,
    "mask": PIL.Image,
    "object": str,
    "class_id": int,
    "bbox_abs": [x1, y1, x2, y2],
    "bbox_norm": [x1/W, y1/H, x2/W, y2/H, w/W, h/H, cx/W, cy/H, area/(W*H)],
    "evidence": list[str],
    "color": str,
}
```

Use this canonical internal format per original row:

```python
{
    "source_row": int,
    "images": list[PIL.Image],
    "masks": list[PIL.Image],
    "regions": list[region],
    "question": str,
    "reason": str,
    "answer": str,
}
```

This separates:

```text
region-level visual supervision
row-level reasoning supervision
```

### What To Fix In The Dataset Generator

If possible, improve the dataset generator with these changes:

1. Mark which evidence supports which question/answer.

```text
region.relevance = direct | supporting | context | unrelated
```

This helps Stage 3 learn which regions matter.

2. Store geometry as structured fields, not only text.

```text
bbox_abs
bbox_norm
center
area
mask_index
```

3. Keep semantic measurements structured only when reliable.

For example:

```text
throw_value: float | None
throw_valid: bool
angle_value: float | None
angle_valid: bool
count_value: int | None
count_valid: bool
```

Do not force heads for values that are inconsistent or optional.

4. Make color explicitly metadata.

```text
color_source = "class_overlay"
```

This prevents treating color as a real seismic visual feature.

5. Keep XML text clean and minimal for early stages.

Early text targets should be easy to learn. Rich formatting belongs later.

## Dataset Cut Summary

For a working first architecture:

```text
Keep:
  image, mask, bbox, object, class_id, evidence, question, reason, answer

Use internally, not mainly text:
  bbox, center, area, mask, position

Derive at output time:
  color from class_id
  bbox from mask when possible

Cut from Stage 1:
  answer, reason, excessive numeric details, color as required generation

Keep for Stage 3:
  full XML-style response with regions + reason + answer
```

This keeps the dataset useful without asking the LLM to do jobs better handled by visual heads or postprocessing.
