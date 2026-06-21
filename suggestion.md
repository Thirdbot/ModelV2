# Dataset Suggestions for Seismic VLM

## Current Dataset

Dataset: `thirdExec/synthetic-seismic-vlm`

Observed schema:

```text
images:      List[Image]
masks:       List[Image]
instruction: string
question:    string
answer:      string
evidence:    string
```

Observed split:

```text
train rows: 117
total image/mask pairs: 429
image size: 100 x 507
image mode: RGBA
mask size: 100 x 507
mask mode: L
mask values: 0 or 255
```

Observed classes:

```text
fault:   class_id 1, red,    313 regions
closure: class_id 2, blue,    95 regions
onlap:   class_id 4, yellow,  21 regions
```

Each row has the same number of `images`, `masks`, and `<region>...</region>` blocks. That is good. The intended design is useful: each image/mask pair teaches the model which visible segment supports each evidence block. Keep that design, but make the relationship explicit so training can always know:

```text
images[i] -> masks[i] -> regions[i]
```

Some rows repeat the same base image for several masks while others contain different images. That is acceptable, but it should be represented explicitly with IDs.

## Main Problem

The planned model expects a clean contract:

```text
pixel_values -> NCS vision encoder -> text/bbox/mask outputs
```

But the current dataset is closer to:

```text
one row -> list of images -> list of masks -> free-form evidence string -> answer string
```

This makes batching, tiling, bbox remapping, mask loss, and multi-object training harder than needed unless the image/mask/region alignment is formalized.

The model should not have to parse dataset geometry during loss. The dataset should already provide tile-local targets, while still preserving the original XML target for user-facing output.

## Recommended Dataset Contract

Create a processed training dataset where each sample is one model input tile.

Recommended fields:

```python
{
    "image_id": str,
    "tile_id": str,
    "pixel_values": image_tile,        # [3, 224, 224]
    "prompt": str,
    "target_text": str,
    "objects": [
        {
            "object_id": str,
            "label": str,              # "fault", "closure", "onlap"
            "class_id": int,
            "bbox": [x1, y1, x2, y2],  # tile-local coords
            "mask": mask_tile,         # [224, 224], binary
            "evidence": list[str]
        }
    ],
    "meta": {
        "x0": int,
        "y0": int,
        "orig_w": int,
        "orig_h": int,
        "pad_w": int,
        "pad_h": int
    }
}
```

The important rule:

```text
pixel_values is what the model sees.
meta tells where this tile came from in the original image.
objects contains only objects visible in this tile.
```

## Image Preprocessing

The NCS encoder should receive image-like tiles, not raw ViT patches.

Use:

```text
tile_size = 224
patch_size = 16
stride = 112
```

Because the source images are `100 x 507`, pad width to 224 and tile vertically.

Recommended padding:

```text
left/top: 0
right:    224 - width
bottom:   enough only if height < 224
```

Do not center-pad at first. Right/bottom padding preserves the current bbox coordinates.

Convert image mode:

```text
RGBA -> RGB
```

Then normalize consistently. For a first pass, use the NCS processor if available. If not, convert to float `[0, 1]`, then standardize per image or per dataset.

## Tiling Logic

For an original image with width `W` and height `H`:

```python
tile_size = 224
stride = 112
```

Generate `x0, y0` starts. Since current width is 100, after right padding width is 224, so `x0 = 0`.

For height 507:

```text
y starts: 0, 112, 224, 283
```

The final start is forced to `H - tile_size = 283` so the bottom edge is covered.

For each tile, save:

```python
meta = {
    "x0": x0,
    "y0": y0,
    "orig_w": W,
    "orig_h": H,
}
```

## Bbox Conversion

Current bboxes are global image coordinates:

```text
[x1, y1, x2, y2]
```

For each tile, intersect the global bbox with the tile rectangle:

```python
ix1 = max(global_x1, x0)
iy1 = max(global_y1, y0)
ix2 = min(global_x2, x0 + tile_size)
iy2 = min(global_y2, y0 + tile_size)
```

Keep the object only if:

```python
ix1 < ix2 and iy1 < iy2
```

Then convert to tile-local coordinates:

```python
local_bbox = [
    ix1 - x0,
    iy1 - y0,
    ix2 - x0,
    iy2 - y0,
]
```

Drop tiny intersections, for example:

```text
local_width >= 8
local_height >= 8
or intersection_area / object_area >= 0.1
```

## Mask Conversion

Each mask already corresponds to one evidence region.

For each object:

```python
mask_tile = full_mask[y0:y0 + 224, x0:x0 + 224]
```

If the original image was padded, pad masks the same way as images.

Convert masks:

```text
0   -> background
255 -> foreground
```

Use:

```python
mask = (mask > 0).float()
```

## Text Target

The current dataset separates `evidence` and `answer`.

For training the language output, combine them into one target:

```text
target_text = evidence + "\n" + answer
```

Use XML as the primary output format. This matches the current dataset and is easier for users to read than JSON, as long as the tags are strict and consistent.

Recommended target format:

```xml
<region>
<object>fault</object>
<class_id>1</class_id>
<color>red</color>
<evidence>Fault 1 occupies the area from x=0 to 99 and y=214 to 230</evidence>
<bbox>[0, 102, 99, 118]</bbox>
<SEG>
</region>
<answer>Yes, Fault 1 shows gouge near the 0 percentile.</answer>
```

For multiple objects, repeat `<region>` blocks:

```xml
<region>
<object>fault</object>
<class_id>1</class_id>
<color>red</color>
<evidence>Fault 1 supports the answer.</evidence>
<bbox>[0, 102, 99, 118]</bbox>
<SEG>
</region>
<region>
<object>fault</object>
<class_id>1</class_id>
<color>red</color>
<evidence>Fault 2 supports the answer.</evidence>
<bbox>[0, 108, 99, 132]</bbox>
<SEG>
</region>
<answer>There are 2 faults present.</answer>
```

Rules for the XML:

```text
1. Every visible object gets exactly one <region>...</region>.
2. Every <region> has exactly one <object>, <class_id>, <color>, <bbox>, and <SEG>.
3. Every <region> has one or more <evidence> tags.
4. <bbox> is always tile-local during tile training.
5. <bbox> always uses integer [x_min, y_min, x_max, y_max].
6. <answer> appears once after all regions.
7. No extra free text outside XML tags.
```

This keeps the output readable for the user and still parseable for metrics/loss helpers.

Optional later format if you need image references in multi-image prompts:

```xml
<region image_id="img_0003" region_id="region_0003">
<object>closure</object>
<class_id>2</class_id>
<color>blue</color>
<evidence>The closure supports the interpretation.</evidence>
<bbox>[12, 44, 88, 120]</bbox>
<SEG>
</region>
<answer>The section contains a closure feature.</answer>
```

Do not require attributes in the first version unless the prompt includes multiple images at once. If each training sample is one tile image, plain `<region>` blocks are enough.

## Handling Multiple Objects

Do not make one image equal one bbox.

Use:

```python
objects: list[object]
```

At training time, each tile may contain:

```text
0 objects
1 object
many objects
```

For the language target, list all visible objects in the tile.

For the mask target, keep per-object masks:

```text
mask_targets: [num_objects, 224, 224]
```

In the collator, pad to the max object count in the batch:

```text
bbox_targets: [B, max_objects, 4]
mask_targets: [B, max_objects, 224, 224]
object_mask:  [B, max_objects]
```

`object_mask` tells the loss which padded slots are real.

## Dataset vs Collator vs Loss

Recommended split:

```text
Dataset/preprocessing:
  - parse evidence regions
  - align each image/mask/region
  - convert RGBA to RGB
  - pad image and masks
  - tile image and masks
  - convert global bboxes to tile-local bboxes
  - create target_text
  - store meta

Collator:
  - stack pixel_values
  - tokenize prompt and target_text
  - pad object lists
  - create object_mask

Loss:
  - text loss from decoder labels
  - bbox loss on tile-local boxes
  - mask loss on tile-local masks
```

The loss should not parse XML, tile images, crop masks, or convert coordinates.

## Suggested Training Stages

Stage 1: region/text formatting

```text
Input: tile image + prompt
Output: strict text with labels and bboxes
Train: text decoder/projector
Freeze: NCS encoder
```

Stage 2: mask decoder

```text
Input: NCS patch tokens from tile
Output: binary masks
Train: mask decoder
Freeze: NCS encoder
```

Stage 3: joint tile training

```text
Loss = text_loss + bbox_loss + mask_loss
```

Stage 4: real image inference

```text
large image -> tiles -> local predictions -> shift to global coords -> merge -> stitch masks
```

## Changes Needed in the Current Dataset

Add or derive these fields:

```text
image_id
region_id / object_id
base_image_group_id
tile_id
x0
y0
orig_w
orig_h
local_bbox
binary_mask_tile
target_text
```

Strongly recommended:

```text
Flatten the current list fields into explicit region records before tiling.
```

Current row:

```text
images: [img0, img1, ...]
masks:  [mask0, mask1, ...]
evidence: <region0>...</region0><region1>...</region1>
```

Intermediate normalized records:

```python
{
    "source_row": 28,
    "region_index": 0,
    "image": img0,
    "mask": mask0,
    "label": "fault",
    "class_id": 1,
    "bbox": [0, 223, 99, 504],
    "evidence": [...]
}
```

Then tile those records.

## Things to Fix or Watch

1. Current images are `RGBA`; the NCS encoder likely expects 3 channels. Convert to RGB.

2. Current images are `100 x 507`; the NCS encoder example expects `224 x 224`. Use pad + vertical tiling.

3. Current `answer` does not include the `<region>` blocks. If training text generation with regions, target should be `evidence + answer`.

4. Some rows have repeated images with different masks, while others have different images in the same row. Add an explicit `image_id` or `base_image_group_id`.

5. Current questions sometimes ask a count, but evidence may include more supporting regions than the answer count. Do not assume `len(regions) == answer count`.

6. Keep original global bboxes for evaluation, but train on tile-local bboxes.

7. Keep `meta` through inference so predicted tile-local bboxes and masks can be moved back to full-image coordinates.

## First Implementation Target

Build a preprocessing script that creates a processed dataset with this shape:

```python
{
    "pixel_values": [3, 224, 224],
    "prompt": "Find and describe seismic regions. Return objects with labels and bboxes.",
    "target_text": "{\"objects\":[...],\"answer\":\"...\"}",
    "bbox_targets": [N, 4],
    "mask_targets": [N, 224, 224],
    "class_ids": [N],
    "object_mask": [N],
    "meta": {
        "source_row": int,
        "tile_id": str,
        "x0": int,
        "y0": int,
        "orig_w": int,
        "orig_h": int
    }
}
```

This is the dataset shape that will line up cleanly with:

```text
NCS encoder -> projector/text decoder -> bbox/text output
NCS patch tokens -> mask decoder -> mask output
```
