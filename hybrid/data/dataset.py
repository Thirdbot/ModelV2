"""Dataset access for the main model.

Two things the model needs:
  - load_local_csv : read the dataset CSV into plain dicts (one per sample).
  - simple_tiling  : cut an image into overlapping tiles for the NCS encoder.
"""
import ast
import json

import pandas as pd
from PIL import ImageOps

from hybrid.data.config import DATASET_CSV

CSV_COLUMNS = {
    "sample_id":   "sample_id",
    "images":      "images",       # JSON list of image paths; only the first is used
    "masks":       "masks",        # JSON list of mask paths; only the first is used
    "instruction": "instruction",
    "question":    "question",
    "answer":      "answer",
    "evidence":    "evidence",
    "reason":      "reason",
    "regions":     "regions",      # JSON list of dicts: class_id, bbox, center, ...
}


def _parse_json_list(value):
    """Read a list cell from the CSV, tolerating JSON ('["a.png"]') and
    Python-literal ("['a.png']") spellings, plus empty/NaN cells -> []."""
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if not isinstance(value, str):
        return list(value)
    text = value.strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return ast.literal_eval(text)


def load_local_csv(csv_path=DATASET_CSV):
    """Read the dataset CSV into plain dicts, one per sample."""
    col = CSV_COLUMNS
    df = pd.read_csv(csv_path)
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "sample_id": row[col["sample_id"]],
            "image_paths": _parse_json_list(row[col["images"]]),
            "mask_paths": _parse_json_list(row[col["masks"]]),
            "instruction": row.get(col["instruction"], "") or "",
            "question": row.get(col["question"], "") or "",
            "answer": row.get(col["answer"], "") or "",
            "evidence": row.get(col["evidence"], "") or "",
            "reason": row.get(col["reason"], "") or "",
            "regions": _parse_json_list(row[col["regions"]]),
        })
    return rows


def simple_tiling(image, height, width, tile_size=224, stride=112):
    """Cut the image into overlapping tile_size x tile_size crops (edge-snapped;
    small images padded black bottom/right). Each tile carries its bbox."""
    if width < tile_size:
        xs = [0]
    else:
        xs = list(range(0, max(width - tile_size + 1, 1), stride))
        if not xs or xs[-1] != width - tile_size:
            xs.append(max(width - tile_size, 0))
    if height < tile_size:
        ys = [0]
    else:
        ys = list(range(0, max(height - tile_size + 1, 1), stride))
        if not ys or ys[-1] != height - tile_size:
            ys.append(max(height - tile_size, 0))

    tiles = []
    for y1 in ys:
        for x1 in xs:
            x2 = min(x1 + tile_size, width)
            y2 = min(y1 + tile_size, height)
            crop = image.crop((x1, y1, x2, y2))
            pad_w = tile_size - crop.size[0]
            pad_h = tile_size - crop.size[1]
            tiles.append({
                "image": ImageOps.expand(crop, border=(0, 0, pad_w, pad_h)),
                "bbox_abs": [x1, y1, x2, y2],
                "bbox_norm": [x1 / width, y1 / height, x2 / width, y2 / height],
            })
    return tiles
