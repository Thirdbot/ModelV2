"""Central config for the main model.

Only the knobs actually read by the model live here:
  - dataset path
  - decoder (Qwen) + LoRA settings
  - image tiling (tied to the frozen NCS encoder)
"""

# ---------- Data ----------
# Point DATASET_CSV at the generator's output; column names are in
# dataset.py's CSV_COLUMNS block.
DATASET_CSV = "/home/third/Desktop/simulationv2/Dataset/multimodal_multi_image_dataset.csv"

# ---------- Decoder (Qwen 4-bit QLoRA) ----------
# Load the decoder in 4-bit to fit a 1.5B model on a ~5.7GB GPU; needs CUDA +
# bitsandbytes, and auto-disables (falls back to fp32) when CUDA is absent.
DECODER_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LOAD_DECODER_IN_4BIT = True
USE_LORA = True
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

# ---------- Image tiling ----------
# Tied to the frozen NCS encoder (224px input, 16px patches -> 14x14 grid);
# don't change without changing the encoder.
TILE_SIZE = 224
TILE_STRIDE = 112   # 50% overlap
PATCH = 16
TILE_GRID = 14
