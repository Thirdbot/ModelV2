import wandb
from pathlib import Path
import os
os.environ["WANDB_MODE"] = "offline"
import wandb
class Wandb:
    def __init__(self):
        self.wandb_path = Path(__file__).parent / 'wandb'
