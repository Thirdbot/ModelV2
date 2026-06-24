import wandb
from pathlib import Path

class Wandb:
    def __init__(self):
        self.wandb_path = Path(__file__).parent / 'wandb'
