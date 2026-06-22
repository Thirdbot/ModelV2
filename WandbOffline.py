from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class WandbOffline:
    project: str = "seismic-vlm"
    entity: str | None = None
    root_dir: str = "runs/wandb"
    enabled: bool = True

    def setup(self, run_name: str) -> list[str]:
        if not self.enabled:
            return ["none"]

        run_dir = Path(self.root_dir).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)

        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_PROJECT"] = self.project
        os.environ["WANDB_NAME"] = run_name
        os.environ["WANDB_DIR"] = str(run_dir)
        if self.entity:
            os.environ["WANDB_ENTITY"] = self.entity

        return ["wandb"]
