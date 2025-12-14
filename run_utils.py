# Checkpoint and results management 

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def seed_dir_name(seed: Optional[int]) -> str:
    if seed is None:
        return "seed_unknown"
    return f"seed_{seed:04d}"


def parse_seed_dir(seed_dir: str) -> Optional[int]:
    if seed_dir.startswith("seed_"):
        try:
            return int(seed_dir.split("seed_", 1)[1])
        except ValueError:
            return None
    return None


def create_run_dir(
    model_type: str,
    base_dir: str = "checkpoints",
    timestamp: str = None,
    experiment_name: str = None,
    seed: Optional[int] = None,
) -> Path:
    if timestamp is None:
        timestamp = get_timestamp()
    
    experiment_name = experiment_name or "default_experiment"
    run_dir = Path(base_dir) / experiment_name / seed_dir_name(seed) / model_type / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    return run_dir


def save_config_json(config: any, save_dir: Path):
    config_dict = {}
    
    for k in dir(config):
        if not k.startswith('_'):
            v = getattr(config, k)
            if not callable(v):
                if isinstance(v, (int, float, str, bool, list, dict, type(None))):
                    config_dict[k] = v
                else:
                    config_dict[k] = str(v)
    
    config_path = save_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Saved config to {config_path}")


def get_checkpoint_path(run_dir: Path, epoch: int) -> Path:
    return run_dir / f"epoch_{epoch:03d}.pt"


def get_latest_checkpoint(run_dir: Path) -> Optional[Path]:
    if not run_dir.exists():
        return None
    
    checkpoints = list(run_dir.glob("epoch_*.pt"))
    if not checkpoints:
        return None
    
    # Sort by epoch number
    checkpoints.sort(key=lambda p: int(p.stem.split('_')[1]))
    return checkpoints[-1]
