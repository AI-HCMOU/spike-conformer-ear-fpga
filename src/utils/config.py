"""
YAML config loader utility.
"""

import yaml
from pathlib import Path
from typing import Any


def load_config(path: str) -> dict:
    """Load YAML configuration file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def save_config(cfg: dict, path: str):
    """Save config dict to YAML."""
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
