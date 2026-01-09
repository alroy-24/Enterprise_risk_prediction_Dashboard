import yaml
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parent.parent


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_config() -> Dict[str, Any]:
    return load_yaml(ROOT / "config" / "model_config.yaml")


def load_weights() -> Dict[str, Any]:
    return load_yaml(ROOT / "config" / "weights.yaml")


def load_scenarios() -> Dict[str, Any]:
    return load_yaml(ROOT / "config" / "scenarios.yaml")


