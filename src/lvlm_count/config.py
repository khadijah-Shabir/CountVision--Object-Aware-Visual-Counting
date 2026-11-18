import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]          # countvision/
PROJECT = ROOT.parent                               # workspace parent
DATASETS = PROJECT / "datasets"


def load_config(path=None) -> dict:
    path = Path(path or ROOT / "configs" / "lvlm_count.yaml")
    return yaml.safe_load(path.read_text())


def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
