from pathlib import Path
import yaml

def load_config(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dirs(cfg):
    for key in ["raw_dir", "processed_dir", "output_dir"]:
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
