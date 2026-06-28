import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physicsai_gnn.config import load_config, ensure_dirs
from physicsai_gnn.dataset import preprocess_cases, make_splits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/generic_cfd_template.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    preprocess_cases(cfg)
    make_splits(cfg)
    print("Preprocessing complete.")

if __name__ == "__main__":
    main()
