import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from physicsai_gnn.config import load_config
from physicsai_gnn.model_registry import register_model

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/generic_cfd_template.yaml'); p.add_argument('--checkpoint',default=None); p.add_argument('--normalizers',default=None); p.add_argument('--metrics',default=None); p.add_argument('--notes',default='')
    a=p.parse_args(); cfg=load_config(a.config)
    checkpoint=a.checkpoint or str(Path(cfg['paths']['output_dir'])/'best_meshgnn.pt')
    normalizers=a.normalizers or cfg['paths'].get('normalizer_file', str(Path(cfg['paths']['output_dir'])/'normalizers.json'))
    metrics=a.metrics or str(Path(cfg['paths']['output_dir'])/'eval_test.json')
    version_dir=register_model(cfg, checkpoint, normalizers, metrics, a.notes)
    print(f'Registered model: {version_dir}')
if __name__=='__main__': main()
