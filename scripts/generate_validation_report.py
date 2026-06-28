import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from physicsai_gnn.config import load_config
from physicsai_gnn.reporting import generate_validation_report

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/generic_cfd_template.yaml'); p.add_argument('--eval_json',default=None); p.add_argument('--training_log',default=None); p.add_argument('--output',default=None)
    a=p.parse_args(); cfg=load_config(a.config)
    eval_json=a.eval_json or str(Path(cfg['paths']['output_dir'])/'eval_test.json')
    training_log=a.training_log or str(Path(cfg['paths']['output_dir'])/'training_log.csv')
    output=a.output or str(Path(cfg['paths']['output_dir'])/'validation_report.md')
    report,csv=generate_validation_report(cfg, eval_json, training_log, output)
    print(f'Report: {report}'); print(f'CSV: {csv}')
if __name__=='__main__': main()
