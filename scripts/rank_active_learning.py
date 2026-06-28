import argparse
from pathlib import Path
import sys
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physicsai_gnn.config import load_config
from physicsai_gnn.mesh_to_graph import mesh_file_to_graph
from physicsai_gnn.dataset import GraphCaseDataset
from physicsai_gnn.train_utils import choose_device, build_model, load_checkpoint
from physicsai_gnn.normalization import load_normalizers
from physicsai_gnn.inference import mc_dropout_predict
from physicsai_gnn.case_schema import (
    case_id_column,
    mesh_file_column,
    point_feature_arrays,
    row_to_global_features,
    target_field_arrays,
)

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/generic_cfd_template.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--candidate_csv", required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = choose_device(cfg["training"].get("device", "auto"))

    dummy_ds = GraphCaseDataset(cfg["paths"]["processed_dir"], split="train")
    model = build_model(cfg, dummy_ds).to(device)

    ckpt = args.checkpoint or str(Path(cfg["paths"]["output_dir"]) / "best_meshgnn.pt")
    load_checkpoint(ckpt, model, map_location=device)

    normalizer_file = cfg["paths"].get("normalizer_file", str(Path(cfg["paths"]["output_dir"]) / "normalizers.json"))
    normalizers = load_normalizers(normalizer_file)

    raw_dir = Path(cfg["paths"]["raw_dir"])
    df = pd.read_csv(args.candidate_csv)
    rows = []

    for _, row in df.iterrows():
        case_id = str(row[case_id_column(cfg)])
        mesh_path = raw_dir / str(row[mesh_file_column(cfg)])
        global_features = row_to_global_features(cfg, row)

        data = mesh_file_to_graph(
            mesh_path=mesh_path,
            global_features=global_features,
            point_feature_arrays=point_feature_arrays(cfg),
            target_field_arrays=target_field_arrays(cfg),
            scalar_targets=None,
            case_id=case_id,
            require_target_fields=False,
        )

        field_mean, field_std, scalar_mean, scalar_std = mc_dropout_predict(
            model=model,
            data=data,
            device=device,
            normalizers=normalizers,
            samples=args.samples,
        )

        # Simple uncertainty score:
        # average normalized field std + average scalar std, if scalar exists.
        field_score = float(field_std.mean().item())
        scalar_score = float(scalar_std.mean().item()) if scalar_std is not None else 0.0
        total_score = field_score + scalar_score

        out = dict(row)
        out["field_uncertainty_score"] = field_score
        out["scalar_uncertainty_score"] = scalar_score
        out["active_learning_score"] = total_score
        rows.append(out)

    ranked = pd.DataFrame(rows).sort_values("active_learning_score", ascending=False)
    out_path = Path(cfg["paths"]["output_dir"]) / "active_learning_rank.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ranked.head(args.top_k).to_csv(out_path, index=False)

    print(ranked.head(args.top_k).to_string(index=False))
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
