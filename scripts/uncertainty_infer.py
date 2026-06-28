import argparse
from pathlib import Path
import sys
import json
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physicsai_gnn.config import load_config
from physicsai_gnn.mesh_to_graph import mesh_file_to_graph
from physicsai_gnn.dataset import GraphCaseDataset
from physicsai_gnn.train_utils import choose_device, build_model, load_checkpoint
from physicsai_gnn.normalization import load_normalizers
from physicsai_gnn.inference import mc_dropout_predict
from physicsai_gnn.export_vtu import export_prediction_vtu
from physicsai_gnn.case_schema import (
    build_global_features,
    load_json_values,
    parse_key_value_pairs,
    point_feature_arrays,
    target_field_arrays,
    target_scalar_columns,
)


def names_for_prediction(data, fallback, width):
    names = list(getattr(data, "target_field_names", fallback))
    if len(names) != width:
        names = names + [f"field_{i}" for i in range(len(names), width)]
    return names[:width]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/generic_cfd_template.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--global", dest="global_pairs", action="append", default=[], help="Global input as name=value. Repeat for each config global feature.")
    parser.add_argument("--globals_json", default=None, help="JSON object or path containing global input values.")
    parser.add_argument("--flow_rate_lpm", type=float, default=None, help="Backward-compatible coldplate shortcut.")
    parser.add_argument("--inlet_temp_c", type=float, default=None, help="Backward-compatible coldplate shortcut.")
    parser.add_argument("--power_w", type=float, default=None, help="Backward-compatible coldplate shortcut.")
    parser.add_argument("--plate_thickness_mm", type=float, default=None, help="Backward-compatible coldplate shortcut.")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--export_vtu", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = choose_device(cfg["training"].get("device", "auto"))

    dummy_ds = GraphCaseDataset(cfg["paths"]["processed_dir"], split="train")
    model = build_model(cfg, dummy_ds).to(device)

    ckpt = args.checkpoint or str(Path(cfg["paths"]["output_dir"]) / "best_meshgnn.pt")
    load_checkpoint(ckpt, model, map_location=device)

    normalizer_file = cfg["paths"].get("normalizer_file", str(Path(cfg["paths"]["output_dir"]) / "normalizers.json"))
    normalizers = load_normalizers(normalizer_file)

    values = load_json_values(args.globals_json)
    values.update(parse_key_value_pairs(args.global_pairs))
    for name in ["flow_rate_lpm", "inlet_temp_c", "power_w", "plate_thickness_mm"]:
        value = getattr(args, name)
        if value is not None:
            values[name] = value
    global_features = build_global_features(cfg, values)

    data = mesh_file_to_graph(
        mesh_path=args.mesh,
        global_features=global_features,
        point_feature_arrays=point_feature_arrays(cfg),
        target_field_arrays=target_field_arrays(cfg),
        scalar_targets=None,
        case_id=Path(args.mesh).stem,
        require_target_fields=False,
    )

    field_mean, field_std, scalar_mean, scalar_std = mc_dropout_predict(
        model=model,
        data=data,
        device=device,
        normalizers=normalizers,
        samples=args.samples,
    )
    field_names = names_for_prediction(data, target_field_arrays(cfg), field_mean.shape[-1])

    result = {
        "global_features": global_features,
        "field_names": field_names,
        "field_prediction_shape": list(field_mean.shape),
        "mean_field_uncertainty_by_output": {
            name: float(field_std[:, i].mean().item())
            for i, name in enumerate(field_names)
        },
    }

    if scalar_mean is not None:
        result["scalar_names"] = target_scalar_columns(cfg)
        result["scalar_mean"] = scalar_mean.reshape(-1).tolist()
        result["scalar_std"] = scalar_std.reshape(-1).tolist()

    if args.export_vtu:
        export_prediction_vtu(
            original_mesh_path=args.mesh,
            output_vtu_path=args.export_vtu,
            field_mean=field_mean,
            field_names=field_names,
            field_std=field_std,
            scalar_mean=scalar_mean,
            scalar_std=scalar_std,
            scalar_names=target_scalar_columns(cfg),
        )
        result["export_vtu"] = args.export_vtu

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
