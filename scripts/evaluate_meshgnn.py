import argparse
from pathlib import Path
import sys
import json
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physicsai_gnn.config import load_config
from physicsai_gnn.dataset import GraphCaseDataset, make_loader
from physicsai_gnn.train_utils import choose_device, build_model, load_checkpoint
from physicsai_gnn.metrics import field_metrics, scalar_metrics
from physicsai_gnn.normalization import load_normalizers, apply_normalizers, inverse_field, inverse_scalar


def field_names_for_batch(data_raw, cfg, width):
    names = getattr(data_raw, "target_field_names", None)
    if names and isinstance(names, (list, tuple)):
        first = names[0]
        if isinstance(first, (list, tuple)):
            names = list(first)
        else:
            names = list(names)
    else:
        names = list(cfg["data"]["target_field_arrays"])
    if len(names) != width:
        names = names + [f"field_{i}" for i in range(len(names), width)]
    return names[:width]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/generic_cfd_template.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = choose_device(cfg["training"].get("device", "auto"))

    ds = GraphCaseDataset(cfg["paths"]["processed_dir"], split=args.split)
    loader = make_loader(ds, int(cfg["training"].get("batch_size", 2)), shuffle=False)

    model = build_model(cfg, ds).to(device)
    ckpt = args.checkpoint or str(Path(cfg["paths"]["output_dir"]) / "best_meshgnn.pt")
    load_checkpoint(ckpt, model, map_location=device)
    model.eval()

    normalizer_file = cfg["paths"].get("normalizer_file", str(Path(cfg["paths"]["output_dir"]) / "normalizers.json"))
    normalizers = load_normalizers(normalizer_file)

    results = []

    for data_raw in loader:
        data = apply_normalizers(data_raw, normalizers).to(device)
        out = model(data)

        pred_field = inverse_field(out["field"], normalizers)
        true_field = inverse_field(data.y, normalizers)
        field_names = field_names_for_batch(data_raw, cfg, pred_field.shape[-1])

        item = {
            "field_physical_units": field_metrics(pred_field, true_field, field_names)
        }

        if "scalar" in out and hasattr(data, "y_scalar"):
            pred_scalar = inverse_scalar(out["scalar"], normalizers)
            true_scalar = inverse_scalar(data.y_scalar, normalizers)
            item["scalar_physical_units"] = scalar_metrics(pred_scalar, true_scalar, cfg["data"]["target_scalar_columns"])

        results.append(item)

    out_path = Path(cfg["paths"]["output_dir"]) / f"eval_{args.split}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
