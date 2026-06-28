from pathlib import Path
import random
import pandas as pd
import torch

try:
    from torch_geometric.loader import DataLoader
except Exception:
    DataLoader = None

from .mesh_to_graph import mesh_file_to_graph
from .case_schema import (
    case_id_column,
    mesh_file_column,
    row_to_global_features,
    row_to_scalar_targets,
    target_field_arrays,
    point_feature_arrays,
    validate_metadata,
    summarize_issues,
)


def preprocess_cases(cfg):
    raw_dir = Path(cfg["paths"]["raw_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    issues = validate_metadata(cfg, check_mesh_arrays=True)
    summary = summarize_issues(issues)
    if summary["errors"]:
        details = "\n".join(f"- [{issue['where']}] {issue['message']}" for issue in issues if issue["level"] == "error")
        raise ValueError(f"Dataset validation failed with {summary['errors']} error(s):\n{details}")

    meta = pd.read_csv(cfg["paths"]["metadata_csv"])
    case_col = case_id_column(cfg)
    mesh_col = mesh_file_column(cfg)

    manifest = []

    for _, row in meta.iterrows():
        case_id = str(row[case_col])
        mesh_file = raw_dir / str(row[mesh_col])

        global_features = row_to_global_features(cfg, row)
        scalar_targets = row_to_scalar_targets(cfg, row, require=True)

        graph = mesh_file_to_graph(
            mesh_path=mesh_file,
            global_features=global_features,
            point_feature_arrays=point_feature_arrays(cfg),
            target_field_arrays=target_field_arrays(cfg),
            scalar_targets=scalar_targets,
            case_id=case_id,
            require_target_fields=True,
        )

        out_path = processed_dir / f"{case_id}.pt"
        torch.save(graph, out_path)
        manifest.append({
            "case_id": case_id,
            "graph_file": out_path.name,
            "mesh_file": str(row[mesh_col]),
            "n_nodes": int(graph.x.shape[0]),
            "n_edges": int(graph.edge_index.shape[1]),
            "n_node_features": int(graph.x.shape[1]),
            "n_field_outputs": int(graph.y.shape[1]),
            "n_scalar_outputs": int(graph.y_scalar.shape[1]) if hasattr(graph, "y_scalar") else 0,
        })

    man = pd.DataFrame(manifest)
    man.to_csv(processed_dir / "manifest.csv", index=False)
    return man


def make_splits(cfg):
    processed_dir = Path(cfg["paths"]["processed_dir"])
    manifest_path = processed_dir / "manifest.csv"
    man = pd.read_csv(manifest_path)

    seed = int(cfg.get("project", {}).get("seed", 42))
    idx = list(man.index)
    random.Random(seed).shuffle(idx)

    n = len(idx)
    n_train = int(n * float(cfg["data"].get("train_fraction", 0.75)))
    n_val = int(n * float(cfg["data"].get("val_fraction", 0.15)))
    n_test = n - n_train - n_val
    if n < 3:
        raise ValueError("At least 3 cases are required to create train/val/test splits.")
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(
            f"Split fractions produce an empty split for {n} cases: "
            f"train={n_train}, val={n_val}, test={n_test}."
        )

    split = ["test"] * n
    for i in idx[:n_train]:
        split[i] = "train"
    for i in idx[n_train:n_train+n_val]:
        split[i] = "val"

    man["split"] = split
    man.to_csv(manifest_path, index=False)
    return man


class GraphCaseDataset(torch.utils.data.Dataset):
    def __init__(self, processed_dir, split=None):
        self.processed_dir = Path(processed_dir)
        man = pd.read_csv(self.processed_dir / "manifest.csv")
        if split is not None and "split" in man.columns:
            man = man[man["split"] == split]
        self.items = man.to_dict("records")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return torch.load(self.processed_dir / self.items[idx]["graph_file"], weights_only=False)


def make_loader(dataset, batch_size, shuffle):
    if DataLoader is None:
        raise ImportError("torch_geometric.loader.DataLoader is required.")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
