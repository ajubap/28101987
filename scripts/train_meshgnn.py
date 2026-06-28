import argparse
from pathlib import Path
import sys
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physicsai_gnn.config import load_config, ensure_dirs
from physicsai_gnn.dataset import GraphCaseDataset, make_loader
from physicsai_gnn.train_utils import (
    seed_everything, choose_device, build_model,
    train_one_epoch, evaluate_loss, save_checkpoint
)
from physicsai_gnn.normalization import fit_normalizers, save_normalizers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/generic_cfd_template.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    seed_everything(int(cfg.get("project", {}).get("seed", 42)))

    device = choose_device(cfg["training"].get("device", "auto"))
    print(f"Using device: {device}")

    train_ds = GraphCaseDataset(cfg["paths"]["processed_dir"], split="train")
    val_ds = GraphCaseDataset(cfg["paths"]["processed_dir"], split="val")

    normalizers = fit_normalizers(train_ds, cfg)
    normalizer_file = cfg["paths"].get("normalizer_file", str(Path(cfg["paths"]["output_dir"]) / "normalizers.json"))
    save_normalizers(normalizer_file, normalizers)
    print(f"Saved normalizers: {normalizer_file}")

    batch_size = int(cfg["training"].get("batch_size", 2))
    train_loader = make_loader(train_ds, batch_size, shuffle=True)
    val_loader = make_loader(val_ds, batch_size, shuffle=False)

    model = build_model(cfg, train_ds).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"].get("learning_rate", 5e-4)),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-6)),
    )

    output_dir = Path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    logs = []

    for epoch in range(1, int(cfg["training"].get("epochs", 100)) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, cfg, device, normalizers)
        val_loss = evaluate_loss(model, val_loader, cfg, device, normalizers)

        logs.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:04d} | train={train_loss:.6f} | val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(output_dir / "best_meshgnn.pt", model, cfg, epoch, best_val)

    pd.DataFrame(logs).to_csv(output_dir / "training_log.csv", index=False)
    print("Training complete.")

if __name__ == "__main__":
    main()
