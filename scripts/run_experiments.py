"""
Grid search de hiperparámetros.

Uso:
    python scripts/run_experiments.py --model unet
    python scripts/run_experiments.py --model residual_unet
    python scripts/run_experiments.py --model attention_unet
"""
import argparse
import json
import os
import random
import sys
from itertools import product
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import Config
from src.dataset import create_dataloaders, make_splits
from src.evaluate import evaluate
from src.models import get_model
from src.trainer import train
from src.visualize import plot_experiment_metrics, plot_experiment_curves


#Grid de hiperparámetros 
LR_VALUES       = [1e-3, 1e-4]
WD_VALUES       = [1e-4, 1e-5]
DROPOUT_VALUES  = [0.0, 0.1]


def setup_device(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    return device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["unet", "residual_unet", "attention_unet"])
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    base_cfg = Config(model_name=args.model)
    if args.epochs:
        base_cfg.epochs = args.epochs

    device = setup_device(base_cfg.seed)
    train_idx, val_idx, test_idx = make_splits()
    train_loader, val_loader, _, _ = create_dataloaders(base_cfg, train_idx, val_idx, test_idx)

    results = []
    grid = list(product(LR_VALUES, WD_VALUES, DROPOUT_VALUES))
    print(f"\nIniciando grid search: {len(grid)} experimentos para '{args.model}'")

    for lr, wd, dropout in grid:
        cfg = Config(model_name=args.model, lr=lr, weight_decay=wd, dropout=dropout)

        print(f"\n{'='*55}")
        print(f"lr={lr} | weight_decay={wd} | dropout={dropout}")
        print(f"{'='*55}")

        model = get_model(cfg.model_name, base_features=cfg.base_features,
                          depth=cfg.depth, dropout=cfg.dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        train_losses, val_losses = train(
            model, train_loader, val_loader, optimizer, device,
            num_epochs=cfg.epochs, patience=cfg.patience,
        )

        dice, iou = evaluate(model, val_loader, device)

        result = {
            "model": args.model,
            "name": f"lr={lr}, wdecay={wd}, dp={dropout}",
            "lr": lr, "weight_decay": wd, "dropout": dropout,
            "dice": dice, "iou": iou,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }
        results.append(result)

        # Guardar resultado individual
        safe_name = result["name"].replace("=", "").replace(",", "").replace(" ", "_")
        exp_dir = f"results/{args.model}/{safe_name}"
        os.makedirs(exp_dir, exist_ok=True)
        with open(f"{exp_dir}/results.json", "w") as f:
            json.dump(result, f, indent=2)

        print(f"Guardado → Dice: {dice:.4f} | IoU: {iou:.4f}")

    # Resumen final
    print(f"\n{'='*55}\nRESUMEN — {args.model}\n{'='*55}")
    for r in sorted(results, key=lambda x: x["dice"], reverse=True):
        print(f"{r['name']:45s} | Dice: {r['dice']:.4f} | IoU: {r['iou']:.4f}")

    plot_experiment_metrics(results)
    plot_experiment_curves(results)


if __name__ == "__main__":
    main()
