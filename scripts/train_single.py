"""
Entrena una sola configuración y guarda el modelo

"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import Config
from src.dataset import create_dataloaders, make_splits
from src.models import get_model
from src.trainer import train


def setup_device(seed):
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
    parser.add_argument("--model", default="unet", choices=["unet", "residual_unet", "attention_unet"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    cfg = Config(
        model_name=args.model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        batch_size=args.batch_size,
    )
    if args.epochs:
        cfg.epochs = args.epochs

    device = setup_device(cfg.seed)
    train_idx, val_idx, test_idx = make_splits()
    train_loader, val_loader, test_loader, test_set = create_dataloaders(cfg, train_idx, val_idx, test_idx)

    model = get_model(cfg.model_name, base_features=cfg.base_features, depth=cfg.depth, dropout=cfg.dropout)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_losses, val_losses = train(
        model, train_loader, val_loader, optimizer, device, cfg.epochs, cfg.patience
    )

    out_path = f"best_{cfg.model_name}.pt"
    torch.save(model.state_dict(), out_path)
    print(f"\nModelo guardado en {out_path}")


if __name__ == "__main__":
    main()
