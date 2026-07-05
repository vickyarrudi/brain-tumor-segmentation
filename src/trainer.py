import time
from copy import deepcopy
import torch
from torch.amp import GradScaler, autocast
from src.losses import combined_loss


def _amp_enabled(device):
    return getattr(device, "type", device) == "cuda"


def train_epoch(model, loader, optimizer, device, scaler):
    model.train()
    amp = _amp_enabled(device)
    total_loss, total_samples = 0.0, 0

    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=amp):
            loss = combined_loss(model(imgs), masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    return total_loss / total_samples


@torch.no_grad()
def val_epoch(model, loader, device):
    model.eval()
    amp = _amp_enabled(device)
    total_loss, total_samples = 0.0, 0

    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
        masks = masks.to(device, non_blocking=True)

        with autocast("cuda", enabled=amp):
            loss = combined_loss(model(imgs), masks)

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    return total_loss / total_samples


def train(model, train_loader, val_loader, optimizer, device, num_epochs, patience):
    scaler = GradScaler("cuda", enabled=_amp_enabled(device))
    best_val_loss = float("inf")
    epochs_sin_mejora = 0
    best_weights = None
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        t0 = time.perf_counter()
        tl = train_epoch(model, train_loader, optimizer, device, scaler)
        vl = val_epoch(model, val_loader, device)
        elapsed = time.perf_counter() - t0

        train_losses.append(tl)
        val_losses.append(vl)
        print(
            f"Epoch {epoch+1:03d}/{num_epochs} | "
            f"Train: {tl:.4f} | Val: {vl:.4f} | {elapsed:.1f}s"
        )

        if vl < best_val_loss:
            best_val_loss = vl
            epochs_sin_mejora = 0
            best_weights = deepcopy(model.state_dict())
            print(f"  ✓ Mejor modelo (val_loss={best_val_loss:.4f})")
        else:
            epochs_sin_mejora += 1
            print(f"  ✗ Sin mejora ({epochs_sin_mejora}/{patience})")
            if epochs_sin_mejora >= patience:
                print(f"\nEarly stopping en epoch {epoch+1}")
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    print(f"\nEntrenamiento finalizado. Mejor val_loss: {best_val_loss:.4f}")
    return train_losses, val_losses
