import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.amp import autocast

from src.losses import combined_loss
from src.metrics import dice_score, iou_score


def _amp_enabled(device):
    return getattr(device, "type", device) == "cuda"


def evaluate(model, loader, device):
    """Devuelve (dice, iou) promediados sobre el loader."""
    model.eval()
    amp = _amp_enabled(device)
    total_loss = total_dice = total_iou = total_samples = 0.0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device, memory_format=torch.channels_last)
            masks = masks.to(device)

            with autocast("cuda", enabled=amp):
                logits = model(imgs)
                loss = combined_loss(logits, masks)

            bs = imgs.size(0)
            total_loss += loss.item() * bs
            total_dice += dice_score(logits, masks) * bs
            total_iou += iou_score(logits, masks) * bs
            total_samples += bs

    print(f"Loss: {total_loss/total_samples:.4f}")
    print(f"Dice: {total_dice/total_samples:.4f}")
    print(f"IoU:  {total_iou/total_samples:.4f}")
    return total_dice / total_samples, total_iou / total_samples


def evaluate_by_cut(model, dataset, device, plot=True):
    """Dice por tipo de corte (Axial / Coronal / Sagital)."""
    model.eval()
    results = {"Axial": [], "Coronal": [], "Sagital": []}

    with torch.no_grad():
        for idx in range(len(dataset)):
            img, mask = dataset[idx]
            real_idx = dataset.indices[idx]
            cut = dataset.dataset.cut_types[real_idx]
            logit = model(img.unsqueeze(0).to(device))
            d = dice_score(logit, mask.unsqueeze(0).to(device))
            results[cut].append(d)

    for cut, scores in results.items():
        print(f"{cut:10s} | n={len(scores):4d} | Dice: {sum(scores)/len(scores):.4f}")

    if plot:
        colors = ["#4C72B0", "#DD8452", "#55A868"]
        fig, ax = plt.subplots(figsize=(8, 5))
        bp = ax.boxplot(
            results.values(),
            labels=results.keys(),
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            flierprops=dict(marker="o", markersize=4, alpha=0.5),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        for flier, color in zip(bp["fliers"], colors):
            flier.set_markerfacecolor(color)
            flier.set_markeredgecolor(color)
        ax.set_ylabel("Dice Score", fontsize=12)
        ax.set_title("Dice por tipo de corte", fontsize=13)
        ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.show()

    return results


def evaluate_by_tumor_size(model, dataset, device):
    """Scatter: Dice vs tamaño del tumor en píxeles."""
    model.eval()
    records = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            img, mask = dataset[idx]
            logit = model(img.unsqueeze(0).to(device))
            d = dice_score(logit, mask.unsqueeze(0).to(device))
            records.append({"dice": d, "tumor_size": mask.sum().item()})

    df = pd.DataFrame(records)
    plt.figure(figsize=(8, 5))
    plt.scatter(df["tumor_size"], df["dice"], alpha=0.4)
    plt.xlabel("Tamaño del tumor (píxeles)")
    plt.ylabel("Dice Score")
    plt.title("Dice Score vs Tamaño del tumor")
    plt.tight_layout()
    plt.show()
    return df
