import random
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from src.metrics import dice_score


#EDA

def plot_samples(dataset, n=4):
    indices = random.sample(range(len(dataset)), n)
    fig, axes = plt.subplots(n, 3, figsize=(10, 3 * n))
    for row, idx in enumerate(indices):
        img, mask = dataset.cache[idx]
        arr_img = np.array(img)
        arr_rgb = np.stack([arr_img] * 3, axis=-1)
        arr_mask = np.array(mask)
        overlay = arr_rgb.copy()
        overlay[arr_mask > 127] = [255, 0, 0]
        axes[row, 0].imshow(arr_img, cmap="gray")
        axes[row, 0].set_title("MRI original")
        axes[row, 1].imshow(arr_mask, cmap="gray")
        axes[row, 1].set_title("Máscara")
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title("Overlay")
        for ax in axes[row]:
            ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_cut_distribution(dataset):
    counts = Counter(dataset.cut_types)
    cortes = {"Axial": counts["Axial"], "Coronal": counts["Coronal"], "Sagital": counts["Sagital"]}
    total = sum(cortes.values())
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(
        list(cortes.keys()), list(cortes.values()),
        color=["#4C72B0", "#DD8452", "#55A868"], edgecolor="white", height=0.5,
    )
    for bar, n in zip(bars, cortes.values()):
        ax.text(bar.get_width() + 15, bar.get_y() + bar.get_height() / 2,
                f"{n/total*100:.1f}%", va="center", fontsize=11)
    ax.set_xlabel("Cantidad de imágenes")
    ax.set_title("Distribución de imágenes por tipo de corte", fontsize=13)
    ax.set_xlim(0, max(cortes.values()) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_tumor_ratio(dataset):
    ratios = np.array([(np.array(dataset.cache[i][1]) > 127).sum() / np.array(dataset.cache[i][1]).size
                       for i in range(len(dataset))])
    print(f"Sin tumor: {(ratios == 0).sum()} ({(ratios == 0).mean()*100:.1f}%)")
    print(f"Con tumor: {(ratios > 0).sum()} ({(ratios > 0).mean()*100:.1f}%)")
    print(f"Área media: {ratios[ratios > 0].mean()*100:.2f}%")
    plt.figure(figsize=(8, 4))
    plt.hist(ratios[ratios > 0], bins=50, color="tomato", edgecolor="black")
    plt.xlabel("Proporción de píxeles tumorales")
    plt.ylabel("Frecuencia")
    plt.title("Distribución del área tumoral")
    plt.tight_layout()
    plt.show()


def plot_tumor_heatmaps(dataset, img_size=256):
    heatmaps = {k: np.zeros((img_size, img_size)) for k in ["Axial", "Coronal", "Sagital"]}
    for idx in range(len(dataset)):
        _, mask = dataset.cache[idx]
        heatmaps[dataset.cut_types[idx]] += (np.array(mask) > 127).astype(float)
    for k in heatmaps:
        heatmaps[k] = gaussian_filter(heatmaps[k], sigma=3)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, cut in zip(axes, ["Axial", "Coronal", "Sagital"]):
        im = ax.imshow(heatmaps[cut], cmap="hot")
        ax.set_title(cut, fontsize=13)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle("Distribución acumulada de tumores por tipo de corte", fontsize=15)
    plt.tight_layout()
    plt.show()


def plot_augmentation_examples(dataset_no_aug, dataset_aug, n=4):
    indices = random.sample(range(len(dataset_no_aug)), n)
    fig, axes = plt.subplots(n, 4, figsize=(12, n * 3))
    for row, idx in enumerate(indices):
        img_o, mask_o = dataset_no_aug[idx]
        img_a, mask_a = dataset_aug[idx]
        axes[row, 0].imshow(img_o.squeeze(), cmap="gray"); axes[row, 0].set_title("Imagen original")
        axes[row, 1].imshow(mask_o.squeeze(), cmap="gray"); axes[row, 1].set_title("Máscara original")
        axes[row, 2].imshow(img_a.squeeze(), cmap="gray"); axes[row, 2].set_title("Imagen aumentada")
        axes[row, 3].imshow(mask_a.squeeze(), cmap="gray"); axes[row, 3].set_title("Máscara aumentada")
        for ax in axes[row]:
            ax.axis("off")
    plt.suptitle("Sin augmentation vs con augmentation", fontsize=13)
    plt.tight_layout()
    plt.show()


#Curvas de entrenamiento

def plot_training_curves(train_losses, val_losses, title="Curvas de entrenamiento"):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss", color="steelblue")
    plt.plot(epochs, val_losses, label="Val Loss", color="tomato")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_experiment_curves(results, top_n=None):
    sorted_r = sorted(results, key=lambda x: x["dice"], reverse=True)
    if top_n:
        sorted_r = sorted_r[:top_n]
    n_cols = 2
    n_rows = (len(sorted_r) + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4 * n_rows))
    axes = axes.flatten()
    for i, r in enumerate(sorted_r):
        ax = axes[i]
        ep = range(1, len(r["train_losses"]) + 1)
        ax.plot(ep, r["train_losses"], label="Train Loss")
        ax.plot(ep, r["val_losses"], label="Val Loss")
        ax.set_title(f"{r['name']} | Dice: {r['dice']:.4f}")
        ax.set_xlabel("Época"); ax.set_ylabel("Loss")
        ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_experiment_metrics(results):
    sorted_r = sorted(results, key=lambda x: x["dice"], reverse=True)
    names = [r["name"] for r in sorted_r]
    dices = [r["dice"] for r in sorted_r]
    ious = [r["iou"] for r in sorted_r]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].barh(names, dices); axes[0].set_xlabel("Dice Score")
    axes[0].set_title("Dice por experimento"); axes[0].set_xlim(0, 1)
    axes[1].barh(names, ious); axes[1].set_xlabel("IoU Score")
    axes[1].set_title("IoU por experimento"); axes[1].set_xlim(0, 1)
    plt.tight_layout()
    plt.show()


#Predicciones

def show_predictions(model, dataset, device, n=4):
    import torch
    model.eval()
    indices = random.sample(range(len(dataset)), n)
    fig, axes = plt.subplots(n, 4, figsize=(14, n * 4))
    with torch.no_grad():
        for row, idx in enumerate(indices):
            img, mask = dataset[idx]
            logit = model(img.unsqueeze(0).to(device))
            pred = (torch.sigmoid(logit) > 0.5).float().squeeze().cpu()
            arr_img = img.squeeze().numpy()
            arr_mask = mask.squeeze().numpy()
            arr_pred = pred.numpy()
            d = dice_score(logit, mask.unsqueeze(0).to(device))
            axes[row, 0].imshow(arr_img, cmap="gray"); axes[row, 0].set_title(f"MRI — Dice: {d:.3f}")
            axes[row, 1].imshow(arr_mask, cmap="gray"); axes[row, 1].set_title("Máscara real")
            axes[row, 2].imshow(arr_pred, cmap="gray"); axes[row, 2].set_title("Predicción")
            axes[row, 3].imshow(arr_img, cmap="gray")
            axes[row, 3].imshow(arr_mask, cmap="Reds", alpha=0.4)
            axes[row, 3].imshow(arr_pred, cmap="Greens", alpha=0.4)
            axes[row, 3].set_title("Overlap (rojo=real, verde=pred)")
            for ax in axes[row]:
                ax.axis("off")
    plt.suptitle("Predicciones del modelo sobre test set", fontsize=13)
    plt.tight_layout(pad=3.0)
    plt.show()


def show_worst_predictions(model, dataset, device, n=4):
    import torch
    model.eval()
    scores = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            img, mask = dataset[idx]
            logit = model(img.unsqueeze(0).to(device))
            scores.append((dice_score(logit, mask.unsqueeze(0).to(device)), idx))
    scores.sort(key=lambda x: x[0])
    peores = scores[:n]
    print("Peores predicciones:")
    for d, idx in peores:
        print(f"  idx={idx} | Dice={d:.4f}")
    fig, axes = plt.subplots(n, 4, figsize=(12, n * 4))
    with torch.no_grad():
        for row, (d_val, idx) in enumerate(peores):
            img, mask = dataset[idx]
            logit = model(img.unsqueeze(0).to(device))
            pred = (torch.sigmoid(logit) > 0.5).float().squeeze().cpu()
            arr_img = img.squeeze().numpy()
            arr_mask = mask.squeeze().numpy()
            arr_pred = pred.numpy()
            axes[row, 0].imshow(arr_img, cmap="gray"); axes[row, 0].set_title(f"MRI — Dice: {d_val:.3f}")
            axes[row, 1].imshow(arr_mask, cmap="gray"); axes[row, 1].set_title("Máscara real")
            axes[row, 2].imshow(arr_pred, cmap="gray"); axes[row, 2].set_title("Predicción")
            axes[row, 3].imshow(arr_img, cmap="gray")
            axes[row, 3].imshow(arr_mask, cmap="Reds", alpha=0.4)
            axes[row, 3].imshow(arr_pred, cmap="Greens", alpha=0.4)
            axes[row, 3].set_title("Overlap (rojo=real, verde=pred)")
            for ax in axes[row]:
                ax.axis("off")
    plt.suptitle("Peores predicciones del modelo (test set)", fontsize=13)
    plt.tight_layout()
    plt.show()
