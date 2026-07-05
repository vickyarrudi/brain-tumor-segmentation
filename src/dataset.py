import os
import random
import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset


class BrainTumorDataset(Dataset):
    def __init__(
        self,
        image_dir="data/images",
        mask_dir="data/masks",
        img_size=256,
        augment=False,
        cuts_csv="data/cuts.csv",
    ):
        self.img_size = img_size
        self.augment = augment
        self.images = sorted(os.listdir(image_dir))

        cuts_df = pd.read_csv(cuts_csv)
        self.cut_map = dict(zip(cuts_df["image"], cuts_df["cut_type"]))

        print("Cargando imágenes en RAM...")
        self.cache = []
        self.cut_types = []

        for img_name in self.images:
            image = Image.open(os.path.join(image_dir, img_name)).convert("L")
            mask = Image.open(os.path.join(mask_dir, img_name)).convert("L")

            image = TF.resize(image, (img_size, img_size))
            mask = TF.resize(
                mask,
                (img_size, img_size),
                interpolation=TF.InterpolationMode.NEAREST,
            )

            self.cache.append((image.copy(), mask.copy()))
            self.cut_types.append(self.cut_map[img_name])

        print(f"{len(self.cache)} imágenes cargadas.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, mask = self.cache[idx]

        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            angle = random.randint(-10, 10)
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

            if random.random() > 0.5:
                image = TF.adjust_contrast(image, random.uniform(0.9, 1.1))

            if random.random() > 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.95, 1.05))

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()

        return image, mask


def make_splits(data_dir="data", seed=42):
    """Devuelve (train_idx, val_idx, test_idx) con split estratificado 70/20/10."""
    cuts_df = pd.read_csv(os.path.join(data_dir, "cuts.csv"))
    cut_map = dict(zip(cuts_df["image"], cuts_df["cut_type"]))
    images = sorted(os.listdir(os.path.join(data_dir, "images")))
    cut_types = [cut_map[img] for img in images]
    indices = list(range(len(images)))

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, stratify=cut_types, random_state=seed
    )
    temp_cuts = [cut_types[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.33, stratify=temp_cuts, random_state=seed
    )
    return train_idx, val_idx, test_idx


def create_dataloaders(cfg, train_idx, val_idx, test_idx):
    train_set = Subset(BrainTumorDataset(augment=True, img_size=cfg.img_size), train_idx)
    val_set = Subset(BrainTumorDataset(augment=False, img_size=cfg.img_size), val_idx)
    test_set = Subset(BrainTumorDataset(augment=False, img_size=cfg.img_size), test_idx)

    loader_kwargs = {
        "num_workers": cfg.num_workers,
        "pin_memory": True,
    }
    if cfg.num_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)

    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
    return train_loader, val_loader, test_loader, test_set
