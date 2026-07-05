from dataclasses import dataclass, field


@dataclass
class Config:
    # Modelo
    model_name: str = "unet"          # "unet" | "residual_unet" | "attention_unet"
    base_features: int = 64
    depth: int = 4
    dropout: float = 0.0

    # Datos
    img_size: int = 256
    batch_size: int = 64
    num_workers: int = 2

    # Entrenamiento
    epochs: int = 300
    patience: int = 10
    lr: float = 1e-4
    weight_decay: float = 0.0

    # Reproducibilidad
    seed: int = 42
