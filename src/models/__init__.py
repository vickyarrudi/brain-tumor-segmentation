from src.models.unet import UNet
from src.models.residual_unet import ResidualUNet
from src.models.attention_unet import AttentionUNet
    

MODEL_REGISTRY = {
    "unet": UNet,
    "residual_unet": ResidualUNet,
    "attention_unet": AttentionUNet
}


def get_model(name: str, **kwargs):
    """Instancia un modelo por nombre"""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Modelo '{name}' no encontrado. Disponibles: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
