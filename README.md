# Segmentación de tumores cerebrales con U-Net

Segmentación semántica binaria de tumores en imágenes MRI usando variantes de U-Net.

## Estructura

```
tumor-segmentation/
├── src/
│   ├── dataset.py          # BrainTumorDataset, splits, dataloaders
│   ├── losses.py           # dice_loss, combined_loss
│   ├── metrics.py          # dice_score, iou_score
│   ├── trainer.py          # train_epoch, val_epoch, train()
│   ├── evaluate.py         # evaluate(), evaluate_by_cut(), evaluate_by_tumor_size()
│   ├── visualize.py        # plots de EDA, predicciones, curvas
│   └── models/
│       ├── __init__.py         # MODEL_REGISTRY + get_model()
│       ├── unet.py             # UNet, UNet++
│       ├── residual_unet.py    # ResidualUNet
│       └── attention_unet.py   # AttentionUNet 
├── scripts/
│   ├── train_single.py     # entrena una config específica
│   └── run_experiments.py  # grid search de hiperparámetros
├── configs/
│   └── config.py           # dataclass Config unificado
├── notebooks/              # notebooks originales de exploración
├── requirements.txt
└── .gitignore
```

## Setup

```bash
pip install -r requirements.txt
```

Los datos deben estar en `data/images/`, `data/masks/` y `data/cuts.csv`.

## Uso

### Entrenar una sola configuración

```bash
python scripts/train_single.py --model unet --lr 1e-4 --dropout 0.1
python scripts/train_single.py --model residual_unet --lr 1e-4 --weight_decay 1e-5
```

### Grid search de hiperparámetros

```bash
python scripts/run_experiments.py --model unet
python scripts/run_experiments.py --model residual_unet
python scripts/run_experiments.py --model attention_unet
```

Los resultados se guardan en `results/<model>/<config>/results.json`.

## Agregar un nuevo modelo

1. Creá `src/models/mi_modelo.py` con una clase que implemente `forward(x)`.
2. Registralo en `src/models/__init__.py`:
   ```python
   from src.models.mi_modelo import MiModelo
   MODEL_REGISTRY["mi_modelo"] = MiModelo
   ```
3. Usalo con `--model mi_modelo` en cualquier script.

## Modelos disponibles

| Nombre | Clase | Descripción |
|---|---|---|
| `unet` | `UNet` | U-Net estándar con DoubleCNN |
| `residual_unet` | `ResidualUNet` | U-Net con bloques residuales |
| `attention_unet` | `AttentionUNet` | U-Net con attention gates |
