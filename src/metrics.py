import torch

def dice_score(logits, targets, threshold=0.5, smooth=1e-6):
    preds = (torch.sigmoid(logits) > threshold).float().flatten(1)
    targets = targets.flatten(1)
    intersection = (preds * targets).sum(dim=1)
    dice = (2 * intersection + smooth) / (preds.sum(dim=1) + targets.sum(dim=1) + smooth)
    return dice.mean().item()


def iou_score(logits, targets, threshold=0.5, smooth=1e-6):
    preds = (torch.sigmoid(logits) > threshold).float().flatten(1)
    targets = targets.flatten(1)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection
    return ((intersection + smooth) / (union + smooth)).mean().item()
