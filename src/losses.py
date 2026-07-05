import torch.nn.functional as F


def dice_loss(logits, targets, smooth=1e-6):
    """Dice loss promediado sobre el batch."""
    import torch
    probs = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    dice = (2 * intersection + smooth) / (probs.sum(dim=1) + targets.sum(dim=1) + smooth)
    return 1 - dice.mean()


def combined_loss(logits, targets):
    """BCE + Dice."""
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    return bce + dice_loss(logits, targets)
