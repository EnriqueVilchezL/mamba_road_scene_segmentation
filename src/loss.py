import torch
import torch.nn.functional as F

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # assumes preds are logits; apply softmax
        preds = torch.softmax(preds, dim=1)
        targets_onehot = F.one_hot(targets, num_classes=preds.shape[1]).permute(0, 3, 1, 2).float()

        intersection = (preds * targets_onehot).sum(dim=(2, 3))
        union = preds.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2, 3))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class CombinedLoss(torch.nn.Module):
    def __init__(self, ce_weight=0.4, dice_weight=0.6):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, preds, targets):
        return self.ce_weight * self.ce(preds, targets) + self.dice_weight * self.dice(preds, targets)