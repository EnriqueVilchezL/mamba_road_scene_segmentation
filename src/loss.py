import torch
import torch.nn as nn
import torch.nn.functional as F

class SymmetricFocalLoss(nn.Module):
    """
    Symmetric Focal Loss for addressing class imbalance in segmentation tasks.

    This loss penalizes both false positives and false negatives using a symmetric formulation.
    """

    def __init__(self, delta=0.6, gamma=0.5, epsilon=1e-6):
        """
        Initializes the Symmetric Focal Loss.

        Args:
            delta (float): Weight for positive samples in the loss.
            gamma (float): Focusing parameter to adjust the rate at which easy examples are down-weighted.
            epsilon (float): Small constant to avoid log(0).
        """
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true_onehot):
        """
        Computes the symmetric focal loss.

        Args:
            y_pred (torch.Tensor): Predicted probabilities of shape (B, C, H, W).
            y_true_onehot (torch.Tensor): One-hot encoded ground truth of shape (B, C, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        pos_term = -self.delta * (1 - y_pred) ** self.gamma * y_true_onehot * torch.log(y_pred + self.epsilon)
        neg_term = -(1 - self.delta) * y_pred ** self.gamma * (1 - y_true_onehot) * torch.log(1 - y_pred + self.epsilon)
        loss = pos_term + neg_term

        return loss.mean()


class SymmetricFocalTverskyLoss(nn.Module):
    """
    Symmetric Focal Tversky Loss, a generalization of Dice loss with control over false positives and negatives.

    Especially useful for highly imbalanced semantic segmentation problems.
    """

    def __init__(self, delta=0.6, gamma=0.5, epsilon=1e-6):
        """
        Initializes the Symmetric Focal Tversky Loss.

        Args:
            delta (float): Balances the penalty between false positives and false negatives.
            gamma (float): Controls the non-linearity of the loss surface.
            epsilon (float): Small constant to avoid division by zero.
        """
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true_onehot):
        """
        Computes the Symmetric Focal Tversky loss.

        Args:
            y_pred (torch.Tensor): Predicted probabilities of shape (B, C, H, W).
            y_true_onehot (torch.Tensor): One-hot encoded ground truth of shape (B, C, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        y_pred_flat = y_pred.view(y_pred.size(0), y_pred.size(1), -1)
        y_true_flat = y_true_onehot.view(y_true_onehot.size(0), y_true_onehot.size(1), -1)

        true_pos = torch.sum(y_true_flat * y_pred_flat, dim=2)
        false_neg = torch.sum(y_true_flat * (1 - y_pred_flat), dim=2)
        false_pos = torch.sum((1 - y_true_flat) * y_pred_flat, dim=2)

        tversky = (true_pos + self.epsilon) / (
            true_pos + self.delta * false_neg + (1 - self.delta) * false_pos + self.epsilon
        )

        loss = torch.pow(torch.clamp(1 - tversky, min=0.0), self.gamma)
        return loss.mean()


class SymmetricUnifiedFocalLoss(nn.Module):
    """
    Combines Symmetric Focal Loss and Symmetric Focal Tversky Loss into a unified loss function.

    Suitable for multi-class semantic segmentation with highly imbalanced classes.

    Attributes:
        weight (float): Weight given to the Focal Tversky Loss. The Focal Loss weight is (1 - weight).
    """

    def __init__(self, weight=0.5, delta=0.6, gamma=0.5, num_classes=20):
        """
        Initializes the unified loss function.

        Args:
            weight (float): Balance between Focal Tversky and Focal loss. Must be between 0 and 1.
            delta (float): Balance parameter between false positives and false negatives.
            gamma (float): Focusing parameter for modulating the easy vs hard examples.
            num_classes (int): Number of output classes in segmentation.
        """
        super().__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma
        self.num_classes = num_classes
        self.epsilon = 1e-6

        self.focal = SymmetricFocalLoss(delta=delta, gamma=gamma, epsilon=self.epsilon)
        self.focal_tversky = SymmetricFocalTverskyLoss(delta=delta, gamma=gamma, epsilon=self.epsilon)

    def forward(self, y_pred, y_true):
        """
        Computes the combined Symmetric Unified Focal Loss.

        Args:
            y_pred (torch.Tensor): Raw logits of shape (B, C, H, W).
            y_true (torch.Tensor): Ground truth class indices of shape (B, H, W).

        Returns:
            torch.Tensor: Scalar loss value combining Focal and Focal Tversky Losses.
        """
        if self.num_classes is None:
            self.num_classes = y_pred.size(1)

        y_pred_prob = torch.softmax(y_pred, dim=1)
        pt = torch.clamp(y_pred_prob, min=self.epsilon, max=1.0 - self.epsilon)

        y_true_onehot = F.one_hot(y_true.long(), num_classes=self.num_classes)
        y_true_onehot = y_true_onehot.permute(0, 3, 1, 2).float()

        focal_loss = self.focal(pt, y_true_onehot)
        focal_tversky_loss = self.focal_tversky(pt, y_true_onehot)

        return self.weight * focal_tversky_loss + (1 - self.weight) * focal_loss