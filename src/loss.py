from abc import abstractmethod, ABC

import torch
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F

_EPSILON = 1e-8

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

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean', task_type='multi-class', num_classes=None):
        """
        Unified Focal Loss class for binary, multi-class, and multi-label classification tasks.
        :param gamma: Focusing parameter, controls the strength of the modulating factor (1 - p_t)^gamma
        :param alpha: Balancing factor, can be a scalar or a tensor for class-wise weights. If None, no class balancing is used.
        :param reduction: Specifies the reduction method: 'none' | 'mean' | 'sum'
        :param task_type: Specifies the type of task: 'binary', 'multi-class', or 'multi-label'
        :param num_classes: Number of classes (only required for multi-class classification)
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.task_type = task_type
        self.num_classes = num_classes

        # Handle alpha for class balancing in multi-class tasks
        if task_type == 'multi-class' and alpha is not None and isinstance(alpha, (list, torch.Tensor)):
            assert num_classes is not None, "num_classes must be specified for multi-class classification"
            if isinstance(alpha, list):
                self.alpha = torch.Tensor(alpha)
            else:
                self.alpha = alpha

    def forward(self, inputs, targets):
        """
        Forward pass to compute the Focal Loss based on the specified task type.
        :param inputs: Predictions (logits) from the model.
                       Shape:
                         - binary/multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size, num_classes)
        :param targets: Ground truth labels.
                        Shape:
                         - binary: (batch_size,)
                         - multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size,)
        """
        if self.task_type == 'binary':
            return self.binary_focal_loss(inputs, targets)
        elif self.task_type == 'multi-class':
            return self.multi_class_focal_loss(inputs, targets)
        elif self.task_type == 'multi-label':
            return self.multi_label_focal_loss(inputs, targets)
        else:
            raise ValueError(
                f"Unsupported task_type '{self.task_type}'. Use 'binary', 'multi-class', or 'multi-label'.")

    def binary_focal_loss(self, inputs, targets):
        """ Focal loss for binary classification. """
        probs = torch.sigmoid(inputs)
        targets = targets.float()

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weighting
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_class_focal_loss(self, inputs, targets):
        eps = 1e-5
        B, C, H, W = inputs.shape
        inputs = inputs.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        targets = targets.view(-1)                         # (B*H*W,)

        probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=C).float()

        ce_loss = -targets_one_hot * torch.log(probs + eps)
        p_t = torch.sum(probs * targets_one_hot, dim=1)
        focal_weight = ((1 - p_t) ** self.gamma)

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha.gather(0, targets)
            ce_loss = alpha_t.unsqueeze(1) * ce_loss

        loss = focal_weight.unsqueeze(1) * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_label_focal_loss(self, inputs, targets):
        """ Focal loss for multi-label classification. """
        probs = torch.sigmoid(inputs)

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weight
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

def soft_dice_loss(output, target, epsilon=1e-6):
    numerator = 2. * torch.sum(output * target, dim=(-2, -1))
    denominator = torch.sum(output + target, dim=(-2, -1))
    return (numerator + epsilon) / (denominator + epsilon)

class ExponentialLogarithmicLoss(nn.Module):
    """
    This loss is focuses on less accurately predicted structures using the combination of Dice Loss and Cross Entropy
    Loss
    
    Original paper: https://arxiv.org/pdf/1809.00076.pdf
    
    See the paper at 2.2 w_l = ((Sum k f_k) / f_l) ** 0.5 is the label weight
    
    Note: 
        - Input for CrossEntropyLoss is the logits - Raw output from the model
    """
    
    def __init__(self, w_dice=0.5, w_cross=0.5, gamma=0.3, use_softmax=True, class_weights=None, epochs=None):
        super(ExponentialLogarithmicLoss, self).__init__()
        self.w_dice = w_dice
        self.gamma = gamma
        self.w_cross = w_cross
        self.use_softmax = use_softmax
        self.class_weights = class_weights
        self.N = epochs

    def forward(self, output, target, epsilon=1e-6, epoch=None):
        assert epoch is not None, "Epoch must be provided for dynamic weighting"
        num_classes = output.shape[1]

        # Update local weights (do not overwrite self.w_cross and self.w_dice!)
        w_cross = (self.N - epoch) / self.N
        w_dice = epoch / self.N

        # Validate class weights
        assert self.class_weights is not None and len(self.class_weights) == num_classes, \
            "Class weights must be a tensor/list with length = num_classes"
        class_weights = self.class_weights.to(target.device)

        # Get weight map per pixel using class weights
        weight_map = class_weights[target]  # shape: (B, H, W)

        # Prepare one-hot targets for Dice
        target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # Softmax ONLY for Dice (not CE)
        if self.use_softmax:
            output_soft = F.softmax(output, dim=1)
        else:
            output_soft = output  # for safety

        # Dice loss (clamped to avoid log(0))
        dice_loss_value = soft_dice_loss(output_soft, target_one_hot).clamp(min=epsilon)
        l_dice = torch.mean(torch.pow(-torch.log(dice_loss_value), self.gamma))

        # Cross-entropy loss (with logits), also clamped
        ce_loss = F.cross_entropy(output, target, reduction='none').clamp(min=epsilon)
        l_cross = torch.mean(weight_map * torch.pow(ce_loss, self.gamma))

        # Final weighted sum
        return w_dice * l_dice + w_cross * l_cross

class UnweightedExponentialLogarithmicLoss(nn.Module):
    """
    This loss is focuses on less accurately predicted structures using the combination of Dice Loss and Cross Entropy
    Loss
    
    Original paper: https://arxiv.org/pdf/1809.00076.pdf
    
    See the paper at 2.2 w_l = ((Sum k f_k) / f_l) ** 0.5 is the label weight
    
    Note: 
        - Input for CrossEntropyLoss is the logits - Raw output from the model
    """
    
    def __init__(self, w_dice=0.5, w_cross=0.5, gamma=0.3, use_softmax=True, class_weights=None, epochs=None):
        super(ExponentialLogarithmicLoss, self).__init__()
        self.w_dice = w_dice
        self.gamma = gamma
        self.w_cross = w_cross
        self.use_softmax = use_softmax
        self.class_weights = class_weights
        self.N = epochs

    def forward(self, output, target, epsilon=1e-6, epoch=None):
        n = epoch
        self.w_cross = (self.N - n) / self.N
        self.w_dice = n / self.N

        num_classes = output.shape[1]
        
        target = F.one_hot(target.to(torch.int64), num_classes=num_classes).permute((0, 3, 1, 2)).to(torch.float)
        if self.use_softmax:
            output = F.softmax(output, dim=1)
        
        l_dice = torch.mean(torch.pow(-torch.log(soft_dice_loss(output, target)), self.gamma))   # mean w.r.t to label
        l_cross = torch.mean(torch.pow(-torch.log(F.cross_entropy(output, target, reduction='none')), self.gamma))
        return self.w_dice * l_dice + self.w_cross * l_cross

class CombinedLoss(torch.nn.Module):
    def __init__(self, ce_weight=1, dice_weight=1, class_weights=None, epochs=None):
        super().__init__()
        if class_weights is not None:
            self.ce = FocalLoss(alpha=class_weights, num_classes=20)
        else:
            self.ce = FocalLoss(num_classes=20)
        
        self.dice = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.N = epochs

    def forward(self, preds, targets, epoch):
        n = epoch
        ce_weight = (self.N - n) / self.N
        dice_weight = n / self.N
        return self.ce(preds, targets) + 0.5 * self.dice(preds, targets)

class SymmetricFocalLoss(nn.Module):
    def __init__(self, delta=0.6, gamma=0.5, epsilon=1e-6):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true_onehot):
        cross_entropy = -y_true_onehot * torch.log(y_pred + self.epsilon)
        # if torch.isnan(cross_entropy).any() == True:
        #     print(f"Ce nan: {torch.isnan(cross_entropy).any()}")

        loss = self.delta * torch.pow(1 - y_pred, self.gamma) * cross_entropy
        # if torch.isnan(loss).any() == True:
        #     print(f"Ce loss nan: {torch.isnan(loss).any()}")

        loss += (1 - self.delta) * torch.pow(y_pred, self.gamma) * cross_entropy
        # if torch.isnan(loss).any() == True:
        #     print(f"Focal loss nan: {torch.isnan(loss).any()}")
        return loss.mean()

class SymmetricFocalTverskyLoss(nn.Module):
    def __init__(self, delta=0.6, gamma=0.5, epsilon=1e-6):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true_onehot):
        # Flatten spatial dims: [B, C, H*W]
        y_pred_flat = y_pred.view(y_pred.size(0), y_pred.size(1), -1)

        # if torch.isnan(y_pred_flat).any() == True:
        #     print(f"Y_pred flat nan: {torch.isnan(y_pred_flat).any()}")

        
        y_true_flat = y_true_onehot.view(y_true_onehot.size(0), y_true_onehot.size(1), -1)

        # if torch.isnan(y_true_flat).any() == True:
        #     print(f"Y_true flat nan: {torch.isnan(y_true_flat).any()}")

        true_pos = torch.sum(y_true_flat * y_pred_flat, dim=2)

        # if torch.isnan(true_pos).any() == True:
        #     print(f"True pos nan: {torch.isnan(true_pos).any()}")
        false_neg = torch.sum(y_true_flat * (1 - y_pred_flat), dim=2)

        # if torch.isnan(false_neg).any() == True:
        #     print(f"False neg nan: {torch.isnan(false_neg).any()}")
        false_pos = torch.sum((1 - y_true_flat) * y_pred_flat, dim=2)

        # if torch.isnan(false_pos).any() == True:
        #     print(f"False pos nan: {torch.isnan(false_pos).any()}")

        tversky = (true_pos + self.epsilon) / (true_pos + self.delta * false_neg + (1 - self.delta) * false_pos + self.epsilon)
        
        # if torch.isnan(tversky).any() == True:
        #     print(f"Tversky nan: {torch.isnan(tversky).any()}")
        loss = torch.pow(torch.max(1 - tversky, torch.tensor(0)), self.gamma)

        # if torch.isnan(loss).any() == True:
        #     print(f"Tversky loss nan: {torch.isnan(loss).any()}")
        return loss.mean()

class SymmetricUnifiedFocalLoss(nn.Module):
    """
    Symmetric Unified Focal Loss for semantic segmentation.
    Inputs:
      - y_pred: probabilities after softmax/sigmoid, shape [B, C, H, W]
      - y_true: class indices, shape [B, H, W]
    """
    def __init__(self, weight=0.5, delta=0.6, gamma=0.5, num_classes=20):
        super().__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma
        self.num_classes = num_classes
        self.epsilon = 1e-6

        self.focal = SymmetricFocalLoss(delta=delta, gamma=gamma, epsilon=self.epsilon)
        self.focal_tversky = SymmetricFocalTverskyLoss(delta=delta, gamma=gamma, epsilon=self.epsilon)

    def forward(self, y_pred, y_true):
        if self.num_classes is None:
            self.num_classes = y_pred.size(1)

        # print('y_pred max:', y_pred.max().item())
        # print('y_pred min:', y_pred.min().item())
        # print('y_pred mean:', y_pred.mean().item())
        # print('Any NaN:', torch.isnan(y_pred).any().item())
        # print('Any Inf:', torch.isinf(y_pred).any().item())

        # if torch.isnan(y_pred).any() == True:
        #     print(f"Y_pred nan: {torch.isnan(y_pred).any()}")
        
        # if torch.isnan(y_true).any() == True:
        #     print(f"Y_true nan: {torch.isnan(y_true).any()}")

        # Apply softmax over channel dim to get probabilities
        y_pred_prob = torch.softmax(y_pred, dim=1)
        pt = torch.clamp(y_pred_prob, min=self.epsilon, max=1.0 - self.epsilon)

        # if torch.isnan(y_pred_prob).any() == True:
        #     print(f"Softmax nan: {torch.isnan(y_pred_prob).any()}")

        # Convert y_true (class indices) to one-hot
        y_true_onehot = F.one_hot(y_true.long(), num_classes=self.num_classes)
        y_true_onehot = y_true_onehot.permute(0, 3, 1, 2).float()

        # if torch.isnan(y_true_onehot).any() == True:
        #     print(f"One shot nan: {torch.isnan(y_true_onehot).any()}")

        focal_loss = self.focal(pt, y_true_onehot)
        focal_tversky_loss = self.focal_tversky(pt, y_true_onehot)

        # if torch.isnan(focal_loss).any() == True:
        #     print(f"Final focal loss nan: {torch.isnan(focal_loss).any()}")

        # if torch.isnan(focal_tversky_loss).any() == True:
        #     print(f"Final tversky loss nan: {torch.isnan(focal_tversky_loss).any()}")

        return self.weight * focal_tversky_loss + (1 - self.weight) * focal_loss
