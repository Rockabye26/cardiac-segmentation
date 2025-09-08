import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import numpy as np

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks
    
    Dice coefficient = 2 * |A ∩ B| / (|A| + |B|)
    Dice loss = 1 - Dice coefficient
    """
    
    def __init__(self, smooth: float = 1e-6, ignore_index: Optional[int] = None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted logits (B, C, H, W)
            target: Ground truth labels (B, H, W)
        """
        # Convert logits to probabilities
        pred = F.softmax(pred, dim=1)
        
        # One-hot encode target
        target_one_hot = F.one_hot(target, num_classes=pred.size(1)).permute(0, 3, 1, 2).float()
        
        # Calculate Dice coefficient for each class
        dice_scores = []
        
        for i in range(pred.size(1)):
            if self.ignore_index is not None and i == self.ignore_index:
                continue
                
            pred_i = pred[:, i]
            target_i = target_one_hot[:, i]
            
            intersection = (pred_i * target_i).sum(dim=(1, 2))
            union = pred_i.sum(dim=(1, 2)) + target_i.sum(dim=(1, 2))
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        # Average across classes and batch
        dice_loss = 1.0 - torch.stack(dice_scores).mean()
        
        return dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """
    
    def __init__(self, alpha: Optional[List[float]] = None, gamma: float = 2.0, ignore_index: int = -100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        
        if alpha is not None:
            self.alpha = torch.tensor(alpha)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted logits (B, C, H, W)
            target: Ground truth labels (B, H, W)
        """
        ce_loss = F.cross_entropy(pred, target, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            if self.alpha.device != target.device:
                self.alpha = self.alpha.to(target.device)
            at = self.alpha.gather(0, target.flatten())
            at = at.view_as(target)
            focal_loss = at * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss: weighted sum of CrossEntropy and Dice losses
    """
    
    def __init__(
        self, 
        ce_weight: float = 0.3, 
        dice_weight: float = 0.7,
        class_weights: Optional[List[float]] = None,
        ignore_index: int = -100
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        # Cross-entropy loss
        weight = None
        if class_weights is not None:
            weight = torch.tensor(class_weights)
        
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.dice_loss = DiceLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        
        return self.ce_weight * ce + self.dice_weight * dice


def calculate_class_weights(dataset, num_classes: int = 4) -> List[float]:
    """
    Calculate class weights based on inverse frequency
    """
    class_counts = np.zeros(num_classes)
    
    for _, mask in dataset:
        mask_np = mask.numpy()
        for class_id in range(num_classes):
            class_counts[class_id] += (mask_np == class_id).sum()
    
    # Inverse frequency weighting
    total_pixels = class_counts.sum()
    class_weights = total_pixels / (num_classes * class_counts + 1e-6)
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * num_classes
    
    return class_weights.tolist()


if __name__ == "__main__":
    # Test loss functions
    batch_size, num_classes, height, width = 2, 4, 256, 256
    
    # Create dummy data
    pred = torch.randn(batch_size, num_classes, height, width)
    target = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Test different losses
    losses = {
        'CrossEntropy': nn.CrossEntropyLoss(),
        'Dice': DiceLoss(),
        'Focal': FocalLoss(),
        'Combined': CombinedLoss()
    }
    
    print("Loss Function Tests:")
    for name, loss_fn in losses.items():
        loss_value = loss_fn(pred, target)
        print(f"  {name}: {loss_value.item():.4f}")
    
    # Test class weights calculation
    print(f"\nExample class weights for ACDC:")
    # Typical ACDC class distribution (approximate)
    example_counts = np.array([0.85, 0.05, 0.05, 0.05])  # Background, RV, Myo, LV
    total = example_counts.sum()
    weights = total / (4 * example_counts + 1e-6)
    weights = weights / weights.sum() * 4
    
    classes = ['Background', 'RV', 'Myocardium', 'LV']
    for i, (cls, weight) in enumerate(zip(classes, weights)):
        print(f"  {cls}: {weight:.3f}")