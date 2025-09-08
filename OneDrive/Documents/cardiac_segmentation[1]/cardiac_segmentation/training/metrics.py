import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class SegmentationMetrics:
    """
    Comprehensive metrics for segmentation evaluation
    """
    
    def __init__(self, num_classes: int = 4, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class_{i}' for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.total_samples = 0
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.dice_scores = []
        self.iou_scores = []
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update metrics with new predictions
        
        Args:
            pred: Predicted logits (B, C, H, W) or predictions (B, H, W)
            target: Ground truth labels (B, H, W)
        """
        # Convert logits to predictions if needed
        if pred.dim() == 4:  # (B, C, H, W)
            pred = torch.argmax(pred, dim=1)
        
        # Move to CPU and convert to numpy
        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()
        
        batch_size = pred_np.shape[0]
        self.total_samples += batch_size
        
        # Update confusion matrix
        for i in range(batch_size):
            cm = confusion_matrix(
                target_np[i].flatten(), 
                pred_np[i].flatten(), 
                labels=list(range(self.num_classes))
            )
            self.confusion_matrix += cm
        
        # Calculate per-image metrics
        for i in range(batch_size):
            dice_scores = self._calculate_dice_per_image(pred_np[i], target_np[i])
            iou_scores = self._calculate_iou_per_image(pred_np[i], target_np[i])
            
            self.dice_scores.append(dice_scores)
            self.iou_scores.append(iou_scores)
    
    def _calculate_dice_per_image(self, pred: np.ndarray, target: np.ndarray) -> List[float]:
        """Calculate Dice coefficient for each class in a single image"""
        dice_scores = []
        
        for class_id in range(self.num_classes):
            pred_mask = (pred == class_id)
            target_mask = (target == class_id)
            
            intersection = (pred_mask & target_mask).sum()
            union = pred_mask.sum() + target_mask.sum()
            
            if union == 0:
                dice = 1.0  # Perfect score when both masks are empty
            else:
                dice = 2.0 * intersection / union
            
            dice_scores.append(dice)
        
        return dice_scores
    
    def _calculate_iou_per_image(self, pred: np.ndarray, target: np.ndarray) -> List[float]:
        """Calculate IoU (Jaccard Index) for each class in a single image"""
        iou_scores = []
        
        for class_id in range(self.num_classes):
            pred_mask = (pred == class_id)
            target_mask = (target == class_id)
            
            intersection = (pred_mask & target_mask).sum()
            union = (pred_mask | target_mask).sum()
            
            if union == 0:
                iou = 1.0  # Perfect score when both masks are empty
            else:
                iou = intersection / union
            
            iou_scores.append(iou)
        
        return iou_scores
    
    def get_metrics(self) -> Dict[str, float]:
        """Get all computed metrics"""
        if self.total_samples == 0:
            return {}
        
        metrics = {}
        
        # Dice coefficients
        dice_array = np.array(self.dice_scores)  # (N_samples, N_classes)
        
        for i, class_name in enumerate(self.class_names):
            metrics[f'dice_{class_name.lower()}'] = dice_array[:, i].mean()
            metrics[f'dice_{class_name.lower()}_std'] = dice_array[:, i].std()
        
        metrics['dice_mean'] = dice_array.mean()
        metrics['dice_std'] = dice_array.mean(axis=1).std()  # Std across samples
        
        # IoU scores
        iou_array = np.array(self.iou_scores)  # (N_samples, N_classes)
        
        for i, class_name in enumerate(self.class_names):
            metrics[f'iou_{class_name.lower()}'] = iou_array[:, i].mean()
            metrics[f'iou_{class_name.lower()}_std'] = iou_array[:, i].std()
        
        metrics['iou_mean'] = iou_array.mean()
        metrics['iou_std'] = iou_array.mean(axis=1).std()
        
        # Accuracy metrics from confusion matrix
        accuracy_metrics = self._calculate_accuracy_metrics()
        metrics.update(accuracy_metrics)
        
        return metrics
    
    def _calculate_accuracy_metrics(self) -> Dict[str, float]:
        """Calculate accuracy metrics from confusion matrix"""
        metrics = {}
        
        # Overall accuracy
        correct_predictions = np.diag(self.confusion_matrix).sum()
        total_predictions = self.confusion_matrix.sum()
        metrics['pixel_accuracy'] = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        # Per-class accuracy (recall)
        for i, class_name in enumerate(self.class_names):
            class_total = self.confusion_matrix[i, :].sum()
            class_correct = self.confusion_matrix[i, i]
            accuracy = class_correct / class_total if class_total > 0 else 0.0
            metrics[f'accuracy_{class_name.lower()}'] = accuracy
        
        # Mean class accuracy
        class_accuracies = []
        for i in range(self.num_classes):
            class_total = self.confusion_matrix[i, :].sum()
            if class_total > 0:
                class_accuracies.append(self.confusion_matrix[i, i] / class_total)
        
        metrics['mean_class_accuracy'] = np.mean(class_accuracies) if class_accuracies else 0.0
        
        return metrics
    
    def get_confusion_matrix(self, normalize: bool = False) -> np.ndarray:
        """Get confusion matrix"""
        if normalize:
            cm = self.confusion_matrix.astype('float')
            cm = cm / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
            return cm
        return self.confusion_matrix
    
    def plot_confusion_matrix(self, normalize: bool = True, figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """Plot confusion matrix"""
        cm = self.get_confusion_matrix(normalize=normalize)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='.3f' if normalize else 'd',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax
        )
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        title = 'Normalized Confusion Matrix' if normalize else 'Confusion Matrix'
        ax.set_title(title)
        
        plt.tight_layout()
        return fig
    
    def print_summary(self):
        """Print a summary of all metrics"""
        metrics = self.get_metrics()
        
        print("=== Segmentation Metrics Summary ===")
        print(f"Total samples evaluated: {self.total_samples}")
        print()
        
        print("Dice Coefficients:")
        for class_name in self.class_names:
            dice_mean = metrics.get(f'dice_{class_name.lower()}', 0)
            dice_std = metrics.get(f'dice_{class_name.lower()}_std', 0)
            print(f"  {class_name}: {dice_mean:.4f} ± {dice_std:.4f}")
        
        print(f"  Overall: {metrics.get('dice_mean', 0):.4f} ± {metrics.get('dice_std', 0):.4f}")
        print()
        
        print("IoU Scores:")
        for class_name in self.class_names:
            iou_mean = metrics.get(f'iou_{class_name.lower()}', 0)
            iou_std = metrics.get(f'iou_{class_name.lower()}_std', 0)
            print(f"  {class_name}: {iou_mean:.4f} ± {iou_std:.4f}")
        
        print(f"  Overall: {metrics.get('iou_mean', 0):.4f} ± {metrics.get('iou_std', 0):.4f}")
        print()
        
        print("Accuracy Metrics:")
        print(f"  Pixel Accuracy: {metrics.get('pixel_accuracy', 0):.4f}")
        print(f"  Mean Class Accuracy: {metrics.get('mean_class_accuracy', 0):.4f}")


def evaluate_model(model, data_loader, device, num_classes: int = 4, class_names: Optional[List[str]] = None):
    """
    Evaluate model on a dataset
    
    Args:
        model: PyTorch model
        data_loader: Data loader for evaluation
        device: Device to run evaluation on
        num_classes: Number of classes
        class_names: List of class names
    
    Returns:
        SegmentationMetrics object with computed metrics
    """
    model.eval()
    metrics = SegmentationMetrics(num_classes, class_names)
    
    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Update metrics
            metrics.update(outputs, masks)
    
    return metrics


if __name__ == "__main__":
    # Test metrics calculation
    num_classes = 4
    class_names = ['Background', 'RV', 'Myocardium', 'LV']
    
    # Create dummy data
    batch_size, height, width = 2, 256, 256
    
    # Simulate predictions and targets
    pred = torch.randint(0, num_classes, (batch_size, height, width))
    target = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Create metrics calculator
    metrics = SegmentationMetrics(num_classes, class_names)
    
    # Update with dummy data
    metrics.update(pred, target)
    
    # Print summary
    metrics.print_summary()
    
    # Get metrics dictionary
    metrics_dict = metrics.get_metrics()
    print(f"\nMetrics dictionary has {len(metrics_dict)} entries:")
    for key, value in list(metrics_dict.items())[:5]:  # Show first 5
        print(f"  {key}: {value:.4f}")