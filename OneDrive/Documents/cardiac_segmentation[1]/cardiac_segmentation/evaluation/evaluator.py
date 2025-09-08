import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import os
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from ..training.metrics import SegmentationMetrics
from ..training.losses import DiceLoss

class ModelEvaluator:
    """
    Comprehensive model evaluation for cardiac segmentation
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        device: torch.device,
        num_classes: int = 4,
        class_names: Optional[List[str]] = None,
        class_colors: Optional[List[str]] = None
    ):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        
        # Default class names and colors for ACDC
        self.class_names = class_names or ['Background', 'RV', 'Myocardium', 'LV']
        self.class_colors = class_colors or ['#000000', '#0000FF', '#00FF00', '#FF0000']
        
        # Initialize metrics
        self.metrics = SegmentationMetrics(num_classes, self.class_names)
        
    def evaluate_dataset(
        self, 
        data_loader: torch.utils.data.DataLoader,
        save_dir: Optional[str] = None
    ) -> Dict:
        """
        Evaluate model on entire dataset
        
        Args:
            data_loader: DataLoader for evaluation
            save_dir: Directory to save evaluation results
            
        Returns:
            Dictionary containing all evaluation metrics and statistics
        """
        self.model.eval()
        self.metrics.reset()
        
        all_predictions = []
        all_targets = []
        all_losses = []
        image_metrics = []
        
        dice_loss_fn = DiceLoss()
        
        print("Evaluating model on dataset...")
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(tqdm(data_loader)):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                predictions = torch.argmax(outputs, dim=1)
                
                # Calculate loss
                loss = dice_loss_fn(outputs, masks)
                all_losses.append(loss.item())
                
                # Store predictions and targets
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(masks.cpu().numpy())
                
                # Update overall metrics
                self.metrics.update(outputs, masks)
                
                # Calculate per-image metrics
                batch_size = images.size(0)
                for i in range(batch_size):
                    pred_i = predictions[i].cpu().numpy()
                    target_i = masks[i].cpu().numpy()
                    
                    # Calculate per-image dice scores
                    image_dice = self._calculate_image_dice(pred_i, target_i)
                    image_metrics.append({
                        'batch_idx': batch_idx,
                        'image_idx': i,
                        'dice_scores': image_dice,
                        'mean_dice': np.mean(image_dice[1:])  # Exclude background
                    })
        
        # Compile results
        results = self._compile_evaluation_results(
            all_predictions, all_targets, all_losses, image_metrics
        )
        
        # Save results if directory provided
        if save_dir:
            self._save_evaluation_results(results, save_dir)
        
        return results
    
    def _calculate_image_dice(self, pred: np.ndarray, target: np.ndarray) -> List[float]:
        """Calculate Dice coefficient for each class in a single image"""
        dice_scores = []
        
        for class_id in range(self.num_classes):
            pred_mask = (pred == class_id)
            target_mask = (target == class_id)
            
            intersection = (pred_mask & target_mask).sum()
            union = pred_mask.sum() + target_mask.sum()
            
            if union == 0:
                dice = 1.0
            else:
                dice = 2.0 * intersection / union
            
            dice_scores.append(dice)
        
        return dice_scores
    
    def _compile_evaluation_results(
        self, 
        all_predictions: List[np.ndarray],
        all_targets: List[np.ndarray],
        all_losses: List[float],
        image_metrics: List[Dict]
    ) -> Dict:
        """Compile all evaluation results"""
        
        # Overall metrics
        overall_metrics = self.metrics.get_metrics()
        
        # Loss statistics
        loss_stats = {
            'mean_loss': np.mean(all_losses),
            'std_loss': np.std(all_losses),
            'min_loss': np.min(all_losses),
            'max_loss': np.max(all_losses)
        }
        
        # Per-image statistics
        image_dice_scores = [img['mean_dice'] for img in image_metrics]
        per_image_stats = {
            'mean_dice': np.mean(image_dice_scores),
            'std_dice': np.std(image_dice_scores),
            'min_dice': np.min(image_dice_scores),
            'max_dice': np.max(image_dice_scores),
            'median_dice': np.median(image_dice_scores),
            'q25_dice': np.percentile(image_dice_scores, 25),
            'q75_dice': np.percentile(image_dice_scores, 75)
        }
        
        # Worst and best cases
        worst_cases = sorted(image_metrics, key=lambda x: x['mean_dice'])[:5]
        best_cases = sorted(image_metrics, key=lambda x: x['mean_dice'], reverse=True)[:5]
        
        # Class-wise statistics
        class_wise_stats = {}
        for i, class_name in enumerate(self.class_names):
            class_dice_scores = [img['dice_scores'][i] for img in image_metrics]
            class_wise_stats[class_name.lower()] = {
                'mean': np.mean(class_dice_scores),
                'std': np.std(class_dice_scores),
                'min': np.min(class_dice_scores),
                'max': np.max(class_dice_scores),
                'median': np.median(class_dice_scores)
            }
        
        return {
            'overall_metrics': overall_metrics,
            'loss_statistics': loss_stats,
            'per_image_statistics': per_image_stats,
            'class_wise_statistics': class_wise_stats,
            'worst_cases': worst_cases,
            'best_cases': best_cases,
            'confusion_matrix': self.metrics.get_confusion_matrix(normalize=True).tolist(),
            'total_images': len(image_metrics)
        }
    
    def _save_evaluation_results(self, results: Dict, save_dir: str):
        """Save evaluation results to files"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save metrics as JSON
        metrics_file = os.path.join(save_dir, 'evaluation_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save confusion matrix plot
        cm_fig = self.metrics.plot_confusion_matrix()
        cm_fig.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close(cm_fig)
        
        # Create summary report
        self._create_summary_report(results, save_dir)
        
        print(f"Evaluation results saved to: {save_dir}")
    
    def _create_summary_report(self, results: Dict, save_dir: str):
        """Create a detailed summary report"""
        report_file = os.path.join(save_dir, 'evaluation_summary.txt')
        
        with open(report_file, 'w') as f:
            f.write("=== Cardiac Segmentation Model Evaluation Report ===\n\n")
            
            # Overall performance
            f.write("OVERALL PERFORMANCE:\n")
            f.write(f"Total images evaluated: {results['total_images']}\n")
            f.write(f"Mean Dice coefficient: {results['per_image_statistics']['mean_dice']:.4f} ± {results['per_image_statistics']['std_dice']:.4f}\n")
            f.write(f"Median Dice coefficient: {results['per_image_statistics']['median_dice']:.4f}\n")
            f.write(f"Range: [{results['per_image_statistics']['min_dice']:.4f}, {results['per_image_statistics']['max_dice']:.4f}]\n\n")
            
            # Class-wise performance
            f.write("CLASS-WISE PERFORMANCE:\n")
            for class_name in self.class_names:
                stats = results['class_wise_statistics'][class_name.lower()]
                f.write(f"{class_name}: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
            f.write("\n")
            
            # Best and worst cases
            f.write("BEST PERFORMING CASES:\n")
            for i, case in enumerate(results['best_cases']):
                f.write(f"{i+1}. Batch {case['batch_idx']}, Image {case['image_idx']}: Dice = {case['mean_dice']:.4f}\n")
            
            f.write("\nWORST PERFORMING CASES:\n")
            for i, case in enumerate(results['worst_cases']):
                f.write(f"{i+1}. Batch {case['batch_idx']}, Image {case['image_idx']}: Dice = {case['mean_dice']:.4f}\n")
            
            f.write(f"\nDetailed metrics saved to: evaluation_metrics.json\n")
    
    def compare_predictions(
        self, 
        images: torch.Tensor, 
        targets: torch.Tensor, 
        save_dir: Optional[str] = None,
        num_samples: int = 5
    ) -> plt.Figure:
        """
        Create comparison visualization between ground truth and predictions
        
        Args:
            images: Input images (B, C, H, W)
            targets: Ground truth masks (B, H, W)
            save_dir: Directory to save visualizations
            num_samples: Number of samples to visualize
            
        Returns:
            Matplotlib figure with comparisons
        """
        self.model.eval()
        
        with torch.no_grad():
            images = images.to(self.device)
            outputs = self.model(images)
            predictions = torch.argmax(outputs, dim=1)
        
        # Move to CPU for visualization
        images = images.cpu()
        targets = targets.cpu()
        predictions = predictions.cpu()
        
        # Select samples to visualize
        batch_size = min(images.size(0), num_samples)
        
        # Create figure
        fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))
        if batch_size == 1:
            axes = axes.reshape(1, -1)
        
        # Create colormap
        colors = ['black'] + [c for c in self.class_colors[1:]]  # Skip background
        cmap = ListedColormap(colors)
        
        for i in range(batch_size):
            # Original image
            if images.size(1) == 1:  # Grayscale
                axes[i, 0].imshow(images[i, 0], cmap='gray')
            else:
                axes[i, 0].imshow(images[i].permute(1, 2, 0))
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            # Ground truth
            axes[i, 1].imshow(targets[i], cmap=cmap, vmin=0, vmax=self.num_classes-1)
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            # Prediction
            axes[i, 2].imshow(predictions[i], cmap=cmap, vmin=0, vmax=self.num_classes-1)
            
            # Calculate and display dice score
            dice_score = self._calculate_image_dice(
                predictions[i].numpy(), 
                targets[i].numpy()
            )
            mean_dice = np.mean(dice_score[1:])  # Exclude background
            axes[i, 2].set_title(f'Prediction (Dice: {mean_dice:.3f})')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, 'prediction_comparison.png'), 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_error_analysis(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        save_dir: Optional[str] = None
    ) -> plt.Figure:
        """
        Create error analysis visualization showing where predictions fail
        """
        self.model.eval()
        
        with torch.no_grad():
            images = images.to(self.device)
            outputs = self.model(images)
            predictions = torch.argmax(outputs, dim=1)
        
        # Move to CPU
        images = images.cpu()
        targets = targets.cpu()
        predictions = predictions.cpu()
        
        # Create error maps
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Sample index to analyze
        sample_idx = 0
        
        img = images[sample_idx, 0] if images.size(1) == 1 else images[sample_idx].permute(1, 2, 0)
        target = targets[sample_idx]
        pred = predictions[sample_idx]
        
        # Original image
        axes[0, 0].imshow(img, cmap='gray' if images.size(1) == 1 else None)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Ground truth
        axes[0, 1].imshow(target, cmap=ListedColormap(self.class_colors))
        axes[0, 1].set_title('Ground Truth')
        axes[0, 1].axis('off')
        
        # Prediction
        axes[0, 2].imshow(pred, cmap=ListedColormap(self.class_colors))
        axes[0, 2].set_title('Prediction')
        axes[0, 2].axis('off')
        
        # Error map (absolute difference)
        error_map = (target != pred).astype(int)
        axes[1, 0].imshow(error_map, cmap='Reds')
        axes[1, 0].set_title('Error Map')
        axes[1, 0].axis('off')
        
        # Per-class error analysis
        for class_id in range(1, min(3, self.num_classes)):  # Show 2 classes max
            target_class = (target == class_id)
            pred_class = (pred == class_id)
            
            # False positives and false negatives
            fp = pred_class & (~target_class)
            fn = (~pred_class) & target_class
            
            error_class = np.zeros_like(target)
            error_class[fp] = 1  # False positives in red
            error_class[fn] = 2   # False negatives in blue
            
            axes[1, class_id].imshow(error_class, cmap=ListedColormap(['black', 'red', 'blue']))
            axes[1, class_id].set_title(f'{self.class_names[class_id]} Errors\n(Red=FP, Blue=FN)')
            axes[1, class_id].axis('off')
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, 'error_analysis.png'), 
                       dpi=300, bbox_inches='tight')
        
        return fig


def load_model_for_evaluation(
    model_path: str, 
    model_class: type,
    device: Optional[torch.device] = None
) -> torch.nn.Module:
    """
    Load trained model for evaluation
    
    Args:
        model_path: Path to saved model checkpoint
        model_class: Model class constructor
        device: Device to load model on
        
    Returns:
        Loaded model in evaluation mode
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model instance
    model = model_class()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from: {model_path}")
    print(f"Best validation Dice: {checkpoint.get('best_val_dice', 'N/A')}")
    
    return model


if __name__ == "__main__":
    # Example usage
    print("Model evaluator ready for use.")
    print("Example usage:")
    print("1. Load trained model")
    print("2. Create evaluator instance")
    print("3. Run evaluation on test dataset")
    print("4. Generate visualization and reports")