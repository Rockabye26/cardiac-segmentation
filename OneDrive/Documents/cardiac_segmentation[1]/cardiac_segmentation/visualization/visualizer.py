import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.widgets import Slider
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
from typing import List, Dict, Optional, Tuple, Union
import os
from pathlib import Path

class CardiacVisualization:
    """
    Comprehensive visualization tools for cardiac MRI segmentation
    """
    
    def __init__(
        self, 
        class_names: Optional[List[str]] = None,
        class_colors: Optional[List[str]] = None
    ):
        # Default ACDC class configuration
        self.class_names = class_names or ['Background', 'RV', 'Myocardium', 'LV']
        self.class_colors = class_colors or ['#000000', '#0000FF', '#00FF00', '#FF0000']
        
        # Create colormap
        self.cmap = ListedColormap(self.class_colors)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'seaborn')
        sns.set_palette("husl")
    
    def visualize_sample(
        self,
        image: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
        prediction: Optional[np.ndarray] = None,
        title: str = "Cardiac MRI Sample",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 5)
    ) -> plt.Figure:
        """
        Visualize a single sample with optional ground truth and prediction
        
        Args:
            image: Input image (H, W)
            ground_truth: Ground truth mask (H, W)
            prediction: Predicted mask (H, W)
            title: Figure title
            save_path: Path to save figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Determine number of subplots
        num_plots = 1 + (ground_truth is not None) + (prediction is not None)
        
        fig, axes = plt.subplots(1, num_plots, figsize=figsize)
        if num_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Original image
        axes[plot_idx].imshow(image, cmap='gray')
        axes[plot_idx].set_title('Original Image')
        axes[plot_idx].axis('off')
        plot_idx += 1
        
        # Ground truth
        if ground_truth is not None:
            im_gt = axes[plot_idx].imshow(ground_truth, cmap=self.cmap, 
                                         vmin=0, vmax=len(self.class_names)-1)
            axes[plot_idx].set_title('Ground Truth')
            axes[plot_idx].axis('off')
            plot_idx += 1
        
        # Prediction
        if prediction is not None:
            im_pred = axes[plot_idx].imshow(prediction, cmap=self.cmap,
                                           vmin=0, vmax=len(self.class_names)-1)
            axes[plot_idx].set_title('Prediction')
            axes[plot_idx].axis('off')
            
            # Add metrics if both GT and prediction available
            if ground_truth is not None:
                dice_scores = self._calculate_dice_scores(ground_truth, prediction)
                metrics_text = "\n".join([
                    f"{name}: {score:.3f}" 
                    for name, score in zip(self.class_names[1:], dice_scores[1:])
                ])
                axes[plot_idx].text(
                    0.02, 0.98, metrics_text,
                    transform=axes[plot_idx].transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=8
                )
        
        # Add colorbar if masks are present
        if ground_truth is not None or prediction is not None:
            cbar = fig.colorbar(im_gt if ground_truth is not None else im_pred, 
                              ax=axes, shrink=0.6)
            cbar.set_ticks(range(len(self.class_names)))
            cbar.set_ticklabels(self.class_names)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_overlay_visualization(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.5,
        title: str = "Segmentation Overlay",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create overlay visualization with adjustable opacity
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Mask only
        axes[1].imshow(mask, cmap=self.cmap, vmin=0, vmax=len(self.class_names)-1)
        axes[1].set_title('Segmentation Mask')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(image, cmap='gray')
        
        # Create colored overlay for each class
        for class_id in range(1, len(self.class_names)):
            class_mask = (mask == class_id)
            if np.any(class_mask):
                # Create colored overlay
                colored_mask = np.zeros((*mask.shape, 4))
                color = plt.cm.colors.to_rgba(self.class_colors[class_id])
                colored_mask[class_mask] = (*color[:3], alpha)
                axes[2].imshow(colored_mask)
        
        axes[2].set_title(f'Overlay (α={alpha})')
        axes[2].axis('off')
        
        # Add colorbar
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap=self.cmap, norm=Normalize(0, len(self.class_names)-1)),
            ax=axes, shrink=0.6
        )
        cbar.set_ticks(range(len(self.class_names)))
        cbar.set_ticklabels(self.class_names)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_training_curves(
        self,
        train_history: Dict[str, List],
        val_history: Dict[str, List],
        metrics: List[str] = ['loss', 'dice'],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot training and validation curves
        """
        fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
        if len(metrics) == 1:
            axes = [axes]
        
        epochs = range(1, len(train_history[metrics[0]]) + 1)
        
        for i, metric in enumerate(metrics):
            if metric in train_history and metric in val_history:
                axes[i].plot(epochs, train_history[metric], 
                           label=f'Train {metric.title()}', marker='o', markersize=3)
                axes[i].plot(epochs, val_history[metric], 
                           label=f'Val {metric.title()}', marker='s', markersize=3)
                
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel(metric.title())
                axes[i].set_title(f'{metric.title()} History')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
                
                # Highlight best epoch for validation
                if metric == 'dice':
                    best_epoch = np.argmax(val_history[metric]) + 1
                    best_value = max(val_history[metric])
                    axes[i].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5)
                    axes[i].annotate(
                        f'Best: {best_value:.4f}', 
                        xy=(best_epoch, best_value),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
                    )
                elif metric == 'loss':
                    best_epoch = np.argmin(val_history[metric]) + 1
                    best_value = min(val_history[metric])
                    axes[i].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5)
                    axes[i].annotate(
                        f'Best: {best_value:.4f}', 
                        xy=(best_epoch, best_value),
                        xytext=(10, -10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
                    )
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_metrics_comparison(
        self,
        metrics_dict: Dict[str, float],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create bar chart comparing different metrics
        """
        # Separate metrics by type
        dice_metrics = {k: v for k, v in metrics_dict.items() if 'dice' in k and 'std' not in k}
        iou_metrics = {k: v for k, v in metrics_dict.items() if 'iou' in k and 'std' not in k}
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Dice scores
        if dice_metrics:
            names = [k.replace('dice_', '').replace('_', ' ').title() for k in dice_metrics.keys()]
            values = list(dice_metrics.values())
            
            bars1 = axes[0].bar(names, values, color=sns.color_palette("husl", len(names)))
            axes[0].set_title('Dice Coefficients')
            axes[0].set_ylabel('Dice Score')
            axes[0].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars1, values):
                axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # IoU scores
        if iou_metrics:
            names = [k.replace('iou_', '').replace('_', ' ').title() for k in iou_metrics.keys()]
            values = list(iou_metrics.values())
            
            bars2 = axes[1].bar(names, values, color=sns.color_palette("husl", len(names)))
            axes[1].set_title('IoU Scores')
            axes[1].set_ylabel('IoU Score')
            axes[1].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars2, values):
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_overlay(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        title: str = "Interactive Segmentation"
    ):
        """
        Create interactive Plotly visualization with opacity slider
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Original Image', 'Segmentation Overlay'],
            specs=[[{"type": "image"}, {"type": "image"}]]
        )
        
        # Original image
        fig.add_trace(
            go.Heatmap(z=image, colorscale='gray', showscale=False),
            row=1, col=1
        )
        
        # Overlay - this is simplified, full implementation would need more complex overlay logic
        fig.add_trace(
            go.Heatmap(z=mask, colorscale='viridis', showscale=True),
            row=1, col=2
        )
        
        fig.update_layout(
            title=title,
            height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            xaxis2=dict(visible=False),
            yaxis2=dict(visible=False)
        )
        
        return fig
    
    def create_batch_visualization(
        self,
        images: np.ndarray,
        ground_truths: Optional[np.ndarray] = None,
        predictions: Optional[np.ndarray] = None,
        max_samples: int = 8,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize multiple samples in a grid
        """
        batch_size = min(images.shape[0], max_samples)
        
        # Determine grid size
        cols = 3 if ground_truths is not None and predictions is not None else 2
        rows = batch_size
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(batch_size):
            col = 0
            
            # Original image
            axes[i, col].imshow(images[i], cmap='gray')
            axes[i, col].set_title(f'Sample {i+1} - Original')
            axes[i, col].axis('off')
            col += 1
            
            # Ground truth
            if ground_truths is not None:
                axes[i, col].imshow(ground_truths[i], cmap=self.cmap,
                                   vmin=0, vmax=len(self.class_names)-1)
                axes[i, col].set_title(f'Sample {i+1} - Ground Truth')
                axes[i, col].axis('off')
                col += 1
            
            # Prediction
            if predictions is not None:
                axes[i, col].imshow(predictions[i], cmap=self.cmap,
                                   vmin=0, vmax=len(self.class_names)-1)
                title = f'Sample {i+1} - Prediction'
                
                # Add dice score if GT available
                if ground_truths is not None:
                    dice_scores = self._calculate_dice_scores(ground_truths[i], predictions[i])
                    mean_dice = np.mean(dice_scores[1:])  # Exclude background
                    title += f'\n(Dice: {mean_dice:.3f})'
                
                axes[i, col].set_title(title)
                axes[i, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _calculate_dice_scores(self, gt: np.ndarray, pred: np.ndarray) -> List[float]:
        """Calculate Dice scores for each class"""
        dice_scores = []
        
        for class_id in range(len(self.class_names)):
            gt_mask = (gt == class_id)
            pred_mask = (pred == class_id)
            
            intersection = (gt_mask & pred_mask).sum()
            union = gt_mask.sum() + pred_mask.sum()
            
            if union == 0:
                dice = 1.0
            else:
                dice = 2.0 * intersection / union
            
            dice_scores.append(dice)
        
        return dice_scores


def save_visualization_report(
    visualizations: Dict[str, plt.Figure],
    save_dir: str,
    title: str = "Cardiac Segmentation Results"
):
    """
    Save all visualizations and create an HTML report
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save all figures
    for name, fig in visualizations.items():
        fig.savefig(os.path.join(save_dir, f"{name}.png"), dpi=300, bbox_inches='tight')
    
    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .section {{ margin-bottom: 30px; }}
            .image {{ text-align: center; margin: 20px 0; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{title}</h1>
            <p>Generated on {plt.datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    """
    
    for name, fig in visualizations.items():
        html_content += f"""
        <div class="section">
            <h2>{name.replace('_', ' ').title()}</h2>
            <div class="image">
                <img src="{name}.png" alt="{name}">
            </div>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    with open(os.path.join(save_dir, "report.html"), 'w') as f:
        f.write(html_content)
    
    print(f"Visualization report saved to: {save_dir}/report.html")


if __name__ == "__main__":
    # Example usage
    print("Cardiac visualization tools ready.")
    
    # Create sample data for testing
    image = np.random.rand(256, 256)
    mask = np.random.randint(0, 4, (256, 256))
    
    # Create visualizer
    viz = CardiacVisualization()
    
    # Test basic visualization
    fig = viz.visualize_sample(image, ground_truth=mask, title="Test Sample")
    plt.show()
    plt.close(fig)