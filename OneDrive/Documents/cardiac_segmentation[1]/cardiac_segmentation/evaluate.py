#!/usr/bin/env python3
"""
Evaluation script for Cardiac MRI Segmentation models
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

from config import get_config, setup_environment, get_device, RESULTS_DIR
from data.acdc_dataset import create_data_loaders
from models.unet import create_unet_model
from evaluation.evaluator import ModelEvaluator, load_model_for_evaluation
from visualization.visualizer import CardiacVisualization
from training.metrics import SegmentationMetrics

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate Cardiac MRI Segmentation Model")
    
    # Required arguments
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--data-path", type=str, required=True,
        help="Path to ACDC data directory"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output-dir", type=str, default=str(RESULTS_DIR / "evaluation"),
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--model-size", choices=["small", "standard", "large"], default="standard",
        help="Model size variant"
    )
    parser.add_argument(
        "--split", choices=["train", "val", "all"], default="val",
        help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--save-predictions", action="store_true",
        help="Save individual predictions as images"
    )
    parser.add_argument(
        "--create-report", action="store_true", default=True,
        help="Create comprehensive evaluation report"
    )
    parser.add_argument(
        "--visualize-samples", type=int, default=5,
        help="Number of samples to visualize (0 to disable)"
    )
    parser.add_argument(
        "--compare-models", nargs="+", default=None,
        help="Paths to additional models for comparison"
    )
    
    return parser.parse_args()

def load_and_evaluate_model(model_path: str, data_loader, device, output_dir: Path, args):
    """Load model and run comprehensive evaluation"""
    logger.info(f"Evaluating model: {model_path}")
    
    # Load model
    model = load_model_for_evaluation(
        model_path=model_path,
        model_class=lambda: create_unet_model(args.model_size),
        device=device
    )
    
    # Create evaluator
    class_names = get_config("acdc")["class_names"]
    class_colors = get_config("acdc")["class_colors"]
    
    evaluator = ModelEvaluator(
        model=model,
        device=device,
        num_classes=4,
        class_names=class_names,
        class_colors=class_colors
    )
    
    # Run evaluation
    model_name = Path(model_path).stem
    model_output_dir = output_dir / model_name
    results = evaluator.evaluate_dataset(data_loader, str(model_output_dir))
    
    return results, evaluator

def create_comparison_report(model_results: dict, output_dir: Path):
    """Create comparison report for multiple models"""
    logger.info("Creating model comparison report...")
    
    # Extract key metrics for comparison
    comparison_data = []
    
    for model_name, results in model_results.items():
        row = {
            'Model': model_name,
            'Mean Dice': results['per_image_statistics']['mean_dice'],
            'Std Dice': results['per_image_statistics']['std_dice'],
            'Median Dice': results['per_image_statistics']['median_dice'],
            'Min Dice': results['per_image_statistics']['min_dice'],
            'Max Dice': results['per_image_statistics']['max_dice'],
        }
        
        # Add class-wise Dice scores
        class_names = ['RV', 'Myocardium', 'LV']
        for class_name in class_names:
            class_key = class_name.lower()
            if class_key in results['class_wise_statistics']:
                row[f'{class_name} Dice'] = results['class_wise_statistics'][class_key]['mean']
        
        comparison_data.append(row)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save as CSV
    comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Overall Dice comparison
    axes[0, 0].bar(comparison_df['Model'], comparison_df['Mean Dice'], 
                   yerr=comparison_df['Std Dice'], capsize=5)
    axes[0, 0].set_title('Mean Dice Score Comparison')
    axes[0, 0].set_ylabel('Dice Score')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Class-wise comparison
    class_columns = [col for col in comparison_df.columns if 'Dice' in col and col != 'Mean Dice' and col != 'Std Dice']
    
    if class_columns:
        class_data = comparison_df[['Model'] + class_columns].set_index('Model')
        class_data.plot(kind='bar', ax=axes[0, 1], rot=45)
        axes[0, 1].set_title('Class-wise Dice Score Comparison')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Dice distribution comparison
    axes[1, 0].bar(comparison_df['Model'], comparison_df['Median Dice'], alpha=0.7, label='Median')
    axes[1, 0].bar(comparison_df['Model'], comparison_df['Mean Dice'], alpha=0.7, label='Mean')
    axes[1, 0].set_title('Dice Score Distribution Comparison')
    axes[1, 0].set_ylabel('Dice Score')
    axes[1, 0].legend()
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Performance range
    models = comparison_df['Model']
    min_dice = comparison_df['Min Dice']
    max_dice = comparison_df['Max Dice']
    mean_dice = comparison_df['Mean Dice']
    
    axes[1, 1].fill_between(range(len(models)), min_dice, max_dice, alpha=0.3, label='Min-Max Range')
    axes[1, 1].plot(range(len(models)), mean_dice, 'o-', label='Mean Dice')
    axes[1, 1].set_title('Performance Range Comparison')
    axes[1, 1].set_ylabel('Dice Score')
    axes[1, 1].set_xticks(range(len(models)))
    axes[1, 1].set_xticklabels(models, rotation=45)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Comparison report saved to: {output_dir}")
    return comparison_df

def create_detailed_visualizations(evaluator: ModelEvaluator, data_loader, output_dir: Path, num_samples: int):
    """Create detailed visualization examples"""
    if num_samples <= 0:
        return
    
    logger.info(f"Creating visualizations for {num_samples} samples...")
    
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Get sample data
    data_iter = iter(data_loader)
    
    sample_count = 0
    batch_count = 0
    
    while sample_count < num_samples:
        try:
            images, targets = next(data_iter)
            batch_size = images.size(0)
            
            # Create comparison visualization
            fig = evaluator.compare_predictions(
                images=images,
                targets=targets,
                num_samples=min(batch_size, num_samples - sample_count),
                save_dir=str(viz_dir)
            )
            
            # Save individual batch visualization
            fig.savefig(viz_dir / f"batch_{batch_count}_comparison.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # Create error analysis for first sample in batch
            if batch_count == 0:
                error_fig = evaluator.create_error_analysis(
                    images=images[:1],
                    targets=targets[:1],
                    save_dir=str(viz_dir)
                )
                plt.close(error_fig)
            
            sample_count += min(batch_size, num_samples - sample_count)
            batch_count += 1
            
        except StopIteration:
            break
    
    logger.info(f"Visualizations saved to: {viz_dir}")

def generate_evaluation_summary(results: dict, model_name: str, output_file: Path):
    """Generate a comprehensive evaluation summary"""
    
    with open(output_file, 'w') as f:
        f.write(f"# Cardiac MRI Segmentation Evaluation Report\n\n")
        f.write(f"**Model:** {model_name}\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Images:** {results['total_images']}\n\n")
        
        f.write("## Overall Performance\n\n")
        stats = results['per_image_statistics']
        f.write(f"- **Mean Dice:** {stats['mean_dice']:.4f} ± {stats['std_dice']:.4f}\n")
        f.write(f"- **Median Dice:** {stats['median_dice']:.4f}\n")
        f.write(f"- **Range:** [{stats['min_dice']:.4f}, {stats['max_dice']:.4f}]\n")
        f.write(f"- **Q25-Q75:** [{stats['q25_dice']:.4f}, {stats['q75_dice']:.4f}]\n\n")
        
        f.write("## Class-wise Performance\n\n")
        class_stats = results['class_wise_statistics']
        for class_name, stats in class_stats.items():
            f.write(f"### {class_name.title()}\n")
            f.write(f"- **Mean:** {stats['mean']:.4f} ± {stats['std']:.4f}\n")
            f.write(f"- **Median:** {stats['median']:.4f}\n")
            f.write(f"- **Range:** [{stats['min']:.4f}, {stats['max']:.4f}]\n\n")
        
        f.write("## Loss Statistics\n\n")
        loss_stats = results['loss_statistics']
        f.write(f"- **Mean Loss:** {loss_stats['mean_loss']:.4f} ± {loss_stats['std_loss']:.4f}\n")
        f.write(f"- **Range:** [{loss_stats['min_loss']:.4f}, {loss_stats['max_loss']:.4f}]\n\n")
        
        f.write("## Best Performing Cases\n\n")
        for i, case in enumerate(results['best_cases'][:3]):
            f.write(f"{i+1}. Batch {case['batch_idx']}, Image {case['image_idx']}: "
                   f"Dice = {case['mean_dice']:.4f}\n")
        
        f.write("\n## Worst Performing Cases\n\n")
        for i, case in enumerate(results['worst_cases'][:3]):
            f.write(f"{i+1}. Batch {case['batch_idx']}, Image {case['image_idx']}: "
                   f"Dice = {case['mean_dice']:.4f}\n")

def main():
    """Main evaluation function"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup environment
    setup_environment()
    device = get_device()
    
    logger.info(f"Using device: {device}")
    logger.info(f"Evaluating model: {args.model_path}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / f"evaluation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        data_dir=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=0.8,
        image_size=(256, 256)
    )
    
    # Select data loader based on split
    if args.split == "train":
        data_loader = train_loader
    elif args.split == "val":
        data_loader = val_loader
    else:  # all
        # Combine train and val loaders (simplified approach)
        data_loader = val_loader
        logger.warning("Using validation set only for 'all' split evaluation")
    
    logger.info(f"Evaluating on {len(data_loader.dataset)} samples")
    
    # Evaluate main model
    model_results = {}
    
    # Main model evaluation
    results, evaluator = load_and_evaluate_model(
        args.model_path, data_loader, device, output_dir, args
    )
    
    main_model_name = Path(args.model_path).stem
    model_results[main_model_name] = results
    
    # Create detailed visualizations
    if args.visualize_samples > 0:
        create_detailed_visualizations(
            evaluator, data_loader, output_dir, args.visualize_samples
        )
    
    # Evaluate comparison models if provided
    if args.compare_models:
        for model_path in args.compare_models:
            if Path(model_path).exists():
                comp_results, _ = load_and_evaluate_model(
                    model_path, data_loader, device, output_dir, args
                )
                comp_model_name = Path(model_path).stem
                model_results[comp_model_name] = comp_results
            else:
                logger.warning(f"Comparison model not found: {model_path}")
    
    # Create comparison report if multiple models
    if len(model_results) > 1:
        comparison_df = create_comparison_report(model_results, output_dir)
        logger.info("Model comparison completed")
    
    # Generate summary report for main model
    if args.create_report:
        summary_file = output_dir / f"{main_model_name}_evaluation_summary.md"
        generate_evaluation_summary(results, main_model_name, summary_file)
        logger.info(f"Evaluation summary saved to: {summary_file}")
    
    # Print final results
    logger.info("\n" + "="*50)
    logger.info("EVALUATION COMPLETED")
    logger.info("="*50)
    
    for model_name, results in model_results.items():
        logger.info(f"\nModel: {model_name}")
        logger.info(f"Mean Dice: {results['per_image_statistics']['mean_dice']:.4f} ± {results['per_image_statistics']['std_dice']:.4f}")
        
        # Class-wise results
        class_stats = results['class_wise_statistics']
        for class_name in ['rv', 'myocardium', 'lv']:
            if class_name in class_stats:
                dice = class_stats[class_name]['mean']
                logger.info(f"{class_name.title()} Dice: {dice:.4f}")
    
    logger.info(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main()