#!/usr/bin/env python3
"""
Main training script for Cardiac MRI Segmentation
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
import json
from datetime import datetime

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

from config import (
    get_config, setup_environment, get_device, 
    PROJECT_ROOT, CHECKPOINTS_DIR, LOGS_DIR
)
from data.acdc_dataset import create_data_loaders, calculate_class_weights
from models.unet import create_unet_model
from training.trainer import create_trainer
from evaluation.evaluator import ModelEvaluator, load_model_for_evaluation
from visualization.visualizer import CardiacVisualization

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Cardiac MRI Segmentation Training")
    
    # Data arguments
    parser.add_argument(
        "--data-path", type=str, required=True,
        help="Path to ACDC training data directory"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size for training (default: 8)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Number of data loading workers (default: 4)"
    )
    
    # Model arguments
    parser.add_argument(
        "--model-size", choices=["small", "standard", "large"], default="standard",
        help="Model size variant (default: standard)"
    )
    parser.add_argument(
        "--pretrained", type=str, default=None,
        help="Path to pretrained model checkpoint"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Maximum number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3,
        help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--patience", type=int, default=15,
        help="Early stopping patience (default: 15)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume training from checkpoint"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir", type=str, default=str(CHECKPOINTS_DIR),
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--log-dir", type=str, default=str(LOGS_DIR),
        help="Directory to save training logs"
    )
    parser.add_argument(
        "--experiment-name", type=str, default=None,
        help="Experiment name for logging and checkpoints"
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--evaluate-only", action="store_true",
        help="Only evaluate model, don't train"
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to trained model for evaluation"
    )
    
    # Other arguments
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--mixed-precision", action="store_true",
        help="Use mixed precision training"
    )
    parser.add_argument(
        "--calculate-class-weights", action="store_true",
        help="Calculate and use class weights for loss function"
    )
    
    return parser.parse_args()

def setup_experiment(args):
    """Setup experiment directory and logging"""
    # Create experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"unet_{args.model_size}_{timestamp}"
    
    # Create experiment directories
    exp_dir = Path(args.output_dir) / args.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path(args.log_dir) / args.experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment configuration
    config_path = exp_dir / "experiment_config.json"
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)
    
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Output directory: {exp_dir}")
    logger.info(f"Log directory: {log_dir}")
    
    return exp_dir, log_dir

def create_model_and_data(args, device):
    """Create model and data loaders"""
    logger.info("Creating model and data loaders...")
    
    # Create model
    model = create_unet_model(
        model_size=args.model_size,
        n_channels=1,
        n_classes=4,
        device=device
    )
    
    # Load pretrained weights if provided
    if args.pretrained:
        logger.info(f"Loading pretrained weights from: {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Print model info
    info = model.get_model_info()
    logger.info(f"Model: {info['architecture']}")
    logger.info(f"Parameters: {info['total_parameters']:,}")
    logger.info(f"Model size: {info['model_size_mb']:.1f} MB")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        data_dir=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=0.8,
        image_size=(256, 256)
    )
    
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    return model, train_loader, val_loader

def train_model(args, model, train_loader, val_loader, exp_dir, log_dir):
    """Train the model"""
    logger.info("Starting model training...")
    
    # Calculate class weights if requested
    class_weights = None
    if args.calculate_class_weights:
        logger.info("Calculating class weights...")
        class_weights = calculate_class_weights(train_loader.dataset)
        logger.info(f"Class weights: {class_weights}")
    
    # Training configuration
    training_config = get_config("training")
    training_config.update({
        'learning_rate': args.learning_rate,
        'max_epochs': args.epochs,
        'patience': args.patience,
        'mixed_precision': args.mixed_precision,
        'save_dir': str(exp_dir),
        'log_dir': str(log_dir),
        'class_weights': class_weights
    })
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config
    )
    
    # Train model
    history = trainer.train(resume_from=args.resume)
    
    logger.info("Training completed!")
    logger.info(f"Best validation Dice: {trainer.best_val_dice:.4f}")
    
    return history

def evaluate_model_performance(args, model_path, val_loader, device):
    """Evaluate trained model"""
    logger.info("Evaluating model performance...")
    
    # Load model for evaluation
    from models.unet import UNet
    model = load_model_for_evaluation(
        model_path=model_path,
        model_class=lambda: create_unet_model(args.model_size),
        device=device
    )
    
    # Create evaluator
    class_names = get_config("acdc")["class_names"]
    evaluator = ModelEvaluator(
        model=model,
        device=device,
        num_classes=4,
        class_names=class_names
    )
    
    # Run evaluation
    results_dir = Path(args.output_dir) / args.experiment_name / "evaluation"
    results = evaluator.evaluate_dataset(val_loader, str(results_dir))
    
    # Print summary
    logger.info("Evaluation Results:")
    logger.info(f"Mean Dice: {results['per_image_statistics']['mean_dice']:.4f} ± {results['per_image_statistics']['std_dice']:.4f}")
    logger.info(f"Median Dice: {results['per_image_statistics']['median_dice']:.4f}")
    
    for class_name in class_names[1:]:  # Skip background
        class_dice = results['class_wise_statistics'][class_name.lower()]['mean']
        logger.info(f"{class_name} Dice: {class_dice:.4f}")
    
    return results

def create_visualizations(args, model_path, val_loader, device):
    """Create visualization examples"""
    logger.info("Creating visualizations...")
    
    # Load model
    from models.unet import UNet
    model = load_model_for_evaluation(
        model_path=model_path,
        model_class=lambda: create_unet_model(args.model_size),
        device=device
    )
    
    # Get some validation samples
    data_iter = iter(val_loader)
    images, targets = next(data_iter)
    
    # Create visualizer
    class_names = get_config("acdc")["class_names"]
    class_colors = get_config("acdc")["class_colors"]
    visualizer = CardiacVisualization(class_names, class_colors)
    
    # Create various visualizations
    viz_dir = Path(args.output_dir) / args.experiment_name / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        images_gpu = images.to(device)
        outputs = model(images_gpu)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
    
    images_np = images.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # Create batch visualization
    fig = visualizer.create_batch_visualization(
        images=images_np[:, 0],  # Remove channel dimension
        ground_truths=targets_np,
        predictions=predictions,
        max_samples=4,
        save_path=str(viz_dir / "batch_results.png")
    )
    fig.close()
    
    # Create individual sample visualizations
    for i in range(min(3, len(images))):
        fig = visualizer.visualize_sample(
            image=images_np[i, 0],
            ground_truth=targets_np[i],
            prediction=predictions[i],
            title=f"Sample {i+1} Results",
            save_path=str(viz_dir / f"sample_{i+1}.png")
        )
        fig.close()
    
    # Create overlay visualization
    fig = visualizer.create_overlay_visualization(
        image=images_np[0, 0],
        mask=predictions[0],
        title="Segmentation Overlay Example",
        save_path=str(viz_dir / "overlay_example.png")
    )
    fig.close()
    
    logger.info(f"Visualizations saved to: {viz_dir}")

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup environment
    setup_environment()
    device = get_device()
    
    logger.info(f"Using device: {device}")
    logger.info(f"Arguments: {vars(args)}")
    
    # Setup experiment
    exp_dir, log_dir = setup_experiment(args)
    
    # Create model and data
    model, train_loader, val_loader = create_model_and_data(args, device)
    
    if args.evaluate_only:
        # Evaluation only mode
        model_path = args.model_path or str(exp_dir / "best_model.pth")
        
        if not Path(model_path).exists():
            logger.error(f"Model file not found: {model_path}")
            sys.exit(1)
        
        # Evaluate model
        results = evaluate_model_performance(args, model_path, val_loader, device)
        
        # Create visualizations
        create_visualizations(args, model_path, val_loader, device)
        
    else:
        # Training mode
        history = train_model(args, model, train_loader, val_loader, exp_dir, log_dir)
        
        # Evaluate best model
        best_model_path = exp_dir / "best_model.pth"
        if best_model_path.exists():
            results = evaluate_model_performance(args, str(best_model_path), val_loader, device)
            create_visualizations(args, str(best_model_path), val_loader, device)
        
    logger.info("Process completed successfully!")

if __name__ == "__main__":
    main()