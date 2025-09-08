"""
Configuration module for Cardiac MRI Segmentation System
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for directory in [DATA_DIR, CHECKPOINTS_DIR, LOGS_DIR, RESULTS_DIR]:
    directory.mkdir(exist_ok=True)

# ACDC Dataset Configuration
ACDC_CONFIG = {
    "num_classes": 4,
    "class_names": ["Background", "RV", "Myocardium", "LV"],
    "class_colors": ["#000000", "#0000FF", "#00FF00", "#FF0000"],
    "image_size": (256, 256),
    "train_ratio": 0.8,
    "random_seed": 42
}

# Model Configuration
MODEL_CONFIG = {
    "architecture": "unet",
    "input_channels": 1,
    "output_classes": 4,
    "base_channels": 64,
    "dropout": 0.1,
    "bilinear": True,
    "pretrained": False
}

# Training Configuration
TRAINING_CONFIG = {
    # Optimizer settings
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "beta1": 0.9,
    "beta2": 0.999,
    
    # Training parameters
    "batch_size": 8,
    "max_epochs": 100,
    "warmup_epochs": 5,
    "patience": 15,
    "min_delta": 0.001,
    
    # Loss function
    "loss_type": "combined",  # 'ce', 'dice', 'focal', 'combined'
    "ce_weight": 0.3,
    "dice_weight": 0.7,
    "focal_alpha": None,
    "focal_gamma": 2.0,
    
    # Regularization
    "gradient_clip_max_norm": 1.0,
    "mixed_precision": True,
    
    # Scheduling
    "scheduler_type": "reduce_on_plateau",  # 'step', 'cosine', 'reduce_on_plateau'
    "scheduler_factor": 0.5,
    "scheduler_patience": 10,
    "min_lr": 1e-7,
    
    # Data loading
    "num_workers": 4,
    "pin_memory": True,
    "augmentation": True
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    "metrics": ["dice", "iou", "accuracy", "sensitivity", "specificity"],
    "save_visualizations": True,
    "save_predictions": True,
    "create_report": True,
    "confusion_matrix": True
}

# Visualization Configuration
VISUALIZATION_CONFIG = {
    "figure_size": (12, 8),
    "dpi": 300,
    "colormap": "viridis",
    "overlay_alpha": 0.5,
    "font_size": 12,
    "save_format": "png"
}

# Web App Configuration
WEBAPP_CONFIG = {
    "title": "Cardiac MRI Segmentation Demo",
    "icon": "🫀",
    "layout": "wide",
    "max_file_size": 50,  # MB
    "allowed_extensions": [".nii", ".nii.gz"],
    "default_opacity": 0.5,
    "processing_timeout": 30  # seconds
}

# Hardware Configuration
HARDWARE_CONFIG = {
    "use_cuda": True,
    "device": "auto",  # 'auto', 'cpu', 'cuda', 'cuda:0', etc.
    "mixed_precision": True,
    "benchmark": True,  # cudnn benchmark
    "deterministic": False  # cudnn deterministic (slower but reproducible)
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_logging": True,
    "console_logging": True,
    "tensorboard_logging": True
}

# File Paths
DEFAULT_PATHS = {
    "acdc_data": "path/to/ACDC/training",  # Update with actual path
    "pretrained_model": CHECKPOINTS_DIR / "best_model.pth",
    "config_file": PROJECT_ROOT / "config.yaml",
    "results_dir": RESULTS_DIR,
    "logs_dir": LOGS_DIR
}

def get_config(config_name: str) -> Dict:
    """
    Get configuration dictionary by name
    
    Args:
        config_name: Name of configuration ('acdc', 'model', 'training', etc.)
        
    Returns:
        Configuration dictionary
    """
    config_map = {
        "acdc": ACDC_CONFIG,
        "model": MODEL_CONFIG,
        "training": TRAINING_CONFIG,
        "evaluation": EVALUATION_CONFIG,
        "visualization": VISUALIZATION_CONFIG,
        "webapp": WEBAPP_CONFIG,
        "hardware": HARDWARE_CONFIG,
        "logging": LOGGING_CONFIG,
        "paths": DEFAULT_PATHS
    }
    
    if config_name not in config_map:
        raise ValueError(f"Unknown configuration: {config_name}. "
                        f"Available: {list(config_map.keys())}")
    
    return config_map[config_name].copy()

def update_config(config_name: str, updates: Dict) -> Dict:
    """
    Update configuration with new values
    
    Args:
        config_name: Name of configuration to update
        updates: Dictionary of updates to apply
        
    Returns:
        Updated configuration dictionary
    """
    config = get_config(config_name)
    config.update(updates)
    return config

def get_device():
    """Get the appropriate device for computation"""
    import torch
    
    hardware_config = get_config("hardware")
    
    if hardware_config["device"] == "auto":
        if hardware_config["use_cuda"] and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(hardware_config["device"])
    
    return device

def setup_environment():
    """Setup the computing environment"""
    import torch
    import numpy as np
    import random
    
    # Get configurations
    hardware_config = get_config("hardware")
    acdc_config = get_config("acdc")
    
    # Set random seeds for reproducibility
    seed = acdc_config["random_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Configure CUDA settings
        if hardware_config["benchmark"]:
            torch.backends.cudnn.benchmark = True
        
        if hardware_config["deterministic"]:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    # Set number of threads for CPU operations
    if torch.get_num_threads() > 1:
        torch.set_num_threads(min(torch.get_num_threads(), 8))

def print_config_summary():
    """Print a summary of all configurations"""
    print("=== Cardiac MRI Segmentation Configuration ===\n")
    
    configs = [
        ("ACDC Dataset", "acdc"),
        ("Model", "model"),
        ("Training", "training"),
        ("Hardware", "hardware")
    ]
    
    for title, config_name in configs:
        print(f"{title} Configuration:")
        config = get_config(config_name)
        for key, value in config.items():
            print(f"  {key}: {value}")
        print()

# Environment variables (override defaults)
def load_env_overrides():
    """Load configuration overrides from environment variables"""
    overrides = {}
    
    # Training overrides
    if "BATCH_SIZE" in os.environ:
        overrides["batch_size"] = int(os.environ["BATCH_SIZE"])
    
    if "LEARNING_RATE" in os.environ:
        overrides["learning_rate"] = float(os.environ["LEARNING_RATE"])
    
    if "MAX_EPOCHS" in os.environ:
        overrides["max_epochs"] = int(os.environ["MAX_EPOCHS"])
    
    # Path overrides
    if "ACDC_DATA_PATH" in os.environ:
        DEFAULT_PATHS["acdc_data"] = os.environ["ACDC_DATA_PATH"]
    
    if "MODEL_PATH" in os.environ:
        DEFAULT_PATHS["pretrained_model"] = Path(os.environ["MODEL_PATH"])
    
    return overrides

# Load environment overrides
ENV_OVERRIDES = load_env_overrides()

if __name__ == "__main__":
    # Print configuration summary
    print_config_summary()
    
    # Setup environment
    setup_environment()
    
    # Print device info
    device = get_device()
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    
    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Checkpoints dir: {CHECKPOINTS_DIR}")
    print(f"Logs dir: {LOGS_DIR}")
    
    # Environment overrides
    if ENV_OVERRIDES:
        print(f"\nEnvironment overrides: {ENV_OVERRIDES}")