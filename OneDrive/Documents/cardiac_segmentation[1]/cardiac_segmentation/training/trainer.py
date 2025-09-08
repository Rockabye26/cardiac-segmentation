import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import time
from tqdm import tqdm
from typing import Dict, Optional, Tuple, Callable
import numpy as np
import json
from datetime import datetime

from .losses import CombinedLoss, DiceLoss
from .metrics import SegmentationMetrics, evaluate_model

class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.wait = 0
        self.stopped_epoch = 0
        
        if mode == 'max':
            self.monitor_op = np.greater
            self.min_delta *= 1
        else:
            self.monitor_op = np.less
            self.min_delta *= -1
    
    def __call__(self, score: float, epoch: int) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.monitor_op(score, self.best_score + self.min_delta):
            self.best_score = score
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            return True
        
        return False


class CardiacSegmentationTrainer:
    """
    Complete training pipeline for cardiac MRI segmentation
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        config: Optional[Dict] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Default configuration
        self.config = {
            'learning_rate': 1e-3,
            'max_epochs': 100,
            'patience': 15,
            'weight_decay': 1e-5,
            'gradient_clip_max_norm': 1.0,
            'warmup_epochs': 5,
            'save_dir': 'checkpoints',
            'log_dir': 'logs',
            'num_classes': 4,
            'class_names': ['Background', 'RV', 'Myocardium', 'LV'],
            'mixed_precision': True,
            'save_best_only': True
        }
        
        if config:
            self.config.update(config)
        
        # Create directories
        os.makedirs(self.config['save_dir'], exist_ok=True)
        os.makedirs(self.config['log_dir'], exist_ok=True)
        
        # Initialize components
        self._setup_training()
        
        # Training state
        self.current_epoch = 0
        self.best_val_dice = 0.0
        self.train_history = {'loss': [], 'dice': []}
        self.val_history = {'loss': [], 'dice': []}
        
    def _setup_training(self):
        """Setup optimizer, scheduler, loss function, and other components"""
        # Loss function
        self.criterion = CombinedLoss(
            ce_weight=0.3, 
            dice_weight=0.7,
            class_weights=None  # Could be calculated from dataset
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            verbose=True,
            min_lr=1e-7
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config['patience'],
            mode='max'
        )
        
        # Mixed precision scaler
        if self.config['mixed_precision'] and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Tensorboard writer
        log_dir = os.path.join(
            self.config['log_dir'], 
            f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.writer = SummaryWriter(log_dir)
        
        # Metrics
        self.train_metrics = SegmentationMetrics(
            self.config['num_classes'], 
            self.config['class_names']
        )
        self.val_metrics = SegmentationMetrics(
            self.config['num_classes'], 
            self.config['class_names']
        )
    
    def _warmup_lr(self, epoch: int):
        """Apply learning rate warmup for the first few epochs"""
        if epoch < self.config['warmup_epochs']:
            lr = self.config['learning_rate'] * (epoch + 1) / self.config['warmup_epochs']
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}/Train')
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['gradient_clip_max_norm']
                )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['gradient_clip_max_norm']
                )
                
                # Optimizer step
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            self.train_metrics.update(outputs.detach(), masks)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # Calculate epoch metrics
        avg_loss = total_loss / num_batches
        metrics = self.train_metrics.get_metrics()
        avg_dice = metrics.get('dice_mean', 0.0)
        
        return {
            'loss': avg_loss,
            'dice': avg_dice,
            'metrics': metrics
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        # Progress bar
        pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch + 1}/Val')
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(pbar):
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)
                
                # Forward pass
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                # Update metrics
                total_loss += loss.item()
                self.val_metrics.update(outputs, masks)
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate epoch metrics
        avg_loss = total_loss / num_batches
        metrics = self.val_metrics.get_metrics()
        avg_dice = metrics.get('dice_mean', 0.0)
        
        return {
            'loss': avg_loss,
            'dice': avg_dice,
            'metrics': metrics
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_dice': self.best_val_dice,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.config['save_dir'], 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config['save_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved new best model with Dice: {self.best_val_dice:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint and return starting epoch"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_dice = checkpoint.get('best_val_dice', 0.0)
        self.train_history = checkpoint.get('train_history', {'loss': [], 'dice': []})
        self.val_history = checkpoint.get('val_history', {'loss': [], 'dice': []})
        
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        return start_epoch
    
    def train(self, resume_from: Optional[str] = None) -> Dict[str, list]:
        """Main training loop"""
        print("=== Starting Training ===")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Max epochs: {self.config['max_epochs']}")
        
        # Resume training if requested
        start_epoch = 0
        if resume_from and os.path.exists(resume_from):
            start_epoch = self.load_checkpoint(resume_from)
        
        # Training loop
        for epoch in range(start_epoch, self.config['max_epochs']):
            self.current_epoch = epoch
            
            # Apply warmup learning rate
            self._warmup_lr(epoch)
            
            print(f"\nEpoch {epoch + 1}/{self.config['max_epochs']}")
            print(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Train
            train_results = self.train_epoch()
            
            # Validate
            val_results = self.validate_epoch()
            
            # Update history
            self.train_history['loss'].append(train_results['loss'])
            self.train_history['dice'].append(train_results['dice'])
            self.val_history['loss'].append(val_results['loss'])
            self.val_history['dice'].append(val_results['dice'])
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/Train', train_results['loss'], epoch)
            self.writer.add_scalar('Loss/Validation', val_results['loss'], epoch)
            self.writer.add_scalar('Dice/Train', train_results['dice'], epoch)
            self.writer.add_scalar('Dice/Validation', val_results['dice'], epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Learning rate scheduling
            self.scheduler.step(val_results['dice'])
            
            # Check for best model
            is_best = val_results['dice'] > self.best_val_dice
            if is_best:
                self.best_val_dice = val_results['dice']
            
            # Save checkpoint
            if not self.config['save_best_only'] or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Print epoch summary
            print(f"Train Loss: {train_results['loss']:.4f}, Dice: {train_results['dice']:.4f}")
            print(f"Val Loss: {val_results['loss']:.4f}, Dice: {val_results['dice']:.4f}")
            if is_best:
                print("*** New best model! ***")
            
            # Early stopping
            if self.early_stopping(val_results['dice'], epoch):
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break
        
        # Close tensorboard writer
        self.writer.close()
        
        # Save final training history
        history_path = os.path.join(self.config['save_dir'], 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump({
                'train_history': self.train_history,
                'val_history': self.val_history,
                'best_val_dice': self.best_val_dice,
                'total_epochs': self.current_epoch + 1
            }, f, indent=2)
        
        print(f"\nTraining completed!")
        print(f"Best validation Dice: {self.best_val_dice:.4f}")
        print(f"Model saved to: {self.config['save_dir']}")
        
        return {
            'train_history': self.train_history,
            'val_history': self.val_history
        }


def create_trainer(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: Optional[Dict] = None
) -> CardiacSegmentationTrainer:
    """
    Factory function to create trainer
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    return CardiacSegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config
    )


if __name__ == "__main__":
    # Example usage
    from ..models.unet import create_unet_model
    from ..data.acdc_dataset import create_data_loaders
    
    # Create model and data loaders (placeholder paths)
    model = create_unet_model('standard')
    # train_loader, val_loader = create_data_loaders('path/to/data')
    
    # Training configuration
    config = {
        'learning_rate': 1e-3,
        'max_epochs': 100,
        'patience': 15,
        'save_dir': 'checkpoints',
        'log_dir': 'logs'
    }
    
    # Create trainer
    # trainer = create_trainer(model, train_loader, val_loader, config)
    
    # Start training
    # history = trainer.train()
    
    print("Trainer setup complete. Ready for training when data is available.")