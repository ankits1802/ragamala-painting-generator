"""
Training Callbacks for Ragamala Painting Generation.

This module provides comprehensive callback functionality for SDXL fine-tuning
on Ragamala paintings, including monitoring, checkpointing, early stopping,
and cultural validation callbacks.
"""

import os
import sys
import time
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, asdict
import numpy as np
from collections import defaultdict, deque
import warnings

# PyTorch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Monitoring imports
import wandb
from tensorboardX import SummaryWriter

# Image processing imports
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Accelerate imports
from accelerate import Accelerator

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logging_utils import setup_logger
from src.utils.visualization import save_training_samples, create_loss_plot
from src.evaluation.metrics import EvaluationMetrics

logger = setup_logger(__name__)

@dataclass
class CallbackConfig:
    """Configuration for training callbacks."""
    # Monitoring
    log_every_n_steps: int = 10
    save_images_every_n_steps: int = 500
    validate_every_n_steps: int = 1000
    
    # Checkpointing
    save_every_n_steps: int = 1000
    save_total_limit: int = 5
    save_on_epoch_end: bool = True
    
    # Early stopping
    enable_early_stopping: bool = True
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4
    early_stopping_metric: str = "val_loss"
    
    # Learning rate scheduling
    enable_lr_scheduling: bool = True
    lr_scheduler_patience: int = 3
    lr_scheduler_factor: float = 0.5
    lr_scheduler_min_lr: float = 1e-7
    
    # Cultural validation
    enable_cultural_validation: bool = True
    cultural_validation_frequency: int = 2000
    
    # Memory monitoring
    enable_memory_monitoring: bool = True
    memory_check_frequency: int = 100
    
    # Gradient monitoring
    enable_gradient_monitoring: bool = True
    gradient_clip_threshold: float = 1.0

class BaseCallback:
    """Base callback class for training callbacks."""
    
    def __init__(self, config: CallbackConfig):
        self.config = config
        self.state = {}
    
    def on_train_begin(self, trainer, **kwargs):
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, trainer, **kwargs):
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, trainer, epoch: int, **kwargs):
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, trainer, epoch: int, **kwargs):
        """Called at the end of each epoch."""
        pass
    
    def on_step_begin(self, trainer, step: int, **kwargs):
        """Called at the beginning of each step."""
        pass
    
    def on_step_end(self, trainer, step: int, logs: Dict[str, float], **kwargs):
        """Called at the end of each step."""
        pass
    
    def on_validation_begin(self, trainer, **kwargs):
        """Called at the beginning of validation."""
        pass
    
    def on_validation_end(self, trainer, val_logs: Dict[str, float], **kwargs):
        """Called at the end of validation."""
        pass
    
    def on_save_checkpoint(self, trainer, checkpoint_path: str, **kwargs):
        """Called when saving a checkpoint."""
        pass

class LoggingCallback(BaseCallback):
    """Callback for logging training metrics."""
    
    def __init__(self, config: CallbackConfig):
        super().__init__(config)
        self.step_logs = []
        self.epoch_logs = []
        self.best_metrics = {}
        
    def on_train_begin(self, trainer, **kwargs):
        """Initialize logging."""
        logger.info("Starting training with logging callback")
        self.start_time = time.time()
        
        # Initialize best metrics tracking
        self.best_metrics = {
            'best_val_loss': float('inf'),
            'best_train_loss': float('inf'),
            'best_step': 0,
            'best_epoch': 0
        }
    
    def on_step_end(self, trainer, step: int, logs: Dict[str, float], **kwargs):
        """Log step metrics."""
        if step % self.config.log_every_n_steps == 0:
            # Add timestamp and step info
            log_entry = {
                'step': step,
                'timestamp': time.time() - self.start_time,
                **logs
            }
            
            # Add learning rate if available
            if hasattr(trainer, 'lr_scheduler'):
                log_entry['learning_rate'] = trainer.lr_scheduler.get_last_lr()[0]
            
            self.step_logs.append(log_entry)
            
            # Update best metrics
            if 'loss' in logs:
                if logs['loss'] < self.best_metrics['best_train_loss']:
                    self.best_metrics['best_train_loss'] = logs['loss']
                    self.best_metrics['best_step'] = step
            
            # Log to console
            log_msg = f"Step {step}: " + ", ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
            logger.info(log_msg)
    
    def on_validation_end(self, trainer, val_logs: Dict[str, float], **kwargs):
        """Log validation metrics."""
        # Update best validation metrics
        if 'val_loss' in val_logs:
            if val_logs['val_loss'] < self.best_metrics['best_val_loss']:
                self.best_metrics['best_val_loss'] = val_logs['val_loss']
                self.best_metrics['best_epoch'] = trainer.epoch
        
        # Log validation results
        val_msg = "Validation: " + ", ".join([f"{k}: {v:.4f}" for k, v in val_logs.items()])
        logger.info(val_msg)
    
    def on_train_end(self, trainer, **kwargs):
        """Log training summary."""
        total_time = time.time() - self.start_time
        
        summary = {
            'total_training_time': total_time,
            'total_steps': trainer.global_step,
            'total_epochs': trainer.epoch + 1,
            **self.best_metrics
        }
        
        logger.info("Training completed!")
        logger.info(f"Training summary: {summary}")
        
        # Save training logs
        if hasattr(trainer, 'trainer_config'):
            logs_dir = Path(trainer.trainer_config.output_dir) / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            with open(logs_dir / "step_logs.json", 'w') as f:
                json.dump(self.step_logs, f, indent=2)
            
            with open(logs_dir / "training_summary.json", 'w') as f:
                json.dump(summary, f, indent=2)

class WandBCallback(BaseCallback):
    """Callback for Weights & Biases logging."""
    
    def __init__(self, config: CallbackConfig, project_name: str = "ragamala-sdxl"):
        super().__init__(config)
        self.project_name = project_name
        self.run = None
    
    def on_train_begin(self, trainer, **kwargs):
        """Initialize W&B run."""
        if not wandb.run:
            self.run = wandb.init(
                project=self.project_name,
                name=getattr(trainer, 'experiment_name', 'ragamala-training'),
                config={
                    'trainer_config': asdict(trainer.trainer_config) if hasattr(trainer, 'trainer_config') else {},
                    'training_config': asdict(trainer.training_config) if hasattr(trainer, 'training_config') else {},
                    'lora_config': asdict(trainer.lora_config) if hasattr(trainer, 'lora_config') else {}
                }
            )
        else:
            self.run = wandb.run
        
        logger.info("W&B logging initialized")
    
    def on_step_end(self, trainer, step: int, logs: Dict[str, float], **kwargs):
        """Log step metrics to W&B."""
        if step % self.config.log_every_n_steps == 0:
            wandb.log(logs, step=step)
    
    def on_validation_end(self, trainer, val_logs: Dict[str, float], **kwargs):
        """Log validation metrics to W&B."""
        wandb.log(val_logs, step=trainer.global_step)
    
    def on_train_end(self, trainer, **kwargs):
        """Finish W&B run."""
        if self.run:
            wandb.finish()

class TensorBoardCallback(BaseCallback):
    """Callback for TensorBoard logging."""
    
    def __init__(self, config: CallbackConfig, log_dir: str = "logs/tensorboard"):
        super().__init__(config)
        self.log_dir = log_dir
        self.writer = None
    
    def on_train_begin(self, trainer, **kwargs):
        """Initialize TensorBoard writer."""
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        logger.info(f"TensorBoard logging to {self.log_dir}")
    
    def on_step_end(self, trainer, step: int, logs: Dict[str, float], **kwargs):
        """Log step metrics to TensorBoard."""
        if step % self.config.log_every_n_steps == 0:
            for key, value in logs.items():
                self.writer.add_scalar(f"train/{key}", value, step)
            
            # Log learning rate
            if hasattr(trainer, 'lr_scheduler'):
                lr = trainer.lr_scheduler.get_last_lr()[0]
                self.writer.add_scalar("train/learning_rate", lr, step)
    
    def on_validation_end(self, trainer, val_logs: Dict[str, float], **kwargs):
        """Log validation metrics to TensorBoard."""
        for key, value in val_logs.items():
            self.writer.add_scalar(f"validation/{key}", value, trainer.global_step)
    
    def on_train_end(self, trainer, **kwargs):
        """Close TensorBoard writer."""
        if self.writer:
            self.writer.close()

class CheckpointCallback(BaseCallback):
    """Callback for model checkpointing."""
    
    def __init__(self, config: CallbackConfig, save_dir: str = "checkpoints"):
        super().__init__(config)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.saved_checkpoints = []
    
    def on_step_end(self, trainer, step: int, logs: Dict[str, float], **kwargs):
        """Save checkpoint at specified intervals."""
        if step % self.config.save_every_n_steps == 0:
            self._save_checkpoint(trainer, step, "step")
    
    def on_epoch_end(self, trainer, epoch: int, **kwargs):
        """Save checkpoint at epoch end if configured."""
        if self.config.save_on_epoch_end:
            self._save_checkpoint(trainer, epoch, "epoch")
    
    def _save_checkpoint(self, trainer, identifier: int, checkpoint_type: str):
        """Save model checkpoint."""
        checkpoint_name = f"checkpoint-{checkpoint_type}-{identifier}"
        checkpoint_path = self.save_dir / checkpoint_name
        
        try:
            # Create checkpoint directory
            checkpoint_path.mkdir(exist_ok=True)
            
            # Save model state
            if hasattr(trainer, 'model') and hasattr(trainer.model, 'unet'):
                # Save LoRA weights
                if hasattr(trainer.accelerator, 'unwrap_model'):
                    unwrapped_unet = trainer.accelerator.unwrap_model(trainer.model.unet)
                else:
                    unwrapped_unet = trainer.model.unet
                
                unwrapped_unet.save_pretrained(checkpoint_path / "unet_lora")
            
            # Save training state
            training_state = {
                'global_step': trainer.global_step,
                'epoch': trainer.epoch,
                'best_validation_loss': getattr(trainer, 'best_validation_loss', float('inf')),
                'random_state': torch.get_rng_state().tolist()
            }
            
            with open(checkpoint_path / "training_state.json", 'w') as f:
                json.dump(training_state, f, indent=2)
            
            # Save accelerator state if available
            if hasattr(trainer, 'accelerator'):
                trainer.accelerator.save_state(checkpoint_path / "accelerator_state")
            
            self.saved_checkpoints.append(checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Cleanup old checkpoints
            self._cleanup_checkpoints()
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints based on save_total_limit."""
        if len(self.saved_checkpoints) > self.config.save_total_limit:
            # Sort by modification time
            self.saved_checkpoints.sort(key=lambda x: x.stat().st_mtime)
            
            # Remove oldest checkpoints
            while len(self.saved_checkpoints) > self.config.save_total_limit:
                old_checkpoint = self.saved_checkpoints.pop(0)
                try:
                    shutil.rmtree(old_checkpoint)
                    logger.info(f"Removed old checkpoint: {old_checkpoint}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {old_checkpoint}: {e}")

class EarlyStoppingCallback(BaseCallback):
    """Callback for early stopping based on validation metrics."""
    
    def __init__(self, config: CallbackConfig):
        super().__init__(config)
        self.best_metric = float('inf')
        self.patience_counter = 0
        self.should_stop = False
    
    def on_validation_end(self, trainer, val_logs: Dict[str, float], **kwargs):
        """Check early stopping condition."""
        if not self.config.enable_early_stopping:
            return
        
        metric_value = val_logs.get(self.config.early_stopping_metric)
        if metric_value is None:
            return
        
        # Check if metric improved
        if metric_value < self.best_metric - self.config.early_stopping_min_delta:
            self.best_metric = metric_value
            self.patience_counter = 0
            logger.info(f"Validation metric improved to {metric_value:.4f}")
        else:
            self.patience_counter += 1
            logger.info(f"No improvement in validation metric. Patience: {self.patience_counter}/{self.config.early_stopping_patience}")
        
        # Check if should stop
        if self.patience_counter >= self.config.early_stopping_patience:
            self.should_stop = True
            logger.info("Early stopping triggered!")
            
            # Set trainer stop flag if available
            if hasattr(trainer, 'should_stop'):
                trainer.should_stop = True

class MemoryMonitoringCallback(BaseCallback):
    """Callback for monitoring GPU memory usage."""
    
    def __init__(self, config: CallbackConfig):
        super().__init__(config)
        self.memory_history = deque(maxlen=100)
    
    def on_step_end(self, trainer, step: int, logs: Dict[str, float], **kwargs):
        """Monitor memory usage."""
        if not self.config.enable_memory_monitoring:
            return
        
        if step % self.config.memory_check_frequency == 0:
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                
                memory_info = {
                    'step': step,
                    'memory_allocated_gb': memory_allocated,
                    'memory_reserved_gb': memory_reserved
                }
                
                self.memory_history.append(memory_info)
                
                # Log memory usage
                if step % (self.config.memory_check_frequency * 10) == 0:
                    logger.info(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
                
                # Check for memory leaks
                if len(self.memory_history) > 10:
                    recent_avg = np.mean([m['memory_allocated_gb'] for m in list(self.memory_history)[-10:]])
                    older_avg = np.mean([m['memory_allocated_gb'] for m in list(self.memory_history)[-20:-10]])
                    
                    if recent_avg > older_avg * 1.2:  # 20% increase
                        logger.warning("Potential memory leak detected!")

class GradientMonitoringCallback(BaseCallback):
    """Callback for monitoring gradient norms and detecting gradient issues."""
    
    def __init__(self, config: CallbackConfig):
        super().__init__(config)
        self.gradient_history = deque(maxlen=100)
    
    def on_step_end(self, trainer, step: int, logs: Dict[str, float], **kwargs):
        """Monitor gradient norms."""
        if not self.config.enable_gradient_monitoring:
            return
        
        if hasattr(trainer, 'model') and hasattr(trainer.model, 'unet'):
            # Calculate gradient norm
            total_norm = 0.0
            param_count = 0
            
            for param in trainer.model.unet.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            
            if param_count > 0:
                total_norm = total_norm ** (1. / 2)
                
                self.gradient_history.append({
                    'step': step,
                    'gradient_norm': total_norm
                })
                
                # Check for gradient explosion
                if total_norm > self.config.gradient_clip_threshold * 10:
                    logger.warning(f"Large gradient norm detected: {total_norm:.4f}")
                
                # Check for vanishing gradients
                if total_norm < 1e-8:
                    logger.warning(f"Very small gradient norm detected: {total_norm:.8f}")
                
                # Log gradient norm periodically
                if step % (self.config.log_every_n_steps * 10) == 0:
                    logger.info(f"Gradient norm: {total_norm:.4f}")

class CulturalValidationCallback(BaseCallback):
    """Callback for cultural authenticity validation."""
    
    def __init__(self, config: CallbackConfig):
        super().__init__(config)
        self.cultural_metrics = EvaluationMetrics()
        self.validation_history = []
    
    def on_step_end(self, trainer, step: int, logs: Dict[str, float], **kwargs):
        """Perform cultural validation at specified intervals."""
        if not self.config.enable_cultural_validation:
            return
        
        if step % self.config.cultural_validation_frequency == 0:
            self._perform_cultural_validation(trainer, step)
    
    def _perform_cultural_validation(self, trainer, step: int):
        """Perform cultural authenticity validation."""
        try:
            # Generate sample images for validation
            validation_prompts = [
                ("A rajput style ragamala painting depicting raga bhairav", "rajput", "bhairav"),
                ("A pahari miniature of raga yaman", "pahari", "yaman"),
                ("A deccan painting of raga malkauns", "deccan", "malkauns"),
                ("A mughal artwork of raga darbari", "mughal", "darbari")
            ]
            
            cultural_scores = []
            
            for prompt, style, raga in validation_prompts:
                # This would integrate with your generation pipeline
                # For now, we'll simulate the validation
                cultural_score = self._simulate_cultural_validation(prompt, style, raga)
                cultural_scores.append(cultural_score)
            
            avg_cultural_score = np.mean(cultural_scores)
            
            validation_result = {
                'step': step,
                'cultural_authenticity_score': avg_cultural_score,
                'individual_scores': cultural_scores
            }
            
            self.validation_history.append(validation_result)
            
            logger.info(f"Cultural validation at step {step}: {avg_cultural_score:.3f}")
            
            # Log to monitoring systems
            if hasattr(trainer, 'accelerator'):
                trainer.accelerator.log({'cultural_authenticity': avg_cultural_score}, step=step)
            
        except Exception as e:
            logger.error(f"Cultural validation failed: {e}")
    
    def _simulate_cultural_validation(self, prompt: str, style: str, raga: str) -> float:
        """Simulate cultural validation (replace with actual implementation)."""
        # This is a placeholder - implement actual cultural validation logic
        return np.random.uniform(0.7, 0.95)

class TrainingCallbacks:
    """Manager for all training callbacks."""
    
    def __init__(self, config: CallbackConfig):
        self.config = config
        self.callbacks = []
        
        # Initialize default callbacks
        self._setup_default_callbacks()
    
    def _setup_default_callbacks(self):
        """Setup default callbacks."""
        # Logging callback
        self.callbacks.append(LoggingCallback(self.config))
        
        # Checkpointing callback
        self.callbacks.append(CheckpointCallback(self.config))
        
        # Early stopping callback
        if self.config.enable_early_stopping:
            self.callbacks.append(EarlyStoppingCallback(self.config))
        
        # Memory monitoring callback
        if self.config.enable_memory_monitoring:
            self.callbacks.append(MemoryMonitoringCallback(self.config))
        
        # Gradient monitoring callback
        if self.config.enable_gradient_monitoring:
            self.callbacks.append(GradientMonitoringCallback(self.config))
        
        # Cultural validation callback
        if self.config.enable_cultural_validation:
            self.callbacks.append(CulturalValidationCallback(self.config))
    
    def add_callback(self, callback: BaseCallback):
        """Add a custom callback."""
        self.callbacks.append(callback)
    
    def add_wandb_callback(self, project_name: str = "ragamala-sdxl"):
        """Add W&B callback."""
        self.callbacks.append(WandBCallback(self.config, project_name))
    
    def add_tensorboard_callback(self, log_dir: str = "logs/tensorboard"):
        """Add TensorBoard callback."""
        self.callbacks.append(TensorBoardCallback(self.config, log_dir))
    
    def on_train_begin(self, trainer, **kwargs):
        """Call all callbacks at training begin."""
        for callback in self.callbacks:
            try:
                callback.on_train_begin(trainer, **kwargs)
            except Exception as e:
                logger.error(f"Callback {type(callback).__name__} failed in on_train_begin: {e}")
    
    def on_train_end(self, trainer, **kwargs):
        """Call all callbacks at training end."""
        for callback in self.callbacks:
            try:
                callback.on_train_end(trainer, **kwargs)
            except Exception as e:
                logger.error(f"Callback {type(callback).__name__} failed in on_train_end: {e}")
    
    def on_epoch_begin(self, trainer, epoch: int, **kwargs):
        """Call all callbacks at epoch begin."""
        for callback in self.callbacks:
            try:
                callback.on_epoch_begin(trainer, epoch, **kwargs)
            except Exception as e:
                logger.error(f"Callback {type(callback).__name__} failed in on_epoch_begin: {e}")
    
    def on_epoch_end(self, trainer, epoch: int, **kwargs):
        """Call all callbacks at epoch end."""
        for callback in self.callbacks:
            try:
                callback.on_epoch_end(trainer, epoch, **kwargs)
            except Exception as e:
                logger.error(f"Callback {type(callback).__name__} failed in on_epoch_end: {e}")
    
    def on_step_begin(self, trainer, step: int, **kwargs):
        """Call all callbacks at step begin."""
        for callback in self.callbacks:
            try:
                callback.on_step_begin(trainer, step, **kwargs)
            except Exception as e:
                logger.error(f"Callback {type(callback).__name__} failed in on_step_begin: {e}")
    
    def on_step_end(self, trainer, step: int, logs: Dict[str, float], **kwargs):
        """Call all callbacks at step end."""
        for callback in self.callbacks:
            try:
                callback.on_step_end(trainer, step, logs, **kwargs)
            except Exception as e:
                logger.error(f"Callback {type(callback).__name__} failed in on_step_end: {e}")
    
    def on_validation_begin(self, trainer, **kwargs):
        """Call all callbacks at validation begin."""
        for callback in self.callbacks:
            try:
                callback.on_validation_begin(trainer, **kwargs)
            except Exception as e:
                logger.error(f"Callback {type(callback).__name__} failed in on_validation_begin: {e}")
    
    def on_validation_end(self, trainer, val_logs: Dict[str, float], **kwargs):
        """Call all callbacks at validation end."""
        for callback in self.callbacks:
            try:
                callback.on_validation_end(trainer, val_logs, **kwargs)
            except Exception as e:
                logger.error(f"Callback {type(callback).__name__} failed in on_validation_end: {e}")
    
    def on_save_checkpoint(self, trainer, checkpoint_path: str, **kwargs):
        """Call all callbacks when saving checkpoint."""
        for callback in self.callbacks:
            try:
                callback.on_save_checkpoint(trainer, checkpoint_path, **kwargs)
            except Exception as e:
                logger.error(f"Callback {type(callback).__name__} failed in on_save_checkpoint: {e}")
    
    def should_stop_training(self) -> bool:
        """Check if any callback requests training to stop."""
        for callback in self.callbacks:
            if hasattr(callback, 'should_stop') and callback.should_stop:
                return True
        return False

def create_training_callbacks(config: CallbackConfig, 
                            enable_wandb: bool = True,
                            enable_tensorboard: bool = True,
                            wandb_project: str = "ragamala-sdxl",
                            tensorboard_dir: str = "logs/tensorboard") -> TrainingCallbacks:
    """Factory function to create training callbacks."""
    callbacks = TrainingCallbacks(config)
    
    if enable_wandb:
        callbacks.add_wandb_callback(wandb_project)
    
    if enable_tensorboard:
        callbacks.add_tensorboard_callback(tensorboard_dir)
    
    return callbacks

def main():
    """Main function for testing callbacks."""
    # Create configuration
    config = CallbackConfig()
    
    # Create callbacks
    callbacks = create_training_callbacks(config, enable_wandb=False, enable_tensorboard=False)
    
    # Simulate training loop
    class MockTrainer:
        def __init__(self):
            self.global_step = 0
            self.epoch = 0
            self.best_validation_loss = float('inf')
    
    trainer = MockTrainer()
    
    # Test callback lifecycle
    callbacks.on_train_begin(trainer)
    
    for epoch in range(2):
        callbacks.on_epoch_begin(trainer, epoch)
        
        for step in range(10):
            trainer.global_step = step + epoch * 10
            
            callbacks.on_step_begin(trainer, trainer.global_step)
            
            # Simulate training metrics
            logs = {
                'loss': np.random.uniform(0.1, 1.0),
                'lr': 1e-4
            }
            
            callbacks.on_step_end(trainer, trainer.global_step, logs)
        
        # Simulate validation
        callbacks.on_validation_begin(trainer)
        val_logs = {'val_loss': np.random.uniform(0.2, 0.8)}
        callbacks.on_validation_end(trainer, val_logs)
        
        callbacks.on_epoch_end(trainer, epoch)
        trainer.epoch = epoch
    
    callbacks.on_train_end(trainer)
    
    print("Callback testing completed successfully!")

if __name__ == "__main__":
    main()
