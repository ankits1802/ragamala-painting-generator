"""
Training Utilities for Ragamala Painting Generation.

This module provides comprehensive utility functions for SDXL fine-tuning
on Ragamala paintings, including model utilities, data utilities, optimization helpers,
and cultural-specific utilities.
"""

import os
import sys
import time
import json
import pickle
import random
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, asdict
import numpy as np
from collections import defaultdict, OrderedDict
import warnings

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

# Diffusers imports
from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler
)
from diffusers.utils import is_xformers_available

# Transformers imports
from transformers import CLIPTokenizer, CLIPTextModel

# PEFT imports
from peft import PeftModel, LoraConfig

# Accelerate imports
from accelerate import Accelerator
from accelerate.utils import set_seed

# Image processing imports
from PIL import Image, ImageDraw, ImageFont
import cv2

# Scientific computing imports
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy import stats

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

@dataclass
class ModelInfo:
    """Information about a trained model."""
    model_name: str
    model_path: str
    lora_path: Optional[str]
    training_config: Dict[str, Any]
    performance_metrics: Dict[str, float]
    cultural_scores: Dict[str, float]
    creation_date: str
    model_size_mb: float

class ModelManager:
    """Manager for model loading, saving, and versioning."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_registry = self._load_model_registry()
    
    def _load_model_registry(self) -> Dict[str, ModelInfo]:
        """Load model registry from disk."""
        registry_file = self.models_dir / "model_registry.json"
        
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    registry_data = json.load(f)
                
                registry = {}
                for name, data in registry_data.items():
                    registry[name] = ModelInfo(**data)
                
                return registry
            except Exception as e:
                logger.error(f"Error loading model registry: {e}")
        
        return {}
    
    def _save_model_registry(self):
        """Save model registry to disk."""
        registry_file = self.models_dir / "model_registry.json"
        
        try:
            registry_data = {}
            for name, info in self.model_registry.items():
                registry_data[name] = asdict(info)
            
            with open(registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving model registry: {e}")
    
    def register_model(self, 
                      model_name: str,
                      model_path: str,
                      lora_path: Optional[str] = None,
                      training_config: Dict[str, Any] = None,
                      performance_metrics: Dict[str, float] = None,
                      cultural_scores: Dict[str, float] = None) -> ModelInfo:
        """Register a new model in the registry."""
        
        # Calculate model size
        model_size = self._calculate_model_size(model_path, lora_path)
        
        model_info = ModelInfo(
            model_name=model_name,
            model_path=model_path,
            lora_path=lora_path,
            training_config=training_config or {},
            performance_metrics=performance_metrics or {},
            cultural_scores=cultural_scores or {},
            creation_date=time.strftime("%Y-%m-%d %H:%M:%S"),
            model_size_mb=model_size
        )
        
        self.model_registry[model_name] = model_info
        self._save_model_registry()
        
        logger.info(f"Registered model: {model_name}")
        return model_info
    
    def _calculate_model_size(self, model_path: str, lora_path: Optional[str] = None) -> float:
        """Calculate total model size in MB."""
        total_size = 0
        
        # Base model size
        if os.path.exists(model_path):
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
        
        # LoRA weights size
        if lora_path and os.path.exists(lora_path):
            for root, dirs, files in os.walk(lora_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a registered model."""
        return self.model_registry.get(model_name)
    
    def list_models(self) -> List[ModelInfo]:
        """List all registered models."""
        return list(self.model_registry.values())
    
    def get_best_model(self, metric: str = "val_loss") -> Optional[ModelInfo]:
        """Get the best model based on a specific metric."""
        best_model = None
        best_score = float('inf') if 'loss' in metric else float('-inf')
        
        for model_info in self.model_registry.values():
            if metric in model_info.performance_metrics:
                score = model_info.performance_metrics[metric]
                
                if ('loss' in metric and score < best_score) or \
                   ('loss' not in metric and score > best_score):
                    best_score = score
                    best_model = model_info
        
        return best_model

class DatasetUtils:
    """Utilities for dataset management and analysis."""
    
    @staticmethod
    def analyze_dataset_distribution(dataset: Dataset) -> Dict[str, Any]:
        """Analyze the distribution of ragas and styles in the dataset."""
        raga_counts = defaultdict(int)
        style_counts = defaultdict(int)
        quality_scores = []
        
        for i in range(len(dataset)):
            try:
                sample = dataset[i]
                
                if 'raga' in sample:
                    raga_counts[sample['raga']] += 1
                
                if 'style' in sample:
                    style_counts[sample['style']] += 1
                
                if 'quality_score' in sample:
                    quality_scores.append(sample['quality_score'].item())
                    
            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                continue
        
        analysis = {
            'total_samples': len(dataset),
            'raga_distribution': dict(raga_counts),
            'style_distribution': dict(style_counts),
            'quality_statistics': {
                'mean': np.mean(quality_scores) if quality_scores else 0,
                'std': np.std(quality_scores) if quality_scores else 0,
                'min': np.min(quality_scores) if quality_scores else 0,
                'max': np.max(quality_scores) if quality_scores else 0
            }
        }
        
        return analysis
    
    @staticmethod
    def create_balanced_sampler_weights(dataset: Dataset) -> torch.Tensor:
        """Create sampling weights for balanced training."""
        raga_counts = defaultdict(int)
        style_counts = defaultdict(int)
        
        # Count occurrences
        for i in range(len(dataset)):
            try:
                sample = dataset[i]
                raga_counts[sample.get('raga', 'unknown')] += 1
                style_counts[sample.get('style', 'unknown')] += 1
            except:
                continue
        
        # Calculate weights
        weights = torch.zeros(len(dataset))
        
        for i in range(len(dataset)):
            try:
                sample = dataset[i]
                raga = sample.get('raga', 'unknown')
                style = sample.get('style', 'unknown')
                
                # Inverse frequency weighting
                raga_weight = 1.0 / raga_counts[raga]
                style_weight = 1.0 / style_counts[style]
                
                weights[i] = (raga_weight + style_weight) / 2
                
            except:
                weights[i] = 1.0
        
        return weights
    
    @staticmethod
    def validate_dataset_integrity(dataset: Dataset) -> Dict[str, Any]:
        """Validate dataset integrity and report issues."""
        issues = []
        valid_samples = 0
        
        required_keys = ['image', 'text', 'raga', 'style']
        
        for i in range(len(dataset)):
            try:
                sample = dataset[i]
                
                # Check required keys
                missing_keys = [key for key in required_keys if key not in sample]
                if missing_keys:
                    issues.append(f"Sample {i}: Missing keys {missing_keys}")
                    continue
                
                # Check image tensor
                if not isinstance(sample['image'], torch.Tensor):
                    issues.append(f"Sample {i}: Image is not a tensor")
                    continue
                
                if sample['image'].dim() != 3 or sample['image'].shape[0] != 3:
                    issues.append(f"Sample {i}: Invalid image shape {sample['image'].shape}")
                    continue
                
                # Check text
                if not isinstance(sample['text'], str) or len(sample['text']) == 0:
                    issues.append(f"Sample {i}: Invalid text")
                    continue
                
                valid_samples += 1
                
            except Exception as e:
                issues.append(f"Sample {i}: Error loading - {e}")
        
        return {
            'total_samples': len(dataset),
            'valid_samples': valid_samples,
            'invalid_samples': len(dataset) - valid_samples,
            'issues': issues[:10],  # Limit to first 10 issues
            'total_issues': len(issues)
        }

class OptimizationUtils:
    """Utilities for optimization and training efficiency."""
    
    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """Count model parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params
        }
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current GPU memory usage."""
        if not torch.cuda.is_available():
            return {'gpu_available': False}
        
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        memory_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3
        
        return {
            'gpu_available': True,
            'memory_allocated_gb': memory_allocated,
            'memory_reserved_gb': memory_reserved,
            'memory_free_gb': memory_free,
            'memory_utilization': memory_allocated / (memory_allocated + memory_free) if (memory_allocated + memory_free) > 0 else 0
        }
    
    @staticmethod
    def optimize_model_for_inference(model: nn.Module) -> nn.Module:
        """Optimize model for inference."""
        model.eval()
        
        # Enable optimizations
        if hasattr(model, 'enable_xformers_memory_efficient_attention'):
            try:
                model.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xFormers memory efficient attention")
            except Exception as e:
                logger.warning(f"Could not enable xFormers: {e}")
        
        # Compile model if available (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode='reduce-overhead')
                logger.info("Model compiled for inference")
            except Exception as e:
                logger.warning(f"Could not compile model: {e}")
        
        return model
    
    @staticmethod
    def calculate_gradient_norm(model: nn.Module) -> float:
        """Calculate the norm of gradients."""
        total_norm = 0.0
        param_count = 0
        
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
        
        return total_norm
    
    @staticmethod
    def find_optimal_batch_size(model: nn.Module, 
                               sample_input: torch.Tensor,
                               max_batch_size: int = 32) -> int:
        """Find optimal batch size for training."""
        model.train()
        device = next(model.parameters()).device
        
        optimal_batch_size = 1
        
        for batch_size in [1, 2, 4, 8, 16, 32]:
            if batch_size > max_batch_size:
                break
            
            try:
                # Create batch
                batch_input = sample_input.repeat(batch_size, 1, 1, 1).to(device)
                
                # Forward pass
                with torch.cuda.amp.autocast():
                    output = model(batch_input)
                
                # Backward pass
                if isinstance(output, dict):
                    loss = output.get('loss', output.get('sample', output[list(output.keys())[0]]))
                else:
                    loss = output
                
                if hasattr(loss, 'mean'):
                    loss = loss.mean()
                
                loss.backward()
                
                # Clear gradients
                model.zero_grad()
                
                optimal_batch_size = batch_size
                logger.info(f"Batch size {batch_size} successful")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.info(f"Batch size {batch_size} caused OOM")
                    break
                else:
                    raise e
        
        return optimal_batch_size

class CulturalUtils:
    """Utilities for cultural aspects of Ragamala paintings."""
    
    @staticmethod
    def get_raga_characteristics() -> Dict[str, Dict[str, Any]]:
        """Get characteristics of different ragas."""
        return {
            'bhairav': {
                'time': 'dawn',
                'season': 'winter',
                'mood': 'devotional',
                'colors': ['white', 'saffron', 'gold'],
                'elements': ['temple', 'ascetic', 'peacocks'],
                'deity': 'shiva'
            },
            'yaman': {
                'time': 'evening',
                'season': 'spring',
                'mood': 'romantic',
                'colors': ['blue', 'white', 'pink'],
                'elements': ['garden', 'lovers', 'moon'],
                'deity': 'krishna'
            },
            'malkauns': {
                'time': 'midnight',
                'season': 'monsoon',
                'mood': 'meditative',
                'colors': ['deep_blue', 'purple', 'black'],
                'elements': ['river', 'meditation', 'stars'],
                'deity': 'shiva'
            },
            'darbari': {
                'time': 'late_evening',
                'season': 'autumn',
                'mood': 'regal',
                'colors': ['purple', 'gold', 'red'],
                'elements': ['court', 'throne', 'courtiers'],
                'deity': 'indra'
            }
        }
    
    @staticmethod
    def get_style_characteristics() -> Dict[str, Dict[str, Any]]:
        """Get characteristics of different painting styles."""
        return {
            'rajput': {
                'period': '16th-18th_century',
                'region': 'rajasthan',
                'characteristics': ['bold_colors', 'geometric_patterns', 'royal_themes'],
                'color_palette': ['red', 'gold', 'white', 'green'],
                'techniques': ['flat_perspective', 'decorative_borders']
            },
            'pahari': {
                'period': '17th-19th_century',
                'region': 'himalayan_foothills',
                'characteristics': ['soft_colors', 'natural_settings', 'romantic_themes'],
                'color_palette': ['soft_blue', 'green', 'pink', 'white'],
                'techniques': ['atmospheric_depth', 'delicate_brushwork']
            },
            'deccan': {
                'period': '16th-18th_century',
                'region': 'deccan_plateau',
                'characteristics': ['persian_influence', 'architectural_elements', 'formal_composition'],
                'color_palette': ['deep_blue', 'purple', 'gold', 'white'],
                'techniques': ['geometric_precision', 'detailed_architecture']
            },
            'mughal': {
                'period': '16th-18th_century',
                'region': 'northern_india',
                'characteristics': ['elaborate_details', 'court_scenes', 'naturalistic_portraiture'],
                'color_palette': ['rich_colors', 'gold', 'jewel_tones'],
                'techniques': ['fine_details', 'realistic_perspective']
            }
        }
    
    @staticmethod
    def validate_cultural_consistency(raga: str, 
                                    style: str, 
                                    time_of_day: str = None,
                                    season: str = None) -> Dict[str, Any]:
        """Validate cultural consistency of raga-style combinations."""
        raga_chars = CulturalUtils.get_raga_characteristics().get(raga.lower(), {})
        style_chars = CulturalUtils.get_style_characteristics().get(style.lower(), {})
        
        consistency_score = 1.0
        issues = []
        
        # Check time consistency
        if time_of_day and raga_chars.get('time'):
            if time_of_day != raga_chars['time']:
                consistency_score -= 0.2
                issues.append(f"Time mismatch: {raga} should be performed at {raga_chars['time']}, not {time_of_day}")
        
        # Check season consistency
        if season and raga_chars.get('season'):
            if season != raga_chars['season']:
                consistency_score -= 0.2
                issues.append(f"Season mismatch: {raga} is associated with {raga_chars['season']}, not {season}")
        
        # Check historical period compatibility
        if raga_chars and style_chars:
            raga_period = raga_chars.get('period', 'classical')
            style_period = style_chars.get('period', 'classical')
            
            # This is a simplified check - in reality, period compatibility is complex
            if 'period' in style_chars and style_period:
                consistency_score += 0.1  # Bonus for having period information
        
        return {
            'consistency_score': max(0.0, consistency_score),
            'issues': issues,
            'raga_characteristics': raga_chars,
            'style_characteristics': style_chars
        }

class VisualizationUtils:
    """Utilities for visualization and analysis."""
    
    @staticmethod
    def create_training_progress_plot(metrics_history: List[Dict[str, float]], 
                                    save_path: str = None) -> None:
        """Create training progress visualization."""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Extract metrics
            steps = [m.get('step', i) for i, m in enumerate(metrics_history)]
            train_losses = [m.get('loss', 0) for m in metrics_history]
            val_losses = [m.get('val_loss', 0) for m in metrics_history if 'val_loss' in m]
            val_steps = [m.get('step', i) for i, m in enumerate(metrics_history) if 'val_loss' in m]
            learning_rates = [m.get('learning_rate', 0) for m in metrics_history]
            
            # Training loss
            axes[0, 0].plot(steps, train_losses, label='Training Loss')
            if val_losses:
                axes[0, 0].plot(val_steps, val_losses, label='Validation Loss')
            axes[0, 0].set_xlabel('Steps')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training Progress')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Learning rate
            axes[0, 1].plot(steps, learning_rates)
            axes[0, 1].set_xlabel('Steps')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].set_title('Learning Rate Schedule')
            axes[0, 1].grid(True)
            
            # Loss distribution
            axes[1, 0].hist(train_losses, bins=30, alpha=0.7, label='Training Loss')
            if val_losses:
                axes[1, 0].hist(val_losses, bins=30, alpha=0.7, label='Validation Loss')
            axes[1, 0].set_xlabel('Loss Value')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Loss Distribution')
            axes[1, 0].legend()
            
            # Moving average
            if len(train_losses) > 10:
                window_size = min(50, len(train_losses) // 10)
                moving_avg = np.convolve(train_losses, np.ones(window_size)/window_size, mode='valid')
                moving_steps = steps[window_size-1:]
                axes[1, 1].plot(moving_steps, moving_avg)
                axes[1, 1].set_xlabel('Steps')
                axes[1, 1].set_ylabel('Moving Average Loss')
                axes[1, 1].set_title(f'Moving Average (window={window_size})')
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Training progress plot saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Error creating training plot: {e}")

class ConfigUtils:
    """Utilities for configuration management."""
    
    @staticmethod
    def save_training_config(config: Dict[str, Any], save_path: str):
        """Save training configuration to file."""
        try:
            with open(save_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            logger.info(f"Training config saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    @staticmethod
    def load_training_config(config_path: str) -> Dict[str, Any]:
        """Load training configuration from file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Training config loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    @staticmethod
    def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configuration dictionaries."""
        merged = {}
        for config in configs:
            merged.update(config)
        return merged
    
    @staticmethod
    def validate_config(config: Dict[str, Any], 
                       required_keys: List[str]) -> Tuple[bool, List[str]]:
        """Validate configuration has required keys."""
        missing_keys = [key for key in required_keys if key not in config]
        is_valid = len(missing_keys) == 0
        return is_valid, missing_keys

def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")

def get_device_info() -> Dict[str, Any]:
    """Get information about available devices."""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
        'device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else None
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version()
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info[f'device_{i}'] = {
                'name': props.name,
                'total_memory_gb': props.total_memory / 1024**3,
                'multi_processor_count': props.multi_processor_count
            }
    
    return info

def create_experiment_name(base_name: str = "ragamala", 
                          config: Dict[str, Any] = None) -> str:
    """Create a unique experiment name."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    if config:
        # Add key config parameters to name
        lr = config.get('learning_rate', 'unknown')
        batch_size = config.get('batch_size', 'unknown')
        lora_rank = config.get('lora_rank', 'unknown')
        
        experiment_name = f"{base_name}_lr{lr}_bs{batch_size}_r{lora_rank}_{timestamp}"
    else:
        experiment_name = f"{base_name}_{timestamp}"
    
    return experiment_name

def calculate_model_hash(model_path: str) -> str:
    """Calculate hash of model files for versioning."""
    hasher = hashlib.md5()
    
    if os.path.isfile(model_path):
        with open(model_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
    elif os.path.isdir(model_path):
        for root, dirs, files in os.walk(model_path):
            for file in sorted(files):
                file_path = os.path.join(root, file)
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hasher.update(chunk)
    
    return hasher.hexdigest()

def main():
    """Main function for testing utilities."""
    # Test device info
    device_info = get_device_info()
    print("Device Information:")
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    
    # Test random seed
    set_random_seed(42)
    
    # Test experiment name generation
    config = {
        'learning_rate': 1e-4,
        'batch_size': 4,
        'lora_rank': 64
    }
    experiment_name = create_experiment_name("test", config)
    print(f"\nExperiment name: {experiment_name}")
    
    # Test cultural validation
    validation_result = CulturalUtils.validate_cultural_consistency(
        raga="bhairav",
        style="rajput",
        time_of_day="dawn",
        season="winter"
    )
    print(f"\nCultural validation: {validation_result}")
    
    print("Training utilities testing completed successfully!")

if __name__ == "__main__":
    main()
