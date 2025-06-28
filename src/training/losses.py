"""
Custom Loss Functions for Ragamala Painting Generation.

This module provides comprehensive loss functions for SDXL fine-tuning
on Ragamala paintings, including cultural conditioning losses, perceptual losses,
and specialized losses for preserving artistic authenticity.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from pathlib import Path

# Computer vision imports
import torchvision.transforms as transforms
from torchvision.models import vgg16, vgg19
from torchvision.models.feature_extraction import create_feature_extractor

# CLIP imports
from transformers import CLIPModel, CLIPProcessor

# LPIPS imports
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    logging.warning("LPIPS not available. Install with: pip install lpips")

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

@dataclass
class LossConfig:
    """Configuration for loss functions."""
    # Main loss settings
    main_loss_type: str = "mse"
    main_loss_weight: float = 1.0
    
    # Perceptual loss
    enable_perceptual_loss: bool = True
    perceptual_loss_weight: float = 0.1
    perceptual_layers: List[str] = None
    
    # Style loss
    enable_style_loss: bool = True
    style_loss_weight: float = 0.05
    style_layers: List[str] = None
    
    # Cultural loss
    enable_cultural_loss: bool = True
    cultural_loss_weight: float = 0.1
    
    # CLIP loss
    enable_clip_loss: bool = True
    clip_loss_weight: float = 0.05
    clip_model_name: str = "openai/clip-vit-base-patch32"
    
    # LPIPS loss
    enable_lpips_loss: bool = False
    lpips_loss_weight: float = 0.1
    lpips_net: str = "alex"
    
    # SNR weighting
    enable_snr_weighting: bool = False
    snr_gamma: float = 5.0
    
    # Temporal consistency
    enable_temporal_consistency: bool = False
    temporal_loss_weight: float = 0.05
    
    def __post_init__(self):
        if self.perceptual_layers is None:
            self.perceptual_layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
        if self.style_layers is None:
            self.style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']

class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features."""
    
    def __init__(self, 
                 layers: List[str] = None,
                 weights: List[float] = None,
                 use_vgg19: bool = False):
        super().__init__()
        
        if layers is None:
            layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
        
        if weights is None:
            weights = [1.0] * len(layers)
        
        self.layers = layers
        self.weights = weights
        
        # Load VGG model
        if use_vgg19:
            vgg = vgg19(pretrained=True)
        else:
            vgg = vgg16(pretrained=True)
        
        # Create feature extractor
        self.feature_extractor = create_feature_extractor(
            vgg.features, return_nodes={
                '3': 'relu1_2',
                '8': 'relu2_2',
                '15': 'relu3_3',
                '22': 'relu4_3',
                '29': 'relu5_3' if use_vgg19 else 'relu4_3'
            }
        )
        
        # Freeze parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # Normalization for VGG
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input for VGG."""
        return (x - self.mean) / self.std
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss."""
        # Normalize inputs
        pred_norm = self.normalize(pred)
        target_norm = self.normalize(target)
        
        # Extract features
        pred_features = self.feature_extractor(pred_norm)
        target_features = self.feature_extractor(target_norm)
        
        # Compute loss
        loss = 0.0
        for layer, weight in zip(self.layers, self.weights):
            if layer in pred_features and layer in target_features:
                pred_feat = pred_features[layer]
                target_feat = target_features[layer]
                loss += weight * F.mse_loss(pred_feat, target_feat)
        
        return loss

class StyleLoss(nn.Module):
    """Style loss using Gram matrices."""
    
    def __init__(self, 
                 layers: List[str] = None,
                 weights: List[float] = None):
        super().__init__()
        
        if layers is None:
            layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
        
        if weights is None:
            weights = [1.0] * len(layers)
        
        self.layers = layers
        self.weights = weights
        
        # Load VGG model
        vgg = vgg19(pretrained=True)
        
        # Create feature extractor
        self.feature_extractor = create_feature_extractor(
            vgg.features, return_nodes={
                '1': 'relu1_1',
                '6': 'relu2_1',
                '11': 'relu3_1',
                '20': 'relu4_1',
                '29': 'relu5_1'
            }
        )
        
        # Freeze parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # Normalization for VGG
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input for VGG."""
        return (x - self.mean) / self.std
    
    def gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix."""
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute style loss."""
        # Normalize inputs
        pred_norm = self.normalize(pred)
        target_norm = self.normalize(target)
        
        # Extract features
        pred_features = self.feature_extractor(pred_norm)
        target_features = self.feature_extractor(target_norm)
        
        # Compute loss
        loss = 0.0
        for layer, weight in zip(self.layers, self.weights):
            if layer in pred_features and layer in target_features:
                pred_gram = self.gram_matrix(pred_features[layer])
                target_gram = self.gram_matrix(target_features[layer])
                loss += weight * F.mse_loss(pred_gram, target_gram)
        
        return loss

class CLIPLoss(nn.Module):
    """CLIP-based semantic loss."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)
        
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
    
    def forward(self, 
                pred: torch.Tensor, 
                target: torch.Tensor,
                text_prompts: Optional[List[str]] = None) -> torch.Tensor:
        """Compute CLIP loss."""
        device = pred.device
        
        # Convert tensors to PIL images for CLIP processing
        pred_images = self._tensor_to_pil(pred)
        target_images = self._tensor_to_pil(target)
        
        # Process images
        pred_inputs = self.clip_processor(images=pred_images, return_tensors="pt").to(device)
        target_inputs = self.clip_processor(images=target_images, return_tensors="pt").to(device)
        
        # Get image features
        pred_features = self.clip_model.get_image_features(**pred_inputs)
        target_features = self.clip_model.get_image_features(**target_inputs)
        
        # Normalize features
        pred_features = F.normalize(pred_features, dim=-1)
        target_features = F.normalize(target_features, dim=-1)
        
        # Compute cosine similarity loss
        similarity = F.cosine_similarity(pred_features, target_features, dim=-1)
        loss = 1.0 - similarity.mean()
        
        # Add text-image alignment loss if prompts provided
        if text_prompts:
            text_inputs = self.clip_processor(text=text_prompts, return_tensors="pt", padding=True).to(device)
            text_features = self.clip_model.get_text_features(**text_inputs)
            text_features = F.normalize(text_features, dim=-1)
            
            # Text-image similarity for predictions
            text_image_similarity = F.cosine_similarity(pred_features, text_features, dim=-1)
            text_loss = 1.0 - text_image_similarity.mean()
            
            loss = loss + 0.5 * text_loss
        
        return loss
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> List:
        """Convert tensor to PIL images."""
        from PIL import Image
        
        # Denormalize if needed (assuming [-1, 1] range)
        tensor = (tensor + 1.0) / 2.0
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to PIL
        images = []
        for i in range(tensor.shape[0]):
            img_tensor = tensor[i].cpu()
            img_array = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            images.append(Image.fromarray(img_array))
        
        return images

class CulturalLoss(nn.Module):
    """Cultural authenticity loss for Ragamala paintings."""
    
    def __init__(self):
        super().__init__()
        
        # Cultural color palettes
        self.cultural_palettes = {
            'rajput': torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.84, 0.0], [1.0, 1.0, 1.0], [0.0, 0.5, 0.0]]),  # Red, Gold, White, Green
            'pahari': torch.tensor([[0.53, 0.81, 0.92], [0.0, 0.5, 0.0], [1.0, 0.75, 0.8], [1.0, 1.0, 1.0]]),  # Sky blue, Green, Pink, White
            'deccan': torch.tensor([[0.0, 0.0, 0.55], [0.5, 0.0, 0.5], [1.0, 0.84, 0.0], [1.0, 1.0, 1.0]]),  # Deep blue, Purple, Gold, White
            'mughal': torch.tensor([[1.0, 0.84, 0.0], [1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.5, 0.0, 0.5]])  # Gold, Red, Green, Purple
        }
        
        # Register palettes as buffers
        for style, palette in self.cultural_palettes.items():
            self.register_buffer(f'{style}_palette', palette)
    
    def extract_dominant_colors(self, image: torch.Tensor, k: int = 4) -> torch.Tensor:
        """Extract dominant colors using k-means clustering."""
        b, c, h, w = image.shape
        
        # Reshape image to (batch_size, num_pixels, channels)
        pixels = image.permute(0, 2, 3, 1).reshape(b, -1, c)
        
        # Simple color extraction (using random sampling for efficiency)
        num_samples = min(1000, h * w)
        indices = torch.randperm(h * w)[:num_samples]
        sampled_pixels = pixels[:, indices, :]
        
        # Get mean colors as approximation
        dominant_colors = sampled_pixels.mean(dim=1)  # (batch_size, channels)
        
        return dominant_colors
    
    def color_palette_loss(self, 
                          pred: torch.Tensor, 
                          style: str) -> torch.Tensor:
        """Compute color palette consistency loss."""
        if style not in self.cultural_palettes:
            return torch.tensor(0.0, device=pred.device)
        
        # Extract dominant colors from prediction
        pred_colors = self.extract_dominant_colors(pred)
        
        # Get target palette
        target_palette = getattr(self, f'{style}_palette')
        
        # Compute minimum distance to palette colors
        distances = torch.cdist(pred_colors.unsqueeze(1), target_palette.unsqueeze(0))
        min_distances = distances.min(dim=-1)[0]
        
        # Average minimum distance as loss
        loss = min_distances.mean()
        
        return loss
    
    def forward(self, 
                pred: torch.Tensor,
                batch: Dict[str, Any]) -> torch.Tensor:
        """Compute cultural loss."""
        total_loss = 0.0
        batch_size = pred.shape[0]
        
        # Get styles from batch
        styles = batch.get('styles', ['unknown'] * batch_size)
        
        # Compute color palette loss for each sample
        for i, style in enumerate(styles):
            if style in self.cultural_palettes:
                sample_pred = pred[i:i+1]
                palette_loss = self.color_palette_loss(sample_pred, style)
                total_loss += palette_loss
        
        return total_loss / batch_size

class LPIPSLoss(nn.Module):
    """LPIPS perceptual loss."""
    
    def __init__(self, net: str = "alex"):
        super().__init__()
        
        if not LPIPS_AVAILABLE:
            raise ImportError("LPIPS not available. Install with: pip install lpips")
        
        self.lpips_fn = lpips.LPIPS(net=net)
        
        # Freeze parameters
        for param in self.lpips_fn.parameters():
            param.requires_grad = False
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute LPIPS loss."""
        # LPIPS expects inputs in [-1, 1] range
        pred_norm = pred * 2.0 - 1.0
        target_norm = target * 2.0 - 1.0
        
        return self.lpips_fn(pred_norm, target_norm).mean()

class TemporalConsistencyLoss(nn.Module):
    """Temporal consistency loss for video-like sequences."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, 
                pred_sequence: torch.Tensor,
                target_sequence: torch.Tensor) -> torch.Tensor:
        """Compute temporal consistency loss."""
        # Compute frame-to-frame differences
        pred_diff = pred_sequence[1:] - pred_sequence[:-1]
        target_diff = target_sequence[1:] - target_sequence[:-1]
        
        # L2 loss on differences
        loss = F.mse_loss(pred_diff, target_diff)
        
        return loss

class SNRWeightedLoss(nn.Module):
    """SNR-weighted loss for diffusion training."""
    
    def __init__(self, gamma: float = 5.0):
        super().__init__()
        self.gamma = gamma
    
    def compute_snr(self, timesteps: torch.Tensor, 
                   noise_scheduler) -> torch.Tensor:
        """Compute signal-to-noise ratio."""
        alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)
        sqrt_alphas_cumprod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod[timesteps]) ** 0.5
        
        # SNR = alpha^2 / (1 - alpha^2)
        snr = (sqrt_alphas_cumprod / sqrt_one_minus_alphas_cumprod) ** 2
        return snr
    
    def forward(self, 
                pred: torch.Tensor,
                target: torch.Tensor,
                timesteps: torch.Tensor,
                noise_scheduler) -> torch.Tensor:
        """Compute SNR-weighted loss."""
        # Compute SNR
        snr = self.compute_snr(timesteps, noise_scheduler)
        
        # Compute loss weights
        mse_loss_weights = torch.stack([snr, self.gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
        
        # Compute MSE loss
        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape))))
        
        # Apply weights
        weighted_loss = loss * mse_loss_weights
        
        return weighted_loss.mean()

class RagamalaLoss(nn.Module):
    """Combined loss function for Ragamala painting generation."""
    
    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config
        
        # Initialize loss components
        self.losses = nn.ModuleDict()
        
        # Perceptual loss
        if config.enable_perceptual_loss:
            self.losses['perceptual'] = PerceptualLoss(
                layers=config.perceptual_layers
            )
        
        # Style loss
        if config.enable_style_loss:
            self.losses['style'] = StyleLoss(
                layers=config.style_layers
            )
        
        # Cultural loss
        if config.enable_cultural_loss:
            self.losses['cultural'] = CulturalLoss()
        
        # CLIP loss
        if config.enable_clip_loss:
            self.losses['clip'] = CLIPLoss(
                model_name=config.clip_model_name
            )
        
        # LPIPS loss
        if config.enable_lpips_loss and LPIPS_AVAILABLE:
            self.losses['lpips'] = LPIPSLoss(net=config.lpips_net)
        
        # Temporal consistency loss
        if config.enable_temporal_consistency:
            self.losses['temporal'] = TemporalConsistencyLoss()
        
        # SNR weighted loss
        if config.enable_snr_weighting:
            self.losses['snr'] = SNRWeightedLoss(gamma=config.snr_gamma)
    
    def forward(self,
                model_pred: torch.Tensor,
                target: torch.Tensor,
                timesteps: Optional[torch.Tensor] = None,
                batch: Optional[Dict[str, Any]] = None,
                noise_scheduler=None,
                **kwargs) -> torch.Tensor:
        """Compute combined loss."""
        total_loss = 0.0
        loss_dict = {}
        
        # Main reconstruction loss
        if self.config.enable_snr_weighting and timesteps is not None:
            main_loss = self.losses['snr'](model_pred, target, timesteps, noise_scheduler)
        else:
            if self.config.main_loss_type == "mse":
                main_loss = F.mse_loss(model_pred, target)
            elif self.config.main_loss_type == "l1":
                main_loss = F.l1_loss(model_pred, target)
            elif self.config.main_loss_type == "huber":
                main_loss = F.huber_loss(model_pred, target)
            else:
                main_loss = F.mse_loss(model_pred, target)
        
        total_loss += self.config.main_loss_weight * main_loss
        loss_dict['main_loss'] = main_loss.item()
        
        # Convert predictions to images for perceptual losses
        # Note: This assumes model_pred and target are in latent space
        # You may need to decode them to image space first
        
        # Perceptual loss
        if 'perceptual' in self.losses:
            try:
                perceptual_loss = self.losses['perceptual'](model_pred, target)
                total_loss += self.config.perceptual_loss_weight * perceptual_loss
                loss_dict['perceptual_loss'] = perceptual_loss.item()
            except Exception as e:
                logger.warning(f"Perceptual loss computation failed: {e}")
        
        # Style loss
        if 'style' in self.losses:
            try:
                style_loss = self.losses['style'](model_pred, target)
                total_loss += self.config.style_loss_weight * style_loss
                loss_dict['style_loss'] = style_loss.item()
            except Exception as e:
                logger.warning(f"Style loss computation failed: {e}")
        
        # Cultural loss
        if 'cultural' in self.losses and batch is not None:
            try:
                cultural_loss = self.losses['cultural'](model_pred, batch)
                total_loss += self.config.cultural_loss_weight * cultural_loss
                loss_dict['cultural_loss'] = cultural_loss.item()
            except Exception as e:
                logger.warning(f"Cultural loss computation failed: {e}")
        
        # CLIP loss
        if 'clip' in self.losses and batch is not None:
            try:
                text_prompts = batch.get('texts', None)
                clip_loss = self.losses['clip'](model_pred, target, text_prompts)
                total_loss += self.config.clip_loss_weight * clip_loss
                loss_dict['clip_loss'] = clip_loss.item()
            except Exception as e:
                logger.warning(f"CLIP loss computation failed: {e}")
        
        # LPIPS loss
        if 'lpips' in self.losses:
            try:
                lpips_loss = self.losses['lpips'](model_pred, target)
                total_loss += self.config.lpips_loss_weight * lpips_loss
                loss_dict['lpips_loss'] = lpips_loss.item()
            except Exception as e:
                logger.warning(f"LPIPS loss computation failed: {e}")
        
        # Store loss components for logging
        if hasattr(self, '_last_loss_dict'):
            self._last_loss_dict = loss_dict
        
        return total_loss
    
    def get_last_loss_dict(self) -> Dict[str, float]:
        """Get the last computed loss components."""
        return getattr(self, '_last_loss_dict', {})

def create_loss_function(config: LossConfig) -> RagamalaLoss:
    """Factory function to create loss function."""
    return RagamalaLoss(config)

def main():
    """Main function for testing loss functions."""
    # Create configuration
    config = LossConfig()
    
    # Create loss function
    loss_fn = create_loss_function(config)
    
    # Test with dummy data
    batch_size = 2
    channels = 4
    height = 64
    width = 64
    
    model_pred = torch.randn(batch_size, channels, height, width)
    target = torch.randn(batch_size, channels, height, width)
    timesteps = torch.randint(0, 1000, (batch_size,))
    
    batch = {
        'texts': ['A rajput style ragamala painting', 'A pahari style artwork'],
        'styles': ['rajput', 'pahari'],
        'ragas': ['bhairav', 'yaman']
    }
    
    # Compute loss
    total_loss = loss_fn(
        model_pred=model_pred,
        target=target,
        timesteps=timesteps,
        batch=batch
    )
    
    print(f"Total loss: {total_loss.item():.4f}")
    
    # Get loss components
    loss_dict = loss_fn.get_last_loss_dict()
    for name, value in loss_dict.items():
        print(f"  {name}: {value:.4f}")
    
    print("Loss function testing completed successfully!")

if __name__ == "__main__":
    main()
