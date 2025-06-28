"""
Diffusion Schedulers for Ragamala Painting Generation.

This module provides comprehensive scheduler implementations for SDXL fine-tuning
on Ragamala paintings, including custom schedulers optimized for cultural art generation
and various noise scheduling strategies.
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
import json
from abc import ABC, abstractmethod

# Diffusers imports
from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler
)
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

@dataclass
class SchedulerConfig:
    """Configuration for diffusion schedulers."""
    # Basic parameters
    num_train_timesteps: int = 1000
    beta_start: float = 0.00085
    beta_end: float = 0.012
    beta_schedule: str = "scaled_linear"
    
    # Prediction type
    prediction_type: str = "epsilon"
    
    # Sampling parameters
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    eta: float = 0.0
    
    # Cultural conditioning
    enable_cultural_conditioning: bool = True
    cultural_guidance_scale: float = 1.0
    
    # Advanced settings
    clip_sample: bool = False
    clip_sample_range: float = 1.0
    thresholding: bool = False
    dynamic_thresholding_ratio: float = 0.995
    sample_max_value: float = 1.0
    
    # Scheduler specific
    algorithm_type: str = "dpmsolver++"
    solver_type: str = "midpoint"
    lower_order_final: bool = True
    use_karras_sigmas: bool = False
    timestep_spacing: str = "linspace"
    steps_offset: int = 0

class CulturalAwareSchedulerMixin:
    """Mixin for cultural-aware scheduling in Ragamala generation."""
    
    def __init__(self, cultural_guidance_scale: float = 1.0):
        self.cultural_guidance_scale = cultural_guidance_scale
        self.raga_time_mapping = {
            'bhairav': 'dawn',
            'yaman': 'evening', 
            'malkauns': 'midnight',
            'darbari': 'late_evening',
            'bageshri': 'late_night',
            'todi': 'morning'
        }
        
    def apply_cultural_conditioning(self, 
                                  noise_pred: torch.Tensor,
                                  timestep: torch.Tensor,
                                  raga: Optional[str] = None,
                                  style: Optional[str] = None) -> torch.Tensor:
        """Apply cultural conditioning to noise prediction."""
        if not raga and not style:
            return noise_pred
            
        # Apply time-based conditioning for ragas
        if raga and raga in self.raga_time_mapping:
            time_factor = self._get_time_conditioning_factor(timestep, raga)
            noise_pred = noise_pred * time_factor
            
        # Apply style-based conditioning
        if style:
            style_factor = self._get_style_conditioning_factor(timestep, style)
            noise_pred = noise_pred * style_factor
            
        return noise_pred
    
    def _get_time_conditioning_factor(self, timestep: torch.Tensor, raga: str) -> torch.Tensor:
        """Get time-based conditioning factor for raga."""
        time_of_day = self.raga_time_mapping.get(raga, 'any')
        
        # Normalize timestep to [0, 1]
        normalized_t = timestep.float() / self.config.num_train_timesteps
        
        if time_of_day == 'dawn':
            # Emphasize early timesteps
            factor = 1.0 + 0.1 * torch.exp(-5 * normalized_t)
        elif time_of_day == 'evening':
            # Emphasize middle timesteps
            factor = 1.0 + 0.1 * torch.exp(-5 * (normalized_t - 0.5) ** 2)
        elif time_of_day == 'midnight':
            # Emphasize late timesteps
            factor = 1.0 + 0.1 * torch.exp(-5 * (1 - normalized_t))
        else:
            factor = torch.ones_like(timestep, dtype=torch.float32)
            
        return factor * self.cultural_guidance_scale
    
    def _get_style_conditioning_factor(self, timestep: torch.Tensor, style: str) -> torch.Tensor:
        """Get style-based conditioning factor."""
        # Normalize timestep to [0, 1]
        normalized_t = timestep.float() / self.config.num_train_timesteps
        
        style_factors = {
            'rajput': 1.0 + 0.05 * torch.sin(2 * math.pi * normalized_t),  # Bold variations
            'pahari': 1.0 + 0.03 * torch.cos(math.pi * normalized_t),      # Gentle variations
            'deccan': 1.0 + 0.04 * (1 - normalized_t),                     # Formal structure
            'mughal': 1.0 + 0.06 * normalized_t                            # Elaborate details
        }
        
        factor = style_factors.get(style, torch.ones_like(timestep, dtype=torch.float32))
        return factor * self.cultural_guidance_scale

class RagamalaDDPMScheduler(DDPMScheduler, CulturalAwareSchedulerMixin):
    """DDPM Scheduler with cultural conditioning for Ragamala paintings."""
    
    @register_to_config
    def __init__(self,
                 num_train_timesteps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 beta_schedule: str = "linear",
                 trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
                 variance_type: str = "fixed_small",
                 clip_sample: bool = True,
                 prediction_type: str = "epsilon",
                 thresholding: bool = False,
                 dynamic_thresholding_ratio: float = 0.995,
                 clip_sample_range: float = 1.0,
                 sample_max_value: float = 1.0,
                 timestep_spacing: str = "leading",
                 steps_offset: int = 0,
                 rescale_betas_zero_snr: bool = False,
                 cultural_guidance_scale: float = 1.0):
        
        # Initialize parent DDPM scheduler
        super(DDPMScheduler, self).__init__(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            trained_betas=trained_betas,
            variance_type=variance_type,
            clip_sample=clip_sample,
            prediction_type=prediction_type,
            thresholding=thresholding,
            dynamic_thresholding_ratio=dynamic_thresholding_ratio,
            clip_sample_range=clip_sample_range,
            sample_max_value=sample_max_value,
            timestep_spacing=timestep_spacing,
            steps_offset=steps_offset,
            rescale_betas_zero_snr=rescale_betas_zero_snr
        )
        
        # Initialize cultural conditioning
        CulturalAwareSchedulerMixin.__init__(self, cultural_guidance_scale)
    
    def step(self,
             model_output: torch.FloatTensor,
             timestep: int,
             sample: torch.FloatTensor,
             generator=None,
             return_dict: bool = True,
             raga: Optional[str] = None,
             style: Optional[str] = None) -> Union[BaseOutput, Tuple]:
        """
        Predict the sample from the previous timestep with cultural conditioning.
        """
        # Apply cultural conditioning
        if raga or style:
            timestep_tensor = torch.tensor([timestep], device=model_output.device)
            model_output = self.apply_cultural_conditioning(
                model_output, timestep_tensor, raga, style
            )
        
        # Call parent step method
        return super().step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            generator=generator,
            return_dict=return_dict
        )

class RagamalaDDIMScheduler(DDIMScheduler, CulturalAwareSchedulerMixin):
    """DDIM Scheduler with cultural conditioning for Ragamala paintings."""
    
    @register_to_config
    def __init__(self,
                 num_train_timesteps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 beta_schedule: str = "linear",
                 trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
                 clip_sample: bool = True,
                 set_alpha_to_one: bool = True,
                 steps_offset: int = 0,
                 prediction_type: str = "epsilon",
                 thresholding: bool = False,
                 dynamic_thresholding_ratio: float = 0.995,
                 clip_sample_range: float = 1.0,
                 sample_max_value: float = 1.0,
                 timestep_spacing: str = "leading",
                 rescale_betas_zero_snr: bool = False,
                 cultural_guidance_scale: float = 1.0):
        
        # Initialize parent DDIM scheduler
        super(DDIMScheduler, self).__init__(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            trained_betas=trained_betas,
            clip_sample=clip_sample,
            set_alpha_to_one=set_alpha_to_one,
            steps_offset=steps_offset,
            prediction_type=prediction_type,
            thresholding=thresholding,
            dynamic_thresholding_ratio=dynamic_thresholding_ratio,
            clip_sample_range=clip_sample_range,
            sample_max_value=sample_max_value,
            timestep_spacing=timestep_spacing,
            rescale_betas_zero_snr=rescale_betas_zero_snr
        )
        
        # Initialize cultural conditioning
        CulturalAwareSchedulerMixin.__init__(self, cultural_guidance_scale)
    
    def step(self,
             model_output: torch.FloatTensor,
             timestep: int,
             sample: torch.FloatTensor,
             eta: float = 0.0,
             use_clipped_model_output: bool = False,
             generator=None,
             variance_noise: Optional[torch.FloatTensor] = None,
             return_dict: bool = True,
             raga: Optional[str] = None,
             style: Optional[str] = None) -> Union[BaseOutput, Tuple]:
        """
        Predict the sample from the previous timestep with cultural conditioning.
        """
        # Apply cultural conditioning
        if raga or style:
            timestep_tensor = torch.tensor([timestep], device=model_output.device)
            model_output = self.apply_cultural_conditioning(
                model_output, timestep_tensor, raga, style
            )
        
        # Call parent step method
        return super().step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            eta=eta,
            use_clipped_model_output=use_clipped_model_output,
            generator=generator,
            variance_noise=variance_noise,
            return_dict=return_dict
        )

class RagamalaDPMSolverMultistepScheduler(DPMSolverMultistepScheduler, CulturalAwareSchedulerMixin):
    """DPM-Solver++ Scheduler with cultural conditioning for Ragamala paintings."""
    
    @register_to_config
    def __init__(self,
                 num_train_timesteps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 beta_schedule: str = "linear",
                 trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
                 solver_order: int = 2,
                 prediction_type: str = "epsilon",
                 thresholding: bool = False,
                 dynamic_thresholding_ratio: float = 0.995,
                 sample_max_value: float = 1.0,
                 algorithm_type: str = "dpmsolver++",
                 solver_type: str = "midpoint",
                 lower_order_final: bool = True,
                 euler_at_final: bool = False,
                 use_karras_sigmas: bool = False,
                 lambda_min_clipped: float = -float("inf"),
                 variance_type: Optional[str] = None,
                 timestep_spacing: str = "linspace",
                 steps_offset: int = 0,
                 cultural_guidance_scale: float = 1.0):
        
        # Initialize parent DPM-Solver scheduler
        super(DPMSolverMultistepScheduler, self).__init__(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            trained_betas=trained_betas,
            solver_order=solver_order,
            prediction_type=prediction_type,
            thresholding=thresholding,
            dynamic_thresholding_ratio=dynamic_thresholding_ratio,
            sample_max_value=sample_max_value,
            algorithm_type=algorithm_type,
            solver_type=solver_type,
            lower_order_final=lower_order_final,
            euler_at_final=euler_at_final,
            use_karras_sigmas=use_karras_sigmas,
            lambda_min_clipped=lambda_min_clipped,
            variance_type=variance_type,
            timestep_spacing=timestep_spacing,
            steps_offset=steps_offset
        )
        
        # Initialize cultural conditioning
        CulturalAwareSchedulerMixin.__init__(self, cultural_guidance_scale)
    
    def step(self,
             model_output: torch.FloatTensor,
             timestep: int,
             sample: torch.FloatTensor,
             generator: Optional[torch.Generator] = None,
             return_dict: bool = True,
             raga: Optional[str] = None,
             style: Optional[str] = None) -> Union[BaseOutput, Tuple]:
        """
        Predict the sample from the previous timestep with cultural conditioning.
        """
        # Apply cultural conditioning
        if raga or style:
            timestep_tensor = torch.tensor([timestep], device=model_output.device)
            model_output = self.apply_cultural_conditioning(
                model_output, timestep_tensor, raga, style
            )
        
        # Call parent step method
        return super().step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            generator=generator,
            return_dict=return_dict
        )

class RagamalaEulerDiscreteScheduler(EulerDiscreteScheduler, CulturalAwareSchedulerMixin):
    """Euler Discrete Scheduler with cultural conditioning for Ragamala paintings."""
    
    @register_to_config
    def __init__(self,
                 num_train_timesteps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 beta_schedule: str = "linear",
                 trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
                 prediction_type: str = "epsilon",
                 interpolation_type: str = "linear",
                 use_karras_sigmas: bool = False,
                 timestep_spacing: str = "linspace",
                 steps_offset: int = 0,
                 cultural_guidance_scale: float = 1.0):
        
        # Initialize parent Euler scheduler
        super(EulerDiscreteScheduler, self).__init__(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            trained_betas=trained_betas,
            prediction_type=prediction_type,
            interpolation_type=interpolation_type,
            use_karras_sigmas=use_karras_sigmas,
            timestep_spacing=timestep_spacing,
            steps_offset=steps_offset
        )
        
        # Initialize cultural conditioning
        CulturalAwareSchedulerMixin.__init__(self, cultural_guidance_scale)
    
    def step(self,
             model_output: torch.FloatTensor,
             timestep: Union[float, torch.FloatTensor],
             sample: torch.FloatTensor,
             s_churn: float = 0.0,
             s_tmin: float = 0.0,
             s_tmax: float = float("inf"),
             s_noise: float = 1.0,
             generator: Optional[torch.Generator] = None,
             return_dict: bool = True,
             raga: Optional[str] = None,
             style: Optional[str] = None) -> Union[BaseOutput, Tuple]:
        """
        Predict the sample from the previous timestep with cultural conditioning.
        """
        # Apply cultural conditioning
        if raga or style:
            if isinstance(timestep, float):
                timestep_tensor = torch.tensor([timestep], device=model_output.device)
            else:
                timestep_tensor = timestep
            model_output = self.apply_cultural_conditioning(
                model_output, timestep_tensor, raga, style
            )
        
        # Call parent step method
        return super().step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            s_churn=s_churn,
            s_tmin=s_tmin,
            s_tmax=s_tmax,
            s_noise=s_noise,
            generator=generator,
            return_dict=return_dict
        )

class AdaptiveScheduler(SchedulerMixin, ConfigMixin):
    """Adaptive scheduler that adjusts based on cultural context."""
    
    @register_to_config
    def __init__(self,
                 num_train_timesteps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 beta_schedule: str = "linear",
                 prediction_type: str = "epsilon",
                 cultural_adaptation: bool = True):
        
        self.num_train_timesteps = num_train_timesteps
        self.cultural_adaptation = cultural_adaptation
        
        # Initialize different schedulers for different contexts
        self.schedulers = {
            'ddpm': RagamalaDDPMScheduler(num_train_timesteps=num_train_timesteps),
            'ddim': RagamalaDDIMScheduler(num_train_timesteps=num_train_timesteps),
            'dpm': RagamalaDPMSolverMultistepScheduler(num_train_timesteps=num_train_timesteps),
            'euler': RagamalaEulerDiscreteScheduler(num_train_timesteps=num_train_timesteps)
        }
        
        # Cultural context mapping
        self.cultural_scheduler_mapping = {
            'rajput': 'ddpm',      # Bold, traditional approach
            'pahari': 'ddim',      # Smooth, refined approach
            'deccan': 'dpm',       # Sophisticated, mathematical approach
            'mughal': 'euler',     # Detailed, precise approach
            'default': 'dpm'
        }
    
    def select_scheduler(self, style: Optional[str] = None, raga: Optional[str] = None) -> str:
        """Select appropriate scheduler based on cultural context."""
        if not self.cultural_adaptation:
            return 'dpm'
        
        if style and style in self.cultural_scheduler_mapping:
            return self.cultural_scheduler_mapping[style]
        
        # Raga-based selection
        if raga:
            raga_scheduler_mapping = {
                'bhairav': 'ddpm',     # Traditional, structured
                'yaman': 'ddim',       # Smooth, romantic
                'malkauns': 'euler',   # Precise, meditative
                'darbari': 'dpm'       # Sophisticated, royal
            }
            return raga_scheduler_mapping.get(raga, 'dpm')
        
        return 'dpm'
    
    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """Set timesteps for all schedulers."""
        for scheduler in self.schedulers.values():
            scheduler.set_timesteps(num_inference_steps, device)
    
    def step(self,
             model_output: torch.FloatTensor,
             timestep: int,
             sample: torch.FloatTensor,
             style: Optional[str] = None,
             raga: Optional[str] = None,
             **kwargs) -> Union[BaseOutput, Tuple]:
        """Adaptive step using selected scheduler."""
        scheduler_name = self.select_scheduler(style, raga)
        scheduler = self.schedulers[scheduler_name]
        
        # Add cultural parameters to kwargs
        kwargs.update({'raga': raga, 'style': style})
        
        return scheduler.step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            **kwargs
        )

class SchedulerFactory:
    """Factory for creating schedulers with cultural conditioning."""
    
    @staticmethod
    def create_scheduler(scheduler_type: str, 
                        config: SchedulerConfig,
                        **kwargs) -> SchedulerMixin:
        """Create scheduler based on type and configuration."""
        
        scheduler_classes = {
            'ddpm': RagamalaDDPMScheduler,
            'ddim': RagamalaDDIMScheduler,
            'dpm_solver': RagamalaDPMSolverMultistepScheduler,
            'euler': RagamalaEulerDiscreteScheduler,
            'adaptive': AdaptiveScheduler
        }
        
        if scheduler_type not in scheduler_classes:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        scheduler_class = scheduler_classes[scheduler_type]
        
        # Prepare initialization arguments
        init_args = {
            'num_train_timesteps': config.num_train_timesteps,
            'beta_start': config.beta_start,
            'beta_end': config.beta_end,
            'beta_schedule': config.beta_schedule,
            'prediction_type': config.prediction_type,
            'cultural_guidance_scale': config.cultural_guidance_scale
        }
        
        # Add scheduler-specific arguments
        if scheduler_type == 'dpm_solver':
            init_args.update({
                'algorithm_type': config.algorithm_type,
                'solver_type': config.solver_type,
                'lower_order_final': config.lower_order_final,
                'use_karras_sigmas': config.use_karras_sigmas
            })
        elif scheduler_type == 'ddim':
            init_args.update({
                'clip_sample': config.clip_sample,
                'set_alpha_to_one': True,
                'steps_offset': config.steps_offset
            })
        elif scheduler_type == 'euler':
            init_args.update({
                'use_karras_sigmas': config.use_karras_sigmas,
                'timestep_spacing': config.timestep_spacing,
                'steps_offset': config.steps_offset
            })
        
        # Override with any additional kwargs
        init_args.update(kwargs)
        
        return scheduler_class(**init_args)

def get_optimal_scheduler_for_raga(raga: str, style: str = None) -> str:
    """Get optimal scheduler recommendation for specific raga and style."""
    
    # Raga-specific recommendations
    raga_recommendations = {
        'bhairav': 'ddpm',        # Traditional, structured approach
        'yaman': 'ddim',          # Smooth, deterministic
        'malkauns': 'euler',      # Precise, mathematical
        'darbari': 'dpm_solver',  # Sophisticated, high-quality
        'bageshri': 'ddim',       # Romantic, smooth
        'todi': 'euler',          # Enchanting, precise
        'puriya': 'ddpm',         # Devotional, traditional
        'marwa': 'dpm_solver',    # Contemplative, refined
        'bhimpalasi': 'ddim',     # Peaceful, gentle
        'kafi': 'adaptive'        # Versatile, folk-like
    }
    
    # Style-specific modifications
    if style == 'rajput':
        # Prefer more traditional, bold approaches
        return raga_recommendations.get(raga, 'ddpm')
    elif style == 'pahari':
        # Prefer smooth, refined approaches
        return raga_recommendations.get(raga, 'ddim')
    elif style == 'deccan':
        # Prefer sophisticated approaches
        return raga_recommendations.get(raga, 'dpm_solver')
    elif style == 'mughal':
        # Prefer detailed, precise approaches
        return raga_recommendations.get(raga, 'euler')
    
    return raga_recommendations.get(raga, 'dpm_solver')

def main():
    """Main function for testing schedulers."""
    # Create configuration
    config = SchedulerConfig()
    
    # Test different schedulers
    scheduler_types = ['ddpm', 'ddim', 'dpm_solver', 'euler', 'adaptive']
    
    for scheduler_type in scheduler_types:
        print(f"Testing {scheduler_type} scheduler...")
        
        try:
            scheduler = SchedulerFactory.create_scheduler(scheduler_type, config)
            
            # Set timesteps
            scheduler.set_timesteps(30, device='cpu')
            
            # Test step (dummy data)
            model_output = torch.randn(1, 4, 64, 64)
            timestep = 500
            sample = torch.randn(1, 4, 64, 64)
            
            # Test with cultural conditioning
            result = scheduler.step(
                model_output=model_output,
                timestep=timestep,
                sample=sample,
                raga='bhairav',
                style='rajput'
            )
            
            print(f"  {scheduler_type} scheduler working correctly")
            
        except Exception as e:
            print(f"  Error with {scheduler_type} scheduler: {e}")
    
    # Test scheduler recommendations
    print("\nTesting scheduler recommendations:")
    test_cases = [
        ('bhairav', 'rajput'),
        ('yaman', 'pahari'),
        ('malkauns', 'deccan'),
        ('darbari', 'mughal')
    ]
    
    for raga, style in test_cases:
        recommended = get_optimal_scheduler_for_raga(raga, style)
        print(f"  {raga} + {style} -> {recommended}")
    
    print("Scheduler testing completed successfully!")

if __name__ == "__main__":
    main()
