"""
SDXL + LoRA Implementation for Ragamala Painting Generation.

This module provides comprehensive SDXL 1.0 fine-tuning implementation with LoRA
for generating Ragamala paintings with cultural conditioning and style control.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import math

# Diffusers imports
from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available

# Transformers imports
from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    AutoTokenizer
)

# PEFT imports for LoRA
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    PeftConfig
)

# Accelerate imports
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logging_utils import setup_logger
from src.data.dataset import RagamalaDataset, DatasetConfig

logger = setup_logger(__name__)

@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""
    # LoRA parameters
    rank: int = 64
    alpha: int = 32
    dropout: float = 0.1
    bias: str = "none"
    
    # Target modules for SDXL UNet
    target_modules: List[str] = None
    
    # LoRA scaling
    lora_scale: float = 1.0
    init_lora_weights: bool = True
    
    # Text encoder LoRA
    train_text_encoder: bool = False
    text_encoder_lr: float = 5e-5
    text_encoder_rank: int = 4
    text_encoder_alpha: int = 16
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "to_k", "to_q", "to_v", "to_out.0",
                "ff.net.0.proj", "ff.net.2"
            ]

@dataclass
class TrainingConfig:
    """Configuration for SDXL training."""
    # Model paths
    pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
    pretrained_vae_model_name_or_path: str = "madebyollin/sdxl-vae-fp16-fix"
    revision: Optional[str] = None
    variant: Optional[str] = None
    
    # Training parameters
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 500
    lr_num_cycles: int = 1
    lr_power: float = 1.0
    
    # Batch and gradient settings
    train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    
    # Training duration
    max_train_steps: int = 10000
    num_train_epochs: int = 2
    
    # Validation and checkpointing
    validation_steps: int = 500
    checkpointing_steps: int = 1000
    save_steps: int = 1000
    save_total_limit: int = 5
    
    # Optimization
    mixed_precision: str = "fp16"
    use_8bit_adam: bool = False
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    
    # Data settings
    resolution: int = 1024
    center_crop: bool = True
    random_flip: bool = True
    
    # Cultural conditioning
    enable_cultural_conditioning: bool = True
    cultural_conditioning_scale: float = 1.0
    
    # Noise scheduler
    noise_scheduler: str = "DDPMScheduler"
    prediction_type: str = "epsilon"
    
    # Memory optimization
    enable_xformers_memory_efficient_attention: bool = True
    allow_tf32: bool = True
    
    # Output settings
    output_dir: str = "outputs/training"
    logging_dir: str = "logs/training"
    
    # Reproducibility
    seed: int = 42

class CulturalConditioningModule(nn.Module):
    """Module for cultural conditioning in Ragamala generation."""
    
    def __init__(self, 
                 raga_vocab_size: int = 50,
                 style_vocab_size: int = 20,
                 embedding_dim: int = 768,
                 hidden_dim: int = 512):
        super().__init__()
        
        # Raga and style embeddings
        self.raga_embedding = nn.Embedding(raga_vocab_size, embedding_dim // 2)
        self.style_embedding = nn.Embedding(style_vocab_size, embedding_dim // 2)
        
        # Fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # Attention mechanism for cultural conditioning
        self.cultural_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
    def forward(self, raga_ids: torch.Tensor, style_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for cultural conditioning.
        
        Args:
            raga_ids: Tensor of raga IDs [batch_size]
            style_ids: Tensor of style IDs [batch_size]
            
        Returns:
            Cultural conditioning embeddings [batch_size, embedding_dim]
        """
        # Get embeddings
        raga_emb = self.raga_embedding(raga_ids)  # [batch_size, embedding_dim//2]
        style_emb = self.style_embedding(style_ids)  # [batch_size, embedding_dim//2]
        
        # Concatenate embeddings
        cultural_emb = torch.cat([raga_emb, style_emb], dim=-1)  # [batch_size, embedding_dim]
        
        # Apply fusion network
        fused_emb = self.fusion_network(cultural_emb)  # [batch_size, embedding_dim]
        
        # Apply self-attention for cultural context
        cultural_context, _ = self.cultural_attention(
            fused_emb.unsqueeze(1),  # [batch_size, 1, embedding_dim]
            fused_emb.unsqueeze(1),
            fused_emb.unsqueeze(1)
        )
        
        return cultural_context.squeeze(1)  # [batch_size, embedding_dim]

class SDXLLoRATrainer:
    """Main trainer class for SDXL + LoRA fine-tuning."""
    
    def __init__(self, 
                 training_config: TrainingConfig,
                 lora_config: LoRAConfig,
                 dataset_config: DatasetConfig):
        self.training_config = training_config
        self.lora_config = lora_config
        self.dataset_config = dataset_config
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            mixed_precision=training_config.mixed_precision,
            log_with="wandb" if is_wandb_available() else None,
            project_config=ProjectConfiguration(
                project_dir=training_config.output_dir,
                logging_dir=training_config.logging_dir
            )
        )
        
        # Set seed for reproducibility
        set_seed(training_config.seed)
        
        # Initialize models
        self.tokenizer = None
        self.tokenizer_2 = None
        self.text_encoder = None
        self.text_encoder_2 = None
        self.vae = None
        self.unet = None
        self.noise_scheduler = None
        
        # Cultural conditioning
        self.cultural_module = None
        
        # Training components
        self.optimizer = None
        self.lr_scheduler = None
        self.train_dataloader = None
        self.val_dataloader = None
        
        self._setup_models()
        self._setup_lora()
        self._setup_cultural_conditioning()
    
    def _setup_models(self):
        """Setup SDXL models."""
        logger.info("Loading SDXL models...")
        
        # Load tokenizers
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.training_config.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=self.training_config.revision
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            self.training_config.pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            revision=self.training_config.revision
        )
        
        # Load text encoders
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.training_config.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=self.training_config.revision,
            variant=self.training_config.variant
        )
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            self.training_config.pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            revision=self.training_config.revision,
            variant=self.training_config.variant
        )
        
        # Load VAE
        if self.training_config.pretrained_vae_model_name_or_path:
            self.vae = AutoencoderKL.from_pretrained(
                self.training_config.pretrained_vae_model_name_or_path,
                revision=self.training_config.revision,
                variant=self.training_config.variant
            )
        else:
            self.vae = AutoencoderKL.from_pretrained(
                self.training_config.pretrained_model_name_or_path,
                subfolder="vae",
                revision=self.training_config.revision,
                variant=self.training_config.variant
            )
        
        # Load UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            self.training_config.pretrained_model_name_or_path,
            subfolder="unet",
            revision=self.training_config.revision,
            variant=self.training_config.variant
        )
        
        # Setup noise scheduler
        if self.training_config.noise_scheduler == "DDPMScheduler":
            self.noise_scheduler = DDPMScheduler.from_pretrained(
                self.training_config.pretrained_model_name_or_path,
                subfolder="scheduler"
            )
        
        # Freeze models
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.unet.requires_grad_(False)
        
        # Enable memory efficient attention
        if self.training_config.enable_xformers_memory_efficient_attention:
            if hasattr(self.unet, "enable_xformers_memory_efficient_attention"):
                self.unet.enable_xformers_memory_efficient_attention()
        
        # Enable gradient checkpointing
        if self.training_config.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            if self.lora_config.train_text_encoder:
                self.text_encoder.gradient_checkpointing_enable()
                self.text_encoder_2.gradient_checkpointing_enable()
        
        logger.info("SDXL models loaded successfully")
    
    def _setup_lora(self):
        """Setup LoRA for UNet and optionally text encoders."""
        logger.info("Setting up LoRA...")
        
        # UNet LoRA configuration
        unet_lora_config = LoraConfig(
            r=self.lora_config.rank,
            lora_alpha=self.lora_config.alpha,
            target_modules=self.lora_config.target_modules,
            lora_dropout=self.lora_config.dropout,
            bias=self.lora_config.bias,
            task_type=TaskType.DIFFUSION
        )
        
        # Apply LoRA to UNet
        self.unet = get_peft_model(self.unet, unet_lora_config)
        
        # Text encoder LoRA (optional)
        if self.lora_config.train_text_encoder:
            text_encoder_lora_config = LoraConfig(
                r=self.lora_config.text_encoder_rank,
                lora_alpha=self.lora_config.text_encoder_alpha,
                target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
                lora_dropout=self.lora_config.dropout,
                bias=self.lora_config.bias,
                task_type=TaskType.FEATURE_EXTRACTION
            )
            
            self.text_encoder = get_peft_model(self.text_encoder, text_encoder_lora_config)
            self.text_encoder_2 = get_peft_model(self.text_encoder_2, text_encoder_lora_config)
        
        logger.info("LoRA setup completed")
    
    def _setup_cultural_conditioning(self):
        """Setup cultural conditioning module."""
        if self.training_config.enable_cultural_conditioning:
            logger.info("Setting up cultural conditioning...")
            
            self.cultural_module = CulturalConditioningModule(
                raga_vocab_size=self.dataset_config.raga_vocab_size,
                style_vocab_size=self.dataset_config.style_vocab_size
            )
            
            logger.info("Cultural conditioning setup completed")
    
    def _setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler."""
        # Collect trainable parameters
        trainable_params = list(self.unet.parameters())
        
        if self.lora_config.train_text_encoder:
            trainable_params.extend(list(self.text_encoder.parameters()))
            trainable_params.extend(list(self.text_encoder_2.parameters()))
        
        if self.cultural_module:
            trainable_params.extend(list(self.cultural_module.parameters()))
        
        # Setup optimizer
        if self.training_config.use_8bit_adam:
            try:
                import bitsandbytes as bnb
                optimizer_cls = bnb.optim.AdamW8bit
            except ImportError:
                raise ImportError("To use 8bit Adam, please install bitsandbytes")
        else:
            optimizer_cls = torch.optim.AdamW
        
        self.optimizer = optimizer_cls(
            trainable_params,
            lr=self.training_config.learning_rate,
            betas=(self.training_config.adam_beta1, self.training_config.adam_beta2),
            weight_decay=self.training_config.adam_weight_decay,
            eps=self.training_config.adam_epsilon
        )
        
        # Setup learning rate scheduler
        self.lr_scheduler = get_scheduler(
            self.training_config.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.training_config.lr_warmup_steps * self.accelerator.num_processes,
            num_training_steps=self.training_config.max_train_steps * self.accelerator.num_processes,
            num_cycles=self.training_config.lr_num_cycles,
            power=self.training_config.lr_power
        )
    
    def _encode_prompt(self, prompt: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text prompt using both text encoders."""
        # Tokenize with both tokenizers
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        text_inputs_2 = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Encode with text encoders
        with torch.no_grad():
            prompt_embeds = self.text_encoder(
                text_inputs.input_ids.to(device),
                output_hidden_states=True
            )
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            
            prompt_embeds_2 = self.text_encoder_2(
                text_inputs_2.input_ids.to(device),
                output_hidden_states=True
            )
            pooled_prompt_embeds_2 = prompt_embeds_2[1]
            prompt_embeds_2 = prompt_embeds_2.hidden_states[-2]
        
        # Concatenate embeddings
        prompt_embeds = torch.concat([prompt_embeds, prompt_embeds_2], dim=-1)
        
        return prompt_embeds, pooled_prompt_embeds_2
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute training loss."""
        device = self.accelerator.device
        
        # Move batch to device
        pixel_values = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        
        # Encode images to latents
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        
        # Sample noise
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        
        # Sample timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=device
        ).long()
        
        # Add noise to latents
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Encode text prompts
        prompt_embeds_list = []
        pooled_prompt_embeds_list = []
        
        for i in range(bsz):
            prompt = batch["texts"][i]
            prompt_embeds, pooled_prompt_embeds = self._encode_prompt(prompt, device)
            prompt_embeds_list.append(prompt_embeds)
            pooled_prompt_embeds_list.append(pooled_prompt_embeds)
        
        encoder_hidden_states = torch.cat(prompt_embeds_list, dim=0)
        pooled_prompt_embeds = torch.cat(pooled_prompt_embeds_list, dim=0)
        
        # Cultural conditioning
        if self.cultural_module and self.training_config.enable_cultural_conditioning:
            raga_ids = batch["raga_ids"].to(device)
            style_ids = batch["style_ids"].to(device)
            
            cultural_conditioning = self.cultural_module(raga_ids, style_ids)
            
            # Add cultural conditioning to text embeddings
            cultural_conditioning = cultural_conditioning.unsqueeze(1)  # [batch_size, 1, embedding_dim]
            encoder_hidden_states = torch.cat([encoder_hidden_states, cultural_conditioning], dim=1)
        
        # Prepare added time ids for SDXL
        add_time_ids = torch.cat([
            torch.tensor([self.training_config.resolution, self.training_config.resolution], device=device).unsqueeze(0).repeat(bsz, 1),
            torch.tensor([0, 0], device=device).unsqueeze(0).repeat(bsz, 1),  # crops_coords_top_left
            torch.tensor([self.training_config.resolution, self.training_config.resolution], device=device).unsqueeze(0).repeat(bsz, 1)  # target_size
        ], dim=1)
        
        # Predict noise
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs={
                "text_embeds": pooled_prompt_embeds,
                "time_ids": add_time_ids
            }
        ).sample
        
        # Compute loss
        if self.training_config.prediction_type == "epsilon":
            target = noise
        elif self.training_config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.training_config.prediction_type}")
        
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        
        return loss
    
    def train(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None):
        """Main training loop."""
        logger.info("Starting training...")
        
        # Setup optimizer and scheduler
        self._setup_optimizer_and_scheduler()
        
        # Prepare everything with accelerator
        if self.cultural_module:
            (
                self.unet,
                self.cultural_module,
                self.optimizer,
                train_dataloader,
                self.lr_scheduler
            ) = self.accelerator.prepare(
                self.unet,
                self.cultural_module,
                self.optimizer,
                train_dataloader,
                self.lr_scheduler
            )
        else:
            (
                self.unet,
                self.optimizer,
                train_dataloader,
                self.lr_scheduler
            ) = self.accelerator.prepare(
                self.unet,
                self.optimizer,
                train_dataloader,
                self.lr_scheduler
            )
        
        # Move other models to device
        self.vae.to(self.accelerator.device)
        self.text_encoder.to(self.accelerator.device)
        self.text_encoder_2.to(self.accelerator.device)
        
        # Training loop
        global_step = 0
        progress_bar = tqdm(
            range(self.training_config.max_train_steps),
            disable=not self.accelerator.is_local_main_process
        )
        
        for epoch in range(self.training_config.num_train_epochs):
            self.unet.train()
            if self.cultural_module:
                self.cultural_module.train()
            
            for step, batch in enumerate(train_dataloader):
                with self.accelerator.accumulate(self.unet):
                    # Compute loss
                    loss = self._compute_loss(batch)
                    
                    # Backward pass
                    self.accelerator.backward(loss)
                    
                    # Gradient clipping
                    if self.accelerator.sync_gradients:
                        params_to_clip = []
                        params_to_clip.extend(self.unet.parameters())
                        if self.cultural_module:
                            params_to_clip.extend(self.cultural_module.parameters())
                        
                        self.accelerator.clip_grad_norm_(params_to_clip, self.training_config.max_grad_norm)
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                
                # Logging
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    
                    if global_step % 10 == 0:
                        logs = {
                            "loss": loss.detach().item(),
                            "lr": self.lr_scheduler.get_last_lr()[0],
                            "step": global_step
                        }
                        progress_bar.set_postfix(**logs)
                        self.accelerator.log(logs, step=global_step)
                
                # Validation
                if val_dataloader and global_step % self.training_config.validation_steps == 0:
                    self._validate(val_dataloader, global_step)
                
                # Save checkpoint
                if global_step % self.training_config.save_steps == 0:
                    self._save_checkpoint(global_step)
                
                if global_step >= self.training_config.max_train_steps:
                    break
        
        # Save final model
        self._save_final_model()
        logger.info("Training completed!")
    
    def _validate(self, val_dataloader: DataLoader, global_step: int):
        """Run validation."""
        logger.info(f"Running validation at step {global_step}")
        
        self.unet.eval()
        if self.cultural_module:
            self.cultural_module.eval()
        
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                loss = self._compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1
                
                if num_batches >= 10:  # Limit validation batches
                    break
        
        avg_loss = total_loss / num_batches
        
        logs = {"val_loss": avg_loss}
        self.accelerator.log(logs, step=global_step)
        
        logger.info(f"Validation loss: {avg_loss:.4f}")
        
        self.unet.train()
        if self.cultural_module:
            self.cultural_module.train()
    
    def _save_checkpoint(self, global_step: int):
        """Save training checkpoint."""
        if self.accelerator.is_main_process:
            checkpoint_dir = Path(self.training_config.output_dir) / f"checkpoint-{global_step}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Save LoRA weights
            unwrapped_unet = self.accelerator.unwrap_model(self.unet)
            unwrapped_unet.save_pretrained(checkpoint_dir / "unet_lora")
            
            if self.cultural_module:
                unwrapped_cultural = self.accelerator.unwrap_model(self.cultural_module)
                torch.save(unwrapped_cultural.state_dict(), checkpoint_dir / "cultural_module.pt")
            
            # Save training state
            self.accelerator.save_state(checkpoint_dir / "accelerator_state")
            
            logger.info(f"Checkpoint saved at step {global_step}")
    
    def _save_final_model(self):
        """Save final trained model."""
        if self.accelerator.is_main_process:
            output_dir = Path(self.training_config.output_dir) / "final_model"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save LoRA weights
            unwrapped_unet = self.accelerator.unwrap_model(self.unet)
            unwrapped_unet.save_pretrained(output_dir / "unet_lora")
            
            if self.cultural_module:
                unwrapped_cultural = self.accelerator.unwrap_model(self.cultural_module)
                torch.save(unwrapped_cultural.state_dict(), output_dir / "cultural_module.pt")
            
            # Save configuration
            with open(output_dir / "training_config.json", "w") as f:
                json.dump(self.training_config.__dict__, f, indent=2)
            
            with open(output_dir / "lora_config.json", "w") as f:
                json.dump(self.lora_config.__dict__, f, indent=2)
            
            logger.info("Final model saved successfully")

class SDXLLoRAInferencePipeline:
    """Inference pipeline for SDXL + LoRA Ragamala generation."""
    
    def __init__(self, 
                 model_path: str,
                 lora_weights_path: str,
                 cultural_module_path: Optional[str] = None,
                 device: str = "cuda"):
        self.model_path = model_path
        self.lora_weights_path = lora_weights_path
        self.cultural_module_path = cultural_module_path
        self.device = device
        
        self._load_pipeline()
    
    def _load_pipeline(self):
        """Load the inference pipeline."""
        logger.info("Loading inference pipeline...")
        
        # Load base pipeline
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        
        # Load LoRA weights
        self.pipeline.load_lora_weights(self.lora_weights_path)
        
        # Load cultural module if available
        if self.cultural_module_path and os.path.exists(self.cultural_module_path):
            self.cultural_module = CulturalConditioningModule()
            self.cultural_module.load_state_dict(torch.load(self.cultural_module_path))
            self.cultural_module.to(self.device)
            self.cultural_module.eval()
        else:
            self.cultural_module = None
        
        # Move to device
        self.pipeline.to(self.device)
        
        # Enable memory efficient attention
        self.pipeline.enable_xformers_memory_efficient_attention()
        
        logger.info("Inference pipeline loaded successfully")
    
    def generate(self,
                prompt: str,
                raga: Optional[str] = None,
                style: Optional[str] = None,
                negative_prompt: str = "blurry, low quality, distorted, modern, western art",
                num_inference_steps: int = 30,
                guidance_scale: float = 7.5,
                width: int = 1024,
                height: int = 1024,
                num_images_per_prompt: int = 1,
                generator: Optional[torch.Generator] = None) -> List[torch.Tensor]:
        """
        Generate Ragamala paintings.
        
        Args:
            prompt: Text prompt
            raga: Raga name for cultural conditioning
            style: Style name for cultural conditioning
            negative_prompt: Negative prompt
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale
            width: Image width
            height: Image height
            num_images_per_prompt: Number of images to generate
            generator: Random generator for reproducibility
            
        Returns:
            List of generated images
        """
        # Enhance prompt with cultural context
        if raga and style:
            enhanced_prompt = f"A {style} style ragamala painting depicting raga {raga}, {prompt}"
        elif raga:
            enhanced_prompt = f"A ragamala painting depicting raga {raga}, {prompt}"
        elif style:
            enhanced_prompt = f"A {style} style ragamala painting, {prompt}"
        else:
            enhanced_prompt = f"A ragamala painting, {prompt}"
        
        # Generate images
        with torch.no_grad():
            images = self.pipeline(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator
            ).images
        
        return images

def main():
    """Main function for training."""
    # Configuration
    training_config = TrainingConfig()
    lora_config = LoRAConfig()
    dataset_config = DatasetConfig()
    
    # Create trainer
    trainer = SDXLLoRATrainer(training_config, lora_config, dataset_config)
    
    # Load datasets
    from src.data.dataset import create_ragamala_datasets
    data_module = create_ragamala_datasets(dataset_config)
    
    train_dataloader = data_module.train_dataloader(
        batch_size=training_config.train_batch_size,
        num_workers=4
    )
    val_dataloader = data_module.val_dataloader(
        batch_size=training_config.train_batch_size,
        num_workers=4
    )
    
    # Start training
    trainer.train(train_dataloader, val_dataloader)

if __name__ == "__main__":
    main()
