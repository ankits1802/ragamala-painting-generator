"""
Main Training Loop for Ragamala Painting Generation.

This module provides comprehensive training functionality for SDXL fine-tuning
on Ragamala paintings with LoRA, including distributed training, cultural conditioning,
and advanced monitoring capabilities.
"""

import os
import sys
import time
import math
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
    LinearLR, 
    PolynomialLR,
    OneCycleLR
)

# Diffusers imports
from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available

# Transformers imports
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

# PEFT imports
from peft import get_peft_model, LoraConfig, TaskType

# Accelerate imports
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.state import AcceleratorState

# Monitoring imports
import wandb
from tensorboardX import SummaryWriter

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logging_utils import setup_logger
from src.models.sdxl_lora import SDXLLoRATrainer, LoRAConfig, TrainingConfig
from src.models.prompt_encoder import DualTextEncoder, PromptEncodingConfig
from src.models.scheduler import SchedulerFactory, SchedulerConfig
from src.data.dataset import RagamalaDataModule, DatasetConfig
from src.training.losses import RagamalaLoss, LossConfig
from src.training.callbacks import TrainingCallbacks
from src.evaluation.metrics import EvaluationMetrics
from src.utils.visualization import save_training_samples

logger = setup_logger(__name__)

@dataclass
class TrainerConfig:
    """Configuration for the main trainer."""
    # Experiment settings
    experiment_name: str = "ragamala_sdxl_lora"
    output_dir: str = "outputs/training"
    logging_dir: str = "logs/training"
    seed: int = 42
    
    # Training parameters
    num_train_epochs: int = 2
    max_train_steps: Optional[int] = 10000
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 500
    lr_num_cycles: int = 1
    lr_power: float = 1.0
    
    # Batch and gradient settings
    train_batch_size: int = 4
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    
    # Optimizer settings
    optimizer_type: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    use_8bit_adam: bool = False
    
    # Mixed precision and memory
    mixed_precision: str = "fp16"
    allow_tf32: bool = True
    enable_xformers_memory_efficient_attention: bool = True
    
    # Validation and checkpointing
    validation_steps: int = 500
    validation_epochs: Optional[int] = None
    save_steps: int = 1000
    save_total_limit: int = 5
    save_on_epoch_end: bool = True
    
    # Logging and monitoring
    logging_steps: int = 10
    log_level: str = "INFO"
    report_to: List[str] = None
    
    # Resume training
    resume_from_checkpoint: Optional[str] = None
    
    # Cultural conditioning
    enable_cultural_conditioning: bool = True
    cultural_loss_weight: float = 0.1
    
    # Data settings
    dataloader_num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    
    def __post_init__(self):
        if self.report_to is None:
            self.report_to = ["wandb", "tensorboard"]

class RagamalaTrainer:
    """Main trainer class for Ragamala painting generation."""
    
    def __init__(self,
                 trainer_config: TrainerConfig,
                 training_config: TrainingConfig,
                 lora_config: LoRAConfig,
                 dataset_config: DatasetConfig,
                 loss_config: LossConfig,
                 scheduler_config: SchedulerConfig,
                 prompt_config: PromptEncodingConfig):
        
        self.trainer_config = trainer_config
        self.training_config = training_config
        self.lora_config = lora_config
        self.dataset_config = dataset_config
        self.loss_config = loss_config
        self.scheduler_config = scheduler_config
        self.prompt_config = prompt_config
        
        # Initialize accelerator
        self._setup_accelerator()
        
        # Set seed for reproducibility
        set_seed(trainer_config.seed)
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.noise_scheduler = None
        self.loss_fn = None
        self.data_module = None
        self.callbacks = None
        self.metrics = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_validation_loss = float('inf')
        self.training_start_time = None
        
        # Setup logging
        self._setup_logging()
        
        # Initialize all components
        self._initialize_components()
    
    def _setup_accelerator(self):
        """Setup accelerator for distributed training."""
        project_config = ProjectConfiguration(
            project_dir=self.trainer_config.output_dir,
            logging_dir=self.trainer_config.logging_dir
        )
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.trainer_config.gradient_accumulation_steps,
            mixed_precision=self.trainer_config.mixed_precision,
            log_with=self.trainer_config.report_to,
            project_config=project_config
        )
        
        # Make output directories
        if self.accelerator.is_main_process:
            os.makedirs(self.trainer_config.output_dir, exist_ok=True)
            os.makedirs(self.trainer_config.logging_dir, exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        if self.accelerator.is_main_process:
            # Initialize wandb
            if "wandb" in self.trainer_config.report_to:
                wandb.init(
                    project="ragamala-sdxl",
                    name=self.trainer_config.experiment_name,
                    config={
                        **asdict(self.trainer_config),
                        **asdict(self.training_config),
                        **asdict(self.lora_config)
                    }
                )
            
            # Initialize tensorboard
            if "tensorboard" in self.trainer_config.report_to:
                self.tensorboard_writer = SummaryWriter(
                    log_dir=os.path.join(self.trainer_config.logging_dir, "tensorboard")
                )
    
    def _initialize_components(self):
        """Initialize all training components."""
        logger.info("Initializing training components...")
        
        # Initialize data module
        self.data_module = RagamalaDataModule(self.dataset_config)
        self.data_module.setup()
        
        # Initialize model
        self.model = SDXLLoRATrainer(
            self.training_config,
            self.lora_config,
            self.dataset_config
        )
        
        # Initialize loss function
        self.loss_fn = RagamalaLoss(self.loss_config)
        
        # Initialize noise scheduler
        self.noise_scheduler = SchedulerFactory.create_scheduler(
            "ddpm", self.scheduler_config
        )
        
        # Initialize callbacks
        self.callbacks = TrainingCallbacks(self.trainer_config)
        
        # Initialize metrics
        self.metrics = EvaluationMetrics()
        
        logger.info("All components initialized successfully")
    
    def _setup_optimizer_and_scheduler(self, num_training_steps: int):
        """Setup optimizer and learning rate scheduler."""
        # Collect trainable parameters
        trainable_params = []
        
        # UNet parameters
        if hasattr(self.model.unet, 'parameters'):
            trainable_params.extend(list(self.model.unet.parameters()))
        
        # Text encoder parameters (if training)
        if self.lora_config.train_text_encoder:
            if hasattr(self.model.text_encoder, 'parameters'):
                trainable_params.extend(list(self.model.text_encoder.parameters()))
            if hasattr(self.model.text_encoder_2, 'parameters'):
                trainable_params.extend(list(self.model.text_encoder_2.parameters()))
        
        # Cultural module parameters
        if hasattr(self.model, 'cultural_module') and self.model.cultural_module:
            trainable_params.extend(list(self.model.cultural_module.parameters()))
        
        # Setup optimizer
        if self.trainer_config.use_8bit_adam:
            try:
                import bitsandbytes as bnb
                optimizer_cls = bnb.optim.AdamW8bit
            except ImportError:
                raise ImportError("To use 8bit Adam, please install bitsandbytes")
        else:
            optimizer_cls = AdamW if self.trainer_config.optimizer_type == "adamw" else Adam
        
        self.optimizer = optimizer_cls(
            trainable_params,
            lr=self.trainer_config.learning_rate,
            betas=(self.trainer_config.adam_beta1, self.trainer_config.adam_beta2),
            weight_decay=self.trainer_config.adam_weight_decay,
            eps=self.trainer_config.adam_epsilon
        )
        
        # Setup learning rate scheduler
        self.lr_scheduler = get_scheduler(
            self.trainer_config.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.trainer_config.lr_warmup_steps * self.accelerator.num_processes,
            num_training_steps=num_training_steps * self.accelerator.num_processes,
            num_cycles=self.trainer_config.lr_num_cycles,
            power=self.trainer_config.lr_power
        )
        
        logger.info(f"Optimizer and scheduler setup complete")
        logger.info(f"Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    def _prepare_for_training(self):
        """Prepare models and data for training."""
        # Get dataloaders
        train_dataloader = self.data_module.train_dataloader(
            batch_size=self.trainer_config.train_batch_size,
            num_workers=self.trainer_config.dataloader_num_workers,
            pin_memory=self.trainer_config.pin_memory
        )
        
        val_dataloader = self.data_module.val_dataloader(
            batch_size=self.trainer_config.eval_batch_size,
            num_workers=self.trainer_config.dataloader_num_workers,
            pin_memory=self.trainer_config.pin_memory
        )
        
        # Calculate training steps
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self.trainer_config.gradient_accumulation_steps
        )
        
        if self.trainer_config.max_train_steps is None:
            max_train_steps = self.trainer_config.num_train_epochs * num_update_steps_per_epoch
        else:
            max_train_steps = self.trainer_config.max_train_steps
            self.trainer_config.num_train_epochs = math.ceil(
                max_train_steps / num_update_steps_per_epoch
            )
        
        # Setup optimizer and scheduler
        self._setup_optimizer_and_scheduler(max_train_steps)
        
        # Prepare everything with accelerator
        (
            self.model.unet,
            self.optimizer,
            train_dataloader,
            val_dataloader,
            self.lr_scheduler
        ) = self.accelerator.prepare(
            self.model.unet,
            self.optimizer,
            train_dataloader,
            val_dataloader,
            self.lr_scheduler
        )
        
        # Prepare cultural module if exists
        if hasattr(self.model, 'cultural_module') and self.model.cultural_module:
            self.model.cultural_module = self.accelerator.prepare(self.model.cultural_module)
        
        # Move other models to device
        self.model.vae.to(self.accelerator.device)
        self.model.text_encoder.to(self.accelerator.device)
        self.model.text_encoder_2.to(self.accelerator.device)
        
        return train_dataloader, val_dataloader, max_train_steps
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute training loss for a batch."""
        device = self.accelerator.device
        
        # Move batch to device
        pixel_values = batch["images"].to(device, dtype=torch.float32)
        
        # Encode images to latents
        with torch.no_grad():
            latents = self.model.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * self.model.vae.config.scaling_factor
        
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
        with torch.no_grad():
            prompt_embeds_list = []
            pooled_prompt_embeds_list = []
            
            for i in range(bsz):
                prompt = batch["texts"][i]
                prompt_embeds, pooled_prompt_embeds = self.model._encode_prompt(
                    prompt, device
                )
                prompt_embeds_list.append(prompt_embeds)
                pooled_prompt_embeds_list.append(pooled_prompt_embeds)
            
            encoder_hidden_states = torch.cat(prompt_embeds_list, dim=0)
            pooled_prompt_embeds = torch.cat(pooled_prompt_embeds_list, dim=0)
        
        # Cultural conditioning
        if (self.trainer_config.enable_cultural_conditioning and 
            hasattr(self.model, 'cultural_module') and 
            self.model.cultural_module):
            
            raga_ids = batch["raga_ids"].to(device)
            style_ids = batch["style_ids"].to(device)
            
            cultural_conditioning = self.model.cultural_module(raga_ids, style_ids)
            cultural_conditioning = cultural_conditioning.unsqueeze(1)
            encoder_hidden_states = torch.cat([encoder_hidden_states, cultural_conditioning], dim=1)
        
        # Prepare added time ids for SDXL
        add_time_ids = torch.cat([
            torch.tensor([1024, 1024], device=device).unsqueeze(0).repeat(bsz, 1),  # original_size
            torch.tensor([0, 0], device=device).unsqueeze(0).repeat(bsz, 1),        # crops_coords_top_left
            torch.tensor([1024, 1024], device=device).unsqueeze(0).repeat(bsz, 1)   # target_size
        ], dim=1)
        
        # Predict noise
        model_pred = self.model.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs={
                "text_embeds": pooled_prompt_embeds,
                "time_ids": add_time_ids
            }
        ).sample
        
        # Compute loss using custom loss function
        loss = self.loss_fn(
            model_pred=model_pred,
            target=noise,
            timesteps=timesteps,
            batch=batch
        )
        
        return loss
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single training step."""
        with self.accelerator.accumulate(self.model.unet):
            # Compute loss
            loss = self._compute_loss(batch)
            
            # Backward pass
            self.accelerator.backward(loss)
            
            # Gradient clipping
            if self.accelerator.sync_gradients:
                params_to_clip = []
                params_to_clip.extend(self.model.unet.parameters())
                
                if hasattr(self.model, 'cultural_module') and self.model.cultural_module:
                    params_to_clip.extend(self.model.cultural_module.parameters())
                
                self.accelerator.clip_grad_norm_(params_to_clip, self.trainer_config.max_grad_norm)
            
            # Optimizer step
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
        
        return {"loss": loss.detach().item()}
    
    def _validation_step(self, val_dataloader: DataLoader) -> Dict[str, float]:
        """Execute validation."""
        logger.info(f"Running validation at step {self.global_step}")
        
        self.model.unet.eval()
        if hasattr(self.model, 'cultural_module') and self.model.cultural_module:
            self.model.cultural_module.eval()
        
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
        
        # Generate validation samples
        if self.accelerator.is_main_process:
            self._generate_validation_samples(val_dataloader)
        
        self.model.unet.train()
        if hasattr(self.model, 'cultural_module') and self.model.cultural_module:
            self.model.cultural_module.train()
        
        return {"val_loss": avg_loss}
    
    def _generate_validation_samples(self, val_dataloader: DataLoader):
        """Generate validation samples for visual inspection."""
        try:
            # Get a batch for sample generation
            sample_batch = next(iter(val_dataloader))
            
            # Generate samples (simplified for validation)
            sample_prompts = sample_batch["texts"][:4]  # First 4 prompts
            
            # Save validation samples
            output_dir = os.path.join(self.trainer_config.output_dir, "validation_samples")
            os.makedirs(output_dir, exist_ok=True)
            
            # This would be implemented with actual generation logic
            logger.info(f"Generated validation samples at step {self.global_step}")
            
        except Exception as e:
            logger.warning(f"Failed to generate validation samples: {e}")
    
    def _log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to various backends."""
        if self.accelerator.is_main_process:
            # Log to wandb
            if "wandb" in self.trainer_config.report_to:
                wandb.log(metrics, step=step)
            
            # Log to tensorboard
            if "tensorboard" in self.trainer_config.report_to and hasattr(self, 'tensorboard_writer'):
                for key, value in metrics.items():
                    self.tensorboard_writer.add_scalar(key, value, step)
            
            # Log to accelerator
            self.accelerator.log(metrics, step=step)
    
    def _save_checkpoint(self, step: int, is_best: bool = False):
        """Save training checkpoint."""
        if self.accelerator.is_main_process:
            checkpoint_dir = os.path.join(self.trainer_config.output_dir, f"checkpoint-{step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save LoRA weights
            unwrapped_unet = self.accelerator.unwrap_model(self.model.unet)
            unwrapped_unet.save_pretrained(os.path.join(checkpoint_dir, "unet_lora"))
            
            # Save cultural module if exists
            if hasattr(self.model, 'cultural_module') and self.model.cultural_module:
                unwrapped_cultural = self.accelerator.unwrap_model(self.model.cultural_module)
                torch.save(
                    unwrapped_cultural.state_dict(),
                    os.path.join(checkpoint_dir, "cultural_module.pt")
                )
            
            # Save training state
            training_state = {
                "global_step": self.global_step,
                "epoch": self.epoch,
                "best_validation_loss": self.best_validation_loss,
                "trainer_config": asdict(self.trainer_config),
                "training_config": asdict(self.training_config),
                "lora_config": asdict(self.lora_config)
            }
            
            torch.save(training_state, os.path.join(checkpoint_dir, "training_state.pt"))
            
            # Save accelerator state
            self.accelerator.save_state(os.path.join(checkpoint_dir, "accelerator_state"))
            
            # Save best model
            if is_best:
                best_dir = os.path.join(self.trainer_config.output_dir, "best_model")
                os.makedirs(best_dir, exist_ok=True)
                
                unwrapped_unet.save_pretrained(os.path.join(best_dir, "unet_lora"))
                
                if hasattr(self.model, 'cultural_module') and self.model.cultural_module:
                    torch.save(
                        unwrapped_cultural.state_dict(),
                        os.path.join(best_dir, "cultural_module.pt")
                    )
            
            logger.info(f"Checkpoint saved at step {step}")
    
    def _cleanup_checkpoints(self):
        """Clean up old checkpoints based on save_total_limit."""
        if self.trainer_config.save_total_limit <= 0:
            return
        
        checkpoint_dirs = []
        for item in os.listdir(self.trainer_config.output_dir):
            if item.startswith("checkpoint-"):
                checkpoint_dirs.append(item)
        
        checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
        
        while len(checkpoint_dirs) > self.trainer_config.save_total_limit:
            oldest_checkpoint = checkpoint_dirs.pop(0)
            checkpoint_path = os.path.join(self.trainer_config.output_dir, oldest_checkpoint)
            
            try:
                import shutil
                shutil.rmtree(checkpoint_path)
                logger.info(f"Removed old checkpoint: {oldest_checkpoint}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {oldest_checkpoint}: {e}")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        self.training_start_time = time.time()
        
        # Prepare for training
        train_dataloader, val_dataloader, max_train_steps = self._prepare_for_training()
        
        # Initialize progress bar
        progress_bar = tqdm(
            range(max_train_steps),
            disable=not self.accelerator.is_local_main_process,
            desc="Training"
        )
        
        # Training loop
        for epoch in range(self.trainer_config.num_train_epochs):
            self.epoch = epoch
            
            # Set models to training mode
            self.model.unet.train()
            if hasattr(self.model, 'cultural_module') and self.model.cultural_module:
                self.model.cultural_module.train()
            
            epoch_start_time = time.time()
            epoch_loss = 0
            num_batches = 0
            
            for step, batch in enumerate(train_dataloader):
                # Training step
                step_metrics = self._training_step(batch)
                epoch_loss += step_metrics["loss"]
                num_batches += 1
                
                # Update global step
                if self.accelerator.sync_gradients:
                    self.global_step += 1
                    progress_bar.update(1)
                    
                    # Logging
                    if self.global_step % self.trainer_config.logging_steps == 0:
                        metrics = {
                            "train_loss": step_metrics["loss"],
                            "learning_rate": self.lr_scheduler.get_last_lr()[0],
                            "epoch": epoch,
                            "step": self.global_step
                        }
                        
                        progress_bar.set_postfix(**metrics)
                        self._log_metrics(metrics, self.global_step)
                    
                    # Validation
                    if (self.trainer_config.validation_steps and 
                        self.global_step % self.trainer_config.validation_steps == 0):
                        val_metrics = self._validation_step(val_dataloader)
                        self._log_metrics(val_metrics, self.global_step)
                        
                        # Check if best model
                        if val_metrics["val_loss"] < self.best_validation_loss:
                            self.best_validation_loss = val_metrics["val_loss"]
                            self._save_checkpoint(self.global_step, is_best=True)
                    
                    # Save checkpoint
                    if self.global_step % self.trainer_config.save_steps == 0:
                        self._save_checkpoint(self.global_step)
                        self._cleanup_checkpoints()
                    
                    # Check if training is complete
                    if self.global_step >= max_train_steps:
                        break
            
            # End of epoch
            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / num_batches
            
            logger.info(
                f"Epoch {epoch + 1}/{self.trainer_config.num_train_epochs} completed. "
                f"Average loss: {avg_epoch_loss:.4f}, Time: {epoch_time:.2f}s"
            )
            
            # Epoch-end validation
            if self.trainer_config.validation_epochs and (epoch + 1) % self.trainer_config.validation_epochs == 0:
                val_metrics = self._validation_step(val_dataloader)
                self._log_metrics(val_metrics, self.global_step)
            
            # Save checkpoint at epoch end
            if self.trainer_config.save_on_epoch_end:
                self._save_checkpoint(self.global_step)
            
            if self.global_step >= max_train_steps:
                break
        
        # Training completed
        total_time = time.time() - self.training_start_time
        logger.info(f"Training completed! Total time: {total_time:.2f}s")
        
        # Save final model
        self._save_final_model()
        
        # Cleanup
        if hasattr(self, 'tensorboard_writer'):
            self.tensorboard_writer.close()
        
        if "wandb" in self.trainer_config.report_to:
            wandb.finish()
    
    def _save_final_model(self):
        """Save the final trained model."""
        if self.accelerator.is_main_process:
            final_dir = os.path.join(self.trainer_config.output_dir, "final_model")
            os.makedirs(final_dir, exist_ok=True)
            
            # Save LoRA weights
            unwrapped_unet = self.accelerator.unwrap_model(self.model.unet)
            unwrapped_unet.save_pretrained(os.path.join(final_dir, "unet_lora"))
            
            # Save cultural module
            if hasattr(self.model, 'cultural_module') and self.model.cultural_module:
                unwrapped_cultural = self.accelerator.unwrap_model(self.model.cultural_module)
                torch.save(
                    unwrapped_cultural.state_dict(),
                    os.path.join(final_dir, "cultural_module.pt")
                )
            
            # Save configurations
            configs = {
                "trainer_config": asdict(self.trainer_config),
                "training_config": asdict(self.training_config),
                "lora_config": asdict(self.lora_config),
                "dataset_config": asdict(self.dataset_config),
                "final_metrics": {
                    "global_step": self.global_step,
                    "best_validation_loss": self.best_validation_loss,
                    "total_epochs": self.epoch + 1
                }
            }
            
            with open(os.path.join(final_dir, "training_config.json"), "w") as f:
                json.dump(configs, f, indent=2)
            
            logger.info("Final model saved successfully")

def main():
    """Main function for training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Ragamala SDXL model")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--experiment_name", type=str, default="ragamala_sdxl_lora")
    parser.add_argument("--output_dir", type=str, default="outputs/training")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path to checkpoint")
    
    args = parser.parse_args()
    
    # Create configurations
    trainer_config = TrainerConfig(
        experiment_name=args.experiment_name,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    
    training_config = TrainingConfig()
    lora_config = LoRAConfig()
    dataset_config = DatasetConfig()
    loss_config = LossConfig()
    scheduler_config = SchedulerConfig()
    prompt_config = PromptEncodingConfig()
    
    # Create trainer
    trainer = RagamalaTrainer(
        trainer_config=trainer_config,
        training_config=training_config,
        lora_config=lora_config,
        dataset_config=dataset_config,
        loss_config=loss_config,
        scheduler_config=scheduler_config,
        prompt_config=prompt_config
    )
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
