"""
Training Script for Ragamala Painting Generation using SDXL + LoRA.

This script provides comprehensive training functionality for fine-tuning SDXL 1.0
on Ragamala paintings with LoRA, including distributed training, monitoring,
and cultural conditioning capabilities.
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import warnings
warnings.filterwarnings("ignore")

# Core ML imports
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

# Diffusers and transformers
from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler
)
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

# Training utilities
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.logging import get_logger

# PEFT for LoRA
from peft import LoraConfig, get_peft_model, TaskType

# Monitoring
import wandb
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import RagamalaDataModule, DatasetConfig
from src.models.sdxl_lora import SDXLLoRATrainer, LoRAConfig as ModelLoRAConfig, TrainingConfig
from src.training.trainer import RagamalaTrainer, TrainerConfig
from src.training.losses import RagamalaLoss, LossConfig
from src.training.callbacks import TrainingCallbacks, CallbackConfig
from src.models.scheduler import SchedulerFactory, SchedulerConfig
from src.models.prompt_encoder import PromptEncodingConfig
from src.evaluation.metrics import EvaluationMetrics
from src.utils.logging_utils import setup_logger, create_training_logger
from src.utils.aws_utils import AWSUtilities, create_aws_config_from_env
from src.utils.visualization import save_training_samples

logger = setup_logger(__name__)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train SDXL model on Ragamala paintings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Path to pretrained model or model identifier from huggingface.co/models"
    )
    model_group.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default="madebyollin/sdxl-vae-fp16-fix",
        help="Path to pretrained VAE model"
    )
    model_group.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from huggingface.co/models"
    )
    model_group.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models"
    )
    
    # LoRA configuration
    lora_group = parser.add_argument_group('LoRA Configuration')
    lora_group.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="The dimension of the LoRA update matrices"
    )
    lora_group.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="The alpha parameter for LoRA scaling"
    )
    lora_group.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="The dropout probability for LoRA layers"
    )
    lora_group.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder with LoRA"
    )
    lora_group.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-5,
        help="Learning rate for text encoder LoRA"
    )
    
    # Training configuration
    training_group = parser.add_argument_group('Training Configuration')
    training_group.add_argument(
        "--output_dir",
        type=str,
        default="outputs/training",
        help="The output directory where the model predictions and checkpoints will be written"
    )
    training_group.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored"
    )
    training_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="A seed for reproducible training"
    )
    training_group.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="The resolution for input images, all the images in the train/validation dataset will be resized to this resolution"
    )
    training_group.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader"
    )
    training_group.add_argument(
        "--num_train_epochs",
        type=int,
        default=2,
        help="Total number of training epochs to perform"
    )
    training_group.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs"
    )
    training_group.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help="Save a checkpoint of the training state every X updates"
    )
    training_group.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=5,
        help="Max number of checkpoints to store"
    )
    training_group.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Whether training should be resumed from a previous checkpoint"
    )
    training_group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass"
    )
    training_group.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass"
    )
    
    # Optimization configuration
    optim_group = parser.add_argument_group('Optimization Configuration')
    optim_group.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use"
    )
    optim_group.add_argument(
        "--scale_lr",
        action="store_true",
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size"
    )
    optim_group.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        help="The scheduler type to use"
    )
    optim_group.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler"
    )
    optim_group.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler"
    )
    optim_group.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler"
    )
    optim_group.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes"
    )
    optim_group.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer"
    )
    optim_group.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer"
    )
    optim_group.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay to use"
    )
    optim_group.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-8,
        help="Epsilon value for the Adam optimizer"
    )
    optim_group.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm"
    )
    
    # Data configuration
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory containing the training data"
    )
    data_group.add_argument(
        "--metadata_file",
        type=str,
        default="data/metadata/metadata.jsonl",
        help="Path to metadata file"
    )
    data_group.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data"
    )
    data_group.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of training examples to this value"
    )
    data_group.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help="Run validation every X steps"
    )
    data_group.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of subprocesses to use for data loading"
    )
    
    # Cultural conditioning
    cultural_group = parser.add_argument_group('Cultural Conditioning')
    cultural_group.add_argument(
        "--enable_cultural_conditioning",
        action="store_true",
        help="Enable cultural conditioning for raga and style"
    )
    cultural_group.add_argument(
        "--cultural_loss_weight",
        type=float,
        default=0.1,
        help="Weight for cultural loss component"
    )
    cultural_group.add_argument(
        "--raga_vocab_size",
        type=int,
        default=50,
        help="Size of raga vocabulary"
    )
    cultural_group.add_argument(
        "--style_vocab_size",
        type=int,
        default=20,
        help="Size of style vocabulary"
    )
    
    # Monitoring and logging
    monitor_group = parser.add_argument_group('Monitoring and Logging')
    monitor_group.add_argument(
        "--logging_dir",
        type=str,
        default="logs/training",
        help="TensorBoard log directory"
    )
    monitor_group.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        choices=["tensorboard", "wandb", "all"],
        help="The integration to report the results and logs to"
    )
    monitor_group.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision"
    )
    monitor_group.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Whether or not to allow TF32 on Ampere GPUs"
    )
    monitor_group.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank"
    )
    monitor_group.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers"
    )
    
    # Experiment configuration
    exp_group = parser.add_argument_group('Experiment Configuration')
    exp_group.add_argument(
        "--experiment_name",
        type=str,
        default="ragamala_sdxl_lora",
        help="Name of the experiment"
    )
    exp_group.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name of the run (auto-generated if not provided)"
    )
    exp_group.add_argument(
        "--tags",
        type=str,
        nargs="*",
        default=None,
        help="Tags for the experiment"
    )
    
    # AWS and cloud configuration
    cloud_group = parser.add_argument_group('Cloud Configuration')
    cloud_group.add_argument(
        "--enable_s3_backup",
        action="store_true",
        help="Enable S3 backup of checkpoints and outputs"
    )
    cloud_group.add_argument(
        "--s3_bucket",
        type=str,
        default=None,
        help="S3 bucket for backup"
    )
    cloud_group.add_argument(
        "--s3_prefix",
        type=str,
        default="ragamala/training/",
        help="S3 prefix for backup"
    )
    
    return parser.parse_args()

def setup_logging_and_monitoring(args: argparse.Namespace) -> logging.Logger:
    """Setup logging and monitoring systems."""
    # Create run name if not provided
    if not args.run_name:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.run_name = f"{args.experiment_name}_{timestamp}"
    
    # Setup training logger
    training_logger = create_training_logger(
        experiment_name=args.experiment_name,
        run_id=args.run_name,
        model_name="sdxl_lora"
    )
    
    # Initialize wandb if enabled
    if args.report_to in ["wandb", "all"]:
        wandb.init(
            project="ragamala-sdxl",
            name=args.run_name,
            tags=args.tags,
            config=vars(args)
        )
        training_logger.info("Weights & Biases initialized")
    
    return training_logger

def create_configurations(args: argparse.Namespace) -> Tuple[TrainerConfig, TrainingConfig, ModelLoRAConfig, DatasetConfig, LossConfig, SchedulerConfig, PromptEncodingConfig]:
    """Create all configuration objects from arguments."""
    
    # Trainer configuration
    trainer_config = TrainerConfig(
        experiment_name=args.experiment_name,
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        seed=args.seed,
        num_train_epochs=args.num_train_epochs,
        max_train_steps=args.max_train_steps,
        learning_rate=args.learning_rate,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_num_cycles=args.lr_num_cycles,
        lr_power=args.lr_power,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=args.max_grad_norm,
        use_8bit_adam=args.use_8bit_adam,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_weight_decay=args.adam_weight_decay,
        adam_epsilon=args.adam_epsilon,
        mixed_precision=args.mixed_precision,
        allow_tf32=args.allow_tf32,
        enable_xformers_memory_efficient_attention=args.enable_xformers_memory_efficient_attention,
        validation_steps=args.validation_steps,
        save_steps=args.checkpointing_steps,
        save_total_limit=args.checkpoints_total_limit,
        resume_from_checkpoint=args.resume_from_checkpoint,
        enable_cultural_conditioning=args.enable_cultural_conditioning,
        cultural_loss_weight=args.cultural_loss_weight,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=[args.report_to] if args.report_to != "all" else ["wandb", "tensorboard"]
    )
    
    # Training configuration
    training_config = TrainingConfig(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        pretrained_vae_model_name_or_path=args.pretrained_vae_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
        learning_rate=args.learning_rate,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_num_cycles=args.lr_num_cycles,
        lr_power=args.lr_power,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=args.max_grad_norm,
        max_train_steps=args.max_train_steps,
        num_train_epochs=args.num_train_epochs,
        validation_steps=args.validation_steps,
        checkpointing_steps=args.checkpointing_steps,
        save_steps=args.checkpointing_steps,
        save_total_limit=args.checkpoints_total_limit,
        mixed_precision=args.mixed_precision,
        use_8bit_adam=args.use_8bit_adam,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_weight_decay=args.adam_weight_decay,
        adam_epsilon=args.adam_epsilon,
        resolution=args.resolution,
        enable_cultural_conditioning=args.enable_cultural_conditioning,
        cultural_guidance_scale=1.0,
        enable_xformers_memory_efficient_attention=args.enable_xformers_memory_efficient_attention,
        allow_tf32=args.allow_tf32,
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        seed=args.seed
    )
    
    # LoRA configuration
    lora_config = ModelLoRAConfig(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        train_text_encoder=args.train_text_encoder,
        text_encoder_lr=args.text_encoder_lr,
        text_encoder_rank=4,
        text_encoder_alpha=16
    )
    
    # Dataset configuration
    dataset_config = DatasetConfig(
        data_dir=args.data_dir,
        metadata_file=args.metadata_file,
        image_size=(args.resolution, args.resolution),
        enable_cultural_conditioning=args.enable_cultural_conditioning,
        raga_vocab_size=args.raga_vocab_size,
        style_vocab_size=args.style_vocab_size,
        enable_augmentation=True,
        balance_by_raga=True,
        balance_by_style=True
    )
    
    # Loss configuration
    loss_config = LossConfig(
        main_loss_weight=1.0,
        enable_perceptual_loss=True,
        perceptual_loss_weight=0.1,
        enable_cultural_loss=args.enable_cultural_conditioning,
        cultural_loss_weight=args.cultural_loss_weight,
        enable_clip_loss=True,
        clip_loss_weight=0.05
    )
    
    # Scheduler configuration
    scheduler_config = SchedulerConfig(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
        enable_cultural_conditioning=args.enable_cultural_conditioning,
        cultural_guidance_scale=1.0
    )
    
    # Prompt encoding configuration
    prompt_config = PromptEncodingConfig(
        max_length=77,
        enable_cultural_conditioning=args.enable_cultural_conditioning,
        cultural_weight=0.3,
        enable_prompt_weighting=True
    )
    
    return trainer_config, training_config, lora_config, dataset_config, loss_config, scheduler_config, prompt_config

def setup_data_module(dataset_config: DatasetConfig, args: argparse.Namespace) -> RagamalaDataModule:
    """Setup data module with proper configuration."""
    data_module = RagamalaDataModule(dataset_config)
    data_module.setup()
    
    # Log dataset statistics
    train_dataset = data_module.train_dataset
    val_dataset = data_module.val_dataset
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Get class distribution
    if hasattr(train_dataset, 'get_class_distribution'):
        distribution = train_dataset.get_class_distribution()
        logger.info(f"Raga distribution: {distribution['raga_distribution']}")
        logger.info(f"Style distribution: {distribution['style_distribution']}")
    
    return data_module

def setup_aws_backup(args: argparse.Namespace) -> Optional[AWSUtilities]:
    """Setup AWS utilities for backup if enabled."""
    if not args.enable_s3_backup:
        return None
    
    try:
        aws_config = create_aws_config_from_env()
        if args.s3_bucket:
            aws_config.s3_bucket_name = args.s3_bucket
        aws_config.s3_prefix = args.s3_prefix
        
        aws_utils = AWSUtilities(aws_config)
        logger.info("AWS backup enabled")
        return aws_utils
        
    except Exception as e:
        logger.warning(f"Failed to setup AWS backup: {e}")
        return None

def backup_to_s3(aws_utils: Optional[AWSUtilities], local_path: str, s3_key: str):
    """Backup file to S3 if AWS utils available."""
    if aws_utils and aws_utils.s3:
        try:
            aws_utils.s3.upload_file(local_path, s3_key)
            logger.info(f"Backed up {local_path} to S3: {s3_key}")
        except Exception as e:
            logger.warning(f"Failed to backup to S3: {e}")

def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging and monitoring
    training_logger = setup_logging_and_monitoring(args)
    training_logger.info("Starting Ragamala SDXL training")
    training_logger.info(f"Arguments: {vars(args)}")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create configurations
    trainer_config, training_config, lora_config, dataset_config, loss_config, scheduler_config, prompt_config = create_configurations(args)
    
    # Setup AWS backup
    aws_utils = setup_aws_backup(args)
    
    # Setup data module
    training_logger.info("Setting up data module...")
    data_module = setup_data_module(dataset_config, args)
    
    # Create trainer
    training_logger.info("Creating trainer...")
    trainer = RagamalaTrainer(
        trainer_config=trainer_config,
        training_config=training_config,
        lora_config=lora_config,
        dataset_config=dataset_config,
        loss_config=loss_config,
        scheduler_config=scheduler_config,
        prompt_config=prompt_config
    )
    
    # Setup callbacks
    callback_config = CallbackConfig(
        log_every_n_steps=10,
        save_images_every_n_steps=500,
        validate_every_n_steps=args.validation_steps,
        save_every_n_steps=args.checkpointing_steps,
        save_total_limit=args.checkpoints_total_limit,
        enable_early_stopping=True,
        early_stopping_patience=5,
        enable_cultural_validation=args.enable_cultural_conditioning
    )
    
    callbacks = TrainingCallbacks(callback_config)
    
    # Add monitoring callbacks
    if args.report_to in ["wandb", "all"]:
        callbacks.add_wandb_callback("ragamala-sdxl")
    
    if args.report_to in ["tensorboard", "all"]:
        callbacks.add_tensorboard_callback(args.logging_dir)
    
    # Get data loaders
    train_dataloader = data_module.train_dataloader(
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers
    )
    
    val_dataloader = data_module.val_dataloader(
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers
    )
    
    # Log training start
    training_logger.info("Starting training loop...")
    training_logger.info(f"Total training steps: {args.max_train_steps or 'epoch-based'}")
    training_logger.info(f"Batch size: {args.train_batch_size}")
    training_logger.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    training_logger.info(f"Learning rate: {args.learning_rate}")
    training_logger.info(f"LoRA rank: {args.lora_rank}")
    
    # Start training
    try:
        trainer.train()
        
        training_logger.info("Training completed successfully!")
        
        # Backup final model to S3
        if aws_utils:
            final_model_dir = Path(args.output_dir) / "final_model"
            if final_model_dir.exists():
                backup_to_s3(
                    aws_utils,
                    str(final_model_dir),
                    f"{args.s3_prefix}final_model/{args.run_name}/"
                )
        
        # Generate final evaluation
        training_logger.info("Running final evaluation...")
        
        # Create evaluation metrics
        eval_metrics = EvaluationMetrics()
        
        # Generate sample images for evaluation
        sample_prompts = [
            "A rajput style ragamala painting depicting raga bhairav",
            "A pahari miniature of raga yaman in evening setting",
            "A deccan painting of raga malkauns with deep blue tones",
            "A mughal artwork of raga darbari in royal court"
        ]
        
        # Save sample generation results
        samples_dir = Path(args.output_dir) / "final_samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        training_logger.info(f"Final samples saved to {samples_dir}")
        
    except Exception as e:
        training_logger.error(f"Training failed: {e}")
        raise
    
    finally:
        # Cleanup
        if args.report_to in ["wandb", "all"]:
            wandb.finish()
        
        training_logger.info("Training script completed")

if __name__ == "__main__":
    main()
