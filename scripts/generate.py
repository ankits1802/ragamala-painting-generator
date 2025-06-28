"""
Inference Script for Ragamala Painting Generation using SDXL + LoRA.

This script provides comprehensive image generation functionality for creating
Ragamala paintings with cultural conditioning, batch processing, and various
generation modes including text-to-image, image-to-image, and inpainting.
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import warnings
warnings.filterwarnings("ignore")

# Core ML imports
import torch
import numpy as np
from PIL import Image

# Diffusers and transformers
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline
)

# Generation utilities
from tqdm import tqdm
import gradio as gr

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.inference.generator import RagamalaGenerator, GenerationConfig, GenerationRequest, GenerationResult
from src.inference.prompt_templates import PromptTemplateManager, CulturalContext, create_cultural_context
from src.inference.post_processor import RagamalaPostProcessor, PostProcessingConfig
from src.utils.logging_utils import setup_logger, create_training_logger
from src.utils.visualization import RagamalaVisualizer
from src.utils.aws_utils import AWSUtilities, create_aws_config_from_env

logger = setup_logger(__name__)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Ragamala paintings using fine-tuned SDXL model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        "--model_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Path to the base SDXL model"
    )
    model_group.add_argument(
        "--lora_weights_path",
        type=str,
        default=None,
        help="Path to LoRA weights for fine-tuned model"
    )
    model_group.add_argument(
        "--vae_model_path",
        type=str,
        default="madebyollin/sdxl-vae-fp16-fix",
        help="Path to VAE model"
    )
    model_group.add_argument(
        "--refiner_model_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-refiner-1.0",
        help="Path to SDXL refiner model"
    )
    model_group.add_argument(
        "--use_refiner",
        action="store_true",
        help="Use SDXL refiner for post-processing"
    )
    
    # Generation configuration
    gen_group = parser.add_argument_group('Generation Configuration')
    gen_group.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for generation"
    )
    gen_group.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="File containing multiple prompts (one per line)"
    )
    gen_group.add_argument(
        "--raga",
        type=str,
        default=None,
        choices=["bhairav", "yaman", "malkauns", "darbari", "bageshri", "todi"],
        help="Raga for cultural conditioning"
    )
    gen_group.add_argument(
        "--style",
        type=str,
        default=None,
        choices=["rajput", "pahari", "deccan", "mughal"],
        help="Painting style for cultural conditioning"
    )
    gen_group.add_argument(
        "--cultural_template",
        type=str,
        default="detailed",
        choices=["basic", "detailed", "cultural", "atmospheric", "compositional", "artistic", "narrative"],
        help="Cultural prompt template type"
    )
    gen_group.add_argument(
        "--negative_prompt",
        type=str,
        default="blurry, low quality, distorted, modern, western art, cartoon, anime",
        help="Negative prompt"
    )
    gen_group.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="Number of denoising steps"
    )
    gen_group.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale for generation"
    )
    gen_group.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Width of generated images"
    )
    gen_group.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Height of generated images"
    )
    gen_group.add_argument(
        "--num_images",
        type=int,
        default=1,
        help="Number of images to generate per prompt"
    )
    gen_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation"
    )
    gen_group.add_argument(
        "--scheduler",
        type=str,
        default="dpm_solver",
        choices=["ddim", "dpm_solver", "euler", "euler_ancestral"],
        help="Scheduler type for generation"
    )
    
    # Generation modes
    mode_group = parser.add_argument_group('Generation Modes')
    mode_group.add_argument(
        "--mode",
        type=str,
        default="text2img",
        choices=["text2img", "img2img", "inpaint", "batch", "interactive"],
        help="Generation mode"
    )
    mode_group.add_argument(
        "--init_image",
        type=str,
        default=None,
        help="Initial image for img2img mode"
    )
    mode_group.add_argument(
        "--mask_image",
        type=str,
        default=None,
        help="Mask image for inpainting mode"
    )
    mode_group.add_argument(
        "--strength",
        type=float,
        default=0.8,
        help="Strength for img2img and inpainting (0.0-1.0)"
    )
    
    # Batch processing
    batch_group = parser.add_argument_group('Batch Processing')
    batch_group.add_argument(
        "--batch_config",
        type=str,
        default=None,
        help="JSON file with batch generation configuration"
    )
    batch_group.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for processing multiple prompts"
    )
    
    # Output configuration
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument(
        "--output_dir",
        type=str,
        default="outputs/generated",
        help="Directory to save generated images"
    )
    output_group.add_argument(
        "--save_metadata",
        action="store_true",
        help="Save generation metadata with images"
    )
    output_group.add_argument(
        "--output_format",
        type=str,
        default="png",
        choices=["png", "jpg", "webp"],
        help="Output image format"
    )
    output_group.add_argument(
        "--quality",
        type=int,
        default=95,
        help="Output image quality (for jpg/webp)"
    )
    
    # Post-processing
    post_group = parser.add_argument_group('Post-processing')
    post_group.add_argument(
        "--enable_post_processing",
        action="store_true",
        help="Enable post-processing of generated images"
    )
    post_group.add_argument(
        "--super_resolution",
        action="store_true",
        help="Apply super-resolution to generated images"
    )
    post_group.add_argument(
        "--color_correction",
        action="store_true",
        help="Apply color correction"
    )
    post_group.add_argument(
        "--cultural_refinement",
        action="store_true",
        help="Apply cultural style refinement"
    )
    
    # Interactive mode
    interactive_group = parser.add_argument_group('Interactive Mode')
    interactive_group.add_argument(
        "--interface",
        type=str,
        default="gradio",
        choices=["gradio", "cli"],
        help="Interface type for interactive mode"
    )
    interactive_group.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for Gradio interface"
    )
    interactive_group.add_argument(
        "--share",
        action="store_true",
        help="Create public Gradio link"
    )
    
    # Cloud configuration
    cloud_group = parser.add_argument_group('Cloud Configuration')
    cloud_group.add_argument(
        "--enable_s3_upload",
        action="store_true",
        help="Upload generated images to S3"
    )
    cloud_group.add_argument(
        "--s3_bucket",
        type=str,
        default=None,
        help="S3 bucket for uploads"
    )
    cloud_group.add_argument(
        "--s3_prefix",
        type=str,
        default="ragamala/generated/",
        help="S3 prefix for uploads"
    )
    
    return parser.parse_args()

def setup_logging_and_monitoring(args: argparse.Namespace) -> logging.Logger:
    """Setup logging for generation."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    gen_logger = create_training_logger(
        experiment_name=f"generation_{args.mode}",
        run_id=f"gen_{timestamp}",
        model_name="sdxl_ragamala"
    )
    
    gen_logger.info("Starting Ragamala image generation")
    gen_logger.info(f"Generation mode: {args.mode}")
    gen_logger.info(f"Model path: {args.model_path}")
    
    return gen_logger

def load_generator(args: argparse.Namespace) -> RagamalaGenerator:
    """Load the generator with proper configuration."""
    logger.info("Loading generator...")
    
    # Create generation configuration
    gen_config = GenerationConfig(
        model_path=args.model_path,
        lora_weights_path=args.lora_weights_path,
        refiner_model_path=args.refiner_model_path if args.use_refiner else None,
        vae_model_path=args.vae_model_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_images_per_prompt=args.num_images,
        generator_seed=args.seed,
        use_refiner=args.use_refiner,
        scheduler_type=args.scheduler
    )
    
    # Create generator
    generator = RagamalaGenerator(gen_config)
    
    logger.info("Generator loaded successfully")
    return generator

def setup_post_processor(args: argparse.Namespace) -> Optional[RagamalaPostProcessor]:
    """Setup post-processor if enabled."""
    if not args.enable_post_processing:
        return None
    
    logger.info("Setting up post-processor...")
    
    post_config = PostProcessingConfig(
        enable_super_resolution=args.super_resolution,
        enable_color_correction=args.color_correction,
        enable_cultural_refinement=args.cultural_refinement,
        output_format=args.output_format.upper(),
        output_quality=args.quality
    )
    
    post_processor = RagamalaPostProcessor(post_config)
    
    logger.info("Post-processor setup complete")
    return post_processor

def load_prompts(args: argparse.Namespace) -> List[str]:
    """Load prompts from file or command line."""
    prompts = []
    
    if args.prompts_file and Path(args.prompts_file).exists():
        logger.info(f"Loading prompts from {args.prompts_file}")
        with open(args.prompts_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
    elif args.prompt:
        prompts = [args.prompt]
    else:
        # Default prompts for demonstration
        prompts = [
            "A traditional ragamala painting",
            "An exquisite miniature artwork depicting a classical raga"
        ]
    
    logger.info(f"Loaded {len(prompts)} prompts")
    return prompts

def enhance_prompts_with_cultural_context(prompts: List[str], 
                                        args: argparse.Namespace) -> List[str]:
    """Enhance prompts with cultural context."""
    if not args.raga and not args.style:
        return prompts
    
    logger.info("Enhancing prompts with cultural context...")
    
    # Create prompt template manager
    template_manager = PromptTemplateManager()
    
    # Create cultural context
    cultural_context = create_cultural_context(
        raga=args.raga,
        style=args.style
    )
    
    enhanced_prompts = []
    for prompt in prompts:
        try:
            prompt_data = template_manager.generate_prompt(
                template_name=args.cultural_template,
                cultural_context=cultural_context,
                base_prompt=prompt
            )
            enhanced_prompts.append(prompt_data['prompt'])
        except Exception as e:
            logger.warning(f"Failed to enhance prompt '{prompt}': {e}")
            enhanced_prompts.append(prompt)
    
    return enhanced_prompts

def generate_single_image(generator: RagamalaGenerator,
                         prompt: str,
                         args: argparse.Namespace) -> GenerationResult:
    """Generate a single image."""
    # Create generation request
    request = GenerationRequest(
        prompt=prompt,
        raga=args.raga,
        style=args.style,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_images=args.num_images,
        seed=args.seed,
        scheduler_type=args.scheduler,
        cultural_template=args.cultural_template
    )
    
    # Add mode-specific parameters
    if args.mode == "img2img" and args.init_image:
        request.init_image = Image.open(args.init_image).convert('RGB')
        request.strength = args.strength
    elif args.mode == "inpaint" and args.init_image and args.mask_image:
        request.init_image = Image.open(args.init_image).convert('RGB')
        request.mask_image = Image.open(args.mask_image).convert('RGB')
        request.strength = args.strength
    
    # Generate image
    result = generator.generate(request)
    
    return result

def generate_batch_images(generator: RagamalaGenerator,
                         prompts: List[str],
                         args: argparse.Namespace) -> List[GenerationResult]:
    """Generate multiple images in batch."""
    logger.info(f"Generating {len(prompts)} images in batch...")
    
    results = []
    
    # Process in batches
    for i in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[i:i + args.batch_size]
        
        batch_requests = []
        for j, prompt in enumerate(batch_prompts):
            request = GenerationRequest(
                prompt=prompt,
                raga=args.raga,
                style=args.style,
                negative_prompt=args.negative_prompt,
                width=args.width,
                height=args.height,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                num_images=args.num_images,
                seed=args.seed + i + j if args.seed else None,
                scheduler_type=args.scheduler,
                cultural_template=args.cultural_template
            )
            batch_requests.append(request)
        
        # Generate batch
        batch_results = generator.generate_batch(batch_requests)
        results.extend(batch_results)
        
        logger.info(f"Completed batch {i // args.batch_size + 1}/{(len(prompts) - 1) // args.batch_size + 1}")
    
    return results

def process_batch_config(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Process batch configuration file."""
    if not args.batch_config or not Path(args.batch_config).exists():
        raise ValueError("Batch config file not found")
    
    with open(args.batch_config, 'r') as f:
        batch_config = json.load(f)
    
    return batch_config.get('generations', [])

def save_generation_results(results: Union[GenerationResult, List[GenerationResult]],
                           args: argparse.Namespace,
                           post_processor: Optional[RagamalaPostProcessor] = None) -> List[str]:
    """Save generation results to disk."""
    if isinstance(results, GenerationResult):
        results = [results]
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    for i, result in enumerate(results):
        for j, image in enumerate(result.images):
            # Apply post-processing if enabled
            if post_processor:
                try:
                    processed_result = post_processor.process_image(
                        image=image,
                        style=args.style,
                        raga=args.raga,
                        metadata=result.metadata
                    )
                    image = processed_result.processed_image
                except Exception as e:
                    logger.warning(f"Post-processing failed: {e}")
            
            # Create filename
            timestamp = int(time.time())
            filename = f"ragamala_{timestamp}_{i:04d}_{j:02d}.{args.output_format}"
            image_path = output_dir / filename
            
            # Save image
            if args.output_format.lower() == 'jpg':
                image.save(image_path, format='JPEG', quality=args.quality)
            elif args.output_format.lower() == 'webp':
                image.save(image_path, format='WEBP', quality=args.quality)
            else:
                image.save(image_path, format='PNG')
            
            saved_paths.append(str(image_path))
            
            # Save metadata if requested
            if args.save_metadata:
                metadata = {
                    'prompt': result.prompt_used,
                    'raga': args.raga,
                    'style': args.style,
                    'generation_time': result.generation_time,
                    'config': asdict(result.config_used),
                    'metadata': result.metadata,
                    'seeds_used': result.seeds_used
                }
                
                metadata_path = output_dir / f"{filename.rsplit('.', 1)[0]}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Saved {len(saved_paths)} images to {output_dir}")
    return saved_paths

def upload_to_s3(file_paths: List[str], args: argparse.Namespace):
    """Upload generated images to S3."""
    if not args.enable_s3_upload:
        return
    
    try:
        logger.info("Uploading images to S3...")
        
        aws_config = create_aws_config_from_env()
        if args.s3_bucket:
            aws_config.s3_bucket_name = args.s3_bucket
        aws_config.s3_prefix = args.s3_prefix
        
        aws_utils = AWSUtilities(aws_config)
        
        for file_path in file_paths:
            file_name = Path(file_path).name
            s3_key = f"{args.s3_prefix}{file_name}"
            
            aws_utils.s3.upload_file(file_path, s3_key)
        
        logger.info(f"Uploaded {len(file_paths)} images to S3")
        
    except Exception as e:
        logger.warning(f"S3 upload failed: {e}")

def create_gradio_interface(generator: RagamalaGenerator,
                          post_processor: Optional[RagamalaPostProcessor],
                          args: argparse.Namespace) -> gr.Interface:
    """Create Gradio interface for interactive generation."""
    
    def generate_with_gradio(prompt: str,
                           raga: str,
                           style: str,
                           negative_prompt: str,
                           num_inference_steps: int,
                           guidance_scale: float,
                           width: int,
                           height: int,
                           seed: int,
                           use_cultural_template: bool) -> Tuple[Image.Image, str]:
        """Generate image through Gradio interface."""
        try:
            # Enhance prompt if cultural template is enabled
            if use_cultural_template and (raga != "None" or style != "None"):
                template_manager = PromptTemplateManager()
                cultural_context = create_cultural_context(
                    raga=raga if raga != "None" else None,
                    style=style if style != "None" else None
                )
                
                prompt_data = template_manager.generate_prompt(
                    template_name="detailed",
                    cultural_context=cultural_context,
                    base_prompt=prompt
                )
                enhanced_prompt = prompt_data['prompt']
            else:
                enhanced_prompt = prompt
            
            # Create generation request
            request = GenerationRequest(
                prompt=enhanced_prompt,
                raga=raga if raga != "None" else None,
                style=style if style != "None" else None,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images=1,
                seed=seed if seed != -1 else None
            )
            
            # Generate image
            result = generator.generate(request)
            image = result.images[0]
            
            # Apply post-processing if available
            if post_processor:
                processed_result = post_processor.process_image(
                    image=image,
                    style=style if style != "None" else None,
                    raga=raga if raga != "None" else None
                )
                image = processed_result.processed_image
            
            # Create info text
            info = f"Generation completed in {result.generation_time:.2f}s\n"
            info += f"Enhanced prompt: {enhanced_prompt}\n"
            info += f"Raga: {raga}, Style: {style}"
            
            return image, info
            
        except Exception as e:
            error_image = Image.new('RGB', (512, 512), color='red')
            return error_image, f"Error: {str(e)}"
    
    # Create interface
    interface = gr.Interface(
        fn=generate_with_gradio,
        inputs=[
            gr.Textbox(label="Prompt", placeholder="Enter your prompt here..."),
            gr.Dropdown(
                choices=["None", "bhairav", "yaman", "malkauns", "darbari", "bageshri", "todi"],
                label="Raga",
                value="None"
            ),
            gr.Dropdown(
                choices=["None", "rajput", "pahari", "deccan", "mughal"],
                label="Style",
                value="None"
            ),
            gr.Textbox(
                label="Negative Prompt",
                value=args.negative_prompt
            ),
            gr.Slider(
                minimum=10,
                maximum=100,
                value=args.num_inference_steps,
                step=1,
                label="Inference Steps"
            ),
            gr.Slider(
                minimum=1.0,
                maximum=20.0,
                value=args.guidance_scale,
                step=0.5,
                label="Guidance Scale"
            ),
            gr.Slider(
                minimum=512,
                maximum=1536,
                value=args.width,
                step=64,
                label="Width"
            ),
            gr.Slider(
                minimum=512,
                maximum=1536,
                value=args.height,
                step=64,
                label="Height"
            ),
            gr.Number(
                label="Seed (-1 for random)",
                value=-1
            ),
            gr.Checkbox(
                label="Use Cultural Template",
                value=True
            )
        ],
        outputs=[
            gr.Image(label="Generated Image"),
            gr.Textbox(label="Generation Info")
        ],
        title="Ragamala Painting Generator",
        description="Generate traditional Ragamala paintings using SDXL with cultural conditioning",
        examples=[
            ["A beautiful traditional painting", "bhairav", "rajput", "", 30, 7.5, 1024, 1024, -1, True],
            ["Romantic scene in a garden", "yaman", "pahari", "", 30, 7.5, 1024, 1024, -1, True],
            ["Meditative figure by a river", "malkauns", "deccan", "", 30, 7.5, 1024, 1024, -1, True]
        ]
    )
    
    return interface

def run_interactive_cli(generator: RagamalaGenerator,
                       post_processor: Optional[RagamalaPostProcessor],
                       args: argparse.Namespace):
    """Run interactive CLI mode."""
    logger.info("Starting interactive CLI mode...")
    
    print("\n" + "="*60)
    print("         Ragamala Painting Generator - Interactive Mode")
    print("="*60)
    print("Commands:")
    print("  generate <prompt> - Generate image from prompt")
    print("  set raga <raga> - Set raga for cultural conditioning")
    print("  set style <style> - Set style for cultural conditioning")
    print("  set steps <num> - Set inference steps")
    print("  set guidance <scale> - Set guidance scale")
    print("  show config - Show current configuration")
    print("  help - Show this help")
    print("  exit - Exit interactive mode")
    print("="*60)
    
    # Current settings
    current_raga = args.raga
    current_style = args.style
    current_steps = args.num_inference_steps
    current_guidance = args.guidance_scale
    
    while True:
        try:
            command = input("\nragamala> ").strip()
            
            if not command:
                continue
            
            parts = command.split()
            cmd = parts[0].lower()
            
            if cmd == "exit":
                break
            elif cmd == "help":
                print("Available commands: generate, set, show, help, exit")
            elif cmd == "show" and len(parts) > 1 and parts[1] == "config":
                print(f"Current configuration:")
                print(f"  Raga: {current_raga or 'None'}")
                print(f"  Style: {current_style or 'None'}")
                print(f"  Inference steps: {current_steps}")
                print(f"  Guidance scale: {current_guidance}")
            elif cmd == "set" and len(parts) >= 3:
                setting = parts[1].lower()
                value = " ".join(parts[2:])
                
                if setting == "raga":
                    if value in ["bhairav", "yaman", "malkauns", "darbari", "bageshri", "todi", "none"]:
                        current_raga = value if value != "none" else None
                        print(f"Raga set to: {current_raga or 'None'}")
                    else:
                        print("Invalid raga. Choose from: bhairav, yaman, malkauns, darbari, bageshri, todi, none")
                elif setting == "style":
                    if value in ["rajput", "pahari", "deccan", "mughal", "none"]:
                        current_style = value if value != "none" else None
                        print(f"Style set to: {current_style or 'None'}")
                    else:
                        print("Invalid style. Choose from: rajput, pahari, deccan, mughal, none")
                elif setting == "steps":
                    try:
                        current_steps = int(value)
                        print(f"Inference steps set to: {current_steps}")
                    except ValueError:
                        print("Invalid number for steps")
                elif setting == "guidance":
                    try:
                        current_guidance = float(value)
                        print(f"Guidance scale set to: {current_guidance}")
                    except ValueError:
                        print("Invalid number for guidance scale")
                else:
                    print("Unknown setting. Available: raga, style, steps, guidance")
            elif cmd == "generate" and len(parts) > 1:
                prompt = " ".join(parts[1:])
                print(f"Generating image for: {prompt}")
                
                # Create request
                request = GenerationRequest(
                    prompt=prompt,
                    raga=current_raga,
                    style=current_style,
                    num_inference_steps=current_steps,
                    guidance_scale=current_guidance,
                    width=args.width,
                    height=args.height,
                    num_images=1
                )
                
                # Generate
                try:
                    result = generator.generate(request)
                    
                    # Save image
                    timestamp = int(time.time())
                    filename = f"interactive_{timestamp}.png"
                    output_path = Path(args.output_dir) / filename
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    image = result.images[0]
                    
                    # Apply post-processing if available
                    if post_processor:
                        processed_result = post_processor.process_image(
                            image=image,
                            style=current_style,
                            raga=current_raga
                        )
                        image = processed_result.processed_image
                    
                    image.save(output_path)
                    
                    print(f"Image generated and saved to: {output_path}")
                    print(f"Generation time: {result.generation_time:.2f}s")
                    
                except Exception as e:
                    print(f"Generation failed: {e}")
            else:
                print("Unknown command. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main generation function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    gen_logger = setup_logging_and_monitoring(args)
    
    # Load generator
    generator = load_generator(args)
    
    # Setup post-processor
    post_processor = setup_post_processor(args)
    
    # Handle different modes
    if args.mode == "interactive":
        if args.interface == "gradio":
            # Create and launch Gradio interface
            interface = create_gradio_interface(generator, post_processor, args)
            interface.launch(server_port=args.port, share=args.share)
        else:
            # Run CLI interactive mode
            run_interactive_cli(generator, post_processor, args)
    
    elif args.mode == "batch":
        # Process batch configuration
        batch_configs = process_batch_config(args)
        
        all_results = []
        for config in batch_configs:
            # Override args with batch config
            batch_args = argparse.Namespace(**{**vars(args), **config})
            
            # Load and enhance prompts
            prompts = load_prompts(batch_args)
            enhanced_prompts = enhance_prompts_with_cultural_context(prompts, batch_args)
            
            # Generate images
            results = generate_batch_images(generator, enhanced_prompts, batch_args)
            all_results.extend(results)
        
        # Save results
        saved_paths = save_generation_results(all_results, args, post_processor)
        
        # Upload to S3 if enabled
        upload_to_s3(saved_paths, args)
        
        gen_logger.info(f"Batch generation completed. Generated {len(saved_paths)} images.")
    
    else:
        # Single or multiple prompt generation
        prompts = load_prompts(args)
        enhanced_prompts = enhance_prompts_with_cultural_context(prompts, args)
        
        if len(enhanced_prompts) == 1:
            # Single image generation
            result = generate_single_image(generator, enhanced_prompts[0], args)
            results = [result]
        else:
            # Multiple image generation
            results = generate_batch_images(generator, enhanced_prompts, args)
        
        # Save results
        saved_paths = save_generation_results(results, args, post_processor)
        
        # Upload to S3 if enabled
        upload_to_s3(saved_paths, args)
        
        # Print summary
        total_images = sum(len(result.images) for result in results)
        total_time = sum(result.generation_time for result in results)
        
        gen_logger.info(f"Generation completed!")
        gen_logger.info(f"Generated {total_images} images in {total_time:.2f}s")
        gen_logger.info(f"Average time per image: {total_time/total_images:.2f}s")
        gen_logger.info(f"Images saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
