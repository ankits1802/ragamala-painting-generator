"""
Image Generation Module for Ragamala Paintings.

This module provides comprehensive image generation functionality using fine-tuned SDXL
models for creating authentic Ragamala paintings with cultural conditioning and style control.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, asdict
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Diffusers imports
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    DiffusionPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler
)

# Transformers imports
from transformers import CLIPTokenizer, CLIPTextModel

# PEFT imports
from peft import PeftModel

# Compel for prompt weighting
try:
    from compel import Compel, ReturnedEmbeddingsType
    COMPEL_AVAILABLE = True
except ImportError:
    COMPEL_AVAILABLE = False
    logging.warning("Compel not available. Install with: pip install compel")

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logging_utils import setup_logger
from src.models.prompt_encoder import CulturalPromptTemplates
from src.models.scheduler import get_optimal_scheduler_for_raga

logger = setup_logger(__name__)

@dataclass
class GenerationConfig:
    """Configuration for image generation."""
    # Model settings
    model_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
    lora_weights_path: Optional[str] = None
    refiner_model_path: Optional[str] = "stabilityai/stable-diffusion-xl-refiner-1.0"
    vae_model_path: Optional[str] = "madebyollin/sdxl-vae-fp16-fix"
    
    # Generation parameters
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    negative_prompt: str = "blurry, low quality, distorted, modern, western art, cartoon, anime"
    width: int = 1024
    height: int = 1024
    num_images_per_prompt: int = 1
    
    # Advanced parameters
    eta: float = 0.0
    generator_seed: Optional[int] = None
    strength: float = 0.8  # For img2img
    
    # Cultural conditioning
    enable_cultural_conditioning: bool = True
    cultural_guidance_scale: float = 1.0
    
    # Refiner settings
    use_refiner: bool = False
    refiner_strength: float = 0.3
    high_noise_frac: float = 0.8
    
    # Scheduler settings
    scheduler_type: str = "dpm_solver"
    use_karras_sigmas: bool = False
    
    # Output settings
    output_type: str = "pil"
    return_dict: bool = True
    
    # Performance settings
    enable_cpu_offload: bool = False
    enable_model_cpu_offload: bool = False
    enable_attention_slicing: bool = True
    enable_vae_slicing: bool = True
    enable_vae_tiling: bool = False
    enable_xformers_memory_efficient_attention: bool = True

@dataclass
class GenerationRequest:
    """Request for image generation."""
    prompt: str
    raga: Optional[str] = None
    style: Optional[str] = None
    negative_prompt: Optional[str] = None
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    num_images: int = 1
    seed: Optional[int] = None
    
    # Advanced options
    use_refiner: bool = False
    scheduler_type: Optional[str] = None
    cultural_template: str = "detailed"
    
    # Image conditioning
    init_image: Optional[Image.Image] = None
    mask_image: Optional[Image.Image] = None
    strength: float = 0.8

@dataclass
class GenerationResult:
    """Result of image generation."""
    images: List[Image.Image]
    prompt_used: str
    generation_time: float
    config_used: GenerationConfig
    metadata: Dict[str, Any]
    seeds_used: List[int]

class RagamalaGenerator:
    """Main generator class for Ragamala paintings."""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.pipeline = None
        self.refiner_pipeline = None
        self.cultural_templates = CulturalPromptTemplates()
        self.compel = None
        
        # Load models
        self._load_models()
        
        # Setup prompt weighting
        if COMPEL_AVAILABLE:
            self._setup_compel()
    
    def _load_models(self):
        """Load SDXL models and LoRA weights."""
        logger.info("Loading SDXL models...")
        
        try:
            # Load main pipeline
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True
            )
            
            # Load custom VAE if specified
            if self.config.vae_model_path:
                from diffusers import AutoencoderKL
                vae = AutoencoderKL.from_pretrained(
                    self.config.vae_model_path,
                    torch_dtype=torch.float16
                )
                self.pipeline.vae = vae
            
            # Load LoRA weights if specified
            if self.config.lora_weights_path and os.path.exists(self.config.lora_weights_path):
                self.pipeline.load_lora_weights(self.config.lora_weights_path)
                logger.info(f"Loaded LoRA weights from {self.config.lora_weights_path}")
            
            # Load refiner if enabled
            if self.config.use_refiner and self.config.refiner_model_path:
                self.refiner_pipeline = DiffusionPipeline.from_pretrained(
                    self.config.refiner_model_path,
                    text_encoder_2=self.pipeline.text_encoder_2,
                    vae=self.pipeline.vae,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16"
                )
                self.refiner_pipeline.to(self.device)
            
            # Move to device
            self.pipeline.to(self.device)
            
            # Apply optimizations
            self._apply_optimizations()
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def _apply_optimizations(self):
        """Apply performance optimizations."""
        try:
            # Memory efficient attention
            if self.config.enable_xformers_memory_efficient_attention:
                self.pipeline.enable_xformers_memory_efficient_attention()
                if self.refiner_pipeline:
                    self.refiner_pipeline.enable_xformers_memory_efficient_attention()
            
            # Attention slicing
            if self.config.enable_attention_slicing:
                self.pipeline.enable_attention_slicing()
                if self.refiner_pipeline:
                    self.refiner_pipeline.enable_attention_slicing()
            
            # VAE optimizations
            if self.config.enable_vae_slicing:
                self.pipeline.enable_vae_slicing()
                if self.refiner_pipeline:
                    self.refiner_pipeline.enable_vae_slicing()
            
            if self.config.enable_vae_tiling:
                self.pipeline.enable_vae_tiling()
                if self.refiner_pipeline:
                    self.refiner_pipeline.enable_vae_tiling()
            
            # CPU offloading
            if self.config.enable_cpu_offload:
                self.pipeline.enable_sequential_cpu_offload()
                if self.refiner_pipeline:
                    self.refiner_pipeline.enable_sequential_cpu_offload()
            elif self.config.enable_model_cpu_offload:
                self.pipeline.enable_model_cpu_offload()
                if self.refiner_pipeline:
                    self.refiner_pipeline.enable_model_cpu_offload()
            
            logger.info("Optimizations applied successfully")
            
        except Exception as e:
            logger.warning(f"Some optimizations failed: {e}")
    
    def _setup_compel(self):
        """Setup Compel for prompt weighting."""
        try:
            self.compel = Compel(
                tokenizer=[self.pipeline.tokenizer, self.pipeline.tokenizer_2],
                text_encoder=[self.pipeline.text_encoder, self.pipeline.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True]
            )
            logger.info("Compel setup for prompt weighting")
        except Exception as e:
            logger.warning(f"Failed to setup Compel: {e}")
    
    def _enhance_prompt_with_cultural_context(self, 
                                            prompt: str,
                                            raga: Optional[str] = None,
                                            style: Optional[str] = None,
                                            template_type: str = "detailed") -> str:
        """Enhance prompt with cultural context."""
        if not raga and not style:
            return prompt
        
        enhanced_prompt = self.cultural_templates.get_enhanced_prompt(
            prompt, raga, style, template_type
        )
        
        logger.info(f"Enhanced prompt: {enhanced_prompt}")
        return enhanced_prompt
    
    def _get_optimal_scheduler(self, 
                             raga: Optional[str] = None,
                             style: Optional[str] = None,
                             scheduler_type: Optional[str] = None) -> str:
        """Get optimal scheduler for generation."""
        if scheduler_type:
            return scheduler_type
        
        if raga or style:
            return get_optimal_scheduler_for_raga(raga or "default", style)
        
        return self.config.scheduler_type
    
    def _set_scheduler(self, scheduler_type: str):
        """Set the scheduler for the pipeline."""
        scheduler_classes = {
            "ddim": DDIMScheduler,
            "dpm_solver": DPMSolverMultistepScheduler,
            "euler": EulerDiscreteScheduler,
            "euler_ancestral": EulerAncestralDiscreteScheduler,
            "lms": LMSDiscreteScheduler,
            "pndm": PNDMScheduler,
            "unipc": UniPCMultistepScheduler
        }
        
        if scheduler_type in scheduler_classes:
            scheduler_class = scheduler_classes[scheduler_type]
            
            # Get scheduler config
            scheduler_config = self.pipeline.scheduler.config
            
            # Create new scheduler
            new_scheduler = scheduler_class.from_config(scheduler_config)
            
            # Set use_karras_sigmas if supported
            if hasattr(new_scheduler, 'use_karras_sigmas'):
                new_scheduler.use_karras_sigmas = self.config.use_karras_sigmas
            
            self.pipeline.scheduler = new_scheduler
            
            if self.refiner_pipeline:
                self.refiner_pipeline.scheduler = scheduler_class.from_config(
                    self.refiner_pipeline.scheduler.config
                )
    
    def _prepare_generation_kwargs(self, request: GenerationRequest) -> Dict[str, Any]:
        """Prepare keyword arguments for generation."""
        # Create generator for reproducibility
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(request.seed)
        elif self.config.generator_seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(self.config.generator_seed)
        
        # Prepare basic kwargs
        kwargs = {
            "width": request.width,
            "height": request.height,
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
            "num_images_per_prompt": request.num_images,
            "eta": self.config.eta,
            "generator": generator,
            "output_type": self.config.output_type,
            "return_dict": self.config.return_dict
        }
        
        # Add negative prompt
        negative_prompt = request.negative_prompt or self.config.negative_prompt
        kwargs["negative_prompt"] = negative_prompt
        
        # Handle prompt weighting with Compel
        if self.compel and COMPEL_AVAILABLE:
            try:
                conditioning, pooled = self.compel(request.prompt_used)
                kwargs["prompt_embeds"] = conditioning
                kwargs["pooled_prompt_embeds"] = pooled
                
                # Handle negative prompt
                negative_conditioning, negative_pooled = self.compel(negative_prompt)
                kwargs["negative_prompt_embeds"] = negative_conditioning
                kwargs["negative_pooled_prompt_embeds"] = negative_pooled
                
                # Remove text prompts since we're using embeddings
                kwargs.pop("negative_prompt", None)
                
            except Exception as e:
                logger.warning(f"Compel processing failed, using text prompts: {e}")
                kwargs["prompt"] = request.prompt_used
        else:
            kwargs["prompt"] = request.prompt_used
        
        return kwargs
    
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate Ragamala paintings based on request."""
        start_time = time.time()
        
        try:
            # Enhance prompt with cultural context
            request.prompt_used = self._enhance_prompt_with_cultural_context(
                request.prompt,
                request.raga,
                request.style,
                request.cultural_template
            )
            
            # Set optimal scheduler
            optimal_scheduler = self._get_optimal_scheduler(
                request.raga,
                request.style,
                request.scheduler_type
            )
            self._set_scheduler(optimal_scheduler)
            
            # Prepare generation arguments
            kwargs = self._prepare_generation_kwargs(request)
            
            # Generate images
            if request.init_image is not None:
                images = self._generate_img2img(request, kwargs)
            elif request.mask_image is not None:
                images = self._generate_inpaint(request, kwargs)
            else:
                images = self._generate_text2img(kwargs)
            
            # Apply refiner if enabled
            if request.use_refiner and self.refiner_pipeline:
                images = self._apply_refiner(images, request, kwargs)
            
            # Calculate generation time
            generation_time = time.time() - start_time
            
            # Prepare metadata
            metadata = {
                "raga": request.raga,
                "style": request.style,
                "scheduler_used": optimal_scheduler,
                "cultural_template": request.cultural_template,
                "generation_config": asdict(self.config),
                "request_config": asdict(request)
            }
            
            # Get seeds used
            seeds_used = []
            if request.seed is not None:
                seeds_used = [request.seed + i for i in range(request.num_images)]
            
            return GenerationResult(
                images=images,
                prompt_used=request.prompt_used,
                generation_time=generation_time,
                config_used=self.config,
                metadata=metadata,
                seeds_used=seeds_used
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def _generate_text2img(self, kwargs: Dict[str, Any]) -> List[Image.Image]:
        """Generate images from text prompt."""
        logger.info("Generating images from text...")
        
        with torch.no_grad():
            result = self.pipeline(**kwargs)
        
        return result.images if hasattr(result, 'images') else [result]
    
    def _generate_img2img(self, request: GenerationRequest, kwargs: Dict[str, Any]) -> List[Image.Image]:
        """Generate images from initial image."""
        logger.info("Generating images from initial image...")
        
        # Create img2img pipeline if needed
        if not hasattr(self, 'img2img_pipeline'):
            self.img2img_pipeline = StableDiffusionXLImg2ImgPipeline(
                vae=self.pipeline.vae,
                text_encoder=self.pipeline.text_encoder,
                text_encoder_2=self.pipeline.text_encoder_2,
                tokenizer=self.pipeline.tokenizer,
                tokenizer_2=self.pipeline.tokenizer_2,
                unet=self.pipeline.unet,
                scheduler=self.pipeline.scheduler
            )
            self.img2img_pipeline.to(self.device)
        
        # Add img2img specific parameters
        kwargs["image"] = request.init_image
        kwargs["strength"] = request.strength
        
        with torch.no_grad():
            result = self.img2img_pipeline(**kwargs)
        
        return result.images if hasattr(result, 'images') else [result]
    
    def _generate_inpaint(self, request: GenerationRequest, kwargs: Dict[str, Any]) -> List[Image.Image]:
        """Generate images with inpainting."""
        logger.info("Generating images with inpainting...")
        
        # Create inpaint pipeline if needed
        if not hasattr(self, 'inpaint_pipeline'):
            self.inpaint_pipeline = StableDiffusionXLInpaintPipeline(
                vae=self.pipeline.vae,
                text_encoder=self.pipeline.text_encoder,
                text_encoder_2=self.pipeline.text_encoder_2,
                tokenizer=self.pipeline.tokenizer,
                tokenizer_2=self.pipeline.tokenizer_2,
                unet=self.pipeline.unet,
                scheduler=self.pipeline.scheduler
            )
            self.inpaint_pipeline.to(self.device)
        
        # Add inpaint specific parameters
        kwargs["image"] = request.init_image
        kwargs["mask_image"] = request.mask_image
        kwargs["strength"] = request.strength
        
        with torch.no_grad():
            result = self.inpaint_pipeline(**kwargs)
        
        return result.images if hasattr(result, 'images') else [result]
    
    def _apply_refiner(self, 
                      images: List[Image.Image],
                      request: GenerationRequest,
                      kwargs: Dict[str, Any]) -> List[Image.Image]:
        """Apply refiner to generated images."""
        logger.info("Applying refiner...")
        
        refined_images = []
        
        for image in images:
            # Prepare refiner kwargs
            refiner_kwargs = {
                "prompt": request.prompt_used,
                "negative_prompt": kwargs.get("negative_prompt", self.config.negative_prompt),
                "image": image,
                "num_inference_steps": max(10, request.num_inference_steps // 3),
                "denoising_start": self.config.high_noise_frac,
                "strength": self.config.refiner_strength,
                "guidance_scale": request.guidance_scale,
                "generator": kwargs.get("generator"),
                "output_type": self.config.output_type
            }
            
            with torch.no_grad():
                refined_result = self.refiner_pipeline(**refiner_kwargs)
            
            refined_image = refined_result.images[0] if hasattr(refined_result, 'images') else refined_result
            refined_images.append(refined_image)
        
        return refined_images
    
    def generate_batch(self, 
                      requests: List[GenerationRequest],
                      show_progress: bool = True) -> List[GenerationResult]:
        """Generate multiple images in batch."""
        results = []
        
        iterator = tqdm(requests, desc="Generating images") if show_progress else requests
        
        for request in iterator:
            try:
                result = self.generate(request)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to generate image for request: {e}")
                continue
        
        return results
    
    def generate_variations(self,
                          base_request: GenerationRequest,
                          num_variations: int = 4,
                          seed_range: Tuple[int, int] = (0, 1000)) -> List[GenerationResult]:
        """Generate variations of a base request."""
        variations = []
        
        for i in range(num_variations):
            variation_request = GenerationRequest(**asdict(base_request))
            
            # Use different seeds for variations
            if seed_range:
                variation_request.seed = np.random.randint(seed_range[0], seed_range[1])
            
            variations.append(variation_request)
        
        return self.generate_batch(variations)
    
    def save_results(self, 
                    results: Union[GenerationResult, List[GenerationResult]],
                    output_dir: str = "outputs/generated",
                    save_metadata: bool = True) -> List[str]:
        """Save generation results to disk."""
        if isinstance(results, GenerationResult):
            results = [results]
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        for i, result in enumerate(results):
            for j, image in enumerate(result.images):
                # Create filename
                timestamp = int(time.time())
                filename = f"ragamala_{timestamp}_{i}_{j}.png"
                image_path = output_dir / filename
                
                # Save image
                image.save(image_path)
                saved_paths.append(str(image_path))
                
                # Save metadata
                if save_metadata:
                    metadata_path = output_dir / f"{filename.replace('.png', '_metadata.json')}"
                    with open(metadata_path, 'w') as f:
                        json.dump(result.metadata, f, indent=2, default=str)
        
        logger.info(f"Saved {len(saved_paths)} images to {output_dir}")
        return saved_paths

class RagamalaGeneratorFactory:
    """Factory for creating Ragamala generators."""
    
    @staticmethod
    def create_generator(model_path: str = None,
                        lora_weights_path: str = None,
                        device: str = "auto",
                        **config_kwargs) -> RagamalaGenerator:
        """Create a Ragamala generator with specified configuration."""
        
        # Auto-detect device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create configuration
        config = GenerationConfig(
            model_path=model_path or "stabilityai/stable-diffusion-xl-base-1.0",
            lora_weights_path=lora_weights_path,
            **config_kwargs
        )
        
        return RagamalaGenerator(config)
    
    @staticmethod
    def create_from_checkpoint(checkpoint_path: str,
                             **config_kwargs) -> RagamalaGenerator:
        """Create generator from saved checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        # Load configuration if available
        config_file = checkpoint_path / "generation_config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                saved_config = json.load(f)
            saved_config.update(config_kwargs)
            config = GenerationConfig(**saved_config)
        else:
            config = GenerationConfig(**config_kwargs)
        
        # Set model paths
        config.model_path = str(checkpoint_path / "base_model")
        config.lora_weights_path = str(checkpoint_path / "lora_weights")
        
        return RagamalaGenerator(config)

def create_sample_requests() -> List[GenerationRequest]:
    """Create sample generation requests for testing."""
    return [
        GenerationRequest(
            prompt="A beautiful traditional painting",
            raga="bhairav",
            style="rajput",
            cultural_template="detailed"
        ),
        GenerationRequest(
            prompt="Romantic scene in a garden",
            raga="yaman",
            style="pahari",
            cultural_template="atmospheric"
        ),
        GenerationRequest(
            prompt="Meditative figure by a river",
            raga="malkauns",
            style="deccan",
            cultural_template="cultural"
        ),
        GenerationRequest(
            prompt="Royal court ceremony",
            raga="darbari",
            style="mughal",
            cultural_template="compositional"
        )
    ]

def main():
    """Main function for testing the generator."""
    # Create generator
    config = GenerationConfig(
        num_inference_steps=20,  # Reduced for testing
        num_images_per_prompt=1
    )
    
    generator = RagamalaGeneratorFactory.create_generator()
    
    # Create sample requests
    requests = create_sample_requests()
    
    # Generate images
    print("Generating sample Ragamala paintings...")
    results = generator.generate_batch(requests[:2])  # Test with first 2 requests
    
    # Save results
    saved_paths = generator.save_results(results)
    
    print(f"Generated {len(saved_paths)} images:")
    for path in saved_paths:
        print(f"  {path}")
    
    # Print generation statistics
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  Prompt: {result.prompt_used}")
        print(f"  Generation time: {result.generation_time:.2f}s")
        print(f"  Images generated: {len(result.images)}")

if __name__ == "__main__":
    main()
