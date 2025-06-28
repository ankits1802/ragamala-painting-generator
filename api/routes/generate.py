"""
Generation endpoints for the Ragamala painting API.
Handles image generation requests, batch processing, and status tracking.
"""

import asyncio
import base64
import io
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import torch
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from PIL import Image
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..models import (
    GenerationRequest,
    GenerationResponse,
    BatchGenerationRequest,
    BatchGenerationResponse,
    GenerationStatus,
    GenerationStatusEnum,
    QualityMetrics,
    CulturalAuthenticity,
    GeneratedImage,
    ImageMetadata,
    ErrorResponse,
)
from ..middleware.auth import verify_api_key
from ...src.inference.generator import RagamalaGenerator
from ...src.evaluation.metrics import EvaluationMetrics
from ...src.evaluation.cultural_evaluator import CulturalAccuracyEvaluator
from ...src.utils.aws_utils import S3Manager
from ...src.utils.logging_utils import setup_logger

# Setup logging
logger = setup_logger(__name__)

# Initialize router
router = APIRouter()

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Security
security = HTTPBearer()

# Global variables (will be injected by dependency injection)
generator: Optional[RagamalaGenerator] = None
evaluation_metrics: Optional[EvaluationMetrics] = None
cultural_evaluator: Optional[CulturalAccuracyEvaluator] = None
s3_manager: Optional[S3Manager] = None

# In-memory storage for generation status (use Redis in production)
generation_status_store: Dict[str, GenerationStatus] = {}
batch_status_store: Dict[str, Dict] = {}

# Dependency injection functions
async def get_generator() -> RagamalaGenerator:
    """Get the global generator instance."""
    global generator
    if generator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    return generator

async def get_evaluation_metrics() -> Optional[EvaluationMetrics]:
    """Get the evaluation metrics instance."""
    global evaluation_metrics
    return evaluation_metrics

async def get_cultural_evaluator() -> Optional[CulturalAccuracyEvaluator]:
    """Get the cultural evaluator instance."""
    global cultural_evaluator
    return cultural_evaluator

async def get_s3_manager() -> Optional[S3Manager]:
    """Get the S3 manager instance."""
    global s3_manager
    return s3_manager

# Utility functions
def create_image_metadata(
    image_id: str,
    raga: str,
    style: str,
    prompt: str,
    generation_params: Dict,
    image: Image.Image,
    model_version: str = "sdxl-1.0-ragamala"
) -> ImageMetadata:
    """Create image metadata object."""
    return ImageMetadata(
        image_id=image_id,
        filename=f"{raga}_{style}_{image_id}.png",
        format="PNG",
        width=image.width,
        height=image.height,
        generation_timestamp=datetime.utcnow(),
        raga=raga,
        style=style,
        prompt_used=prompt,
        negative_prompt_used=generation_params.get('negative_prompt'),
        generation_parameters=generation_params,
        model_version=model_version
    )

async def calculate_quality_metrics(
    image: Image.Image,
    evaluator: Optional[EvaluationMetrics]
) -> Optional[QualityMetrics]:
    """Calculate quality metrics for generated image."""
    if not evaluator:
        return None
    
    try:
        metrics = await evaluator.calculate_comprehensive_metrics(image)
        
        # Determine quality level
        overall_score = metrics.get('overall_score', 0.5)
        if overall_score >= 0.85:
            quality_level = "excellent"
        elif overall_score >= 0.7:
            quality_level = "good"
        elif overall_score >= 0.5:
            quality_level = "fair"
        else:
            quality_level = "poor"
        
        return QualityMetrics(
            overall_score=overall_score,
            sharpness=metrics.get('sharpness', 0.5),
            contrast=metrics.get('contrast', 0.5),
            color_harmony=metrics.get('color_harmony', 0.5),
            composition_balance=metrics.get('composition_balance', 0.5),
            noise_level=metrics.get('noise_level', 0.3),
            quality_level=quality_level,
            technical_issues=metrics.get('technical_issues', [])
        )
        
    except Exception as e:
        logger.warning(f"Failed to calculate quality metrics: {e}")
        return None

async def assess_cultural_authenticity(
    image: Image.Image,
    raga: str,
    style: str,
    evaluator: Optional[CulturalAccuracyEvaluator]
) -> Optional[CulturalAuthenticity]:
    """Assess cultural authenticity of generated image."""
    if not evaluator:
        return None
    
    try:
        assessment = await evaluator.assess_cultural_authenticity(
            image, raga, style
        )
        
        return CulturalAuthenticity(
            overall_authenticity=assessment.get('overall_authenticity', 0.7),
            iconographic_accuracy=assessment.get('iconographic_accuracy', 0.7),
            temporal_consistency=assessment.get('temporal_consistency', 0.7),
            color_appropriateness=assessment.get('color_appropriateness', 0.7),
            style_consistency=assessment.get('style_consistency', 0.7),
            cultural_violations=assessment.get('cultural_violations', []),
            authenticity_level=assessment.get('authenticity_level', 'moderate')
        )
        
    except Exception as e:
        logger.warning(f"Failed to assess cultural authenticity: {e}")
        return None

async def upload_to_s3(
    image: Image.Image,
    image_id: str,
    raga: str,
    style: str,
    s3_manager: Optional[S3Manager]
) -> Optional[str]:
    """Upload image to S3 and return URL."""
    if not s3_manager:
        return None
    
    try:
        # Create S3 key with organized structure
        date_prefix = datetime.utcnow().strftime('%Y/%m/%d')
        s3_key = f"generated/{date_prefix}/{raga}/{style}/{image_id}.png"
        
        # Upload image
        s3_url = await s3_manager.upload_image(image, s3_key)
        logger.info(f"Uploaded image to S3: {s3_url}")
        
        return s3_url
        
    except Exception as e:
        logger.error(f"Failed to upload to S3: {e}")
        return None

def update_generation_status(
    request_id: str,
    status: GenerationStatusEnum,
    progress: float = 0.0,
    current_step: Optional[str] = None,
    error_message: Optional[str] = None
):
    """Update generation status in store."""
    if request_id not in generation_status_store:
        generation_status_store[request_id] = GenerationStatus(
            request_id=request_id,
            status=status,
            progress=progress
        )
    else:
        generation_status_store[request_id].status = status
        generation_status_store[request_id].progress = progress
        
        if current_step:
            generation_status_store[request_id].current_step = current_step
        if error_message:
            generation_status_store[request_id].error_message = error_message
        
        if status == GenerationStatusEnum.PROCESSING and not generation_status_store[request_id].started_at:
            generation_status_store[request_id].started_at = datetime.utcnow()
        elif status in [GenerationStatusEnum.COMPLETED, GenerationStatusEnum.FAILED]:
            generation_status_store[request_id].completed_at = datetime.utcnow()

# Main generation endpoints
@router.post("/generate", response_model=GenerationResponse)
@limiter.limit("10/minute")
async def generate_single_image(
    request: Request,
    generation_request: GenerationRequest,
    background_tasks: BackgroundTasks,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    generator: RagamalaGenerator = Depends(get_generator),
    evaluation_metrics: Optional[EvaluationMetrics] = Depends(get_evaluation_metrics),
    cultural_evaluator: Optional[CulturalAccuracyEvaluator] = Depends(get_cultural_evaluator),
    s3_manager: Optional[S3Manager] = Depends(get_s3_manager)
):
    """Generate a single Ragamala painting image."""
    
    # Verify API key
    await verify_api_key(credentials)
    
    request_id = str(uuid.uuid4())
    logger.info(f"Starting generation request {request_id}: {generation_request.raga} - {generation_request.style}")
    
    # Initialize status tracking
    update_generation_status(request_id, GenerationStatusEnum.PROCESSING, 0.1, "Initializing generation")
    
    try:
        # Generate prompt
        update_generation_status(request_id, GenerationStatusEnum.PROCESSING, 0.2, "Creating prompt")
        
        if generation_request.prompt_config.custom_prompt:
            prompt = generation_request.prompt_config.custom_prompt
        else:
            prompt = await generator.create_culturally_aware_prompt(
                raga=generation_request.raga,
                style=generation_request.style,
                template=generation_request.prompt_config.template,
                cultural_config=generation_request.cultural_config
            )
        
        # Prepare generation parameters
        update_generation_status(request_id, GenerationStatusEnum.PROCESSING, 0.3, "Preparing generation parameters")
        
        generation_params = {
            'prompt': prompt,
            'negative_prompt': generation_request.prompt_config.negative_prompt,
            'num_inference_steps': generation_request.generation_params.num_inference_steps,
            'guidance_scale': generation_request.generation_params.guidance_scale,
            'width': generation_request.generation_params.width,
            'height': generation_request.generation_params.height,
            'num_images_per_prompt': generation_request.num_images,
            'generator': torch.Generator().manual_seed(generation_request.generation_params.seed) if generation_request.generation_params.seed else None,
        }
        
        # Generate images
        update_generation_status(request_id, GenerationStatusEnum.PROCESSING, 0.4, "Generating images")
        
        start_time = time.time()
        images = await generator.generate_with_progress_callback(
            **generation_params,
            progress_callback=lambda progress: update_generation_status(
                request_id, GenerationStatusEnum.PROCESSING, 0.4 + (progress * 0.4), "Generating images"
            )
        )
        generation_time = time.time() - start_time
        
        # Process generated images
        update_generation_status(request_id, GenerationStatusEnum.PROCESSING, 0.8, "Processing results")
        
        generated_images = []
        
        for i, image in enumerate(images):
            image_id = f"{request_id}_{i}"
            
            # Create metadata
            metadata = create_image_metadata(
                image_id=image_id,
                raga=generation_request.raga,
                style=generation_request.style,
                prompt=prompt,
                generation_params=generation_params,
                image=image
            )
            
            # Calculate quality metrics if requested
            quality_metrics = None
            if generation_request.output_config.calculate_quality_metrics:
                quality_metrics = await calculate_quality_metrics(image, evaluation_metrics)
            
            # Assess cultural authenticity
            cultural_authenticity = None
            if generation_request.cultural_config.strict_authenticity:
                cultural_authenticity = await assess_cultural_authenticity(
                    image, generation_request.raga, generation_request.style, cultural_evaluator
                )
            
            # Convert to base64 if requested
            image_data = None
            if generation_request.output_config.return_base64:
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                image_data = base64.b64encode(buffered.getvalue()).decode()
            
            # Upload to S3 if requested
            s3_url = None
            if generation_request.output_config.save_to_s3:
                s3_url = await upload_to_s3(
                    image, image_id, generation_request.raga, generation_request.style, s3_manager
                )
                metadata.s3_url = s3_url
                metadata.s3_key = f"generated/{datetime.utcnow().strftime('%Y/%m/%d')}/{generation_request.raga}/{generation_request.style}/{image_id}.png"
            
            # Create generated image object
            generated_image = GeneratedImage(
                metadata=metadata,
                image_data=image_data,
                quality_metrics=quality_metrics,
                cultural_authenticity=cultural_authenticity,
                download_url=s3_url
            )
            
            generated_images.append(generated_image)
        
        # Update final status
        update_generation_status(request_id, GenerationStatusEnum.COMPLETED, 1.0, "Generation completed")
        
        # Log successful generation
        background_tasks.add_task(
            log_generation_request,
            request_id,
            generation_request,
            generation_time,
            len(generated_images),
            "success"
        )
        
        # Calculate cost estimate
        cost_estimate = calculate_generation_cost(
            num_images=len(generated_images),
            inference_steps=generation_request.generation_params.num_inference_steps,
            resolution=(generation_request.generation_params.width, generation_request.generation_params.height)
        )
        
        return GenerationResponse(
            request_id=request_id,
            status=GenerationStatusEnum.COMPLETED,
            message="Images generated successfully",
            images=generated_images,
            generation_time=generation_time,
            total_images=len(generated_images),
            request_metadata={
                'raga': generation_request.raga,
                'style': generation_request.style,
                'prompt_template': generation_request.prompt_config.template,
                'cultural_config': generation_request.cultural_config.dict(),
                'generation_params': generation_request.generation_params.dict()
            },
            cost_estimate=cost_estimate
        )
        
    except Exception as e:
        logger.error(f"Generation request {request_id} failed: {e}")
        
        # Update error status
        update_generation_status(request_id, GenerationStatusEnum.FAILED, 0.0, error_message=str(e))
        
        # Log failed generation
        background_tasks.add_task(
            log_generation_request,
            request_id,
            generation_request,
            0.0,
            0,
            "failed",
            str(e)
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image generation failed: {str(e)}"
        )

@router.post("/batch-generate", response_model=BatchGenerationResponse)
@limiter.limit("2/minute")
async def generate_batch_images(
    request: Request,
    batch_request: BatchGenerationRequest,
    background_tasks: BackgroundTasks,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    generator: RagamalaGenerator = Depends(get_generator),
    evaluation_metrics: Optional[EvaluationMetrics] = Depends(get_evaluation_metrics),
    cultural_evaluator: Optional[CulturalAccuracyEvaluator] = Depends(get_cultural_evaluator),
    s3_manager: Optional[S3Manager] = Depends(get_s3_manager)
):
    """Generate multiple Ragamala paintings in batch."""
    
    # Verify API key
    await verify_api_key(credentials)
    
    batch_id = batch_request.batch_id or str(uuid.uuid4())
    logger.info(f"Starting batch generation {batch_id} with {len(batch_request.requests)} requests")
    
    # Initialize batch status
    batch_status_store[batch_id] = {
        'status': GenerationStatusEnum.PROCESSING,
        'total_requests': len(batch_request.requests),
        'completed': 0,
        'failed': 0,
        'start_time': time.time()
    }
    
    results = []
    
    try:
        if batch_request.parallel_processing:
            # Process requests in parallel
            tasks = []
            for i, gen_request in enumerate(batch_request.requests):
                task = asyncio.create_task(
                    process_single_generation(
                        gen_request, f"{batch_id}_{i}", generator,
                        evaluation_metrics, cultural_evaluator, s3_manager
                    )
                )
                tasks.append(task)
            
            # Wait for all tasks to complete
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(task_results):
                if isinstance(result, Exception):
                    batch_status_store[batch_id]['failed'] += 1
                    error_response = GenerationResponse(
                        request_id=f"{batch_id}_{i}",
                        status=GenerationStatusEnum.FAILED,
                        message=f"Generation failed: {str(result)}",
                        images=[],
                        generation_time=0.0,
                        total_images=0
                    )
                    results.append(error_response)
                else:
                    batch_status_store[batch_id]['completed'] += 1
                    results.append(result)
        else:
            # Process requests sequentially
            for i, gen_request in enumerate(batch_request.requests):
                try:
                    result = await process_single_generation(
                        gen_request, f"{batch_id}_{i}", generator,
                        evaluation_metrics, cultural_evaluator, s3_manager
                    )
                    results.append(result)
                    batch_status_store[batch_id]['completed'] += 1
                    
                except Exception as e:
                    logger.error(f"Batch item {i} failed: {e}")
                    batch_status_store[batch_id]['failed'] += 1
                    
                    error_response = GenerationResponse(
                        request_id=f"{batch_id}_{i}",
                        status=GenerationStatusEnum.FAILED,
                        message=f"Generation failed: {str(e)}",
                        images=[],
                        generation_time=0.0,
                        total_images=0
                    )
                    results.append(error_response)
                    
                    if batch_request.stop_on_error:
                        break
        
        # Calculate batch statistics
        total_generation_time = time.time() - batch_status_store[batch_id]['start_time']
        completed = batch_status_store[batch_id]['completed']
        failed = batch_status_store[batch_id]['failed']
        total_requests = len(batch_request.requests)
        success_rate = completed / total_requests if total_requests > 0 else 0
        
        # Calculate total cost
        total_cost = sum(
            result.cost_estimate for result in results 
            if result.cost_estimate is not None
        )
        
        # Log batch completion
        background_tasks.add_task(
            log_batch_request,
            batch_id,
            total_requests,
            completed,
            failed,
            total_generation_time
        )
        
        return BatchGenerationResponse(
            request_id=batch_id,
            batch_id=batch_id,
            status=GenerationStatusEnum.COMPLETED,
            total_requests=total_requests,
            completed=completed,
            failed=failed,
            pending=0,
            results=results,
            total_generation_time=total_generation_time,
            success_rate=success_rate,
            total_cost_estimate=total_cost
        )
        
    except Exception as e:
        logger.error(f"Batch generation {batch_id} failed: {e}")
        
        # Update batch status
        batch_status_store[batch_id]['status'] = GenerationStatusEnum.FAILED
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch generation failed: {str(e)}"
        )

@router.get("/status/{request_id}", response_model=GenerationStatus)
async def get_generation_status(
    request_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get the status of a generation request."""
    
    # Verify API key
    await verify_api_key(credentials)
    
    if request_id not in generation_status_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Generation request not found"
        )
    
    return generation_status_store[request_id]

@router.get("/batch-status/{batch_id}")
async def get_batch_status(
    batch_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get the status of a batch generation request."""
    
    # Verify API key
    await verify_api_key(credentials)
    
    if batch_id not in batch_status_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Batch request not found"
        )
    
    return batch_status_store[batch_id]

@router.get("/download/{image_id}")
async def download_image(
    image_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    s3_manager: Optional[S3Manager] = Depends(get_s3_manager)
):
    """Download a generated image by ID."""
    
    # Verify API key
    await verify_api_key(credentials)
    
    if not s3_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="S3 storage not available"
        )
    
    try:
        # Generate presigned URL for download
        download_url = await s3_manager.generate_presigned_url(
            f"generated/{image_id}.png",
            expiration=3600  # 1 hour
        )
        
        return {"download_url": download_url}
        
    except Exception as e:
        logger.error(f"Failed to generate download URL for {image_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found or download failed"
        )

@router.post("/generate-from-image", response_model=GenerationResponse)
@limiter.limit("5/minute")
async def generate_from_reference_image(
    request: Request,
    raga: str = Form(...),
    style: str = Form(...),
    reference_image: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    strength: float = Form(0.8),
    num_inference_steps: int = Form(30),
    guidance_scale: float = Form(7.5),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    generator: RagamalaGenerator = Depends(get_generator)
):
    """Generate Ragamala painting from reference image."""
    
    # Verify API key
    await verify_api_key(credentials)
    
    request_id = str(uuid.uuid4())
    logger.info(f"Starting image-to-image generation {request_id}: {raga} - {style}")
    
    try:
        # Validate file type
        if not reference_image.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an image"
            )
        
        # Load reference image
        image_data = await reference_image.read()
        reference_img = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Resize if necessary
        if reference_img.width > 1024 or reference_img.height > 1024:
            reference_img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        
        # Generate prompt if not provided
        if not prompt:
            prompt = await generator.create_culturally_aware_prompt(raga, style)
        
        # Generate image
        start_time = time.time()
        images = await generator.generate_from_image(
            prompt=prompt,
            image=reference_img,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
        generation_time = time.time() - start_time
        
        # Process result (simplified for image-to-image)
        image = images[0]
        image_id = f"{request_id}_0"
        
        # Create metadata
        metadata = create_image_metadata(
            image_id=image_id,
            raga=raga,
            style=style,
            prompt=prompt,
            generation_params={
                'strength': strength,
                'num_inference_steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'reference_image': reference_image.filename
            },
            image=image
        )
        
        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        image_data = base64.b64encode(buffered.getvalue()).decode()
        
        generated_image = GeneratedImage(
            metadata=metadata,
            image_data=image_data
        )
        
        return GenerationResponse(
            request_id=request_id,
            status=GenerationStatusEnum.COMPLETED,
            message="Image-to-image generation completed successfully",
            images=[generated_image],
            generation_time=generation_time,
            total_images=1
        )
        
    except Exception as e:
        logger.error(f"Image-to-image generation {request_id} failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image-to-image generation failed: {str(e)}"
        )

# Helper functions
async def process_single_generation(
    generation_request: GenerationRequest,
    request_id: str,
    generator: RagamalaGenerator,
    evaluation_metrics: Optional[EvaluationMetrics],
    cultural_evaluator: Optional[CulturalAccuracyEvaluator],
    s3_manager: Optional[S3Manager]
) -> GenerationResponse:
    """Process a single generation request (used in batch processing)."""
    
    # Generate prompt
    if generation_request.prompt_config.custom_prompt:
        prompt = generation_request.prompt_config.custom_prompt
    else:
        prompt = await generator.create_culturally_aware_prompt(
            raga=generation_request.raga,
            style=generation_request.style,
            template=generation_request.prompt_config.template,
            cultural_config=generation_request.cultural_config
        )
    
    # Prepare generation parameters
    generation_params = {
        'prompt': prompt,
        'negative_prompt': generation_request.prompt_config.negative_prompt,
        'num_inference_steps': generation_request.generation_params.num_inference_steps,
        'guidance_scale': generation_request.generation_params.guidance_scale,
        'width': generation_request.generation_params.width,
        'height': generation_request.generation_params.height,
        'num_images_per_prompt': generation_request.num_images,
        'generator': torch.Generator().manual_seed(generation_request.generation_params.seed) if generation_request.generation_params.seed else None,
    }
    
    # Generate images
    start_time = time.time()
    images = await generator.generate(**generation_params)
    generation_time = time.time() - start_time
    
    # Process results
    generated_images = []
    
    for i, image in enumerate(images):
        image_id = f"{request_id}_{i}"
        
        # Create metadata
        metadata = create_image_metadata(
            image_id=image_id,
            raga=generation_request.raga,
            style=generation_request.style,
            prompt=prompt,
            generation_params=generation_params,
            image=image
        )
        
        # Convert to base64 if requested
        image_data = None
        if generation_request.output_config.return_base64:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            image_data = base64.b64encode(buffered.getvalue()).decode()
        
        # Upload to S3 if requested
        s3_url = None
        if generation_request.output_config.save_to_s3:
            s3_url = await upload_to_s3(
                image, image_id, generation_request.raga, generation_request.style, s3_manager
            )
            metadata.s3_url = s3_url
        
        generated_image = GeneratedImage(
            metadata=metadata,
            image_data=image_data,
            download_url=s3_url
        )
        
        generated_images.append(generated_image)
    
    return GenerationResponse(
        request_id=request_id,
        status=GenerationStatusEnum.COMPLETED,
        message="Images generated successfully",
        images=generated_images,
        generation_time=generation_time,
        total_images=len(generated_images)
    )

def calculate_generation_cost(
    num_images: int,
    inference_steps: int,
    resolution: tuple
) -> float:
    """Calculate estimated cost for generation."""
    # Base cost per image (simplified pricing model)
    base_cost = 0.02  # $0.02 per image
    
    # Step multiplier
    step_multiplier = inference_steps / 30  # 30 steps as baseline
    
    # Resolution multiplier
    width, height = resolution
    resolution_multiplier = (width * height) / (1024 * 1024)  # 1024x1024 as baseline
    
    total_cost = num_images * base_cost * step_multiplier * resolution_multiplier
    
    return round(total_cost, 4)

async def log_generation_request(
    request_id: str,
    generation_request: GenerationRequest,
    generation_time: float,
    num_images: int,
    status: str,
    error: Optional[str] = None
):
    """Log generation request for analytics."""
    try:
        log_data = {
            'request_id': request_id,
            'timestamp': datetime.utcnow().isoformat(),
            'raga': generation_request.raga,
            'style': generation_request.style,
            'num_images': num_images,
            'generation_time': generation_time,
            'status': status,
            'error': error,
            'user_id': generation_request.user_id,
            'priority': generation_request.priority,
            'generation_params': generation_request.generation_params.dict(),
            'cultural_config': generation_request.cultural_config.dict()
        }
        
        # In production, send to analytics service or database
        logger.info(f"Generation analytics: {json.dumps(log_data)}")
        
    except Exception as e:
        logger.error(f"Failed to log generation request: {e}")

async def log_batch_request(
    batch_id: str,
    total_requests: int,
    completed: int,
    failed: int,
    total_time: float
):
    """Log batch request for analytics."""
    try:
        log_data = {
            'batch_id': batch_id,
            'timestamp': datetime.utcnow().isoformat(),
            'total_requests': total_requests,
            'completed': completed,
            'failed': failed,
            'success_rate': completed / total_requests if total_requests > 0 else 0,
            'total_time': total_time,
            'avg_time_per_request': total_time / total_requests if total_requests > 0 else 0
        }
        
        logger.info(f"Batch analytics: {json.dumps(log_data)}")
        
    except Exception as e:
        logger.error(f"Failed to log batch request: {e}")
