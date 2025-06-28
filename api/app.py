"""
FastAPI application for Ragamala painting generation using SDXL 1.0.
Provides endpoints for image generation, model management, and health checks.
"""

import asyncio
import base64
import io
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import uvicorn
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from PIL import Image
from pydantic import BaseModel, Field, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# Import project modules
from models import (
    GenerationRequest,
    GenerationResponse,
    BatchGenerationRequest,
    BatchGenerationResponse,
    ModelInfo,
    HealthResponse,
    ErrorResponse,
    GenerationStatus,
    StyleInfo,
    RagaInfo,
)
from routes.generate import router as generate_router
from routes.health import router as health_router
from middleware.auth import verify_api_key
from middleware.rate_limiting import get_user_id
from ..src.inference.generator import RagamalaGenerator, GenerationConfig
from ..src.evaluation.metrics import EvaluationMetrics
from ..src.utils.logging_utils import setup_logger
from ..src.utils.aws_utils import S3Manager
from ..config.deployment_config import DeploymentConfig

# Setup logging
logger = setup_logger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Global variables for model management
generator = None
evaluation_metrics = None
s3_manager = None
config = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info("Starting Ragamala painting generation API")
    
    global generator, evaluation_metrics, s3_manager, config
    
    try:
        # Load configuration
        config = DeploymentConfig()
        logger.info(f"Loaded configuration: {config.model_name}")
        
        # Initialize S3 manager
        s3_manager = S3Manager(
            bucket_name=config.s3_bucket,
            region=config.aws_region
        )
        logger.info("Initialized S3 manager")
        
        # Initialize generator
        generation_config = GenerationConfig(
            model_path=config.model_path,
            lora_path=config.lora_path,
            device=config.device,
            torch_dtype=torch.float16 if config.use_fp16 else torch.float32,
            enable_cpu_offload=config.enable_cpu_offload,
            enable_attention_slicing=config.enable_attention_slicing,
        )
        
        generator = RagamalaGenerator(generation_config)
        await generator.load_model()
        logger.info("Model loaded successfully")
        
        # Initialize evaluation metrics
        evaluation_metrics = EvaluationMetrics(device=config.device)
        logger.info("Evaluation metrics initialized")
        
        # Warm up the model
        if config.warmup_on_startup:
            await warmup_model()
        
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Ragamala painting generation API")
    
    if generator:
        generator.cleanup()
    
    logger.info("API shutdown completed")

# Initialize FastAPI app
app = FastAPI(
    title="Ragamala Painting Generator API",
    description="Generate authentic Ragamala paintings using fine-tuned SDXL 1.0",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Security
security = HTTPBearer()

# Pydantic models for request/response validation
class GenerateImageRequest(BaseModel):
    """Request model for image generation."""
    
    raga: str = Field(..., description="Raga name (e.g., 'bhairav', 'yaman', 'malkauns')")
    style: str = Field(..., description="Painting style (e.g., 'rajput', 'pahari', 'deccan', 'mughal')")
    prompt_template: Optional[str] = Field(None, description="Custom prompt template")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt")
    num_inference_steps: int = Field(30, ge=10, le=100, description="Number of inference steps")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Guidance scale")
    width: int = Field(1024, ge=512, le=1024, description="Image width")
    height: int = Field(1024, ge=512, le=1024, description="Image height")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    num_images: int = Field(1, ge=1, le=4, description="Number of images to generate")
    return_base64: bool = Field(False, description="Return image as base64 string")
    save_to_s3: bool = Field(True, description="Save generated image to S3")
    
    @validator('raga')
    def validate_raga(cls, v):
        valid_ragas = ['bhairav', 'yaman', 'malkauns', 'darbari', 'bageshri', 'todi']
        if v.lower() not in valid_ragas:
            raise ValueError(f"Raga must be one of: {valid_ragas}")
        return v.lower()
    
    @validator('style')
    def validate_style(cls, v):
        valid_styles = ['rajput', 'pahari', 'deccan', 'mughal']
        if v.lower() not in valid_styles:
            raise ValueError(f"Style must be one of: {valid_styles}")
        return v.lower()

class GenerateImageResponse(BaseModel):
    """Response model for image generation."""
    
    request_id: str
    status: str
    message: str
    images: Optional[List[Dict]] = None
    generation_time: Optional[float] = None
    metadata: Optional[Dict] = None

class BatchGenerateRequest(BaseModel):
    """Request model for batch image generation."""
    
    requests: List[GenerateImageRequest] = Field(..., max_items=10)
    batch_id: Optional[str] = Field(None, description="Custom batch ID")

class BatchGenerateResponse(BaseModel):
    """Response model for batch generation."""
    
    batch_id: str
    status: str
    total_requests: int
    completed: int
    failed: int
    results: List[GenerateImageResponse]

class ModelStatusResponse(BaseModel):
    """Response model for model status."""
    
    model_loaded: bool
    model_name: str
    device: str
    memory_usage: Dict
    uptime: float

# Utility functions
async def warmup_model():
    """Warm up the model with a simple generation."""
    try:
        logger.info("Warming up model...")
        
        warmup_request = GenerateImageRequest(
            raga="bhairav",
            style="rajput",
            num_inference_steps=10,
            num_images=1,
            return_base64=False,
            save_to_s3=False
        )
        
        start_time = time.time()
        await generate_image_internal(warmup_request, warmup=True)
        warmup_time = time.time() - start_time
        
        logger.info(f"Model warmup completed in {warmup_time:.2f} seconds")
        
    except Exception as e:
        logger.warning(f"Model warmup failed: {e}")

def get_memory_usage():
    """Get current memory usage statistics."""
    memory_info = {}
    
    if torch.cuda.is_available():
        memory_info['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3
        memory_info['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**3
        memory_info['gpu_max_allocated'] = torch.cuda.max_memory_allocated() / 1024**3
    
    import psutil
    process = psutil.Process(os.getpid())
    memory_info['cpu_memory'] = process.memory_info().rss / 1024**3
    
    return memory_info

async def generate_image_internal(request: GenerateImageRequest, warmup: bool = False) -> Dict:
    """Internal image generation function."""
    if not generator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Generate prompt
        prompt = generator.create_prompt(
            raga=request.raga,
            style=request.style,
            template=request.prompt_template
        )
        
        # Set generation parameters
        generation_params = {
            'prompt': prompt,
            'negative_prompt': request.negative_prompt,
            'num_inference_steps': request.num_inference_steps,
            'guidance_scale': request.guidance_scale,
            'width': request.width,
            'height': request.height,
            'num_images_per_prompt': request.num_images,
            'generator': torch.Generator().manual_seed(request.seed) if request.seed else None,
        }
        
        # Generate images
        start_time = time.time()
        images = await generator.generate(**generation_params)
        generation_time = time.time() - start_time
        
        if warmup:
            return {"status": "success", "generation_time": generation_time}
        
        # Process results
        results = []
        for i, image in enumerate(images):
            image_id = str(uuid.uuid4())
            
            # Convert to base64 if requested
            image_data = None
            if request.return_base64:
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                image_data = base64.b64encode(buffered.getvalue()).decode()
            
            # Save to S3 if requested
            s3_url = None
            if request.save_to_s3 and s3_manager:
                try:
                    s3_key = f"generated/{datetime.now().strftime('%Y/%m/%d')}/{image_id}.png"
                    s3_url = await s3_manager.upload_image(image, s3_key)
                except Exception as e:
                    logger.warning(f"Failed to upload to S3: {e}")
            
            # Calculate quality metrics if available
            quality_metrics = None
            if evaluation_metrics:
                try:
                    quality_metrics = await evaluation_metrics.calculate_image_metrics(image)
                except Exception as e:
                    logger.warning(f"Failed to calculate quality metrics: {e}")
            
            result = {
                'image_id': image_id,
                'image_data': image_data,
                's3_url': s3_url,
                'quality_metrics': quality_metrics,
                'metadata': {
                    'raga': request.raga,
                    'style': request.style,
                    'prompt': prompt,
                    'generation_params': {
                        'steps': request.num_inference_steps,
                        'guidance_scale': request.guidance_scale,
                        'seed': request.seed,
                    }
                }
            }
            results.append(result)
        
        return {
            'status': 'success',
            'images': results,
            'generation_time': generation_time,
            'total_images': len(results)
        }
        
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image generation failed: {str(e)}"
        )

# API Routes

@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Ragamala Painting Generator API",
        "version": "1.0.0",
        "description": "Generate authentic Ragamala paintings using fine-tuned SDXL 1.0",
        "endpoints": {
            "generate": "/generate",
            "batch_generate": "/batch-generate",
            "health": "/health",
            "model_status": "/model/status",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        memory_info = get_memory_usage()
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "model_loaded": generator is not None,
            "memory_usage": memory_info,
            "gpu_available": torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            health_status["gpu_count"] = torch.cuda.device_count()
            health_status["gpu_name"] = torch.cuda.get_device_name(0)
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )

@app.get("/model/status", response_model=ModelStatusResponse)
async def model_status():
    """Get model status and information."""
    if not generator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        uptime = time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
        
        return ModelStatusResponse(
            model_loaded=True,
            model_name=config.model_name,
            device=str(config.device),
            memory_usage=get_memory_usage(),
            uptime=uptime
        )
        
    except Exception as e:
        logger.error(f"Failed to get model status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get model status"
        )

@app.post("/generate", response_model=GenerateImageResponse)
@limiter.limit("10/minute")
async def generate_image(
    request: Request,
    generation_request: GenerateImageRequest,
    background_tasks: BackgroundTasks,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Generate Ragamala painting images."""
    
    # Verify API key
    await verify_api_key(credentials)
    
    request_id = str(uuid.uuid4())
    logger.info(f"Received generation request {request_id}: {generation_request.raga} - {generation_request.style}")
    
    try:
        # Generate images
        result = await generate_image_internal(generation_request)
        
        # Log successful generation
        background_tasks.add_task(
            log_generation_request,
            request_id,
            generation_request,
            result,
            "success"
        )
        
        return GenerateImageResponse(
            request_id=request_id,
            status="success",
            message="Images generated successfully",
            images=result['images'],
            generation_time=result['generation_time'],
            metadata={
                'total_images': result['total_images'],
                'raga': generation_request.raga,
                'style': generation_request.style
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generation request {request_id} failed: {e}")
        
        # Log failed generation
        background_tasks.add_task(
            log_generation_request,
            request_id,
            generation_request,
            None,
            "failed",
            str(e)
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image generation failed: {str(e)}"
        )

@app.post("/batch-generate", response_model=BatchGenerateResponse)
@limiter.limit("2/minute")
async def batch_generate_images(
    request: Request,
    batch_request: BatchGenerateRequest,
    background_tasks: BackgroundTasks,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Generate multiple Ragamala paintings in batch."""
    
    # Verify API key
    await verify_api_key(credentials)
    
    batch_id = batch_request.batch_id or str(uuid.uuid4())
    logger.info(f"Received batch generation request {batch_id} with {len(batch_request.requests)} images")
    
    if len(batch_request.requests) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 10 images per batch request"
        )
    
    results = []
    completed = 0
    failed = 0
    
    for i, gen_request in enumerate(batch_request.requests):
        try:
            request_id = f"{batch_id}_{i}"
            logger.info(f"Processing batch item {i+1}/{len(batch_request.requests)}")
            
            result = await generate_image_internal(gen_request)
            
            response = GenerateImageResponse(
                request_id=request_id,
                status="success",
                message="Image generated successfully",
                images=result['images'],
                generation_time=result['generation_time'],
                metadata={
                    'batch_id': batch_id,
                    'batch_index': i,
                    'raga': gen_request.raga,
                    'style': gen_request.style
                }
            )
            
            results.append(response)
            completed += 1
            
        except Exception as e:
            logger.error(f"Batch item {i} failed: {e}")
            
            error_response = GenerateImageResponse(
                request_id=f"{batch_id}_{i}",
                status="failed",
                message=f"Generation failed: {str(e)}",
                images=None,
                generation_time=None,
                metadata={
                    'batch_id': batch_id,
                    'batch_index': i,
                    'error': str(e)
                }
            )
            
            results.append(error_response)
            failed += 1
    
    # Log batch completion
    background_tasks.add_task(
        log_batch_request,
        batch_id,
        len(batch_request.requests),
        completed,
        failed
    )
    
    return BatchGenerateResponse(
        batch_id=batch_id,
        status="completed",
        total_requests=len(batch_request.requests),
        completed=completed,
        failed=failed,
        results=results
    )

@app.get("/ragas", response_model=List[RagaInfo])
async def list_ragas():
    """Get list of supported ragas with descriptions."""
    ragas = [
        RagaInfo(
            name="bhairav",
            description="Dawn raga with devotional and solemn mood",
            time_of_day="dawn",
            emotions=["devotional", "spiritual", "awakening"]
        ),
        RagaInfo(
            name="yaman",
            description="Evening raga with romantic and serene mood",
            time_of_day="evening",
            emotions=["romantic", "beautiful", "serene"]
        ),
        RagaInfo(
            name="malkauns",
            description="Midnight raga with meditative and mysterious mood",
            time_of_day="midnight",
            emotions=["meditative", "mysterious", "contemplative"]
        ),
        RagaInfo(
            name="darbari",
            description="Late evening raga with regal and dignified mood",
            time_of_day="late_evening",
            emotions=["regal", "majestic", "dignified"]
        ),
        RagaInfo(
            name="bageshri",
            description="Night raga with yearning and devotional mood",
            time_of_day="night",
            emotions=["yearning", "devotional", "patient"]
        ),
        RagaInfo(
            name="todi",
            description="Morning raga with enchanting and charming mood",
            time_of_day="morning",
            emotions=["enchanting", "charming", "musical"]
        )
    ]
    return ragas

@app.get("/styles", response_model=List[StyleInfo])
async def list_styles():
    """Get list of supported painting styles with descriptions."""
    styles = [
        StyleInfo(
            name="rajput",
            description="Bold colors, geometric patterns, royal themes",
            period="16th-18th century",
            region="Rajasthan",
            characteristics=["bold colors", "geometric patterns", "flat perspective"]
        ),
        StyleInfo(
            name="pahari",
            description="Soft colors, naturalistic, lyrical quality",
            period="17th-19th century",
            region="Himalayan foothills",
            characteristics=["soft colors", "naturalistic", "lyrical"]
        ),
        StyleInfo(
            name="deccan",
            description="Persian influence, architectural elements, formal composition",
            period="16th-18th century",
            region="Deccan plateau",
            characteristics=["persian influence", "architectural", "formal"]
        ),
        StyleInfo(
            name="mughal",
            description="Elaborate details, naturalistic portraiture, imperial grandeur",
            period="16th-18th century",
            region="Northern India",
            characteristics=["elaborate details", "naturalistic", "imperial"]
        )
    ]
    return styles

# Background task functions
async def log_generation_request(
    request_id: str,
    request: GenerateImageRequest,
    result: Optional[Dict],
    status: str,
    error: Optional[str] = None
):
    """Log generation request for analytics."""
    try:
        log_data = {
            'request_id': request_id,
            'timestamp': datetime.utcnow().isoformat(),
            'raga': request.raga,
            'style': request.style,
            'num_images': request.num_images,
            'status': status,
            'generation_time': result.get('generation_time') if result else None,
            'error': error
        }
        
        # Save to database or analytics service
        logger.info(f"Generation log: {json.dumps(log_data)}")
        
    except Exception as e:
        logger.error(f"Failed to log generation request: {e}")

async def log_batch_request(
    batch_id: str,
    total_requests: int,
    completed: int,
    failed: int
):
    """Log batch request for analytics."""
    try:
        log_data = {
            'batch_id': batch_id,
            'timestamp': datetime.utcnow().isoformat(),
            'total_requests': total_requests,
            'completed': completed,
            'failed': failed,
            'success_rate': completed / total_requests if total_requests > 0 else 0
        }
        
        logger.info(f"Batch log: {json.dumps(log_data)}")
        
    except Exception as e:
        logger.error(f"Failed to log batch request: {e}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Exception",
            "detail": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Include routers
app.include_router(generate_router, prefix="/api/v1", tags=["generation"])
app.include_router(health_router, prefix="/api/v1", tags=["health"])

# Startup event to record start time
@app.on_event("startup")
async def startup_event():
    app.state.start_time = time.time()
    logger.info("FastAPI application started")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info",
        access_log=True
    )
