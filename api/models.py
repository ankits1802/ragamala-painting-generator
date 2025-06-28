"""
Pydantic models for the Ragamala painting generation API.
Defines request/response models, validation schemas, and data structures.
"""

import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator, root_validator
from pydantic.color import Color


class RagaEnum(str, Enum):
    """Enumeration of supported ragas."""
    
    BHAIRAV = "bhairav"
    YAMAN = "yaman"
    MALKAUNS = "malkauns"
    DARBARI = "darbari"
    BAGESHRI = "bageshri"
    TODI = "todi"


class StyleEnum(str, Enum):
    """Enumeration of supported painting styles."""
    
    RAJPUT = "rajput"
    PAHARI = "pahari"
    DECCAN = "deccan"
    MUGHAL = "mughal"


class GenerationStatusEnum(str, Enum):
    """Enumeration of generation status values."""
    
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class QualityLevelEnum(str, Enum):
    """Enumeration of quality assessment levels."""
    
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class PromptTemplateEnum(str, Enum):
    """Enumeration of available prompt templates."""
    
    BASIC = "basic"
    DETAILED = "detailed"
    CULTURAL = "cultural"
    ATMOSPHERIC = "atmospheric"
    NARRATIVE = "narrative"
    TECHNICAL = "technical"


# Base Models
class BaseResponse(BaseModel):
    """Base response model with common fields."""
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = Field(None, description="Unique request identifier")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ErrorDetail(BaseModel):
    """Error detail model for structured error responses."""
    
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    field: Optional[str] = Field(None, description="Field that caused the error")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional error context")


class ErrorResponse(BaseResponse):
    """Standard error response model."""
    
    error: str = Field(..., description="Error type")
    detail: Union[str, List[ErrorDetail]] = Field(..., description="Error details")
    status_code: int = Field(..., description="HTTP status code")


# Generation Request Models
class GenerationParameters(BaseModel):
    """Model for generation parameters."""
    
    num_inference_steps: int = Field(
        30, 
        ge=10, 
        le=100, 
        description="Number of denoising steps"
    )
    guidance_scale: float = Field(
        7.5, 
        ge=1.0, 
        le=20.0, 
        description="Classifier-free guidance scale"
    )
    width: int = Field(
        1024, 
        ge=512, 
        le=1024, 
        description="Image width in pixels"
    )
    height: int = Field(
        1024, 
        ge=512, 
        le=1024, 
        description="Image height in pixels"
    )
    seed: Optional[int] = Field(
        None, 
        ge=0, 
        le=2**32-1, 
        description="Random seed for reproducibility"
    )
    
    @validator('width', 'height')
    def validate_dimensions(cls, v):
        """Validate that dimensions are multiples of 64."""
        if v % 64 != 0:
            raise ValueError("Width and height must be multiples of 64")
        return v


class PromptConfiguration(BaseModel):
    """Model for prompt configuration."""
    
    template: Optional[PromptTemplateEnum] = Field(
        None, 
        description="Prompt template to use"
    )
    custom_prompt: Optional[str] = Field(
        None, 
        max_length=500, 
        description="Custom prompt override"
    )
    negative_prompt: Optional[str] = Field(
        None, 
        max_length=300, 
        description="Negative prompt to avoid certain elements"
    )
    prompt_strength: float = Field(
        1.0, 
        ge=0.1, 
        le=2.0, 
        description="Strength of prompt conditioning"
    )
    
    @validator('custom_prompt', 'negative_prompt')
    def validate_prompt_content(cls, v):
        """Validate prompt content for inappropriate terms."""
        if v is None:
            return v
        
        # Basic content filtering
        inappropriate_terms = [
            'nsfw', 'explicit', 'violence', 'hate'
        ]
        
        v_lower = v.lower()
        for term in inappropriate_terms:
            if term in v_lower:
                raise ValueError(f"Inappropriate content detected: {term}")
        
        return v


class CulturalConfiguration(BaseModel):
    """Model for cultural authenticity configuration."""
    
    strict_authenticity: bool = Field(
        True, 
        description="Enforce strict cultural authenticity"
    )
    include_iconography: bool = Field(
        True, 
        description="Include traditional iconographic elements"
    )
    temporal_accuracy: bool = Field(
        True, 
        description="Ensure temporal accuracy (time of day, season)"
    )
    color_palette_adherence: float = Field(
        0.8, 
        ge=0.0, 
        le=1.0, 
        description="Adherence to traditional color palettes"
    )


class OutputConfiguration(BaseModel):
    """Model for output configuration."""
    
    return_base64: bool = Field(
        False, 
        description="Return image as base64 encoded string"
    )
    save_to_s3: bool = Field(
        True, 
        description="Save generated images to S3"
    )
    include_metadata: bool = Field(
        True, 
        description="Include generation metadata"
    )
    calculate_quality_metrics: bool = Field(
        False, 
        description="Calculate quality assessment metrics"
    )
    watermark: bool = Field(
        False, 
        description="Add watermark to generated images"
    )


class GenerationRequest(BaseModel):
    """Main generation request model."""
    
    raga: RagaEnum = Field(..., description="Raga to depict in the painting")
    style: StyleEnum = Field(..., description="Painting style to use")
    
    num_images: int = Field(
        1, 
        ge=1, 
        le=4, 
        description="Number of images to generate"
    )
    
    generation_params: GenerationParameters = Field(
        default_factory=GenerationParameters,
        description="Generation parameters"
    )
    
    prompt_config: PromptConfiguration = Field(
        default_factory=PromptConfiguration,
        description="Prompt configuration"
    )
    
    cultural_config: CulturalConfiguration = Field(
        default_factory=CulturalConfiguration,
        description="Cultural authenticity configuration"
    )
    
    output_config: OutputConfiguration = Field(
        default_factory=OutputConfiguration,
        description="Output configuration"
    )
    
    priority: int = Field(
        0, 
        ge=0, 
        le=10, 
        description="Request priority (0=normal, 10=highest)"
    )
    
    user_id: Optional[str] = Field(
        None, 
        description="User identifier for tracking"
    )
    
    tags: Optional[List[str]] = Field(
        None, 
        description="Custom tags for categorization"
    )
    
    @validator('tags')
    def validate_tags(cls, v):
        """Validate tags format and content."""
        if v is None:
            return v
        
        if len(v) > 10:
            raise ValueError("Maximum 10 tags allowed")
        
        for tag in v:
            if not re.match(r'^[a-zA-Z0-9_-]+$', tag):
                raise ValueError(f"Invalid tag format: {tag}")
            if len(tag) > 50:
                raise ValueError(f"Tag too long: {tag}")
        
        return v


# Response Models
class QualityMetrics(BaseModel):
    """Model for image quality metrics."""
    
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall quality score")
    sharpness: float = Field(..., ge=0.0, le=1.0, description="Image sharpness score")
    contrast: float = Field(..., ge=0.0, le=1.0, description="Image contrast score")
    color_harmony: float = Field(..., ge=0.0, le=1.0, description="Color harmony score")
    composition_balance: float = Field(..., ge=0.0, le=1.0, description="Composition balance score")
    noise_level: float = Field(..., ge=0.0, le=1.0, description="Noise level (lower is better)")
    
    quality_level: QualityLevelEnum = Field(..., description="Overall quality assessment")
    
    technical_issues: Optional[List[str]] = Field(
        None, 
        description="List of detected technical issues"
    )


class CulturalAuthenticity(BaseModel):
    """Model for cultural authenticity assessment."""
    
    overall_authenticity: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Overall cultural authenticity score"
    )
    
    iconographic_accuracy: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Iconographic accuracy score"
    )
    
    temporal_consistency: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Temporal consistency score"
    )
    
    color_appropriateness: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Color palette appropriateness score"
    )
    
    style_consistency: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Style consistency score"
    )
    
    cultural_violations: List[str] = Field(
        default_factory=list,
        description="List of detected cultural violations"
    )
    
    authenticity_level: str = Field(
        ..., 
        description="Overall authenticity assessment"
    )


class ImageMetadata(BaseModel):
    """Model for image metadata."""
    
    image_id: str = Field(..., description="Unique image identifier")
    filename: Optional[str] = Field(None, description="Generated filename")
    format: str = Field("PNG", description="Image format")
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    
    generation_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the image was generated"
    )
    
    raga: RagaEnum = Field(..., description="Raga depicted")
    style: StyleEnum = Field(..., description="Painting style used")
    
    prompt_used: str = Field(..., description="Final prompt used for generation")
    negative_prompt_used: Optional[str] = Field(None, description="Negative prompt used")
    
    generation_parameters: Dict[str, Any] = Field(
        ..., 
        description="Parameters used for generation"
    )
    
    model_version: Optional[str] = Field(None, description="Model version used")
    
    s3_url: Optional[str] = Field(None, description="S3 URL if uploaded")
    s3_key: Optional[str] = Field(None, description="S3 key if uploaded")


class GeneratedImage(BaseModel):
    """Model for a generated image."""
    
    metadata: ImageMetadata = Field(..., description="Image metadata")
    
    image_data: Optional[str] = Field(
        None, 
        description="Base64 encoded image data (if requested)"
    )
    
    quality_metrics: Optional[QualityMetrics] = Field(
        None, 
        description="Quality assessment metrics"
    )
    
    cultural_authenticity: Optional[CulturalAuthenticity] = Field(
        None, 
        description="Cultural authenticity assessment"
    )
    
    download_url: Optional[str] = Field(
        None, 
        description="Temporary download URL"
    )
    
    thumbnail_url: Optional[str] = Field(
        None, 
        description="Thumbnail image URL"
    )


class GenerationResponse(BaseResponse):
    """Response model for image generation."""
    
    status: GenerationStatusEnum = Field(..., description="Generation status")
    message: str = Field(..., description="Status message")
    
    images: List[GeneratedImage] = Field(
        default_factory=list,
        description="Generated images"
    )
    
    generation_time: Optional[float] = Field(
        None, 
        description="Total generation time in seconds"
    )
    
    total_images: int = Field(0, description="Total number of images generated")
    
    request_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Request metadata and parameters"
    )
    
    cost_estimate: Optional[float] = Field(
        None, 
        description="Estimated cost in USD"
    )


# Batch Generation Models
class BatchGenerationRequest(BaseModel):
    """Model for batch generation requests."""
    
    requests: List[GenerationRequest] = Field(
        ..., 
        min_items=1, 
        max_items=10, 
        description="List of generation requests"
    )
    
    batch_id: Optional[str] = Field(
        None, 
        description="Custom batch identifier"
    )
    
    priority: int = Field(
        0, 
        ge=0, 
        le=10, 
        description="Batch priority"
    )
    
    parallel_processing: bool = Field(
        False, 
        description="Process requests in parallel"
    )
    
    stop_on_error: bool = Field(
        False, 
        description="Stop batch processing on first error"
    )
    
    @validator('requests')
    def validate_batch_requests(cls, v):
        """Validate batch request constraints."""
        if len(v) > 10:
            raise ValueError("Maximum 10 requests per batch")
        
        # Check for duplicate raga-style combinations
        combinations = [(req.raga, req.style) for req in v]
        if len(combinations) != len(set(combinations)):
            raise ValueError("Duplicate raga-style combinations in batch")
        
        return v


class BatchGenerationResponse(BaseResponse):
    """Response model for batch generation."""
    
    batch_id: str = Field(..., description="Batch identifier")
    status: GenerationStatusEnum = Field(..., description="Batch status")
    
    total_requests: int = Field(..., description="Total number of requests")
    completed: int = Field(0, description="Number of completed requests")
    failed: int = Field(0, description="Number of failed requests")
    pending: int = Field(0, description="Number of pending requests")
    
    results: List[GenerationResponse] = Field(
        default_factory=list,
        description="Individual generation results"
    )
    
    total_generation_time: Optional[float] = Field(
        None, 
        description="Total batch processing time"
    )
    
    success_rate: float = Field(
        0.0, 
        ge=0.0, 
        le=1.0, 
        description="Batch success rate"
    )
    
    total_cost_estimate: Optional[float] = Field(
        None, 
        description="Total estimated cost in USD"
    )


# Information Models
class RagaInfo(BaseModel):
    """Model for raga information."""
    
    name: RagaEnum = Field(..., description="Raga name")
    display_name: str = Field(..., description="Display name")
    description: str = Field(..., description="Raga description")
    
    time_of_day: str = Field(..., description="Associated time of day")
    season: Optional[str] = Field(None, description="Associated season")
    mood: str = Field(..., description="Emotional mood")
    
    emotions: List[str] = Field(..., description="Associated emotions")
    colors: List[str] = Field(..., description="Traditional color associations")
    iconography: List[str] = Field(..., description="Traditional iconographic elements")
    
    deity: Optional[str] = Field(None, description="Associated deity")
    musical_notes: Optional[List[str]] = Field(None, description="Musical notes (swaras)")
    
    difficulty_level: str = Field(..., description="Generation difficulty level")
    popularity: float = Field(0.0, ge=0.0, le=1.0, description="Popularity score")


class StyleInfo(BaseModel):
    """Model for painting style information."""
    
    name: StyleEnum = Field(..., description="Style name")
    display_name: str = Field(..., description="Display name")
    description: str = Field(..., description="Style description")
    
    period: str = Field(..., description="Historical period")
    region: str = Field(..., description="Geographic region")
    
    characteristics: List[str] = Field(..., description="Style characteristics")
    techniques: List[str] = Field(..., description="Painting techniques")
    color_palette: List[str] = Field(..., description="Typical color palette")
    
    typical_subjects: List[str] = Field(..., description="Typical subjects depicted")
    cultural_context: str = Field(..., description="Cultural context")
    
    difficulty_level: str = Field(..., description="Generation difficulty level")
    popularity: float = Field(0.0, ge=0.0, le=1.0, description="Popularity score")


# Health and Status Models
class HealthResponse(BaseModel):
    """Model for health check response."""
    
    status: str = Field(..., description="Service health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    model_name: Optional[str] = Field(None, description="Loaded model name")
    
    gpu_available: bool = Field(..., description="GPU availability")
    gpu_count: Optional[int] = Field(None, description="Number of available GPUs")
    gpu_name: Optional[str] = Field(None, description="GPU model name")
    
    memory_usage: Dict[str, float] = Field(
        default_factory=dict,
        description="Memory usage statistics"
    )
    
    uptime: Optional[float] = Field(None, description="Service uptime in seconds")
    version: Optional[str] = Field(None, description="API version")


class ModelInfo(BaseModel):
    """Model for model information."""
    
    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    model_type: str = Field(..., description="Model type (e.g., SDXL)")
    
    loaded: bool = Field(..., description="Whether model is loaded")
    loading_time: Optional[float] = Field(None, description="Model loading time")
    
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Model parameters"
    )
    
    capabilities: List[str] = Field(
        default_factory=list,
        description="Model capabilities"
    )
    
    supported_ragas: List[RagaEnum] = Field(
        default_factory=list,
        description="Supported ragas"
    )
    
    supported_styles: List[StyleEnum] = Field(
        default_factory=list,
        description="Supported styles"
    )
    
    performance_metrics: Optional[Dict[str, float]] = Field(
        None,
        description="Performance metrics"
    )


class GenerationStatus(BaseModel):
    """Model for generation status tracking."""
    
    request_id: str = Field(..., description="Request identifier")
    batch_id: Optional[str] = Field(None, description="Batch identifier if applicable")
    
    status: GenerationStatusEnum = Field(..., description="Current status")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Progress percentage")
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    
    estimated_completion: Optional[datetime] = Field(
        None, 
        description="Estimated completion time"
    )
    
    current_step: Optional[str] = Field(None, description="Current processing step")
    total_steps: Optional[int] = Field(None, description="Total number of steps")
    
    error_message: Optional[str] = Field(None, description="Error message if failed")
    retry_count: int = Field(0, description="Number of retries attempted")
    
    queue_position: Optional[int] = Field(None, description="Position in processing queue")


# Analytics and Metrics Models
class UsageStatistics(BaseModel):
    """Model for usage statistics."""
    
    total_requests: int = Field(0, description="Total number of requests")
    successful_requests: int = Field(0, description="Number of successful requests")
    failed_requests: int = Field(0, description="Number of failed requests")
    
    average_generation_time: float = Field(0.0, description="Average generation time")
    total_images_generated: int = Field(0, description="Total images generated")
    
    raga_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Distribution of requests by raga"
    )
    
    style_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Distribution of requests by style"
    )
    
    quality_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Distribution by quality level"
    )
    
    period_start: datetime = Field(..., description="Statistics period start")
    period_end: datetime = Field(..., description="Statistics period end")


# Configuration Models
class APIConfiguration(BaseModel):
    """Model for API configuration."""
    
    max_requests_per_minute: int = Field(60, description="Rate limit per minute")
    max_batch_size: int = Field(10, description="Maximum batch size")
    max_image_size: int = Field(1024, description="Maximum image dimension")
    
    default_quality_metrics: bool = Field(False, description="Calculate quality metrics by default")
    default_cultural_assessment: bool = Field(True, description="Perform cultural assessment by default")
    
    s3_storage_enabled: bool = Field(True, description="S3 storage enabled")
    watermark_enabled: bool = Field(False, description="Watermarking enabled")
    
    supported_formats: List[str] = Field(
        default_factory=lambda: ["PNG", "JPEG"],
        description="Supported image formats"
    )
    
    maintenance_mode: bool = Field(False, description="Maintenance mode enabled")
    debug_mode: bool = Field(False, description="Debug mode enabled")


# Update forward references for all models
GenerationRequest.update_forward_refs()
GenerationResponse.update_forward_refs()
BatchGenerationRequest.update_forward_refs()
BatchGenerationResponse.update_forward_refs()
GeneratedImage.update_forward_refs()
ImageMetadata.update_forward_refs()
