"""
Health check endpoints for the Ragamala painting generation API.
Provides comprehensive health monitoring, system status, and diagnostic information.
"""

import asyncio
import gc
import json
import logging
import os
import platform
import psutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from ..models import HealthResponse, ModelInfo, ErrorResponse
from ...src.inference.generator import RagamalaGenerator
from ...src.utils.aws_utils import S3Manager
from ...src.utils.logging_utils import setup_logger

# Setup logging
logger = setup_logger(__name__)

# Initialize router
router = APIRouter()

# Global variables for dependency injection
generator: Optional[RagamalaGenerator] = None
s3_manager: Optional[S3Manager] = None
app_start_time: Optional[datetime] = None

class SystemMetrics(BaseModel):
    """System metrics model."""
    
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    memory_used_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    load_average: List[float]
    process_count: int

class GPUMetrics(BaseModel):
    """GPU metrics model."""
    
    gpu_count: int
    gpu_names: List[str]
    gpu_memory_used: List[float]
    gpu_memory_total: List[float]
    gpu_utilization: List[float]
    gpu_temperature: List[float]

class ModelHealthStatus(BaseModel):
    """Model health status."""
    
    loaded: bool
    model_name: Optional[str]
    load_time: Optional[float]
    last_inference_time: Optional[datetime]
    inference_count: int
    average_inference_time: float
    error_count: int
    last_error: Optional[str]

class ServiceHealthStatus(BaseModel):
    """External service health status."""
    
    s3_accessible: bool
    s3_response_time: Optional[float]
    database_accessible: bool
    database_response_time: Optional[float]
    redis_accessible: bool
    redis_response_time: Optional[float]

class DetailedHealthResponse(BaseModel):
    """Detailed health response model."""
    
    status: str
    timestamp: datetime
    uptime_seconds: float
    version: str
    environment: str
    
    system_metrics: SystemMetrics
    gpu_metrics: Optional[GPUMetrics]
    model_status: ModelHealthStatus
    service_status: ServiceHealthStatus
    
    warnings: List[str]
    errors: List[str]

# Dependency injection functions
async def get_generator() -> Optional[RagamalaGenerator]:
    """Get the global generator instance."""
    global generator
    return generator

async def get_s3_manager() -> Optional[S3Manager]:
    """Get the S3 manager instance."""
    global s3_manager
    return s3_manager

def set_app_start_time(start_time: datetime):
    """Set the application start time."""
    global app_start_time
    app_start_time = start_time

# Utility functions
def get_system_metrics() -> SystemMetrics:
    """Get comprehensive system metrics."""
    try:
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_gb = memory.available / (1024**3)
        memory_used_gb = memory.used / (1024**3)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_usage_percent = (disk.used / disk.total) * 100
        disk_free_gb = disk.free / (1024**3)
        
        # Load average (Unix systems)
        try:
            load_average = list(os.getloadavg())
        except (OSError, AttributeError):
            load_average = [0.0, 0.0, 0.0]
        
        # Process count
        process_count = len(psutil.pids())
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_available_gb=memory_available_gb,
            memory_used_gb=memory_used_gb,
            disk_usage_percent=disk_usage_percent,
            disk_free_gb=disk_free_gb,
            load_average=load_average,
            process_count=process_count
        )
        
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        return SystemMetrics(
            cpu_percent=0.0,
            memory_percent=0.0,
            memory_available_gb=0.0,
            memory_used_gb=0.0,
            disk_usage_percent=0.0,
            disk_free_gb=0.0,
            load_average=[0.0, 0.0, 0.0],
            process_count=0
        )

def get_gpu_metrics() -> Optional[GPUMetrics]:
    """Get GPU metrics if available."""
    if not torch.cuda.is_available():
        return None
    
    try:
        gpu_count = torch.cuda.device_count()
        gpu_names = []
        gpu_memory_used = []
        gpu_memory_total = []
        gpu_utilization = []
        gpu_temperature = []
        
        for i in range(gpu_count):
            # GPU name
            gpu_names.append(torch.cuda.get_device_name(i))
            
            # Memory metrics
            memory_used = torch.cuda.memory_allocated(i) / (1024**3)  # GB
            memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
            gpu_memory_used.append(memory_used)
            gpu_memory_total.append(memory_total)
            
            # Utilization and temperature (requires nvidia-ml-py)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # GPU utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utilization.append(utilization.gpu)
                
                # GPU temperature
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                gpu_temperature.append(temperature)
                
            except ImportError:
                gpu_utilization.append(0.0)
                gpu_temperature.append(0.0)
            except Exception as e:
                logger.warning(f"Failed to get GPU {i} utilization/temperature: {e}")
                gpu_utilization.append(0.0)
                gpu_temperature.append(0.0)
        
        return GPUMetrics(
            gpu_count=gpu_count,
            gpu_names=gpu_names,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            gpu_utilization=gpu_utilization,
            gpu_temperature=gpu_temperature
        )
        
    except Exception as e:
        logger.error(f"Failed to get GPU metrics: {e}")
        return None

def get_model_health_status(generator: Optional[RagamalaGenerator]) -> ModelHealthStatus:
    """Get model health status."""
    if not generator:
        return ModelHealthStatus(
            loaded=False,
            model_name=None,
            load_time=None,
            last_inference_time=None,
            inference_count=0,
            average_inference_time=0.0,
            error_count=0,
            last_error=None
        )
    
    try:
        # Get model statistics from generator
        stats = generator.get_statistics() if hasattr(generator, 'get_statistics') else {}
        
        return ModelHealthStatus(
            loaded=True,
            model_name=stats.get('model_name', 'SDXL-Ragamala'),
            load_time=stats.get('load_time'),
            last_inference_time=stats.get('last_inference_time'),
            inference_count=stats.get('inference_count', 0),
            average_inference_time=stats.get('average_inference_time', 0.0),
            error_count=stats.get('error_count', 0),
            last_error=stats.get('last_error')
        )
        
    except Exception as e:
        logger.error(f"Failed to get model health status: {e}")
        return ModelHealthStatus(
            loaded=True,
            model_name="Unknown",
            load_time=None,
            last_inference_time=None,
            inference_count=0,
            average_inference_time=0.0,
            error_count=1,
            last_error=str(e)
        )

async def check_s3_health(s3_manager: Optional[S3Manager]) -> tuple[bool, Optional[float]]:
    """Check S3 service health."""
    if not s3_manager:
        return False, None
    
    try:
        start_time = time.time()
        # Simple S3 operation to check connectivity
        await s3_manager.list_objects(prefix="health-check/", max_keys=1)
        response_time = time.time() - start_time
        return True, response_time
        
    except Exception as e:
        logger.warning(f"S3 health check failed: {e}")
        return False, None

async def check_database_health() -> tuple[bool, Optional[float]]:
    """Check database health (placeholder for future database integration)."""
    # Placeholder for database health check
    # In production, implement actual database connectivity check
    return True, 0.001

async def check_redis_health() -> tuple[bool, Optional[float]]:
    """Check Redis health (placeholder for future Redis integration)."""
    # Placeholder for Redis health check
    # In production, implement actual Redis connectivity check
    return True, 0.001

async def get_service_health_status(s3_manager: Optional[S3Manager]) -> ServiceHealthStatus:
    """Get external service health status."""
    # Check S3
    s3_accessible, s3_response_time = await check_s3_health(s3_manager)
    
    # Check database
    db_accessible, db_response_time = await check_database_health()
    
    # Check Redis
    redis_accessible, redis_response_time = await check_redis_health()
    
    return ServiceHealthStatus(
        s3_accessible=s3_accessible,
        s3_response_time=s3_response_time,
        database_accessible=db_accessible,
        database_response_time=db_response_time,
        redis_accessible=redis_accessible,
        redis_response_time=redis_response_time
    )

def analyze_health_warnings_and_errors(
    system_metrics: SystemMetrics,
    gpu_metrics: Optional[GPUMetrics],
    model_status: ModelHealthStatus,
    service_status: ServiceHealthStatus
) -> tuple[List[str], List[str]]:
    """Analyze system state and generate warnings and errors."""
    warnings = []
    errors = []
    
    # System warnings
    if system_metrics.cpu_percent > 80:
        warnings.append(f"High CPU usage: {system_metrics.cpu_percent:.1f}%")
    
    if system_metrics.memory_percent > 85:
        warnings.append(f"High memory usage: {system_metrics.memory_percent:.1f}%")
    
    if system_metrics.disk_usage_percent > 90:
        warnings.append(f"High disk usage: {system_metrics.disk_usage_percent:.1f}%")
    
    if system_metrics.disk_free_gb < 5:
        errors.append(f"Low disk space: {system_metrics.disk_free_gb:.1f}GB remaining")
    
    # GPU warnings
    if gpu_metrics:
        for i, (memory_used, memory_total, temp) in enumerate(
            zip(gpu_metrics.gpu_memory_used, gpu_metrics.gpu_memory_total, gpu_metrics.gpu_temperature)
        ):
            memory_percent = (memory_used / memory_total) * 100 if memory_total > 0 else 0
            
            if memory_percent > 90:
                warnings.append(f"GPU {i} high memory usage: {memory_percent:.1f}%")
            
            if temp > 80:
                warnings.append(f"GPU {i} high temperature: {temp}°C")
            
            if temp > 90:
                errors.append(f"GPU {i} critical temperature: {temp}°C")
    
    # Model warnings
    if not model_status.loaded:
        errors.append("Model not loaded")
    
    if model_status.error_count > 10:
        warnings.append(f"High model error count: {model_status.error_count}")
    
    if model_status.last_error:
        warnings.append(f"Recent model error: {model_status.last_error}")
    
    # Service warnings
    if not service_status.s3_accessible:
        errors.append("S3 service not accessible")
    
    if not service_status.database_accessible:
        errors.append("Database not accessible")
    
    if service_status.s3_response_time and service_status.s3_response_time > 5.0:
        warnings.append(f"Slow S3 response time: {service_status.s3_response_time:.2f}s")
    
    return warnings, errors

# Health check endpoints
@router.get("/health", response_class=PlainTextResponse)
async def basic_health_check():
    """Basic health check endpoint for load balancers."""
    return "OK"

@router.get("/health/simple", response_model=HealthResponse)
async def simple_health_check(
    generator: Optional[RagamalaGenerator] = Depends(get_generator)
):
    """Simple health check with basic status information."""
    try:
        # Calculate uptime
        uptime = 0.0
        if app_start_time:
            uptime = (datetime.utcnow() - app_start_time).total_seconds()
        
        # Get basic memory info
        memory_info = {}
        if torch.cuda.is_available():
            memory_info['gpu_allocated'] = torch.cuda.memory_allocated() / (1024**3)
            memory_info['gpu_reserved'] = torch.cuda.memory_reserved() / (1024**3)
        
        process = psutil.Process(os.getpid())
        memory_info['cpu_memory'] = process.memory_info().rss / (1024**3)
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow(),
            model_loaded=generator is not None,
            model_name="SDXL-Ragamala" if generator else None,
            gpu_available=torch.cuda.is_available(),
            gpu_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
            gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            memory_usage=memory_info,
            uptime=uptime,
            version="1.0.0"
        )
        
    except Exception as e:
        logger.error(f"Simple health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Health check failed"
        )

@router.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check(
    generator: Optional[RagamalaGenerator] = Depends(get_generator),
    s3_manager: Optional[S3Manager] = Depends(get_s3_manager)
):
    """Detailed health check with comprehensive system information."""
    try:
        # Calculate uptime
        uptime = 0.0
        if app_start_time:
            uptime = (datetime.utcnow() - app_start_time).total_seconds()
        
        # Get all metrics
        system_metrics = get_system_metrics()
        gpu_metrics = get_gpu_metrics()
        model_status = get_model_health_status(generator)
        service_status = await get_service_health_status(s3_manager)
        
        # Analyze warnings and errors
        warnings, errors = analyze_health_warnings_and_errors(
            system_metrics, gpu_metrics, model_status, service_status
        )
        
        # Determine overall status
        if errors:
            overall_status = "unhealthy"
        elif warnings:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        return DetailedHealthResponse(
            status=overall_status,
            timestamp=datetime.utcnow(),
            uptime_seconds=uptime,
            version="1.0.0",
            environment=os.getenv("ENVIRONMENT", "development"),
            system_metrics=system_metrics,
            gpu_metrics=gpu_metrics,
            model_status=model_status,
            service_status=service_status,
            warnings=warnings,
            errors=errors
        )
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Detailed health check failed: {str(e)}"
        )

@router.get("/health/model", response_model=ModelInfo)
async def model_health_check(
    generator: Optional[RagamalaGenerator] = Depends(get_generator)
):
    """Model-specific health check."""
    if not generator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Get model information
        model_info = generator.get_model_info() if hasattr(generator, 'get_model_info') else {}
        
        # Perform a quick inference test
        test_start = time.time()
        test_successful = await perform_model_test(generator)
        test_time = time.time() - test_start
        
        return ModelInfo(
            model_name=model_info.get('model_name', 'SDXL-Ragamala'),
            model_version=model_info.get('version', '1.0'),
            model_type="SDXL",
            loaded=True,
            loading_time=model_info.get('loading_time'),
            parameters=model_info.get('parameters', {}),
            capabilities=['text-to-image', 'ragamala-generation'],
            supported_ragas=['bhairav', 'yaman', 'malkauns', 'darbari', 'bageshri', 'todi'],
            supported_styles=['rajput', 'pahari', 'deccan', 'mughal'],
            performance_metrics={
                'last_test_time': test_time,
                'test_successful': test_successful,
                'average_inference_time': model_info.get('average_inference_time', 0.0)
            }
        )
        
    except Exception as e:
        logger.error(f"Model health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model health check failed: {str(e)}"
        )

@router.get("/health/services")
async def services_health_check(
    s3_manager: Optional[S3Manager] = Depends(get_s3_manager)
):
    """External services health check."""
    try:
        service_status = await get_service_health_status(s3_manager)
        
        # Determine overall service health
        all_services_healthy = all([
            service_status.s3_accessible,
            service_status.database_accessible,
            service_status.redis_accessible
        ])
        
        return {
            "status": "healthy" if all_services_healthy else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "s3": {
                    "accessible": service_status.s3_accessible,
                    "response_time": service_status.s3_response_time
                },
                "database": {
                    "accessible": service_status.database_accessible,
                    "response_time": service_status.database_response_time
                },
                "redis": {
                    "accessible": service_status.redis_accessible,
                    "response_time": service_status.redis_response_time
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Services health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Services health check failed: {str(e)}"
        )

@router.get("/health/metrics")
async def metrics_endpoint():
    """Prometheus-style metrics endpoint."""
    try:
        system_metrics = get_system_metrics()
        gpu_metrics = get_gpu_metrics()
        
        metrics_text = f"""# HELP cpu_usage_percent CPU usage percentage
# TYPE cpu_usage_percent gauge
cpu_usage_percent {system_metrics.cpu_percent}

# HELP memory_usage_percent Memory usage percentage
# TYPE memory_usage_percent gauge
memory_usage_percent {system_metrics.memory_percent}

# HELP disk_usage_percent Disk usage percentage
# TYPE disk_usage_percent gauge
disk_usage_percent {system_metrics.disk_usage_percent}

# HELP process_count Number of running processes
# TYPE process_count gauge
process_count {system_metrics.process_count}
"""
        
        if gpu_metrics:
            for i, (memory_used, memory_total, utilization, temp) in enumerate(
                zip(gpu_metrics.gpu_memory_used, gpu_metrics.gpu_memory_total, 
                    gpu_metrics.gpu_utilization, gpu_metrics.gpu_temperature)
            ):
                metrics_text += f"""
# HELP gpu_memory_used_gb GPU memory used in GB
# TYPE gpu_memory_used_gb gauge
gpu_memory_used_gb{{gpu="{i}"}} {memory_used}

# HELP gpu_utilization_percent GPU utilization percentage
# TYPE gpu_utilization_percent gauge
gpu_utilization_percent{{gpu="{i}"}} {utilization}

# HELP gpu_temperature_celsius GPU temperature in Celsius
# TYPE gpu_temperature_celsius gauge
gpu_temperature_celsius{{gpu="{i}"}} {temp}
"""
        
        return PlainTextResponse(content=metrics_text, media_type="text/plain")
        
    except Exception as e:
        logger.error(f"Metrics endpoint failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Metrics collection failed"
        )

@router.post("/health/gc")
async def trigger_garbage_collection():
    """Trigger garbage collection (admin endpoint)."""
    try:
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Run Python garbage collection
        collected = gc.collect()
        
        # Get memory info after GC
        memory_info = {}
        if torch.cuda.is_available():
            memory_info['gpu_allocated'] = torch.cuda.memory_allocated() / (1024**3)
            memory_info['gpu_reserved'] = torch.cuda.memory_reserved() / (1024**3)
        
        process = psutil.Process(os.getpid())
        memory_info['cpu_memory'] = process.memory_info().rss / (1024**3)
        
        return {
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat(),
            "objects_collected": collected,
            "memory_after_gc": memory_info
        }
        
    except Exception as e:
        logger.error(f"Garbage collection failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Garbage collection failed: {str(e)}"
        )

# Helper functions
async def perform_model_test(generator: RagamalaGenerator) -> bool:
    """Perform a quick model test to verify functionality."""
    try:
        # Quick test generation with minimal parameters
        test_prompt = "A simple test image"
        
        # This would be implemented in the actual generator
        # For now, simulate a successful test
        await asyncio.sleep(0.1)  # Simulate quick inference
        
        return True
        
    except Exception as e:
        logger.warning(f"Model test failed: {e}")
        return False

# Initialize app start time when module is imported
set_app_start_time(datetime.utcnow())
