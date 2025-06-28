# =============================================================================
# RAGAMALA PAINTING GENERATOR - DOCKERFILE
# =============================================================================
# Multi-stage Docker build for SDXL 1.0 fine-tuning on Ragamala paintings
# Optimized for EC2 deployment with GPU support

# =============================================================================
# BUILD ARGUMENTS
# =============================================================================
ARG PYTHON_VERSION=3.10
ARG CUDA_VERSION=11.8
ARG UBUNTU_VERSION=20.04
ARG PYTORCH_VERSION=2.0.0
ARG TORCHVISION_VERSION=0.15.0

# =============================================================================
# BASE STAGE - CUDA RUNTIME WITH PYTHON
# =============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV TOKENIZERS_PARALLELISM=false

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Basic utilities
    curl \
    wget \
    git \
    unzip \
    vim \
    htop \
    tree \
    # Build tools
    build-essential \
    cmake \
    pkg-config \
    # Python dependencies
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils \
    python3-pip \
    # Image processing libraries
    libopencv-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Audio processing (for cultural integration)
    libsndfile1 \
    libasound2-dev \
    # Font support for cultural text
    fonts-noto \
    fonts-noto-cjk \
    fonts-noto-color-emoji \
    # Network utilities
    netcat \
    telnet \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# =============================================================================
# PYTHON DEPENDENCIES STAGE
# =============================================================================
FROM base as python-deps

# Copy requirements file
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch==${PYTORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

# Install additional ML dependencies
RUN pip install --no-cache-dir \
    # Hugging Face ecosystem
    transformers>=4.30.0 \
    diffusers>=0.21.0 \
    accelerate>=0.20.0 \
    datasets>=2.14.0 \
    huggingface-hub>=0.16.0 \
    peft>=0.4.0 \
    # SDXL specific
    safetensors>=0.3.1 \
    compel>=2.0.0 \
    invisible-watermark>=0.2.0 \
    omegaconf>=2.3.0 \
    xformers>=0.0.20 \
    # Evaluation metrics
    clip-by-openai>=1.0 \
    lpips>=0.1.4 \
    pytorch-fid>=0.3.0 \
    torchmetrics>=1.0.0 \
    cleanfid>=0.1.35

# =============================================================================
# DEVELOPMENT STAGE
# =============================================================================
FROM python-deps as development

# Install development tools
RUN pip install --no-cache-dir \
    jupyter>=1.0.0 \
    jupyterlab>=4.0.0 \
    ipykernel>=6.25.0 \
    ipywidgets>=8.1.0 \
    notebook>=7.0.0 \
    # Code quality
    black>=23.7.0 \
    isort>=5.12.0 \
    flake8>=6.0.0 \
    mypy>=1.5.0 \
    pre-commit>=3.3.0 \
    # Testing
    pytest>=7.4.0 \
    pytest-asyncio>=0.21.0 \
    pytest-cov>=4.1.0 \
    # Debugging
    ipdb>=0.13.0 \
    line_profiler>=4.1.0 \
    memory_profiler>=0.61.0

# =============================================================================
# APPLICATION STAGE
# =============================================================================
FROM python-deps as application

# Create application user
RUN groupadd -r ragamala && useradd -r -g ragamala ragamala

# Create necessary directories
RUN mkdir -p /app/{src,config,data,models,outputs,logs,cache} \
    && mkdir -p /app/cache/{huggingface,transformers,datasets} \
    && chown -R ragamala:ragamala /app

# Copy application code
COPY --chown=ragamala:ragamala src/ /app/src/
COPY --chown=ragamala:ragamala config/ /app/config/
COPY --chown=ragamala:ragamala scripts/ /app/scripts/
COPY --chown=ragamala:ragamala api/ /app/api/
COPY --chown=ragamala:ragamala frontend/ /app/frontend/

# Copy configuration files
COPY --chown=ragamala:ragamala .env.example /app/.env.example
COPY --chown=ragamala:ragamala docker-compose.yml /app/docker-compose.yml

# Set Python path
ENV PYTHONPATH=/app/src:/app

# Set cache directories
ENV HUGGINGFACE_HUB_CACHE=/app/cache/huggingface
ENV TRANSFORMERS_CACHE=/app/cache/transformers
ENV HF_HOME=/app/cache/hf_home
ENV HF_DATASETS_CACHE=/app/cache/datasets

# =============================================================================
# CULTURAL PROCESSING STAGE
# =============================================================================
FROM application as cultural

# Install cultural-specific dependencies
RUN pip install --no-cache-dir \
    # Sanskrit and Indic text processing
    indic-transliteration>=2.3.0 \
    # Music theory for raga integration
    musicpy>=1.5.0 \
    # NLP for cultural context
    spacy>=3.6.0 \
    nltk>=3.8.0

# Download spaCy models
RUN python -m spacy download en_core_web_sm

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Copy cultural data processing scripts
COPY --chown=ragamala:ragamala src/cultural/ /app/src/cultural/

# =============================================================================
# PRODUCTION STAGE
# =============================================================================
FROM cultural as production

# Install production web server
RUN pip install --no-cache-dir \
    gunicorn>=21.2.0 \
    uvicorn[standard]>=0.23.0

# Copy startup scripts
COPY --chown=ragamala:ragamala scripts/start.sh /app/start.sh
COPY --chown=ragamala:ragamala scripts/healthcheck.sh /app/healthcheck.sh

# Make scripts executable
RUN chmod +x /app/start.sh /app/healthcheck.sh

# Create startup script
RUN cat > /app/start.sh << 'EOF'
#!/bin/bash
set -e

# Function to check if GPU is available
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU detected:"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    else
        echo "No GPU detected, running in CPU mode"
    fi
}

# Function to download models if not present
download_models() {
    echo "Checking for required models..."
    python -c "
from huggingface_hub import snapshot_download
import os

models = [
    'stabilityai/stable-diffusion-xl-base-1.0',
    'stabilityai/stable-diffusion-xl-refiner-1.0',
    'madebyollin/sdxl-vae-fp16-fix'
]

for model in models:
    try:
        snapshot_download(repo_id=model, cache_dir=os.environ.get('HUGGINGFACE_HUB_CACHE'))
        print(f'✓ Downloaded {model}')
    except Exception as e:
        print(f'✗ Failed to download {model}: {e}')
"
}

# Function to start the appropriate service
start_service() {
    case "${SERVICE_TYPE:-api}" in
        "training")
            echo "Starting training service..."
            python /app/scripts/train.py
            ;;
        "inference")
            echo "Starting inference service..."
            python /app/scripts/generate.py --server
            ;;
        "api")
            echo "Starting API server..."
            cd /app/api
            uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
            ;;
        "gradio")
            echo "Starting Gradio interface..."
            python /app/frontend/gradio_app.py
            ;;
        "jupyter")
            echo "Starting Jupyter Lab..."
            jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
                --NotebookApp.token='' --NotebookApp.password=''
            ;;
        *)
            echo "Unknown service type: ${SERVICE_TYPE}"
            echo "Available options: training, inference, api, gradio, jupyter"
            exit 1
            ;;
    esac
}

# Main execution
echo "Starting Ragamala Painting Generator..."
check_gpu
download_models
start_service
EOF

# Create healthcheck script
RUN cat > /app/healthcheck.sh << 'EOF'
#!/bin/bash
set -e

# Check if the main service is running
case "${SERVICE_TYPE:-api}" in
    "api")
        curl -f http://localhost:8000/health || exit 1
        ;;
    "gradio")
        curl -f http://localhost:7860 || exit 1
        ;;
    "jupyter")
        curl -f http://localhost:8888 || exit 1
        ;;
    *)
        # For training/inference, check if Python process is running
        pgrep -f python || exit 1
        ;;
esac

echo "Service is healthy"
EOF

# Switch to application user
USER ragamala

# Expose ports
EXPOSE 8000 7860 8501 8888 6006

# Set default service type
ENV SERVICE_TYPE=api

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /app/healthcheck.sh

# Default command
CMD ["/app/start.sh"]

# =============================================================================
# JUPYTER DEVELOPMENT STAGE
# =============================================================================
FROM development as jupyter

# Install additional Jupyter extensions
RUN pip install --no-cache-dir \
    jupyterlab-git \
    jupyterlab-lsp \
    python-lsp-server[all] \
    jupyterlab_code_formatter \
    black[jupyter] \
    isort

# Configure Jupyter
RUN jupyter lab --generate-config

# Create Jupyter configuration
RUN cat > /home/ragamala/.jupyter/jupyter_lab_config.py << 'EOF'
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True
c.ServerApp.token = ''
c.ServerApp.password = ''
c.ServerApp.allow_origin = '*'
c.ServerApp.disable_check_xsrf = True
EOF

# Set Jupyter as default service
ENV SERVICE_TYPE=jupyter

# =============================================================================
# TRAINING STAGE
# =============================================================================
FROM production as training

# Install additional training dependencies
RUN pip install --no-cache-dir \
    # Monitoring
    wandb>=0.15.0 \
    tensorboard>=2.13.0 \
    mlflow>=2.5.0 \
    # Optimization
    bitsandbytes>=0.41.0 \
    deepspeed>=0.10.0

# Copy training-specific scripts
COPY --chown=ragamala:ragamala scripts/train.py /app/scripts/train.py
COPY --chown=ragamala:ragamala src/training/ /app/src/training/

# Set training as default service
ENV SERVICE_TYPE=training

# =============================================================================
# INFERENCE STAGE
# =============================================================================
FROM production as inference

# Install inference optimization libraries
RUN pip install --no-cache-dir \
    # Model optimization
    torch-tensorrt \
    # Serving
    torchserve \
    torch-model-archiver

# Copy inference-specific scripts
COPY --chown=ragamala:ragamala scripts/generate.py /app/scripts/generate.py
COPY --chown=ragamala:ragamala src/inference/ /app/src/inference/

# Set inference as default service
ENV SERVICE_TYPE=inference

# =============================================================================
# FINAL STAGE SELECTION
# =============================================================================
FROM production as final

# Labels for metadata
LABEL maintainer="Ragamala AI Team"
LABEL version="1.0.0"
LABEL description="SDXL 1.0 fine-tuning for Ragamala painting generation"
LABEL gpu.required="true"
LABEL cuda.version="11.8"
LABEL python.version="3.10"

# Final setup message
RUN echo "Ragamala Painting Generator Docker image built successfully!" > /app/build_info.txt
RUN echo "Built on: $(date)" >> /app/build_info.txt
RUN echo "CUDA Version: ${CUDA_VERSION}" >> /app/build_info.txt
RUN echo "Python Version: ${PYTHON_VERSION}" >> /app/build_info.txt
