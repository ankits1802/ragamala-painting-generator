#!/bin/bash

# EC2 Instance Setup Script for Ragamala Painting Generation
# This script sets up a complete environment for SDXL fine-tuning on EC2
# Optimized for g5.2xlarge or g4dn.xlarge instances with NVIDIA GPUs

set -e  # Exit on any error
set -u  # Exit on undefined variables

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Configuration variables
PYTHON_VERSION="3.10"
CUDA_VERSION="11.8"
DOCKER_COMPOSE_VERSION="2.21.0"
PROJECT_DIR="/home/ubuntu/ragamala-painting-generator"
CONDA_ENV_NAME="ragamala"
LOG_FILE="/var/log/ec2_setup.log"

# Create log file
sudo touch $LOG_FILE
sudo chmod 666 $LOG_FILE

log "Starting EC2 setup for Ragamala Painting Generation" | tee -a $LOG_FILE

# Function to check if running on correct instance type
check_instance_type() {
    log "Checking instance type..."
    INSTANCE_TYPE=$(curl -s http://169.254.169.254/latest/meta-data/instance-type)
    log "Instance type: $INSTANCE_TYPE"
    
    case $INSTANCE_TYPE in
        g5.*|g4dn.*|p3.*|p4d.*)
            log "GPU instance detected: $INSTANCE_TYPE"
            ;;
        *)
            warn "This script is optimized for GPU instances. Current: $INSTANCE_TYPE"
            read -p "Continue anyway? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
            ;;
    esac
}

# Function to update system packages
update_system() {
    log "Updating system packages..."
    sudo apt update -y | tee -a $LOG_FILE
    sudo apt upgrade -y | tee -a $LOG_FILE
    
    # Install essential packages
    sudo apt install -y \
        curl \
        wget \
        git \
        vim \
        htop \
        tree \
        unzip \
        build-essential \
        software-properties-common \
        apt-transport-https \
        ca-certificates \
        gnupg \
        lsb-release \
        jq \
        awscli \
        screen \
        tmux \
        ncdu \
        iotop \
        nethogs 2>&1 | tee -a $LOG_FILE
    
    log "System packages updated successfully"
}

# Function to install NVIDIA drivers
install_nvidia_drivers() {
    log "Installing NVIDIA drivers..."
    
    # Check if NVIDIA drivers are already installed
    if command -v nvidia-smi &> /dev/null; then
        log "NVIDIA drivers already installed"
        nvidia-smi | tee -a $LOG_FILE
        return 0
    fi
    
    # Remove any existing NVIDIA installations
    sudo apt remove --purge -y nvidia-* libnvidia-* 2>/dev/null || true
    sudo apt autoremove -y
    
    # Install NVIDIA drivers
    sudo apt update
    sudo apt install -y ubuntu-drivers-common
    
    # Install recommended driver
    RECOMMENDED_DRIVER=$(ubuntu-drivers devices | grep recommended | awk '{print $3}')
    if [ -n "$RECOMMENDED_DRIVER" ]; then
        log "Installing recommended driver: $RECOMMENDED_DRIVER"
        sudo apt install -y $RECOMMENDED_DRIVER
    else
        log "Installing generic NVIDIA driver"
        sudo apt install -y nvidia-driver-525
    fi
    
    # Install NVIDIA utilities
    sudo apt install -y nvidia-utils-525
    
    log "NVIDIA drivers installed. Reboot required."
}

# Function to install CUDA toolkit
install_cuda() {
    log "Installing CUDA toolkit..."
    
    # Check if CUDA is already installed
    if command -v nvcc &> /dev/null; then
        log "CUDA already installed"
        nvcc --version | tee -a $LOG_FILE
        return 0
    fi
    
    # Download and install CUDA
    CUDA_REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64"
    CUDA_REPO_KEY="3bf863cc.pub"
    CUDA_REPO_PIN="cuda-ubuntu2004.pin"
    
    # Add CUDA repository
    wget -O /tmp/$CUDA_REPO_KEY $CUDA_REPO_URL/$CUDA_REPO_KEY
    sudo apt-key add /tmp/$CUDA_REPO_KEY
    
    wget -O /tmp/$CUDA_REPO_PIN $CUDA_REPO_URL/$CUDA_REPO_PIN
    sudo mv /tmp/$CUDA_REPO_PIN /etc/apt/preferences.d/cuda-repository-pin-600
    
    sudo add-apt-repository "deb $CUDA_REPO_URL /"
    sudo apt update
    
    # Install CUDA toolkit
    sudo apt install -y cuda-toolkit-${CUDA_VERSION//./-}
    
    # Add CUDA to PATH
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    
    log "CUDA toolkit installed"
}

# Function to install Docker
install_docker() {
    log "Installing Docker..."
    
    # Check if Docker is already installed
    if command -v docker &> /dev/null; then
        log "Docker already installed"
        docker --version | tee -a $LOG_FILE
        return 0
    fi
    
    # Remove old Docker installations
    sudo apt remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true
    
    # Install Docker
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    rm get-docker.sh
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    
    # Install Docker Compose
    sudo curl -L "https://github.com/docker/compose/releases/download/v${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    
    # Start and enable Docker
    sudo systemctl start docker
    sudo systemctl enable docker
    
    log "Docker installed successfully"
}

# Function to install NVIDIA Container Toolkit
install_nvidia_docker() {
    log "Installing NVIDIA Container Toolkit..."
    
    # Check if nvidia-docker is already installed
    if docker info 2>/dev/null | grep -q nvidia; then
        log "NVIDIA Container Toolkit already installed"
        return 0
    fi
    
    # Add NVIDIA Container Toolkit repository
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    sudo apt update
    sudo apt install -y nvidia-container-toolkit
    
    # Configure Docker to use NVIDIA runtime
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    
    log "NVIDIA Container Toolkit installed"
}

# Function to install Python and Conda
install_python_conda() {
    log "Installing Python and Conda..."
    
    # Check if conda is already installed
    if command -v conda &> /dev/null; then
        log "Conda already installed"
        conda --version | tee -a $LOG_FILE
        return 0
    fi
    
    # Download and install Miniconda
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    wget -O /tmp/miniconda.sh $MINICONDA_URL
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    rm /tmp/miniconda.sh
    
    # Initialize conda
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    
    # Add conda to PATH
    echo 'export PATH=$HOME/miniconda3/bin:$PATH' >> ~/.bashrc
    
    log "Conda installed successfully"
}

# Function to create Python environment
create_python_environment() {
    log "Creating Python environment for Ragamala project..."
    
    # Source conda
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    
    # Create conda environment
    conda create -n $CONDA_ENV_NAME python=$PYTHON_VERSION -y
    conda activate $CONDA_ENV_NAME
    
    # Install PyTorch with CUDA support
    conda install pytorch torchvision torchaudio pytorch-cuda=$CUDA_VERSION -c pytorch -c nvidia -y
    
    # Install essential packages
    pip install --upgrade pip
    pip install \
        diffusers[torch] \
        transformers \
        accelerate \
        xformers \
        peft \
        datasets \
        wandb \
        tensorboard \
        opencv-python \
        pillow \
        numpy \
        pandas \
        matplotlib \
        seaborn \
        scikit-learn \
        tqdm \
        rich \
        typer \
        fastapi \
        uvicorn \
        gradio \
        streamlit \
        boto3 \
        python-multipart \
        python-dotenv \
        pydantic \
        requests \
        aiofiles \
        jinja2 \
        omegaconf \
        hydra-core \
        albumentations \
        lpips \
        clip-by-openai \
        compel
    
    # Install additional ML packages
    pip install \
        timm \
        einops \
        safetensors \
        invisible-watermark \
        bitsandbytes \
        scipy \
        scikit-image \
        imageio \
        imageio-ffmpeg
    
    log "Python environment created successfully"
}

# Function to setup project directory
setup_project_directory() {
    log "Setting up project directory..."
    
    # Create project directory structure
    mkdir -p $PROJECT_DIR
    cd $PROJECT_DIR
    
    # Create directory structure
    mkdir -p {config,data/{raw,processed,metadata,splits},src/{data,models,training,inference,evaluation,utils},scripts,notebooks,api/{routes,middleware},frontend,models/{checkpoints,lora_weights,final_model},outputs/{training_samples,evaluation_results,production_outputs},logs/{training,evaluation,api},tests,deployment/{terraform,kubernetes,monitoring}}
    
    # Create basic files
    touch {README.md,requirements.txt,environment.yml,.env.example,.gitignore,docker-compose.yml,Dockerfile}
    
    # Create basic Python __init__.py files
    find src -type d -exec touch {}/__init__.py \;
    touch api/__init__.py
    touch tests/__init__.py
    
    # Set permissions
    sudo chown -R ubuntu:ubuntu $PROJECT_DIR
    chmod -R 755 $PROJECT_DIR
    
    log "Project directory structure created"
}

# Function to configure AWS CLI
configure_aws() {
    log "Configuring AWS CLI..."
    
    # Check if AWS CLI is configured
    if aws sts get-caller-identity &>/dev/null; then
        log "AWS CLI already configured"
        return 0
    fi
    
    # Get instance metadata for IAM role
    INSTANCE_PROFILE=$(curl -s http://169.254.169.254/latest/meta-data/iam/security-credentials/ 2>/dev/null || echo "")
    
    if [ -n "$INSTANCE_PROFILE" ]; then
        log "Using IAM instance profile: $INSTANCE_PROFILE"
    else
        warn "No IAM instance profile found. Please configure AWS credentials manually:"
        echo "aws configure"
    fi
}

# Function to setup monitoring
setup_monitoring() {
    log "Setting up monitoring tools..."
    
    # Install htop, iotop, and other monitoring tools
    sudo apt install -y htop iotop nethogs ncdu
    
    # Install nvidia-ml-py for GPU monitoring
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda activate $CONDA_ENV_NAME
    pip install nvidia-ml-py3 psutil
    
    # Create GPU monitoring script
    cat > $PROJECT_DIR/scripts/monitor_gpu.py << 'EOF'
#!/usr/bin/env python3
import time
import psutil
import subprocess
import json
from datetime import datetime

def get_gpu_info():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpus = []
            for line in lines:
                parts = line.split(', ')
                if len(parts) == 6:
                    gpus.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'temperature': int(parts[2]),
                        'utilization': int(parts[3]),
                        'memory_used': int(parts[4]),
                        'memory_total': int(parts[5])
                    })
            return gpus
    except Exception as e:
        print(f"Error getting GPU info: {e}")
    return []

def main():
    while True:
        timestamp = datetime.now().isoformat()
        
        # CPU and Memory info
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # GPU info
        gpus = get_gpu_info()
        
        # Print status
        print(f"\n[{timestamp}]")
        print(f"CPU: {cpu_percent:.1f}%")
        print(f"Memory: {memory.percent:.1f}% ({memory.used // (1024**3):.1f}GB / {memory.total // (1024**3):.1f}GB)")
        
        for gpu in gpus:
            print(f"GPU {gpu['index']} ({gpu['name']}): {gpu['utilization']}% | {gpu['temperature']}°C | {gpu['memory_used']}MB / {gpu['memory_total']}MB")
        
        time.sleep(5)

if __name__ == "__main__":
    main()
EOF
    
    chmod +x $PROJECT_DIR/scripts/monitor_gpu.py
    
    log "Monitoring tools setup complete"
}

# Function to setup automatic shutdown
setup_auto_shutdown() {
    log "Setting up automatic shutdown protection..."
    
    # Create auto-shutdown script
    cat > /tmp/auto_shutdown.sh << 'EOF'
#!/bin/bash
# Auto-shutdown script to prevent accidental costs
# Shuts down instance if idle for more than 2 hours

IDLE_TIME=7200  # 2 hours in seconds
LOG_FILE="/var/log/auto_shutdown.log"

check_activity() {
    # Check GPU utilization
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
    
    # Check CPU utilization
    CPU_UTIL=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    
    # Check if training processes are running
    TRAINING_PROCS=$(pgrep -f "python.*train" | wc -l)
    
    if [ "$GPU_UTIL" -gt 10 ] || [ "${CPU_UTIL%.*}" -gt 20 ] || [ "$TRAINING_PROCS" -gt 0 ]; then
        echo "$(date): Activity detected - GPU: ${GPU_UTIL}%, CPU: ${CPU_UTIL}%, Training procs: $TRAINING_PROCS" >> $LOG_FILE
        return 0
    else
        echo "$(date): No activity detected - GPU: ${GPU_UTIL}%, CPU: ${CPU_UTIL}%, Training procs: $TRAINING_PROCS" >> $LOG_FILE
        return 1
    fi
}

# Check activity every 30 minutes
IDLE_COUNT=0
while true; do
    if check_activity; then
        IDLE_COUNT=0
    else
        IDLE_COUNT=$((IDLE_COUNT + 1800))  # 30 minutes
        
        if [ $IDLE_COUNT -ge $IDLE_TIME ]; then
            echo "$(date): Shutting down due to inactivity" >> $LOG_FILE
            sudo shutdown -h now
        fi
    fi
    
    sleep 1800  # Check every 30 minutes
done
EOF
    
    sudo mv /tmp/auto_shutdown.sh /usr/local/bin/auto_shutdown.sh
    sudo chmod +x /usr/local/bin/auto_shutdown.sh
    
    # Create systemd service
    cat > /tmp/auto-shutdown.service << 'EOF'
[Unit]
Description=Auto Shutdown Service
After=multi-user.target

[Service]
Type=simple
ExecStart=/usr/local/bin/auto_shutdown.sh
Restart=always
User=root

[Install]
WantedBy=multi-user.target
EOF
    
    sudo mv /tmp/auto-shutdown.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable auto-shutdown.service
    
    log "Auto-shutdown protection enabled"
}

# Function to create useful aliases and shortcuts
setup_aliases() {
    log "Setting up useful aliases..."
    
    cat >> ~/.bashrc << 'EOF'

# Ragamala project aliases
alias ragamala='cd /home/ubuntu/ragamala-painting-generator'
alias activate-ragamala='conda activate ragamala'
alias gpu-status='watch -n 1 nvidia-smi'
alias gpu-monitor='python /home/ubuntu/ragamala-painting-generator/scripts/monitor_gpu.py'
alias logs-training='tail -f /home/ubuntu/ragamala-painting-generator/logs/training/*.log'
alias docker-gpu='docker run --gpus all'
alias disk-usage='ncdu /'

# Useful shortcuts
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias ..='cd ..'
alias ...='cd ../..'
alias grep='grep --color=auto'
alias fgrep='fgrep --color=auto'
alias egrep='egrep --color=auto'

# Git shortcuts
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git log --oneline'

# Python shortcuts
alias py='python'
alias pip-list='pip list --format=columns'
alias jupyter-lab='jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root'

EOF
    
    log "Aliases setup complete"
}

# Function to setup Jupyter Lab
setup_jupyter() {
    log "Setting up Jupyter Lab..."
    
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda activate $CONDA_ENV_NAME
    
    # Install Jupyter Lab and extensions
    pip install jupyterlab ipywidgets
    pip install jupyter-dash plotly
    
    # Generate Jupyter config
    jupyter lab --generate-config
    
    # Create Jupyter startup script
    cat > $PROJECT_DIR/scripts/start_jupyter.sh << 'EOF'
#!/bin/bash
cd /home/ubuntu/ragamala-painting-generator
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ragamala
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
EOF
    
    chmod +x $PROJECT_DIR/scripts/start_jupyter.sh
    
    log "Jupyter Lab setup complete"
}

# Function to test installation
test_installation() {
    log "Testing installation..."
    
    # Test NVIDIA drivers
    if command -v nvidia-smi &> /dev/null; then
        log "NVIDIA drivers test:"
        nvidia-smi | tee -a $LOG_FILE
    else
        error "NVIDIA drivers not found"
    fi
    
    # Test CUDA
    if command -v nvcc &> /dev/null; then
        log "CUDA test:"
        nvcc --version | tee -a $LOG_FILE
    else
        warn "CUDA not found in PATH"
    fi
    
    # Test Docker
    if command -v docker &> /dev/null; then
        log "Docker test:"
        docker --version | tee -a $LOG_FILE
        
        # Test NVIDIA Docker
        log "Testing NVIDIA Docker..."
        docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi | tee -a $LOG_FILE
    else
        error "Docker not found"
    fi
    
    # Test Python environment
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda activate $CONDA_ENV_NAME
    
    log "Testing Python packages..."
    python -c "
import torch
import torchvision
import diffusers
import transformers
import accelerate

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')

print(f'Diffusers version: {diffusers.__version__}')
print(f'Transformers version: {transformers.__version__}')
print('All packages imported successfully!')
" | tee -a $LOG_FILE
    
    log "Installation test completed successfully"
}

# Function to create startup script
create_startup_script() {
    log "Creating startup script..."
    
    cat > $PROJECT_DIR/scripts/startup.sh << 'EOF'
#!/bin/bash
# Startup script for Ragamala project

echo "Starting Ragamala Painting Generator environment..."

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ragamala

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/ubuntu/ragamala-painting-generator:$PYTHONPATH

# Change to project directory
cd /home/ubuntu/ragamala-painting-generator

echo "Environment ready!"
echo "Available commands:"
echo "  python scripts/train.py - Start training"
echo "  python scripts/generate.py - Generate images"
echo "  python scripts/evaluate.py - Run evaluation"
echo "  ./scripts/start_jupyter.sh - Start Jupyter Lab"
echo "  gpu-monitor - Monitor GPU usage"

bash
EOF
    
    chmod +x $PROJECT_DIR/scripts/startup.sh
    
    # Add to .bashrc for automatic activation
    echo "alias ragamala-start='$PROJECT_DIR/scripts/startup.sh'" >> ~/.bashrc
    
    log "Startup script created"
}

# Function to display final instructions
display_final_instructions() {
    log "Setup completed successfully!"
    
    cat << 'EOF'

================================================================================
                    RAGAMALA PAINTING GENERATOR - SETUP COMPLETE
================================================================================

Your EC2 instance is now ready for SDXL fine-tuning on Ragamala paintings!

NEXT STEPS:
1. Reboot the instance to ensure all drivers are loaded:
   sudo reboot

2. After reboot, test the installation:
   ragamala-start

3. Start your project:
   cd /home/ubuntu/ragamala-painting-generator
   conda activate ragamala

USEFUL COMMANDS:
- ragamala              : Go to project directory
- activate-ragamala     : Activate conda environment
- gpu-status           : Monitor GPU usage
- gpu-monitor          : Detailed GPU monitoring
- ragamala-start       : Start project environment

JUPYTER LAB:
- Start: ./scripts/start_jupyter.sh
- Access: http://YOUR_EC2_IP:8888

MONITORING:
- Auto-shutdown is enabled (2 hours idle)
- GPU monitoring available
- Logs in /var/log/ec2_setup.log

IMPORTANT:
- Remember to stop your instance when not in use
- Monitor costs in AWS console
- Backup important data to S3

Happy training!
================================================================================
EOF
}

# Main execution
main() {
    log "Starting EC2 setup script..."
    
    # Check if running as root
    if [ "$EUID" -eq 0 ]; then
        error "Please run this script as ubuntu user, not root"
    fi
    
    # Run setup functions
    check_instance_type
    update_system
    install_nvidia_drivers
    install_cuda
    install_docker
    install_nvidia_docker
    install_python_conda
    create_python_environment
    setup_project_directory
    configure_aws
    setup_monitoring
    setup_auto_shutdown
    setup_aliases
    setup_jupyter
    create_startup_script
    test_installation
    display_final_instructions
    
    log "EC2 setup completed successfully!"
    log "Please reboot the instance to complete the setup: sudo reboot"
}

# Run main function
main "$@"
