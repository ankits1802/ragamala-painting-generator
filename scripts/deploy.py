"""
Deployment Script for Ragamala Painting Generation Project.

This script provides comprehensive deployment functionality for the SDXL fine-tuned
Ragamala painting generation model, including EC2 deployment, model serving setup,
API deployment, and infrastructure management.
"""

import os
import sys
import json
import time
import argparse
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import warnings
warnings.filterwarnings("ignore")

# Cloud and deployment imports
import boto3
from botocore.exceptions import ClientError
import docker
import yaml

# FastAPI and serving imports
import uvicorn
from fastapi import FastAPI
import gradio as gr

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logging_utils import setup_logger
from src.utils.aws_utils import AWSUtilities, AWSConfig, create_aws_config_from_env
from src.api.app import create_app
from src.inference.generator import RagamalaGenerator, GenerationConfig

logger = setup_logger(__name__)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Deploy Ragamala painting generation model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Deployment configuration
    deploy_group = parser.add_argument_group('Deployment Configuration')
    deploy_group.add_argument(
        "--deployment_type",
        type=str,
        choices=["local", "ec2", "docker", "kubernetes", "api", "gradio"],
        default="local",
        help="Type of deployment"
    )
    deploy_group.add_argument(
        "--environment",
        type=str,
        choices=["development", "staging", "production"],
        default="development",
        help="Deployment environment"
    )
    deploy_group.add_argument(
        "--config_file",
        type=str,
        default="config/deployment_config.yaml",
        help="Path to deployment configuration file"
    )
    
    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model"
    )
    model_group.add_argument(
        "--lora_weights_path",
        type=str,
        default=None,
        help="Path to LoRA weights"
    )
    model_group.add_argument(
        "--model_name",
        type=str,
        default="ragamala-generator",
        help="Name for the deployed model"
    )
    model_group.add_argument(
        "--model_version",
        type=str,
        default="v1.0",
        help="Model version"
    )
    
    # EC2 deployment
    ec2_group = parser.add_argument_group('EC2 Deployment')
    ec2_group.add_argument(
        "--instance_type",
        type=str,
        default="g4dn.xlarge",
        help="EC2 instance type for deployment"
    )
    ec2_group.add_argument(
        "--ami_id",
        type=str,
        default="ami-0c02fb55956c7d316",
        help="AMI ID for EC2 instance"
    )
    ec2_group.add_argument(
        "--key_pair_name",
        type=str,
        default=None,
        help="EC2 key pair name"
    )
    ec2_group.add_argument(
        "--security_group_ids",
        type=str,
        nargs="+",
        default=None,
        help="Security group IDs"
    )
    ec2_group.add_argument(
        "--subnet_id",
        type=str,
        default=None,
        help="Subnet ID for EC2 instance"
    )
    
    # Docker configuration
    docker_group = parser.add_argument_group('Docker Configuration')
    docker_group.add_argument(
        "--docker_image_name",
        type=str,
        default="ragamala-generator",
        help="Docker image name"
    )
    docker_group.add_argument(
        "--docker_tag",
        type=str,
        default="latest",
        help="Docker image tag"
    )
    docker_group.add_argument(
        "--dockerfile_path",
        type=str,
        default="Dockerfile",
        help="Path to Dockerfile"
    )
    docker_group.add_argument(
        "--push_to_registry",
        action="store_true",
        help="Push Docker image to registry"
    )
    docker_group.add_argument(
        "--registry_url",
        type=str,
        default=None,
        help="Docker registry URL"
    )
    
    # API deployment
    api_group = parser.add_argument_group('API Deployment')
    api_group.add_argument(
        "--api_host",
        type=str,
        default="0.0.0.0",
        help="API host address"
    )
    api_group.add_argument(
        "--api_port",
        type=int,
        default=8000,
        help="API port"
    )
    api_group.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of API workers"
    )
    api_group.add_argument(
        "--enable_auth",
        action="store_true",
        help="Enable API authentication"
    )
    api_group.add_argument(
        "--rate_limit",
        type=int,
        default=100,
        help="API rate limit per minute"
    )
    
    # Gradio deployment
    gradio_group = parser.add_argument_group('Gradio Deployment')
    gradio_group.add_argument(
        "--gradio_port",
        type=int,
        default=7860,
        help="Gradio interface port"
    )
    gradio_group.add_argument(
        "--gradio_share",
        action="store_true",
        help="Create public Gradio link"
    )
    gradio_group.add_argument(
        "--gradio_auth",
        type=str,
        default=None,
        help="Gradio authentication (username:password)"
    )
    
    # Storage and backup
    storage_group = parser.add_argument_group('Storage Configuration')
    storage_group.add_argument(
        "--s3_bucket",
        type=str,
        default=None,
        help="S3 bucket for model storage"
    )
    storage_group.add_argument(
        "--s3_model_prefix",
        type=str,
        default="models/",
        help="S3 prefix for model files"
    )
    storage_group.add_argument(
        "--backup_models",
        action="store_true",
        help="Backup models to S3"
    )
    
    # Monitoring and logging
    monitor_group = parser.add_argument_group('Monitoring Configuration')
    monitor_group.add_argument(
        "--enable_monitoring",
        action="store_true",
        help="Enable monitoring and metrics"
    )
    monitor_group.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    return parser.parse_args()

class DeploymentConfig:
    """Deployment configuration manager."""
    
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load deployment configuration from file."""
        if not Path(self.config_file).exists():
            logger.warning(f"Config file {self.config_file} not found, using defaults")
            return self._get_default_config()
        
        try:
            with open(self.config_file, 'r') as f:
                if self.config_file.endswith('.yaml') or self.config_file.endswith('.yml'):
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
            
            logger.info(f"Loaded configuration from {self.config_file}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default deployment configuration."""
        return {
            "model": {
                "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
                "vae_model": "madebyollin/sdxl-vae-fp16-fix",
                "scheduler": "dpm_solver",
                "inference_steps": 30,
                "guidance_scale": 7.5
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 1,
                "timeout": 300,
                "max_requests": 1000
            },
            "docker": {
                "base_image": "nvidia/cuda:11.8-devel-ubuntu20.04",
                "python_version": "3.10",
                "requirements_file": "requirements.txt"
            },
            "ec2": {
                "instance_type": "g4dn.xlarge",
                "storage_size": 100,
                "auto_scaling": False
            },
            "monitoring": {
                "enable_metrics": True,
                "log_retention_days": 30,
                "alert_endpoints": []
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value

class ModelDeployer:
    """Main model deployment orchestrator."""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config = DeploymentConfig(args.config_file)
        self.aws_utils = None
        
        # Initialize AWS utilities if needed
        if args.deployment_type in ["ec2", "s3"] or args.backup_models:
            try:
                aws_config = create_aws_config_from_env()
                if args.s3_bucket:
                    aws_config.s3_bucket_name = args.s3_bucket
                self.aws_utils = AWSUtilities(aws_config)
            except Exception as e:
                logger.warning(f"Failed to initialize AWS utilities: {e}")
    
    def deploy(self):
        """Main deployment orchestrator."""
        logger.info(f"Starting {self.args.deployment_type} deployment...")
        
        # Backup models if requested
        if self.args.backup_models:
            self._backup_models()
        
        # Execute deployment based on type
        if self.args.deployment_type == "local":
            self._deploy_local()
        elif self.args.deployment_type == "ec2":
            self._deploy_ec2()
        elif self.args.deployment_type == "docker":
            self._deploy_docker()
        elif self.args.deployment_type == "kubernetes":
            self._deploy_kubernetes()
        elif self.args.deployment_type == "api":
            self._deploy_api()
        elif self.args.deployment_type == "gradio":
            self._deploy_gradio()
        else:
            raise ValueError(f"Unknown deployment type: {self.args.deployment_type}")
        
        logger.info("Deployment completed successfully!")
    
    def _backup_models(self):
        """Backup models to S3."""
        if not self.aws_utils:
            logger.warning("AWS utilities not available for backup")
            return
        
        logger.info("Backing up models to S3...")
        
        model_files = [
            self.args.model_path,
            self.args.lora_weights_path
        ]
        
        for model_file in model_files:
            if model_file and Path(model_file).exists():
                s3_key = f"{self.args.s3_model_prefix}{Path(model_file).name}"
                
                try:
                    self.aws_utils.s3.upload_file(model_file, s3_key)
                    logger.info(f"Backed up {model_file} to S3: {s3_key}")
                except Exception as e:
                    logger.error(f"Failed to backup {model_file}: {e}")
    
    def _deploy_local(self):
        """Deploy model locally."""
        logger.info("Setting up local deployment...")
        
        # Create local deployment structure
        deploy_dir = Path("deployment/local")
        deploy_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model files
        if Path(self.args.model_path).exists():
            import shutil
            shutil.copy2(self.args.model_path, deploy_dir / "model")
            
            if self.args.lora_weights_path and Path(self.args.lora_weights_path).exists():
                shutil.copy2(self.args.lora_weights_path, deploy_dir / "lora_weights")
        
        # Create startup script
        startup_script = deploy_dir / "start.sh"
        with open(startup_script, 'w') as f:
            f.write(f"""#!/bin/bash
# Ragamala Model Local Deployment Startup Script

echo "Starting Ragamala Painting Generator..."

# Activate environment
source venv/bin/activate

# Set environment variables
export MODEL_PATH="{deploy_dir}/model"
export LORA_WEIGHTS_PATH="{deploy_dir}/lora_weights"
export PYTHONPATH="$PWD:$PYTHONPATH"

# Start API server
python scripts/generate.py --mode interactive --interface gradio --port {self.args.gradio_port}
""")
        
        startup_script.chmod(0o755)
        
        logger.info(f"Local deployment ready at {deploy_dir}")
        logger.info(f"Run: {startup_script} to start the service")
    
    def _deploy_ec2(self):
        """Deploy model to EC2 instance."""
        if not self.aws_utils:
            raise RuntimeError("AWS utilities required for EC2 deployment")
        
        logger.info("Deploying to EC2...")
        
        # Create user data script for EC2 instance
        user_data_script = self._create_ec2_user_data()
        
        # Create EC2 instance
        instance_name = f"ragamala-{self.args.model_name}-{self.args.environment}"
        
        instance_id = self.aws_utils.ec2.create_instance(
            instance_name=instance_name,
            instance_type=self.args.instance_type,
            ami_id=self.args.ami_id,
            key_pair_name=self.args.key_pair_name,
            security_group_ids=self.args.security_group_ids,
            subnet_id=self.args.subnet_id,
            user_data_script=user_data_script,
            tags={
                "Project": "RagamalaPainting",
                "Environment": self.args.environment,
                "ModelName": self.args.model_name,
                "ModelVersion": self.args.model_version
            }
        )
        
        if instance_id:
            logger.info(f"EC2 instance created: {instance_id}")
            
            # Wait for instance to be running
            if self.aws_utils.ec2.wait_for_instance_state(instance_id, 'running'):
                instance_info = self.aws_utils.ec2.get_instance_info(instance_id)
                
                if instance_info:
                    public_ip = instance_info.get('PublicIpAddress')
                    if public_ip:
                        logger.info(f"Instance running at: {public_ip}")
                        logger.info(f"API will be available at: http://{public_ip}:{self.args.api_port}")
                        logger.info(f"Gradio interface at: http://{public_ip}:{self.args.gradio_port}")
            
            # Save deployment info
            deployment_info = {
                "instance_id": instance_id,
                "instance_type": self.args.instance_type,
                "model_name": self.args.model_name,
                "model_version": self.args.model_version,
                "deployment_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "environment": self.args.environment
            }
            
            with open(f"deployment/ec2_deployment_{instance_id}.json", 'w') as f:
                json.dump(deployment_info, f, indent=2)
        else:
            raise RuntimeError("Failed to create EC2 instance")
    
    def _create_ec2_user_data(self) -> str:
        """Create user data script for EC2 instance."""
        return f"""#!/bin/bash
# EC2 User Data Script for Ragamala Deployment

# Update system
apt update && apt upgrade -y

# Install dependencies
apt install -y python3.10 python3-pip git nginx

# Install NVIDIA drivers (if not in AMI)
apt install -y nvidia-driver-470 nvidia-cuda-toolkit

# Clone project repository
cd /home/ubuntu
git clone https://github.com/your-repo/ragamala-painting-generator.git
cd ragamala-painting-generator

# Install Python dependencies
pip3 install -r requirements.txt

# Download model from S3 if specified
if [ -n "{self.args.s3_bucket}" ]; then
    aws s3 cp s3://{self.args.s3_bucket}/{self.args.s3_model_prefix} ./models/ --recursive
fi

# Create systemd service for API
cat > /etc/systemd/system/ragamala-api.service << 'EOF'
[Unit]
Description=Ragamala Painting Generator API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/ragamala-painting-generator
Environment=MODEL_PATH={self.args.model_path}
Environment=LORA_WEIGHTS_PATH={self.args.lora_weights_path}
ExecStart=/usr/bin/python3 scripts/deploy.py --deployment_type api --api_port {self.args.api_port}
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Create systemd service for Gradio
cat > /etc/systemd/system/ragamala-gradio.service << 'EOF'
[Unit]
Description=Ragamala Painting Generator Gradio Interface
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/ragamala-painting-generator
Environment=MODEL_PATH={self.args.model_path}
Environment=LORA_WEIGHTS_PATH={self.args.lora_weights_path}
ExecStart=/usr/bin/python3 scripts/generate.py --mode interactive --interface gradio --port {self.args.gradio_port} --share
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Enable and start services
systemctl daemon-reload
systemctl enable ragamala-api
systemctl enable ragamala-gradio
systemctl start ragamala-api
systemctl start ragamala-gradio

# Configure nginx reverse proxy
cat > /etc/nginx/sites-available/ragamala << 'EOF'
server {{
    listen 80;
    server_name _;
    
    location /api/ {{
        proxy_pass http://localhost:{self.args.api_port}/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }}
    
    location / {{
        proxy_pass http://localhost:{self.args.gradio_port}/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }}
}}
EOF

ln -s /etc/nginx/sites-available/ragamala /etc/nginx/sites-enabled/
systemctl restart nginx

echo "Ragamala deployment completed on EC2" > /var/log/deployment.log
"""
    
    def _deploy_docker(self):
        """Deploy model using Docker."""
        logger.info("Building and deploying Docker container...")
        
        try:
            client = docker.from_env()
        except Exception as e:
            raise RuntimeError(f"Docker not available: {e}")
        
        # Create Dockerfile if it doesn't exist
        dockerfile_path = Path(self.args.dockerfile_path)
        if not dockerfile_path.exists():
            self._create_dockerfile(dockerfile_path)
        
        # Build Docker image
        image_tag = f"{self.args.docker_image_name}:{self.args.docker_tag}"
        
        logger.info(f"Building Docker image: {image_tag}")
        
        try:
            image, build_logs = client.images.build(
                path=str(Path.cwd()),
                dockerfile=str(dockerfile_path),
                tag=image_tag,
                rm=True
            )
            
            for log in build_logs:
                if 'stream' in log:
                    logger.info(log['stream'].strip())
            
            logger.info(f"Docker image built successfully: {image_tag}")
            
        except Exception as e:
            raise RuntimeError(f"Docker build failed: {e}")
        
        # Push to registry if requested
        if self.args.push_to_registry and self.args.registry_url:
            self._push_to_registry(client, image_tag)
        
        # Run container
        self._run_docker_container(client, image_tag)
    
    def _create_dockerfile(self, dockerfile_path: Path):
        """Create Dockerfile for deployment."""
        dockerfile_content = f"""# Ragamala Painting Generator Dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/app:$PYTHONPATH

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    python3.10 \\
    python3-pip \\
    git \\
    wget \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create model directory
RUN mkdir -p /app/models

# Set permissions
RUN chmod +x scripts/*.py

# Expose ports
EXPOSE {self.args.api_port}
EXPOSE {self.args.gradio_port}

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{self.args.api_port}/health || exit 1

# Default command
CMD ["python3", "scripts/deploy.py", "--deployment_type", "api", "--api_port", "{self.args.api_port}"]
"""
        
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        logger.info(f"Created Dockerfile at {dockerfile_path}")
    
    def _push_to_registry(self, client: docker.DockerClient, image_tag: str):
        """Push Docker image to registry."""
        logger.info(f"Pushing image to registry: {self.args.registry_url}")
        
        try:
            # Tag for registry
            registry_tag = f"{self.args.registry_url}/{image_tag}"
            client.images.get(image_tag).tag(registry_tag)
            
            # Push to registry
            push_logs = client.images.push(registry_tag, stream=True, decode=True)
            
            for log in push_logs:
                if 'status' in log:
                    logger.info(f"Push: {log['status']}")
            
            logger.info(f"Image pushed successfully: {registry_tag}")
            
        except Exception as e:
            logger.error(f"Failed to push to registry: {e}")
    
    def _run_docker_container(self, client: docker.DockerClient, image_tag: str):
        """Run Docker container."""
        logger.info("Starting Docker container...")
        
        container_name = f"{self.args.docker_image_name}-{self.args.environment}"
        
        # Remove existing container if it exists
        try:
            existing_container = client.containers.get(container_name)
            existing_container.stop()
            existing_container.remove()
            logger.info(f"Removed existing container: {container_name}")
        except docker.errors.NotFound:
            pass
        
        # Run new container
        try:
            container = client.containers.run(
                image_tag,
                name=container_name,
                ports={
                    f'{self.args.api_port}/tcp': self.args.api_port,
                    f'{self.args.gradio_port}/tcp': self.args.gradio_port
                },
                environment={
                    'MODEL_PATH': self.args.model_path,
                    'LORA_WEIGHTS_PATH': self.args.lora_weights_path,
                    'ENVIRONMENT': self.args.environment
                },
                volumes={
                    str(Path(self.args.model_path).parent): {'bind': '/app/models', 'mode': 'ro'}
                } if Path(self.args.model_path).exists() else {},
                runtime='nvidia' if torch.cuda.is_available() else None,
                detach=True,
                restart_policy={"Name": "unless-stopped"}
            )
            
            logger.info(f"Container started: {container.id}")
            logger.info(f"API available at: http://localhost:{self.args.api_port}")
            logger.info(f"Gradio interface at: http://localhost:{self.args.gradio_port}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to run container: {e}")
    
    def _deploy_kubernetes(self):
        """Deploy model to Kubernetes cluster."""
        logger.info("Deploying to Kubernetes...")
        
        # Create Kubernetes manifests
        k8s_dir = Path("deployment/kubernetes")
        k8s_dir.mkdir(parents=True, exist_ok=True)
        
        # Create deployment manifest
        deployment_manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"ragamala-{self.args.model_name}",
                "labels": {
                    "app": "ragamala-generator",
                    "version": self.args.model_version
                }
            },
            "spec": {
                "replicas": 1,
                "selector": {
                    "matchLabels": {
                        "app": "ragamala-generator"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "ragamala-generator"
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "ragamala-generator",
                            "image": f"{self.args.docker_image_name}:{self.args.docker_tag}",
                            "ports": [
                                {"containerPort": self.args.api_port},
                                {"containerPort": self.args.gradio_port}
                            ],
                            "env": [
                                {"name": "MODEL_PATH", "value": self.args.model_path},
                                {"name": "LORA_WEIGHTS_PATH", "value": self.args.lora_weights_path}
                            ],
                            "resources": {
                                "requests": {
                                    "nvidia.com/gpu": 1,
                                    "memory": "8Gi",
                                    "cpu": "2"
                                },
                                "limits": {
                                    "nvidia.com/gpu": 1,
                                    "memory": "16Gi",
                                    "cpu": "4"
                                }
                            }
                        }]
                    }
                }
            }
        }
        
        # Create service manifest
        service_manifest = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"ragamala-{self.args.model_name}-service"
            },
            "spec": {
                "selector": {
                    "app": "ragamala-generator"
                },
                "ports": [
                    {
                        "name": "api",
                        "port": 80,
                        "targetPort": self.args.api_port
                    },
                    {
                        "name": "gradio",
                        "port": 7860,
                        "targetPort": self.args.gradio_port
                    }
                ],
                "type": "LoadBalancer"
            }
        }
        
        # Save manifests
        with open(k8s_dir / "deployment.yaml", 'w') as f:
            yaml.dump(deployment_manifest, f, default_flow_style=False)
        
        with open(k8s_dir / "service.yaml", 'w') as f:
            yaml.dump(service_manifest, f, default_flow_style=False)
        
        logger.info(f"Kubernetes manifests created in {k8s_dir}")
        logger.info("Apply with: kubectl apply -f deployment/kubernetes/")
    
    def _deploy_api(self):
        """Deploy FastAPI server."""
        logger.info("Starting FastAPI server...")
        
        # Create FastAPI app
        app = create_app(
            model_path=self.args.model_path,
            lora_weights_path=self.args.lora_weights_path,
            enable_auth=self.args.enable_auth,
            rate_limit=self.args.rate_limit
        )
        
        # Configure uvicorn
        config = uvicorn.Config(
            app=app,
            host=self.args.api_host,
            port=self.args.api_port,
            workers=self.args.workers,
            log_level=self.args.log_level.lower(),
            access_log=True
        )
        
        server = uvicorn.Server(config)
        
        logger.info(f"API server starting at http://{self.args.api_host}:{self.args.api_port}")
        server.run()
    
    def _deploy_gradio(self):
        """Deploy Gradio interface."""
        logger.info("Starting Gradio interface...")
        
        # Load generator
        gen_config = GenerationConfig(
            model_path=self.args.model_path,
            lora_weights_path=self.args.lora_weights_path
        )
        
        generator = RagamalaGenerator(gen_config)
        
        # Create Gradio interface
        interface = self._create_gradio_interface(generator)
        
        # Configure authentication
        auth = None
        if self.args.gradio_auth:
            username, password = self.args.gradio_auth.split(':')
            auth = (username, password)
        
        # Launch interface
        interface.launch(
            server_port=self.args.gradio_port,
            share=self.args.gradio_share,
            auth=auth,
            server_name="0.0.0.0"
        )
    
    def _create_gradio_interface(self, generator: RagamalaGenerator) -> gr.Interface:
        """Create Gradio interface for the model."""
        def generate_image(prompt, raga, style, steps, guidance_scale):
            try:
                from src.inference.generator import GenerationRequest
                
                request = GenerationRequest(
                    prompt=prompt,
                    raga=raga if raga != "None" else None,
                    style=style if style != "None" else None,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    width=1024,
                    height=1024,
                    num_images=1
                )
                
                result = generator.generate(request)
                return result.images[0], f"Generated in {result.generation_time:.2f}s"
                
            except Exception as e:
                return None, f"Error: {str(e)}"
        
        interface = gr.Interface(
            fn=generate_image,
            inputs=[
                gr.Textbox(label="Prompt", placeholder="Describe the Ragamala painting..."),
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
                gr.Slider(10, 100, value=30, label="Inference Steps"),
                gr.Slider(1.0, 20.0, value=7.5, label="Guidance Scale")
            ],
            outputs=[
                gr.Image(label="Generated Image"),
                gr.Textbox(label="Status")
            ],
            title="Ragamala Painting Generator",
            description="Generate traditional Ragamala paintings using SDXL with cultural conditioning"
        )
        
        return interface

def main():
    """Main deployment function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting Ragamala model deployment")
    logger.info(f"Deployment type: {args.deployment_type}")
    logger.info(f"Environment: {args.environment}")
    logger.info(f"Model: {args.model_name} v{args.model_version}")
    
    # Validate model path
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    
    # Create deployer and deploy
    deployer = ModelDeployer(args)
    deployer.deploy()

if __name__ == "__main__":
    main()
