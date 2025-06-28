# Variables for Ragamala Painting Generator Terraform Configuration
# This file defines all input variables for the infrastructure deployment

# Project Configuration
variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "ragamala-painting-generator"
  
  validation {
    condition     = can(regex("^[a-z0-9-]+$", var.project_name))
    error_message = "Project name must contain only lowercase letters, numbers, and hyphens."
  }
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "project_owner" {
  description = "Owner of the project for tagging purposes"
  type        = string
  default     = "ai-research-team"
}

# AWS Configuration
variable "aws_region" {
  description = "AWS region for resource deployment"
  type        = string
  default     = "us-west-2"
  
  validation {
    condition = can(regex("^[a-z]{2}-[a-z]+-[0-9]$", var.aws_region))
    error_message = "AWS region must be in the format like us-west-2."
  }
}

variable "availability_zones" {
  description = "List of availability zones to use"
  type        = list(string)
  default     = ["us-west-2a", "us-west-2b", "us-west-2c"]
}

# Network Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
  
  validation {
    condition     = can(cidrhost(var.vpc_cidr, 0))
    error_message = "VPC CIDR must be a valid IPv4 CIDR block."
  }
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  
  validation {
    condition = length(var.public_subnet_cidrs) >= 2
    error_message = "At least 2 public subnets are required for high availability."
  }
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.10.0/24", "10.0.20.0/24", "10.0.30.0/24"]
  
  validation {
    condition = length(var.private_subnet_cidrs) >= 2
    error_message = "At least 2 private subnets are required for high availability."
  }
}

variable "allowed_ssh_cidrs" {
  description = "CIDR blocks allowed to SSH into instances"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

# Training Instance Configuration
variable "enable_training_instance" {
  description = "Whether to create training instances"
  type        = bool
  default     = true
}

variable "training_instance_type" {
  description = "EC2 instance type for training workloads"
  type        = string
  default     = "g5.2xlarge"
  
  validation {
    condition = contains([
      "g4dn.xlarge", "g4dn.2xlarge", "g4dn.4xlarge",
      "g5.xlarge", "g5.2xlarge", "g5.4xlarge", "g5.8xlarge",
      "p3.2xlarge", "p3.8xlarge", "p4d.24xlarge"
    ], var.training_instance_type)
    error_message = "Training instance type must be a GPU-enabled instance."
  }
}

variable "training_volume_size" {
  description = "EBS volume size for training instances (GB)"
  type        = number
  default     = 500
  
  validation {
    condition     = var.training_volume_size >= 100 && var.training_volume_size <= 2000
    error_message = "Training volume size must be between 100 and 2000 GB."
  }
}

variable "training_spot_instances" {
  description = "Use spot instances for training to reduce costs"
  type        = bool
  default     = false
}

variable "training_max_spot_price" {
  description = "Maximum price for spot instances (USD per hour)"
  type        = string
  default     = "1.50"
}

# Inference Instance Configuration
variable "inference_instance_type" {
  description = "EC2 instance type for inference workloads"
  type        = string
  default     = "g4dn.xlarge"
  
  validation {
    condition = contains([
      "g4dn.xlarge", "g4dn.2xlarge", "g4dn.4xlarge",
      "g5.xlarge", "g5.2xlarge", "g5.4xlarge",
      "m5.large", "m5.xlarge", "m5.2xlarge", "m5.4xlarge",
      "c5.large", "c5.xlarge", "c5.2xlarge", "c5.4xlarge"
    ], var.inference_instance_type)
    error_message = "Inference instance type must be suitable for ML inference."
  }
}

variable "inference_volume_size" {
  description = "EBS volume size for inference instances (GB)"
  type        = number
  default     = 200
  
  validation {
    condition     = var.inference_volume_size >= 50 && var.inference_volume_size <= 1000
    error_message = "Inference volume size must be between 50 and 1000 GB."
  }
}

# Auto Scaling Configuration
variable "inference_min_capacity" {
  description = "Minimum number of inference instances"
  type        = number
  default     = 1
  
  validation {
    condition     = var.inference_min_capacity >= 0 && var.inference_min_capacity <= 10
    error_message = "Minimum capacity must be between 0 and 10."
  }
}

variable "inference_max_capacity" {
  description = "Maximum number of inference instances"
  type        = number
  default     = 5
  
  validation {
    condition     = var.inference_max_capacity >= 1 && var.inference_max_capacity <= 20
    error_message = "Maximum capacity must be between 1 and 20."
  }
}

variable "inference_desired_capacity" {
  description = "Desired number of inference instances"
  type        = number
  default     = 2
  
  validation {
    condition     = var.inference_desired_capacity >= var.inference_min_capacity
    error_message = "Desired capacity must be greater than or equal to minimum capacity."
  }
}

# Storage Configuration
variable "s3_bucket_name" {
  description = "Name for S3 bucket (will be made unique with random suffix)"
  type        = string
  default     = ""
}

variable "s3_versioning_enabled" {
  description = "Enable versioning on S3 bucket"
  type        = bool
  default     = true
}

variable "s3_lifecycle_enabled" {
  description = "Enable lifecycle management on S3 bucket"
  type        = bool
  default     = true
}

variable "s3_transition_to_ia_days" {
  description = "Days after which objects transition to IA storage class"
  type        = number
  default     = 30
}

variable "s3_transition_to_glacier_days" {
  description = "Days after which objects transition to Glacier storage class"
  type        = number
  default     = 90
}

variable "s3_expiration_days" {
  description = "Days after which objects are deleted (0 to disable)"
  type        = number
  default     = 365
}

# Database Configuration
variable "enable_rds" {
  description = "Whether to create RDS instance for metadata storage"
  type        = bool
  default     = false
}

variable "rds_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
}

variable "rds_allocated_storage" {
  description = "RDS allocated storage in GB"
  type        = number
  default     = 20
}

variable "rds_engine_version" {
  description = "PostgreSQL engine version"
  type        = string
  default     = "14.9"
}

variable "rds_backup_retention_period" {
  description = "RDS backup retention period in days"
  type        = number
  default     = 7
}

# Redis Configuration
variable "enable_redis" {
  description = "Whether to create ElastiCache Redis cluster"
  type        = bool
  default     = false
}

variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.t3.micro"
}

variable "redis_num_cache_nodes" {
  description = "Number of cache nodes in Redis cluster"
  type        = number
  default     = 1
}

variable "redis_parameter_group_name" {
  description = "Redis parameter group name"
  type        = string
  default     = "default.redis7"
}

# Monitoring Configuration
variable "enable_cloudwatch_logs" {
  description = "Enable CloudWatch logs for instances"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "CloudWatch logs retention period in days"
  type        = number
  default     = 14
  
  validation {
    condition = contains([1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1827, 3653], var.log_retention_days)
    error_message = "Log retention days must be a valid CloudWatch retention period."
  }
}

variable "enable_detailed_monitoring" {
  description = "Enable detailed CloudWatch monitoring for EC2 instances"
  type        = bool
  default     = false
}

variable "sns_email_endpoint" {
  description = "Email address for SNS notifications"
  type        = string
  default     = ""
}

# Security Configuration
variable "enable_encryption" {
  description = "Enable encryption for EBS volumes and S3 bucket"
  type        = bool
  default     = true
}

variable "kms_key_deletion_window" {
  description = "KMS key deletion window in days"
  type        = number
  default     = 7
  
  validation {
    condition     = var.kms_key_deletion_window >= 7 && var.kms_key_deletion_window <= 30
    error_message = "KMS key deletion window must be between 7 and 30 days."
  }
}

variable "enable_flow_logs" {
  description = "Enable VPC flow logs"
  type        = bool
  default     = false
}

# Application Configuration
variable "api_port" {
  description = "Port for the API service"
  type        = number
  default     = 8000
  
  validation {
    condition     = var.api_port >= 1024 && var.api_port <= 65535
    error_message = "API port must be between 1024 and 65535."
  }
}

variable "gradio_port" {
  description = "Port for the Gradio interface"
  type        = number
  default     = 7860
}

variable "jupyter_port" {
  description = "Port for Jupyter notebook"
  type        = number
  default     = 8888
}

variable "tensorboard_port" {
  description = "Port for TensorBoard"
  type        = number
  default     = 6006
}

variable "mlflow_port" {
  description = "Port for MLflow tracking server"
  type        = number
  default     = 5000
}

# Model Configuration
variable "model_artifacts_path" {
  description = "S3 path for storing model artifacts"
  type        = string
  default     = "models/"
}

variable "dataset_path" {
  description = "S3 path for storing datasets"
  type        = string
  default     = "datasets/"
}

variable "outputs_path" {
  description = "S3 path for storing generated outputs"
  type        = string
  default     = "outputs/"
}

variable "logs_path" {
  description = "S3 path for storing application logs"
  type        = string
  default     = "logs/"
}

# Load Balancer Configuration
variable "enable_load_balancer" {
  description = "Whether to create Application Load Balancer"
  type        = bool
  default     = true
}

variable "load_balancer_type" {
  description = "Type of load balancer (application or network)"
  type        = string
  default     = "application"
  
  validation {
    condition     = contains(["application", "network"], var.load_balancer_type)
    error_message = "Load balancer type must be either 'application' or 'network'."
  }
}

variable "enable_ssl" {
  description = "Enable SSL/TLS for load balancer"
  type        = bool
  default     = false
}

variable "ssl_certificate_arn" {
  description = "ARN of SSL certificate for HTTPS"
  type        = string
  default     = ""
}

variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = ""
}

# Backup Configuration
variable "enable_automated_backups" {
  description = "Enable automated backups for EBS volumes"
  type        = bool
  default     = true
}

variable "backup_retention_days" {
  description = "Number of days to retain automated backups"
  type        = number
  default     = 7
  
  validation {
    condition     = var.backup_retention_days >= 1 && var.backup_retention_days <= 35
    error_message = "Backup retention days must be between 1 and 35."
  }
}

variable "backup_schedule" {
  description = "Cron expression for backup schedule"
  type        = string
  default     = "cron(0 2 * * ? *)"  # Daily at 2 AM UTC
}

# Cost Optimization
variable "enable_spot_instances" {
  description = "Use spot instances where possible to reduce costs"
  type        = bool
  default     = false
}

variable "enable_scheduled_scaling" {
  description = "Enable scheduled scaling to reduce costs during off-hours"
  type        = bool
  default     = false
}

variable "scale_down_schedule" {
  description = "Cron expression for scaling down (UTC)"
  type        = string
  default     = "cron(0 18 * * MON-FRI)"  # 6 PM weekdays
}

variable "scale_up_schedule" {
  description = "Cron expression for scaling up (UTC)"
  type        = string
  default     = "cron(0 8 * * MON-FRI)"   # 8 AM weekdays
}

# Development Configuration
variable "enable_bastion_host" {
  description = "Create bastion host for secure access to private instances"
  type        = bool
  default     = false
}

variable "bastion_instance_type" {
  description = "Instance type for bastion host"
  type        = string
  default     = "t3.micro"
}

variable "enable_jupyter_notebook" {
  description = "Install and configure Jupyter notebook on training instances"
  type        = bool
  default     = true
}

variable "enable_code_server" {
  description = "Install VS Code server on training instances"
  type        = bool
  default     = false
}

# Resource Tags
variable "additional_tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}

variable "cost_center" {
  description = "Cost center for billing purposes"
  type        = string
  default     = "ai-research"
}

variable "team" {
  description = "Team responsible for the resources"
  type        = string
  default     = "ml-engineering"
}

# Feature Flags
variable "enable_gpu_monitoring" {
  description = "Enable GPU-specific monitoring and alerting"
  type        = bool
  default     = true
}

variable "enable_model_versioning" {
  description = "Enable model versioning in S3"
  type        = bool
  default     = true
}

variable "enable_experiment_tracking" {
  description = "Enable MLflow for experiment tracking"
  type        = bool
  default     = true
}

variable "enable_data_validation" {
  description = "Enable data validation pipelines"
  type        = bool
  default     = false
}

# Performance Configuration
variable "ebs_volume_type" {
  description = "EBS volume type for instances"
  type        = string
  default     = "gp3"
  
  validation {
    condition     = contains(["gp2", "gp3", "io1", "io2"], var.ebs_volume_type)
    error_message = "EBS volume type must be one of: gp2, gp3, io1, io2."
  }
}

variable "ebs_iops" {
  description = "IOPS for EBS volumes (only for io1, io2, gp3)"
  type        = number
  default     = 3000
  
  validation {
    condition     = var.ebs_iops >= 100 && var.ebs_iops <= 64000
    error_message = "EBS IOPS must be between 100 and 64000."
  }
}

variable "ebs_throughput" {
  description = "Throughput for gp3 EBS volumes (MiB/s)"
  type        = number
  default     = 125
  
  validation {
    condition     = var.ebs_throughput >= 125 && var.ebs_throughput <= 1000
    error_message = "EBS throughput must be between 125 and 1000 MiB/s."
  }
}

# Disaster Recovery
variable "enable_multi_az" {
  description = "Deploy resources across multiple availability zones"
  type        = bool
  default     = true
}

variable "enable_cross_region_backup" {
  description = "Enable cross-region backup for critical data"
  type        = bool
  default     = false
}

variable "backup_region" {
  description = "AWS region for cross-region backups"
  type        = string
  default     = "us-east-1"
}

# Compliance and Governance
variable "enable_config_rules" {
  description = "Enable AWS Config rules for compliance monitoring"
  type        = bool
  default     = false
}

variable "enable_cloudtrail" {
  description = "Enable CloudTrail for audit logging"
  type        = bool
  default     = false
}

variable "enable_guardduty" {
  description = "Enable GuardDuty for security monitoring"
  type        = bool
  default     = false
}

# Local Variables for Computed Values
locals {
  # Common tags applied to all resources
  common_tags = merge(
    {
      Project     = var.project_name
      Environment = var.environment
      Owner       = var.project_owner
      ManagedBy   = "terraform"
      CostCenter  = var.cost_center
      Team        = var.team
    },
    var.additional_tags
  )
  
  # S3 bucket name with uniqueness
  s3_bucket_name = var.s3_bucket_name != "" ? var.s3_bucket_name : "${var.project_name}-data-${var.environment}"
  
  # Instance name prefixes
  training_instance_name   = "${var.project_name}-training-${var.environment}"
  inference_instance_name  = "${var.project_name}-inference-${var.environment}"
  bastion_instance_name    = "${var.project_name}-bastion-${var.environment}"
  
  # Security group names
  training_sg_name   = "${var.project_name}-training-sg-${var.environment}"
  inference_sg_name  = "${var.project_name}-inference-sg-${var.environment}"
  alb_sg_name        = "${var.project_name}-alb-sg-${var.environment}"
  bastion_sg_name    = "${var.project_name}-bastion-sg-${var.environment}"
  
  # Load balancer configuration
  enable_https = var.enable_ssl && var.ssl_certificate_arn != ""
  
  # Monitoring configuration
  enable_monitoring = var.enable_cloudwatch_logs || var.enable_detailed_monitoring || var.enable_gpu_monitoring
}
