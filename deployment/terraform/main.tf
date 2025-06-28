# Terraform configuration for Ragamala Painting Generator on AWS EC2
# This configuration sets up the complete infrastructure for SDXL 1.0 fine-tuning
# and inference deployment optimized for GPU workloads

terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.1"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }

  backend "s3" {
    bucket         = "ragamala-terraform-state"
    key            = "infrastructure/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    dynamodb_table = "ragamala-terraform-locks"
  }
}

# Configure the AWS Provider
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "ragamala-painting-generator"
      Environment = var.environment
      ManagedBy   = "terraform"
      Owner       = var.project_owner
      CostCenter  = "ai-research"
    }
  }
}

# Data sources for availability zones and AMI
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning AMI GPU PyTorch*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# Random password for database
resource "random_password" "db_password" {
  length  = 16
  special = true
}

# Generate SSH key pair
resource "tls_private_key" "ragamala_key" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "aws_key_pair" "ragamala_key_pair" {
  key_name   = "${var.project_name}-key-${var.environment}"
  public_key = tls_private_key.ragamala_key.public_key_openssh

  tags = {
    Name = "${var.project_name}-key-${var.environment}"
  }
}

# Store private key in AWS Secrets Manager
resource "aws_secretsmanager_secret" "ssh_private_key" {
  name                    = "${var.project_name}-ssh-key-${var.environment}"
  description             = "SSH private key for Ragamala project instances"
  recovery_window_in_days = 7

  tags = {
    Name = "${var.project_name}-ssh-key"
  }
}

resource "aws_secretsmanager_secret_version" "ssh_private_key" {
  secret_id     = aws_secretsmanager_secret.ssh_private_key.id
  secret_string = tls_private_key.ragamala_key.private_key_pem
}

# VPC Configuration
resource "aws_vpc" "ragamala_vpc" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "${var.project_name}-vpc-${var.environment}"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "ragamala_igw" {
  vpc_id = aws_vpc.ragamala_vpc.id

  tags = {
    Name = "${var.project_name}-igw-${var.environment}"
  }
}

# Public Subnets
resource "aws_subnet" "public_subnets" {
  count = length(var.public_subnet_cidrs)

  vpc_id                  = aws_vpc.ragamala_vpc.id
  cidr_block              = var.public_subnet_cidrs[count.index]
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.project_name}-public-subnet-${count.index + 1}-${var.environment}"
    Type = "public"
  }
}

# Private Subnets
resource "aws_subnet" "private_subnets" {
  count = length(var.private_subnet_cidrs)

  vpc_id            = aws_vpc.ragamala_vpc.id
  cidr_block        = var.private_subnet_cidrs[count.index]
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name = "${var.project_name}-private-subnet-${count.index + 1}-${var.environment}"
    Type = "private"
  }
}

# NAT Gateway for private subnets
resource "aws_eip" "nat_eips" {
  count = length(var.public_subnet_cidrs)

  domain = "vpc"
  depends_on = [aws_internet_gateway.ragamala_igw]

  tags = {
    Name = "${var.project_name}-nat-eip-${count.index + 1}-${var.environment}"
  }
}

resource "aws_nat_gateway" "nat_gateways" {
  count = length(var.public_subnet_cidrs)

  allocation_id = aws_eip.nat_eips[count.index].id
  subnet_id     = aws_subnet.public_subnets[count.index].id

  tags = {
    Name = "${var.project_name}-nat-gateway-${count.index + 1}-${var.environment}"
  }

  depends_on = [aws_internet_gateway.ragamala_igw]
}

# Route Tables
resource "aws_route_table" "public_route_table" {
  vpc_id = aws_vpc.ragamala_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.ragamala_igw.id
  }

  tags = {
    Name = "${var.project_name}-public-rt-${var.environment}"
  }
}

resource "aws_route_table" "private_route_tables" {
  count = length(var.private_subnet_cidrs)

  vpc_id = aws_vpc.ragamala_vpc.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.nat_gateways[count.index].id
  }

  tags = {
    Name = "${var.project_name}-private-rt-${count.index + 1}-${var.environment}"
  }
}

# Route Table Associations
resource "aws_route_table_association" "public_subnet_associations" {
  count = length(var.public_subnet_cidrs)

  subnet_id      = aws_subnet.public_subnets[count.index].id
  route_table_id = aws_route_table.public_route_table.id
}

resource "aws_route_table_association" "private_subnet_associations" {
  count = length(var.private_subnet_cidrs)

  subnet_id      = aws_subnet.private_subnets[count.index].id
  route_table_id = aws_route_table.private_route_tables[count.index].id
}

# Security Groups
resource "aws_security_group" "training_instance_sg" {
  name_prefix = "${var.project_name}-training-sg-"
  vpc_id      = aws_vpc.ragamala_vpc.id
  description = "Security group for SDXL training instances"

  # SSH access
  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_cidrs
  }

  # Jupyter notebook access
  ingress {
    description = "Jupyter"
    from_port   = 8888
    to_port     = 8888
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_cidrs
  }

  # TensorBoard access
  ingress {
    description = "TensorBoard"
    from_port   = 6006
    to_port     = 6006
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_cidrs
  }

  # MLflow tracking server
  ingress {
    description = "MLflow"
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_cidrs
  }

  # All outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-training-sg-${var.environment}"
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_security_group" "inference_instance_sg" {
  name_prefix = "${var.project_name}-inference-sg-"
  vpc_id      = aws_vpc.ragamala_vpc.id
  description = "Security group for inference API instances"

  # SSH access
  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_cidrs
  }

  # API access
  ingress {
    description = "API"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Gradio interface
  ingress {
    description = "Gradio"
    from_port   = 7860
    to_port     = 7860
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Health check
  ingress {
    description = "Health Check"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # HTTPS
  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # All outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-inference-sg-${var.environment}"
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_security_group" "alb_sg" {
  name_prefix = "${var.project_name}-alb-sg-"
  vpc_id      = aws_vpc.ragamala_vpc.id
  description = "Security group for Application Load Balancer"

  # HTTP
  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # HTTPS
  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # All outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-alb-sg-${var.environment}"
  }

  lifecycle {
    create_before_destroy = true
  }
}

# IAM Roles and Policies
resource "aws_iam_role" "ec2_role" {
  name = "${var.project_name}-ec2-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name = "${var.project_name}-ec2-role-${var.environment}"
  }
}

resource "aws_iam_policy" "ec2_policy" {
  name        = "${var.project_name}-ec2-policy-${var.environment}"
  description = "Policy for Ragamala EC2 instances"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.ragamala_bucket.arn,
          "${aws_s3_bucket.ragamala_bucket.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = [
          aws_secretsmanager_secret.ssh_private_key.arn,
          aws_secretsmanager_secret.api_keys.arn
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "cloudwatch:GetMetricStatistics",
          "cloudwatch:ListMetrics"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ec2_policy_attachment" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = aws_iam_policy.ec2_policy.arn
}

resource "aws_iam_instance_profile" "ec2_profile" {
  name = "${var.project_name}-ec2-profile-${var.environment}"
  role = aws_iam_role.ec2_role.name

  tags = {
    Name = "${var.project_name}-ec2-profile-${var.environment}"
  }
}

# S3 Bucket for datasets and models
resource "aws_s3_bucket" "ragamala_bucket" {
  bucket = "${var.project_name}-data-${var.environment}-${random_id.bucket_suffix.hex}"

  tags = {
    Name = "${var.project_name}-data-bucket-${var.environment}"
  }
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "aws_s3_bucket_versioning" "ragamala_bucket_versioning" {
  bucket = aws_s3_bucket.ragamala_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "ragamala_bucket_encryption" {
  bucket = aws_s3_bucket.ragamala_bucket.id

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "aws_s3_bucket_public_access_block" "ragamala_bucket_pab" {
  bucket = aws_s3_bucket.ragamala_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Launch Template for Training Instances
resource "aws_launch_template" "training_template" {
  name_prefix   = "${var.project_name}-training-template-"
  image_id      = data.aws_ami.deep_learning.id
  instance_type = var.training_instance_type
  key_name      = aws_key_pair.ragamala_key_pair.key_name

  vpc_security_group_ids = [aws_security_group.training_instance_sg.id]

  iam_instance_profile {
    name = aws_iam_instance_profile.ec2_profile.name
  }

  block_device_mappings {
    device_name = "/dev/sda1"
    ebs {
      volume_size           = var.training_volume_size
      volume_type           = "gp3"
      iops                  = 3000
      throughput            = 125
      delete_on_termination = true
      encrypted             = true
    }
  }

  user_data = base64encode(templatefile("${path.module}/user_data/training_setup.sh", {
    s3_bucket = aws_s3_bucket.ragamala_bucket.bucket
    region    = var.aws_region
  }))

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name = "${var.project_name}-training-instance-${var.environment}"
      Type = "training"
    }
  }

  lifecycle {
    create_before_destroy = true
  }
}

# Training Instance
resource "aws_instance" "training_instance" {
  count = var.enable_training_instance ? 1 : 0

  launch_template {
    id      = aws_launch_template.training_template.id
    version = "$Latest"
  }

  subnet_id = aws_subnet.private_subnets[0].id

  tags = {
    Name = "${var.project_name}-training-${var.environment}"
    Type = "training"
  }
}

# Launch Template for Inference Instances
resource "aws_launch_template" "inference_template" {
  name_prefix   = "${var.project_name}-inference-template-"
  image_id      = data.aws_ami.deep_learning.id
  instance_type = var.inference_instance_type
  key_name      = aws_key_pair.ragamala_key_pair.key_name

  vpc_security_group_ids = [aws_security_group.inference_instance_sg.id]

  iam_instance_profile {
    name = aws_iam_instance_profile.ec2_profile.name
  }

  block_device_mappings {
    device_name = "/dev/sda1"
    ebs {
      volume_size           = var.inference_volume_size
      volume_type           = "gp3"
      iops                  = 3000
      throughput            = 125
      delete_on_termination = true
      encrypted             = true
    }
  }

  user_data = base64encode(templatefile("${path.module}/user_data/inference_setup.sh", {
    s3_bucket = aws_s3_bucket.ragamala_bucket.bucket
    region    = var.aws_region
  }))

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name = "${var.project_name}-inference-instance-${var.environment}"
      Type = "inference"
    }
  }

  lifecycle {
    create_before_destroy = true
  }
}

# Auto Scaling Group for Inference
resource "aws_autoscaling_group" "inference_asg" {
  name                = "${var.project_name}-inference-asg-${var.environment}"
  vpc_zone_identifier = aws_subnet.private_subnets[*].id
  target_group_arns   = [aws_lb_target_group.inference_tg.arn]
  health_check_type   = "ELB"
  health_check_grace_period = 300

  min_size         = var.inference_min_capacity
  max_size         = var.inference_max_capacity
  desired_capacity = var.inference_desired_capacity

  launch_template {
    id      = aws_launch_template.inference_template.id
    version = "$Latest"
  }

  tag {
    key                 = "Name"
    value               = "${var.project_name}-inference-asg-${var.environment}"
    propagate_at_launch = true
  }

  tag {
    key                 = "Type"
    value               = "inference"
    propagate_at_launch = true
  }

  instance_refresh {
    strategy = "Rolling"
    preferences {
      min_healthy_percentage = 50
    }
  }
}

# Application Load Balancer
resource "aws_lb" "inference_alb" {
  name               = "${var.project_name}-alb-${var.environment}"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_sg.id]
  subnets            = aws_subnet.public_subnets[*].id

  enable_deletion_protection = false

  tags = {
    Name = "${var.project_name}-alb-${var.environment}"
  }
}

resource "aws_lb_target_group" "inference_tg" {
  name     = "${var.project_name}-tg-${var.environment}"
  port     = 8000
  protocol = "HTTP"
  vpc_id   = aws_vpc.ragamala_vpc.id

  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = "/health"
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 2
  }

  tags = {
    Name = "${var.project_name}-tg-${var.environment}"
  }
}

resource "aws_lb_listener" "inference_listener" {
  load_balancer_arn = aws_lb.inference_alb.arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.inference_tg.arn
  }
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "training_logs" {
  name              = "/aws/ec2/${var.project_name}/training"
  retention_in_days = 14

  tags = {
    Name = "${var.project_name}-training-logs-${var.environment}"
  }
}

resource "aws_cloudwatch_log_group" "inference_logs" {
  name              = "/aws/ec2/${var.project_name}/inference"
  retention_in_days = 7

  tags = {
    Name = "${var.project_name}-inference-logs-${var.environment}"
  }
}

# API Keys Secret
resource "aws_secretsmanager_secret" "api_keys" {
  name                    = "${var.project_name}-api-keys-${var.environment}"
  description             = "API keys for Ragamala painting generator"
  recovery_window_in_days = 7

  tags = {
    Name = "${var.project_name}-api-keys"
  }
}

resource "aws_secretsmanager_secret_version" "api_keys" {
  secret_id = aws_secretsmanager_secret.api_keys.id
  secret_string = jsonencode({
    admin_key = random_password.admin_api_key.result
    user_key  = random_password.user_api_key.result
  })
}

resource "random_password" "admin_api_key" {
  length  = 32
  special = false
}

resource "random_password" "user_api_key" {
  length  = 32
  special = false
}

# Elastic IP for NAT Gateway (if needed for static IP)
resource "aws_eip" "training_instance_eip" {
  count = var.enable_training_instance ? 1 : 0

  domain   = "vpc"
  instance = aws_instance.training_instance[0].id

  tags = {
    Name = "${var.project_name}-training-eip-${var.environment}"
  }

  depends_on = [aws_internet_gateway.ragamala_igw]
}

# CloudWatch Alarms
resource "aws_cloudwatch_metric_alarm" "high_cpu" {
  alarm_name          = "${var.project_name}-high-cpu-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = "120"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "This metric monitors ec2 cpu utilization"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    AutoScalingGroupName = aws_autoscaling_group.inference_asg.name
  }

  tags = {
    Name = "${var.project_name}-high-cpu-alarm-${var.environment}"
  }
}

# SNS Topic for Alerts
resource "aws_sns_topic" "alerts" {
  name = "${var.project_name}-alerts-${var.environment}"

  tags = {
    Name = "${var.project_name}-alerts-${var.environment}"
  }
}

# Auto Scaling Policies
resource "aws_autoscaling_policy" "scale_up" {
  name                   = "${var.project_name}-scale-up-${var.environment}"
  scaling_adjustment     = 1
  adjustment_type        = "ChangeInCapacity"
  cooldown               = 300
  autoscaling_group_name = aws_autoscaling_group.inference_asg.name
}

resource "aws_autoscaling_policy" "scale_down" {
  name                   = "${var.project_name}-scale-down-${var.environment}"
  scaling_adjustment     = -1
  adjustment_type        = "ChangeInCapacity"
  cooldown               = 300
  autoscaling_group_name = aws_autoscaling_group.inference_asg.name
}

# CloudWatch Alarms for Auto Scaling
resource "aws_cloudwatch_metric_alarm" "cpu_high" {
  alarm_name          = "${var.project_name}-cpu-high-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = "300"
  statistic           = "Average"
  threshold           = "75"
  alarm_description   = "This metric monitors ec2 cpu utilization for scaling up"
  alarm_actions       = [aws_autoscaling_policy.scale_up.arn]

  dimensions = {
    AutoScalingGroupName = aws_autoscaling_group.inference_asg.name
  }
}

resource "aws_cloudwatch_metric_alarm" "cpu_low" {
  alarm_name          = "${var.project_name}-cpu-low-${var.environment}"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = "300"
  statistic           = "Average"
  threshold           = "25"
  alarm_description   = "This metric monitors ec2 cpu utilization for scaling down"
  alarm_actions       = [aws_autoscaling_policy.scale_down.arn]

  dimensions = {
    AutoScalingGroupName = aws_autoscaling_group.inference_asg.name
  }
}
