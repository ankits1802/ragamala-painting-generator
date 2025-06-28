# Outputs for Ragamala Painting Generator Terraform Configuration
# This file defines all output values for the infrastructure deployment

# VPC and Network Outputs
output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.ragamala_vpc.id
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = aws_vpc.ragamala_vpc.cidr_block
}

output "internet_gateway_id" {
  description = "ID of the Internet Gateway"
  value       = aws_internet_gateway.ragamala_igw.id
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = aws_subnet.public_subnets[*].id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = aws_subnet.private_subnets[*].id
}

output "public_subnet_cidrs" {
  description = "CIDR blocks of the public subnets"
  value       = aws_subnet.public_subnets[*].cidr_block
}

output "private_subnet_cidrs" {
  description = "CIDR blocks of the private subnets"
  value       = aws_subnet.private_subnets[*].cidr_block
}

output "nat_gateway_ids" {
  description = "IDs of the NAT Gateways"
  value       = aws_nat_gateway.nat_gateways[*].id
}

output "nat_gateway_public_ips" {
  description = "Public IP addresses of the NAT Gateways"
  value       = aws_eip.nat_eips[*].public_ip
}

# Security Group Outputs
output "training_security_group_id" {
  description = "ID of the training instances security group"
  value       = aws_security_group.training_instance_sg.id
}

output "inference_security_group_id" {
  description = "ID of the inference instances security group"
  value       = aws_security_group.inference_instance_sg.id
}

output "alb_security_group_id" {
  description = "ID of the Application Load Balancer security group"
  value       = aws_security_group.alb_sg.id
}

# EC2 Instance Outputs
output "training_instance_id" {
  description = "ID of the training instance"
  value       = var.enable_training_instance ? aws_instance.training_instance[0].id : null
}

output "training_instance_private_ip" {
  description = "Private IP address of the training instance"
  value       = var.enable_training_instance ? aws_instance.training_instance[0].private_ip : null
}

output "training_instance_public_ip" {
  description = "Public IP address of the training instance (if EIP attached)"
  value       = var.enable_training_instance ? aws_eip.training_instance_eip[0].public_ip : null
}

output "training_instance_dns" {
  description = "Public DNS name of the training instance"
  value       = var.enable_training_instance ? aws_instance.training_instance[0].public_dns : null
}

# Auto Scaling Group Outputs
output "inference_autoscaling_group_name" {
  description = "Name of the inference Auto Scaling Group"
  value       = aws_autoscaling_group.inference_asg.name
}

output "inference_autoscaling_group_arn" {
  description = "ARN of the inference Auto Scaling Group"
  value       = aws_autoscaling_group.inference_asg.arn
}

output "inference_launch_template_id" {
  description = "ID of the inference launch template"
  value       = aws_launch_template.inference_template.id
}

output "training_launch_template_id" {
  description = "ID of the training launch template"
  value       = aws_launch_template.training_template.id
}

# Load Balancer Outputs
output "load_balancer_dns_name" {
  description = "DNS name of the Application Load Balancer"
  value       = aws_lb.inference_alb.dns_name
}

output "load_balancer_zone_id" {
  description = "Zone ID of the Application Load Balancer"
  value       = aws_lb.inference_alb.zone_id
}

output "load_balancer_arn" {
  description = "ARN of the Application Load Balancer"
  value       = aws_lb.inference_alb.arn
}

output "target_group_arn" {
  description = "ARN of the target group"
  value       = aws_lb_target_group.inference_tg.arn
}

output "api_endpoint_url" {
  description = "URL for the API endpoint"
  value       = "http://${aws_lb.inference_alb.dns_name}"
}

output "api_health_check_url" {
  description = "URL for the API health check endpoint"
  value       = "http://${aws_lb.inference_alb.dns_name}/health"
}

# S3 Bucket Outputs
output "s3_bucket_name" {
  description = "Name of the S3 bucket for data storage"
  value       = aws_s3_bucket.ragamala_bucket.bucket
}

output "s3_bucket_arn" {
  description = "ARN of the S3 bucket"
  value       = aws_s3_bucket.ragamala_bucket.arn
}

output "s3_bucket_domain_name" {
  description = "Domain name of the S3 bucket"
  value       = aws_s3_bucket.ragamala_bucket.bucket_domain_name
}

output "s3_bucket_regional_domain_name" {
  description = "Regional domain name of the S3 bucket"
  value       = aws_s3_bucket.ragamala_bucket.bucket_regional_domain_name
}

# IAM Outputs
output "ec2_instance_profile_name" {
  description = "Name of the EC2 instance profile"
  value       = aws_iam_instance_profile.ec2_profile.name
}

output "ec2_instance_profile_arn" {
  description = "ARN of the EC2 instance profile"
  value       = aws_iam_instance_profile.ec2_profile.arn
}

output "ec2_role_name" {
  description = "Name of the EC2 IAM role"
  value       = aws_iam_role.ec2_role.name
}

output "ec2_role_arn" {
  description = "ARN of the EC2 IAM role"
  value       = aws_iam_role.ec2_role.arn
}

# Key Pair Outputs
output "key_pair_name" {
  description = "Name of the EC2 key pair"
  value       = aws_key_pair.ragamala_key_pair.key_name
}

output "key_pair_fingerprint" {
  description = "Fingerprint of the EC2 key pair"
  value       = aws_key_pair.ragamala_key_pair.fingerprint
}

# Secrets Manager Outputs
output "ssh_private_key_secret_arn" {
  description = "ARN of the SSH private key secret in Secrets Manager"
  value       = aws_secretsmanager_secret.ssh_private_key.arn
  sensitive   = true
}

output "api_keys_secret_arn" {
  description = "ARN of the API keys secret in Secrets Manager"
  value       = aws_secretsmanager_secret.api_keys.arn
  sensitive   = true
}

# CloudWatch Outputs
output "training_log_group_name" {
  description = "Name of the training CloudWatch log group"
  value       = aws_cloudwatch_log_group.training_logs.name
}

output "inference_log_group_name" {
  description = "Name of the inference CloudWatch log group"
  value       = aws_cloudwatch_log_group.inference_logs.name
}

output "training_log_group_arn" {
  description = "ARN of the training CloudWatch log group"
  value       = aws_cloudwatch_log_group.training_logs.arn
}

output "inference_log_group_arn" {
  description = "ARN of the inference CloudWatch log group"
  value       = aws_cloudwatch_log_group.inference_logs.arn
}

# SNS Outputs
output "alerts_topic_arn" {
  description = "ARN of the SNS topic for alerts"
  value       = aws_sns_topic.alerts.arn
}

output "alerts_topic_name" {
  description = "Name of the SNS topic for alerts"
  value       = aws_sns_topic.alerts.name
}

# Auto Scaling Policy Outputs
output "scale_up_policy_arn" {
  description = "ARN of the scale up policy"
  value       = aws_autoscaling_policy.scale_up.arn
}

output "scale_down_policy_arn" {
  description = "ARN of the scale down policy"
  value       = aws_autoscaling_policy.scale_down.arn
}

# CloudWatch Alarm Outputs
output "high_cpu_alarm_arn" {
  description = "ARN of the high CPU alarm"
  value       = aws_cloudwatch_metric_alarm.high_cpu.arn
}

output "cpu_high_alarm_arn" {
  description = "ARN of the CPU high alarm for scaling"
  value       = aws_cloudwatch_metric_alarm.cpu_high.arn
}

output "cpu_low_alarm_arn" {
  description = "ARN of the CPU low alarm for scaling"
  value       = aws_cloudwatch_metric_alarm.cpu_low.arn
}

# Environment Information Outputs
output "environment" {
  description = "Environment name"
  value       = var.environment
}

output "project_name" {
  description = "Project name"
  value       = var.project_name
}

output "aws_region" {
  description = "AWS region"
  value       = var.aws_region
}

# Resource Tags Output
output "common_tags" {
  description = "Common tags applied to all resources"
  value       = local.common_tags
}

# Connection Information Outputs
output "ssh_connection_command" {
  description = "SSH command to connect to training instance"
  value = var.enable_training_instance ? "ssh -i ~/.ssh/${aws_key_pair.ragamala_key_pair.key_name}.pem ubuntu@${aws_eip.training_instance_eip[0].public_ip}" : "No training instance created"
}

output "jupyter_notebook_url" {
  description = "URL for Jupyter notebook on training instance"
  value = var.enable_training_instance ? "<http://${aws_eip.training_instance_eip>[0].public_ip}:8888" : "No training instance created"
}

output "tensorboard_url" {
  description = "URL for TensorBoard on training instance"
  value = var.enable_training_instance ? "<http://${aws_eip.training_instance_eip>[0].public_ip}:6006" : "No training instance created"
}

output "mlflow_url" {
  description = "URL for MLflow tracking server"
  value = var.enable_training_instance ? "<http://${aws_eip.training_instance_eip>[0].public_ip}:5000" : "No training instance created"
}

# API Endpoints
output "api_generate_endpoint" {
  description = "API endpoint for image generation"
  value       = "http://${aws_lb.inference_alb.dns_name}/generate"
}

output "api_batch_generate_endpoint" {
  description = "API endpoint for batch image generation"
  value       = "http://${aws_lb.inference_alb.dns_name}/batch-generate"
}

output "api_docs_url" {
  description = "URL for API documentation"
  value       = "http://${aws_lb.inference_alb.dns_name}/docs"
}

output "gradio_interface_url" {
  description = "URL for Gradio web interface"
  value       = "http://${aws_lb.inference_alb.dns_name}:7860"
}

# Cost Optimization Information
output "estimated_monthly_cost" {
  description = "Estimated monthly cost breakdown (approximate)"
  value = {
    training_instance = var.enable_training_instance ? "~$200-400/month (g5.2xlarge)" : "$0"
    inference_instances = "~$150-300/month (2x g4dn.xlarge)"
    storage = "~$50-100/month (EBS + S3)"
    networking = "~$20-50/month (NAT Gateway + Data Transfer)"
    total_estimated = "~$420-850/month"
  }
}

# Security Information
output "security_recommendations" {
  description = "Security recommendations for the deployment"
  value = {
    ssh_access = "Restrict SSH access to specific IP ranges in security groups"
    api_keys = "Rotate API keys stored in Secrets Manager regularly"
    ssl_certificate = "Consider adding SSL certificate for HTTPS access"
    vpc_flow_logs = "Enable VPC Flow Logs for network monitoring"
    cloudtrail = "Enable CloudTrail for audit logging"
  }
}

# Monitoring URLs
output "cloudwatch_dashboard_url" {
  description = "URL for CloudWatch dashboard"
  value       = "https://${var.aws_region}.console.aws.amazon.com/cloudwatch/home?region=${var.aws_region}#dashboards:"
}

output "ec2_console_url" {
  description = "URL for EC2 console"
  value       = "https://${var.aws_region}.console.aws.amazon.com/ec2/home?region=${var.aws_region}#Instances:"
}

output "s3_console_url" {
  description = "URL for S3 console"
  value       = "https://s3.console.aws.amazon.com/s3/buckets/${aws_s3_bucket.ragamala_bucket.bucket}"
}

# Deployment Information
output "deployment_timestamp" {
  description = "Timestamp of the deployment"
  value       = timestamp()
}

output "terraform_workspace" {
  description = "Terraform workspace used for deployment"
  value       = terraform.workspace
}

# Instance Type Information
output "instance_types" {
  description = "Instance types used in the deployment"
  value = {
    training = var.training_instance_type
    inference = var.inference_instance_type
  }
}

# Storage Information
output "storage_configuration" {
  description = "Storage configuration details"
  value = {
    training_volume_size = "${var.training_volume_size}GB"
    inference_volume_size = "${var.inference_volume_size}GB"
    s3_bucket = aws_s3_bucket.ragamala_bucket.bucket
    ebs_encryption = "Enabled"
    s3_versioning = "Enabled"
  }
}

# Network Configuration Summary
output "network_configuration" {
  description = "Network configuration summary"
  value = {
    vpc_cidr = aws_vpc.ragamala_vpc.cidr_block
    public_subnets = length(aws_subnet.public_subnets)
    private_subnets = length(aws_subnet.private_subnets)
    availability_zones = length(data.aws_availability_zones.available.names)
    nat_gateways = length(aws_nat_gateway.nat_gateways)
  }
}

# Auto Scaling Configuration
output "autoscaling_configuration" {
  description = "Auto Scaling configuration details"
  value = {
    min_capacity = var.inference_min_capacity
    max_capacity = var.inference_max_capacity
    desired_capacity = var.inference_desired_capacity
    health_check_type = "ELB"
    health_check_grace_period = "300 seconds"
  }
}

# Load Balancer Configuration
output "load_balancer_configuration" {
  description = "Load balancer configuration details"
  value = {
    type = "application"
    scheme = "internet-facing"
    target_port = 8000
    health_check_path = "/health"
    health_check_interval = "30 seconds"
    healthy_threshold = 2
    unhealthy_threshold = 2
  }
}

# API Configuration
output "api_configuration" {
  description = "API configuration details"
  value = {
    base_url = "http://${aws_lb.inference_alb.dns_name}"
    health_endpoint = "/health"
    generate_endpoint = "/generate"
    batch_generate_endpoint = "/batch-generate"
    docs_endpoint = "/docs"
    authentication = "API Key based"
  }
}

# Quick Start Commands
output "quick_start_commands" {
  description = "Quick start commands for using the deployment"
  value = {
    ssh_to_training = var.enable_training_instance ? "ssh -i ~/.ssh/${aws_key_pair.ragamala_key_pair.key_name}.pem ubuntu@${aws_eip.training_instance_eip[0].public_ip}" : "No training instance"
    test_api = "curl -X GET ${aws_lb.inference_alb.dns_name}/health"
    view_logs = "aws logs tail /aws/ec2/${var.project_name}/inference --follow"
    download_ssh_key = "aws secretsmanager get-secret-value --secret-id ${aws_secretsmanager_secret.ssh_private_key.name} --query SecretString --output text"
  }
}

# Resource Counts
output "resource_summary" {
  description = "Summary of created resources"
  value = {
    vpc_count = 1
    subnet_count = length(aws_subnet.public_subnets) + length(aws_subnet.private_subnets)
    security_group_count = 3
    instance_count = var.enable_training_instance ? 1 : 0
    autoscaling_group_count = 1
    load_balancer_count = 1
    s3_bucket_count = 1
    iam_role_count = 1
    secrets_count = 2
    cloudwatch_log_group_count = 2
  }
}
