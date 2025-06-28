"""
AWS Utilities for Ragamala Painting Generation.

This module provides comprehensive AWS utilities for S3 storage, EC2 management,
and cloud infrastructure operations for the Ragamala painting generation project.
"""

import os
import sys
import json
import time
import logging
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, asdict
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from botocore.config import Config
import uuid
import math
from tqdm import tqdm
import mimetypes
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

@dataclass
class AWSConfig:
    """Configuration for AWS services."""
    region_name: str = "us-east-1"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None
    profile_name: Optional[str] = None
    
    # S3 Configuration
    s3_bucket_name: Optional[str] = None
    s3_prefix: str = "ragamala/"
    
    # EC2 Configuration
    ec2_instance_type: str = "g5.2xlarge"
    ec2_ami_id: str = "ami-0c02fb55956c7d316"  # Ubuntu 20.04 LTS
    ec2_key_pair_name: Optional[str] = None
    ec2_security_group_ids: List[str] = None
    ec2_subnet_id: Optional[str] = None
    
    def __post_init__(self):
        if self.ec2_security_group_ids is None:
            self.ec2_security_group_ids = []

class ProgressCallback:
    """Progress callback for file transfers."""
    
    def __init__(self, filename: str, total_size: int):
        self.filename = filename
        self.total_size = total_size
        self.transferred = 0
        self.lock = threading.Lock()
        self.pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc=filename)
    
    def __call__(self, bytes_transferred: int):
        with self.lock:
            self.transferred += bytes_transferred
            self.pbar.update(bytes_transferred)
    
    def close(self):
        self.pbar.close()

class S3Manager:
    """Comprehensive S3 management utilities."""
    
    def __init__(self, config: Optional[AWSConfig] = None):
        self.config = config or AWSConfig()
        self.s3_client = None
        self.s3_resource = None
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize S3 clients with proper configuration."""
        try:
            # Create session with configuration
            session_kwargs = {
                'region_name': self.config.region_name
            }
            
            if self.config.profile_name:
                session_kwargs['profile_name'] = self.config.profile_name
            
            session = boto3.Session(**session_kwargs)
            
            # Client configuration for better performance
            client_config = Config(
                region_name=self.config.region_name,
                retries={'max_attempts': 3, 'mode': 'adaptive'},
                max_pool_connections=50
            )
            
            # Create clients
            client_kwargs = {'config': client_config}
            
            if self.config.aws_access_key_id and self.config.aws_secret_access_key:
                client_kwargs.update({
                    'aws_access_key_id': self.config.aws_access_key_id,
                    'aws_secret_access_key': self.config.aws_secret_access_key
                })
                
                if self.config.aws_session_token:
                    client_kwargs['aws_session_token'] = self.config.aws_session_token
            
            self.s3_client = session.client('s3', **client_kwargs)
            self.s3_resource = session.resource('s3', **client_kwargs)
            
            logger.info("S3 clients initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize S3 clients: {e}")
            raise
    
    def create_bucket(self, bucket_name: str, region: Optional[str] = None) -> bool:
        """Create S3 bucket with proper configuration."""
        try:
            region = region or self.config.region_name
            
            if region == 'us-east-1':
                # us-east-1 doesn't require LocationConstraint
                self.s3_client.create_bucket(Bucket=bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': region}
                )
            
            # Set bucket versioning
            self.s3_client.put_bucket_versioning(
                Bucket=bucket_name,
                VersioningConfiguration={'Status': 'Enabled'}
            )
            
            # Set bucket lifecycle configuration
            lifecycle_config = {
                'Rules': [
                    {
                        'ID': 'DeleteIncompleteMultipartUploads',
                        'Status': 'Enabled',
                        'Filter': {'Prefix': ''},
                        'AbortIncompleteMultipartUpload': {'DaysAfterInitiation': 7}
                    }
                ]
            }
            
            self.s3_client.put_bucket_lifecycle_configuration(
                Bucket=bucket_name,
                LifecycleConfiguration=lifecycle_config
            )
            
            logger.info(f"Bucket {bucket_name} created successfully")
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
                logger.info(f"Bucket {bucket_name} already exists")
                return True
            else:
                logger.error(f"Failed to create bucket {bucket_name}: {e}")
                return False
    
    def upload_file(self, 
                   local_path: Union[str, Path],
                   s3_key: str,
                   bucket_name: Optional[str] = None,
                   show_progress: bool = True,
                   metadata: Optional[Dict[str, str]] = None) -> bool:
        """Upload file to S3 with progress tracking."""
        try:
            local_path = Path(local_path)
            bucket_name = bucket_name or self.config.s3_bucket_name
            
            if not bucket_name:
                raise ValueError("Bucket name must be provided")
            
            if not local_path.exists():
                raise FileNotFoundError(f"Local file not found: {local_path}")
            
            # Prepare upload arguments
            extra_args = {}
            
            # Set content type
            content_type, _ = mimetypes.guess_type(str(local_path))
            if content_type:
                extra_args['ContentType'] = content_type
            
            # Add metadata
            if metadata:
                extra_args['Metadata'] = metadata
            
            # Setup progress callback
            file_size = local_path.stat().st_size
            progress_callback = None
            
            if show_progress:
                progress_callback = ProgressCallback(local_path.name, file_size)
                extra_args['Callback'] = progress_callback
            
            # Upload file
            self.s3_client.upload_file(
                str(local_path),
                bucket_name,
                s3_key,
                ExtraArgs=extra_args
            )
            
            if progress_callback:
                progress_callback.close()
            
            logger.info(f"File uploaded successfully: {local_path} -> s3://{bucket_name}/{s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload file {local_path}: {e}")
            return False
    
    def download_file(self,
                     s3_key: str,
                     local_path: Union[str, Path],
                     bucket_name: Optional[str] = None,
                     show_progress: bool = True) -> bool:
        """Download file from S3 with progress tracking."""
        try:
            local_path = Path(local_path)
            bucket_name = bucket_name or self.config.s3_bucket_name
            
            if not bucket_name:
                raise ValueError("Bucket name must be provided")
            
            # Create local directory if it doesn't exist
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Get file size for progress tracking
            file_size = 0
            progress_callback = None
            
            if show_progress:
                try:
                    response = self.s3_client.head_object(Bucket=bucket_name, Key=s3_key)
                    file_size = response['ContentLength']
                    progress_callback = ProgressCallback(local_path.name, file_size)
                except ClientError:
                    pass
            
            # Download file
            extra_args = {}
            if progress_callback:
                extra_args['Callback'] = progress_callback
            
            self.s3_client.download_file(
                bucket_name,
                s3_key,
                str(local_path),
                ExtraArgs=extra_args
            )
            
            if progress_callback:
                progress_callback.close()
            
            logger.info(f"File downloaded successfully: s3://{bucket_name}/{s3_key} -> {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download file {s3_key}: {e}")
            return False
    
    def upload_directory(self,
                        local_dir: Union[str, Path],
                        s3_prefix: str,
                        bucket_name: Optional[str] = None,
                        exclude_patterns: Optional[List[str]] = None,
                        max_workers: int = 10) -> Dict[str, bool]:
        """Upload entire directory to S3 with parallel processing."""
        local_dir = Path(local_dir)
        bucket_name = bucket_name or self.config.s3_bucket_name
        exclude_patterns = exclude_patterns or []
        
        if not local_dir.exists():
            raise FileNotFoundError(f"Local directory not found: {local_dir}")
        
        # Find all files to upload
        files_to_upload = []
        for file_path in local_dir.rglob('*'):
            if file_path.is_file():
                # Check exclude patterns
                skip_file = False
                for pattern in exclude_patterns:
                    if pattern in str(file_path):
                        skip_file = True
                        break
                
                if not skip_file:
                    relative_path = file_path.relative_to(local_dir)
                    s3_key = f"{s3_prefix.rstrip('/')}/{relative_path}"
                    files_to_upload.append((file_path, s3_key))
        
        logger.info(f"Uploading {len(files_to_upload)} files from {local_dir}")
        
        # Upload files in parallel
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(
                    self.upload_file,
                    file_path,
                    s3_key,
                    bucket_name,
                    show_progress=False
                ): (file_path, s3_key)
                for file_path, s3_key in files_to_upload
            }
            
            for future in tqdm(as_completed(future_to_file), total=len(files_to_upload)):
                file_path, s3_key = future_to_file[future]
                try:
                    result = future.result()
                    results[str(file_path)] = result
                except Exception as e:
                    logger.error(f"Failed to upload {file_path}: {e}")
                    results[str(file_path)] = False
        
        successful_uploads = sum(1 for success in results.values() if success)
        logger.info(f"Directory upload completed: {successful_uploads}/{len(files_to_upload)} files uploaded")
        
        return results
    
    def download_directory(self,
                          s3_prefix: str,
                          local_dir: Union[str, Path],
                          bucket_name: Optional[str] = None,
                          max_workers: int = 10) -> Dict[str, bool]:
        """Download entire directory from S3 with parallel processing."""
        local_dir = Path(local_dir)
        bucket_name = bucket_name or self.config.s3_bucket_name
        
        # List all objects with the prefix
        objects = self.list_objects(s3_prefix, bucket_name)
        
        if not objects:
            logger.warning(f"No objects found with prefix: {s3_prefix}")
            return {}
        
        logger.info(f"Downloading {len(objects)} files to {local_dir}")
        
        # Download files in parallel
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_object = {
                executor.submit(
                    self.download_file,
                    obj['Key'],
                    local_dir / obj['Key'].replace(s3_prefix.rstrip('/') + '/', ''),
                    bucket_name,
                    show_progress=False
                ): obj['Key']
                for obj in objects
            }
            
            for future in tqdm(as_completed(future_to_object), total=len(objects)):
                s3_key = future_to_object[future]
                try:
                    result = future.result()
                    results[s3_key] = result
                except Exception as e:
                    logger.error(f"Failed to download {s3_key}: {e}")
                    results[s3_key] = False
        
        successful_downloads = sum(1 for success in results.values() if success)
        logger.info(f"Directory download completed: {successful_downloads}/{len(objects)} files downloaded")
        
        return results
    
    def list_objects(self,
                    prefix: str = "",
                    bucket_name: Optional[str] = None,
                    max_keys: Optional[int] = None) -> List[Dict[str, Any]]:
        """List objects in S3 bucket with optional prefix filter."""
        try:
            bucket_name = bucket_name or self.config.s3_bucket_name
            
            if not bucket_name:
                raise ValueError("Bucket name must be provided")
            
            objects = []
            paginator = self.s3_client.get_paginator('list_objects_v2')
            
            page_iterator = paginator.paginate(
                Bucket=bucket_name,
                Prefix=prefix
            )
            
            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        objects.append(obj)
                        
                        if max_keys and len(objects) >= max_keys:
                            return objects[:max_keys]
            
            logger.info(f"Found {len(objects)} objects with prefix '{prefix}'")
            return objects
            
        except Exception as e:
            logger.error(f"Failed to list objects: {e}")
            return []
    
    def delete_object(self, s3_key: str, bucket_name: Optional[str] = None) -> bool:
        """Delete single object from S3."""
        try:
            bucket_name = bucket_name or self.config.s3_bucket_name
            
            if not bucket_name:
                raise ValueError("Bucket name must be provided")
            
            self.s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
            logger.info(f"Object deleted: s3://{bucket_name}/{s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete object {s3_key}: {e}")
            return False
    
    def delete_objects_batch(self,
                           s3_keys: List[str],
                           bucket_name: Optional[str] = None,
                           batch_size: int = 1000) -> Dict[str, bool]:
        """Delete multiple objects from S3 in batches."""
        bucket_name = bucket_name or self.config.s3_bucket_name
        
        if not bucket_name:
            raise ValueError("Bucket name must be provided")
        
        results = {}
        
        # Process in batches
        for i in range(0, len(s3_keys), batch_size):
            batch = s3_keys[i:i + batch_size]
            
            try:
                # Prepare delete request
                delete_request = {
                    'Objects': [{'Key': key} for key in batch]
                }
                
                response = self.s3_client.delete_objects(
                    Bucket=bucket_name,
                    Delete=delete_request
                )
                
                # Process successful deletions
                for deleted in response.get('Deleted', []):
                    results[deleted['Key']] = True
                
                # Process errors
                for error in response.get('Errors', []):
                    results[error['Key']] = False
                    logger.error(f"Failed to delete {error['Key']}: {error['Message']}")
                
            except Exception as e:
                logger.error(f"Batch delete failed: {e}")
                for key in batch:
                    results[key] = False
        
        successful_deletes = sum(1 for success in results.values() if success)
        logger.info(f"Batch delete completed: {successful_deletes}/{len(s3_keys)} objects deleted")
        
        return results
    
    def object_exists(self, s3_key: str, bucket_name: Optional[str] = None) -> bool:
        """Check if object exists in S3."""
        try:
            bucket_name = bucket_name or self.config.s3_bucket_name
            
            if not bucket_name:
                raise ValueError("Bucket name must be provided")
            
            self.s3_client.head_object(Bucket=bucket_name, Key=s3_key)
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                logger.error(f"Error checking object existence: {e}")
                return False
    
    def get_object_size(self, s3_key: str, bucket_name: Optional[str] = None) -> Optional[int]:
        """Get size of object in S3."""
        try:
            bucket_name = bucket_name or self.config.s3_bucket_name
            
            if not bucket_name:
                raise ValueError("Bucket name must be provided")
            
            response = self.s3_client.head_object(Bucket=bucket_name, Key=s3_key)
            return response['ContentLength']
            
        except Exception as e:
            logger.error(f"Failed to get object size for {s3_key}: {e}")
            return None
    
    def generate_presigned_url(self,
                             s3_key: str,
                             bucket_name: Optional[str] = None,
                             expiration: int = 3600,
                             http_method: str = 'GET') -> Optional[str]:
        """Generate presigned URL for S3 object."""
        try:
            bucket_name = bucket_name or self.config.s3_bucket_name
            
            if not bucket_name:
                raise ValueError("Bucket name must be provided")
            
            url = self.s3_client.generate_presigned_url(
                'get_object' if http_method == 'GET' else 'put_object',
                Params={'Bucket': bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            
            return url
            
        except Exception as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            return None

class EC2Manager:
    """Comprehensive EC2 management utilities."""
    
    def __init__(self, config: Optional[AWSConfig] = None):
        self.config = config or AWSConfig()
        self.ec2_client = None
        self.ec2_resource = None
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize EC2 clients."""
        try:
            session_kwargs = {
                'region_name': self.config.region_name
            }
            
            if self.config.profile_name:
                session_kwargs['profile_name'] = self.config.profile_name
            
            session = boto3.Session(**session_kwargs)
            
            client_kwargs = {}
            if self.config.aws_access_key_id and self.config.aws_secret_access_key:
                client_kwargs.update({
                    'aws_access_key_id': self.config.aws_access_key_id,
                    'aws_secret_access_key': self.config.aws_secret_access_key
                })
                
                if self.config.aws_session_token:
                    client_kwargs['aws_session_token'] = self.config.aws_session_token
            
            self.ec2_client = session.client('ec2', **client_kwargs)
            self.ec2_resource = session.resource('ec2', **client_kwargs)
            
            logger.info("EC2 clients initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize EC2 clients: {e}")
            raise
    
    def create_instance(self,
                       instance_name: str,
                       instance_type: Optional[str] = None,
                       ami_id: Optional[str] = None,
                       key_pair_name: Optional[str] = None,
                       security_group_ids: Optional[List[str]] = None,
                       subnet_id: Optional[str] = None,
                       user_data_script: Optional[str] = None,
                       tags: Optional[Dict[str, str]] = None) -> Optional[str]:
        """Create EC2 instance with specified configuration."""
        try:
            # Use config defaults if not provided
            instance_type = instance_type or self.config.ec2_instance_type
            ami_id = ami_id or self.config.ec2_ami_id
            key_pair_name = key_pair_name or self.config.ec2_key_pair_name
            security_group_ids = security_group_ids or self.config.ec2_security_group_ids
            subnet_id = subnet_id or self.config.ec2_subnet_id
            
            # Prepare instance parameters
            instance_params = {
                'ImageId': ami_id,
                'MinCount': 1,
                'MaxCount': 1,
                'InstanceType': instance_type
            }
            
            if key_pair_name:
                instance_params['KeyName'] = key_pair_name
            
            if security_group_ids:
                instance_params['SecurityGroupIds'] = security_group_ids
            
            if subnet_id:
                instance_params['SubnetId'] = subnet_id
            
            if user_data_script:
                instance_params['UserData'] = user_data_script
            
            # Create instance
            response = self.ec2_client.run_instances(**instance_params)
            instance_id = response['Instances'][0]['InstanceId']
            
            # Add tags
            if tags or instance_name:
                tag_list = [{'Key': 'Name', 'Value': instance_name}]
                if tags:
                    tag_list.extend([{'Key': k, 'Value': v} for k, v in tags.items()])
                
                self.ec2_client.create_tags(
                    Resources=[instance_id],
                    Tags=tag_list
                )
            
            logger.info(f"EC2 instance created: {instance_id}")
            return instance_id
            
        except Exception as e:
            logger.error(f"Failed to create EC2 instance: {e}")
            return None
    
    def start_instance(self, instance_id: str) -> bool:
        """Start EC2 instance."""
        try:
            self.ec2_client.start_instances(InstanceIds=[instance_id])
            logger.info(f"EC2 instance started: {instance_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start instance {instance_id}: {e}")
            return False
    
    def stop_instance(self, instance_id: str) -> bool:
        """Stop EC2 instance."""
        try:
            self.ec2_client.stop_instances(InstanceIds=[instance_id])
            logger.info(f"EC2 instance stopped: {instance_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop instance {instance_id}: {e}")
            return False
    
    def terminate_instance(self, instance_id: str) -> bool:
        """Terminate EC2 instance."""
        try:
            self.ec2_client.terminate_instances(InstanceIds=[instance_id])
            logger.info(f"EC2 instance terminated: {instance_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to terminate instance {instance_id}: {e}")
            return False
    
    def get_instance_info(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about EC2 instance."""
        try:
            response = self.ec2_client.describe_instances(InstanceIds=[instance_id])
            
            if response['Reservations']:
                instance = response['Reservations'][0]['Instances'][0]
                return {
                    'InstanceId': instance['InstanceId'],
                    'State': instance['State']['Name'],
                    'InstanceType': instance['InstanceType'],
                    'PublicIpAddress': instance.get('PublicIpAddress'),
                    'PrivateIpAddress': instance.get('PrivateIpAddress'),
                    'LaunchTime': instance['LaunchTime'],
                    'Tags': {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get instance info for {instance_id}: {e}")
            return None
    
    def list_instances(self, filters: Optional[Dict[str, List[str]]] = None) -> List[Dict[str, Any]]:
        """List EC2 instances with optional filters."""
        try:
            params = {}
            if filters:
                params['Filters'] = [
                    {'Name': key, 'Values': values}
                    for key, values in filters.items()
                ]
            
            response = self.ec2_client.describe_instances(**params)
            
            instances = []
            for reservation in response['Reservations']:
                for instance in reservation['Instances']:
                    instances.append({
                        'InstanceId': instance['InstanceId'],
                        'State': instance['State']['Name'],
                        'InstanceType': instance['InstanceType'],
                        'PublicIpAddress': instance.get('PublicIpAddress'),
                        'PrivateIpAddress': instance.get('PrivateIpAddress'),
                        'LaunchTime': instance['LaunchTime'],
                        'Tags': {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}
                    })
            
            return instances
            
        except Exception as e:
            logger.error(f"Failed to list instances: {e}")
            return []
    
    def wait_for_instance_state(self,
                               instance_id: str,
                               desired_state: str,
                               max_wait_time: int = 300) -> bool:
        """Wait for instance to reach desired state."""
        try:
            waiter_name = f"instance_{desired_state}"
            if hasattr(self.ec2_client, 'get_waiter'):
                waiter = self.ec2_client.get_waiter(waiter_name)
                waiter.wait(
                    InstanceIds=[instance_id],
                    WaiterConfig={'MaxAttempts': max_wait_time // 15}
                )
                return True
            else:
                # Manual waiting
                start_time = time.time()
                while time.time() - start_time < max_wait_time:
                    instance_info = self.get_instance_info(instance_id)
                    if instance_info and instance_info['State'] == desired_state:
                        return True
                    time.sleep(15)
                
                return False
                
        except Exception as e:
            logger.error(f"Failed to wait for instance state: {e}")
            return False

class CloudWatchManager:
    """CloudWatch utilities for monitoring and logging."""
    
    def __init__(self, config: Optional[AWSConfig] = None):
        self.config = config or AWSConfig()
        self.cloudwatch_client = None
        self.logs_client = None
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize CloudWatch clients."""
        try:
            session_kwargs = {
                'region_name': self.config.region_name
            }
            
            if self.config.profile_name:
                session_kwargs['profile_name'] = self.config.profile_name
            
            session = boto3.Session(**session_kwargs)
            
            client_kwargs = {}
            if self.config.aws_access_key_id and self.config.aws_secret_access_key:
                client_kwargs.update({
                    'aws_access_key_id': self.config.aws_access_key_id,
                    'aws_secret_access_key': self.config.aws_secret_access_key
                })
                
                if self.config.aws_session_token:
                    client_kwargs['aws_session_token'] = self.config.aws_session_token
            
            self.cloudwatch_client = session.client('cloudwatch', **client_kwargs)
            self.logs_client = session.client('logs', **client_kwargs)
            
            logger.info("CloudWatch clients initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize CloudWatch clients: {e}")
            raise
    
    def put_metric_data(self,
                       namespace: str,
                       metric_name: str,
                       value: float,
                       unit: str = 'Count',
                       dimensions: Optional[Dict[str, str]] = None):
        """Put custom metric data to CloudWatch."""
        try:
            metric_data = {
                'MetricName': metric_name,
                'Value': value,
                'Unit': unit
            }
            
            if dimensions:
                metric_data['Dimensions'] = [
                    {'Name': key, 'Value': value}
                    for key, value in dimensions.items()
                ]
            
            self.cloudwatch_client.put_metric_data(
                Namespace=namespace,
                MetricData=[metric_data]
            )
            
            logger.debug(f"Metric sent: {namespace}/{metric_name} = {value}")
            
        except Exception as e:
            logger.error(f"Failed to put metric data: {e}")

class AWSUtilities:
    """Main AWS utilities class combining all services."""
    
    def __init__(self, config: Optional[AWSConfig] = None):
        self.config = config or AWSConfig()
        self.s3 = S3Manager(self.config)
        self.ec2 = EC2Manager(self.config)
        self.cloudwatch = CloudWatchManager(self.config)
    
    def setup_training_environment(self,
                                 instance_name: str,
                                 setup_script_path: Optional[str] = None) -> Optional[str]:
        """Setup complete training environment on EC2."""
        try:
            # Default setup script for SDXL training
            default_setup_script = """#!/bin/bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install NVIDIA drivers and CUDA
sudo apt install -y nvidia-driver-470 nvidia-cuda-toolkit

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Install Python and dependencies
sudo apt install -y python3.9 python3-pip git
pip3 install --upgrade pip

# Install PyTorch with CUDA support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install diffusers and related packages
pip3 install diffusers transformers accelerate xformers

# Clone project repository (replace with actual repo)
# git clone https://github.com/your-repo/ragamala-painting-generator.git

# Setup project dependencies
# cd ragamala-painting-generator
# pip3 install -r requirements.txt

echo "Setup completed successfully" > /tmp/setup_complete.log
"""
            
            user_data = setup_script_path or default_setup_script
            
            # Create instance
            instance_id = self.ec2.create_instance(
                instance_name=instance_name,
                user_data_script=user_data,
                tags={
                    'Project': 'RagamalaPainting',
                    'Purpose': 'Training',
                    'Environment': 'Development'
                }
            )
            
            if instance_id:
                logger.info(f"Training environment setup initiated: {instance_id}")
                
                # Wait for instance to be running
                if self.ec2.wait_for_instance_state(instance_id, 'running'):
                    logger.info(f"Instance {instance_id} is now running")
                    return instance_id
                else:
                    logger.error("Instance failed to reach running state")
                    return None
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to setup training environment: {e}")
            return None
    
    def sync_training_data(self,
                          local_data_dir: str,
                          s3_data_prefix: str = "ragamala/data/") -> bool:
        """Sync training data between local and S3."""
        try:
            # Upload local data to S3
            logger.info("Syncing training data to S3...")
            upload_results = self.s3.upload_directory(
                local_data_dir,
                s3_data_prefix,
                exclude_patterns=['.git', '__pycache__', '*.pyc', '.DS_Store']
            )
            
            successful_uploads = sum(1 for success in upload_results.values() if success)
            total_files = len(upload_results)
            
            if successful_uploads == total_files:
                logger.info("All training data synced successfully")
                return True
            else:
                logger.warning(f"Partial sync: {successful_uploads}/{total_files} files uploaded")
                return False
                
        except Exception as e:
            logger.error(f"Failed to sync training data: {e}")
            return False
    
    def monitor_training_metrics(self,
                               instance_id: str,
                               metrics: Dict[str, float]):
        """Send training metrics to CloudWatch."""
        try:
            for metric_name, value in metrics.items():
                self.cloudwatch.put_metric_data(
                    namespace='RagamalaTraining',
                    metric_name=metric_name,
                    value=value,
                    dimensions={'InstanceId': instance_id}
                )
            
            logger.debug(f"Training metrics sent for instance {instance_id}")
            
        except Exception as e:
            logger.error(f"Failed to send training metrics: {e}")

def create_aws_config_from_env() -> AWSConfig:
    """Create AWS configuration from environment variables."""
    return AWSConfig(
        region_name=os.getenv('AWS_REGION', 'us-east-1'),
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
        profile_name=os.getenv('AWS_PROFILE'),
        s3_bucket_name=os.getenv('S3_BUCKET_NAME'),
        s3_prefix=os.getenv('S3_PREFIX', 'ragamala/'),
        ec2_instance_type=os.getenv('EC2_INSTANCE_TYPE', 'g5.2xlarge'),
        ec2_key_pair_name=os.getenv('EC2_KEY_PAIR_NAME')
    )

def main():
    """Main function for testing AWS utilities."""
    # Create configuration
    config = create_aws_config_from_env()
    
    # Initialize AWS utilities
    aws_utils = AWSUtilities(config)
    
    # Test S3 operations
    print("Testing S3 operations...")
    
    # List buckets (if accessible)
    try:
        buckets = aws_utils.s3.s3_client.list_buckets()
        print(f"Available buckets: {[b['Name'] for b in buckets['Buckets']]}")
    except Exception as e:
        print(f"Could not list buckets: {e}")
    
    # Test EC2 operations
    print("Testing EC2 operations...")
    
    # List instances
    try:
        instances = aws_utils.ec2.list_instances()
        print(f"Found {len(instances)} EC2 instances")
        for instance in instances[:3]:  # Show first 3
            print(f"  {instance['InstanceId']}: {instance['State']}")
    except Exception as e:
        print(f"Could not list instances: {e}")
    
    print("AWS utilities testing completed!")

if __name__ == "__main__":
    main()
