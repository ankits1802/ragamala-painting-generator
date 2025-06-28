"""
Comprehensive test suite for API components in the Ragamala painting generation project.
Tests FastAPI endpoints, authentication, rate limiting, and API functionality.
"""

import asyncio
import base64
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from httpx import AsyncClient
from PIL import Image

# Import modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent))

from api.app import app
from api.models import (
    GenerationRequest,
    GenerationResponse,
    BatchGenerationRequest,
    HealthResponse,
    User,
    APIKey
)
from api.middleware.auth import verify_api_key, create_access_token
from api.middleware.rate_limiting import RateLimitingMiddleware
from api.routes.generate import generate_single_image
from api.routes.health import health_check


class TestAPIModels(unittest.TestCase):
    """Test suite for API Pydantic models."""
    
    def test_generation_request_validation(self):
        """Test generation request model validation."""
        # Valid request
        valid_request = {
            "raga": "bhairav",
            "style": "rajput",
            "generation_params": {
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
                "width": 1024,
                "height": 1024
            }
        }
        
        request = GenerationRequest(**valid_request)
        self.assertEqual(request.raga, "bhairav")
        self.assertEqual(request.style, "rajput")
        self.assertEqual(request.generation_params.num_inference_steps, 30)
    
    def test_generation_request_invalid_raga(self):
        """Test generation request with invalid raga."""
        invalid_request = {
            "raga": "invalid_raga",
            "style": "rajput"
        }
        
        with self.assertRaises(ValueError):
            GenerationRequest(**invalid_request)
    
    def test_generation_request_invalid_style(self):
        """Test generation request with invalid style."""
        invalid_request = {
            "raga": "bhairav",
            "style": "invalid_style"
        }
        
        with self.assertRaises(ValueError):
            GenerationRequest(**invalid_request)
    
    def test_generation_request_parameter_validation(self):
        """Test generation request parameter validation."""
        # Invalid inference steps
        with self.assertRaises(ValueError):
            GenerationRequest(
                raga="bhairav",
                style="rajput",
                generation_params={
                    "num_inference_steps": 5  # Too low
                }
            )
        
        # Invalid guidance scale
        with self.assertRaises(ValueError):
            GenerationRequest(
                raga="bhairav",
                style="rajput",
                generation_params={
                    "guidance_scale": 25.0  # Too high
                }
            )
    
    def test_batch_generation_request_validation(self):
        """Test batch generation request validation."""
        # Valid batch request
        valid_requests = [
            {
                "raga": "bhairav",
                "style": "rajput"
            },
            {
                "raga": "yaman",
                "style": "pahari"
            }
        ]
        
        batch_request = BatchGenerationRequest(requests=valid_requests)
        self.assertEqual(len(batch_request.requests), 2)
        
        # Too many requests
        too_many_requests = [{"raga": "bhairav", "style": "rajput"}] * 15
        
        with self.assertRaises(ValueError):
            BatchGenerationRequest(requests=too_many_requests)
    
    def test_user_model_validation(self):
        """Test user model validation."""
        # Valid user
        valid_user = {
            "username": "test_user",
            "email": "test@example.com",
            "role": "user"
        }
        
        user = User(**valid_user)
        self.assertEqual(user.username, "test_user")
        self.assertEqual(user.role, "user")
        
        # Invalid email
        with self.assertRaises(ValueError):
            User(
                username="test_user",
                email="invalid_email",
                role="user"
            )


class TestAPIEndpoints(unittest.TestCase):
    """Test suite for API endpoints."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
        self.valid_api_key = "test_api_key_12345"
        self.headers = {"Authorization": f"Bearer {self.valid_api_key}"}
    
    @patch('api.middleware.auth.verify_api_key')
    def test_root_endpoint(self, mock_verify):
        """Test root endpoint."""
        mock_verify.return_value = {"user_id": 1, "role": "user"}
        
        response = self.client.get("/")
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertIn("name", data)
        self.assertIn("version", data)
        self.assertIn("endpoints", data)
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("timestamp", data)
    
    @patch('api.middleware.auth.verify_api_key')
    @patch('api.routes.generate.generate_image_internal')
    def test_generate_endpoint_success(self, mock_generate, mock_verify):
        """Test successful image generation endpoint."""
        mock_verify.return_value = {"user_id": 1, "role": "user"}
        
        # Mock generation result
        mock_image = Mock(spec=Image.Image)
        mock_image.width = 1024
        mock_image.height = 1024
        
        mock_generate.return_value = {
            'status': 'success',
            'images': [{
                'image_id': 'test_123',
                'image_data': 'base64_encoded_data',
                'metadata': {
                    'raga': 'bhairav',
                    'style': 'rajput'
                }
            }],
            'generation_time': 15.5,
            'total_images': 1
        }
        
        request_data = {
            "raga": "bhairav",
            "style": "rajput",
            "generation_params": {
                "num_inference_steps": 30,
                "guidance_scale": 7.5
            },
            "output_config": {
                "return_base64": True
            }
        }
        
        response = self.client.post(
            "/generate",
            json=request_data,
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(data["status"], "completed")
        self.assertIn("images", data)
        self.assertEqual(len(data["images"]), 1)
    
    @patch('api.middleware.auth.verify_api_key')
    def test_generate_endpoint_invalid_request(self, mock_verify):
        """Test generation endpoint with invalid request."""
        mock_verify.return_value = {"user_id": 1, "role": "user"}
        
        invalid_request = {
            "raga": "invalid_raga",
            "style": "rajput"
        }
        
        response = self.client.post(
            "/generate",
            json=invalid_request,
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, status.HTTP_422_UNPROCESSABLE_ENTITY)
    
    def test_generate_endpoint_no_auth(self):
        """Test generation endpoint without authentication."""
        request_data = {
            "raga": "bhairav",
            "style": "rajput"
        }
        
        response = self.client.post("/generate", json=request_data)
        
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
    
    @patch('api.middleware.auth.verify_api_key')
    @patch('api.routes.generate.process_single_generation')
    def test_batch_generate_endpoint(self, mock_process, mock_verify):
        """Test batch generation endpoint."""
        mock_verify.return_value = {"user_id": 1, "role": "user"}
        
        # Mock batch processing
        mock_process.side_effect = [
            {
                'request_id': 'batch_1_0',
                'status': 'completed',
                'images': [{'image_id': 'img_1'}],
                'generation_time': 12.0
            },
            {
                'request_id': 'batch_1_1',
                'status': 'completed',
                'images': [{'image_id': 'img_2'}],
                'generation_time': 13.5
            }
        ]
        
        batch_request = {
            "requests": [
                {"raga": "bhairav", "style": "rajput"},
                {"raga": "yaman", "style": "pahari"}
            ]
        }
        
        response = self.client.post(
            "/batch-generate",
            json=batch_request,
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(data["total_requests"], 2)
        self.assertEqual(data["completed"], 2)
        self.assertEqual(data["failed"], 0)
    
    @patch('api.middleware.auth.verify_api_key')
    def test_get_generation_status(self, mock_verify):
        """Test generation status endpoint."""
        mock_verify.return_value = {"user_id": 1, "role": "user"}
        
        # Mock status store
        with patch('api.routes.generate.generation_status_store') as mock_store:
            mock_store.__contains__ = Mock(return_value=True)
            mock_store.__getitem__ = Mock(return_value={
                'request_id': 'test_123',
                'status': 'completed',
                'progress': 1.0
            })
            
            response = self.client.get(
                "/status/test_123",
                headers=self.headers
            )
            
            self.assertEqual(response.status_code, status.HTTP_200_OK)
            data = response.json()
            self.assertEqual(data["request_id"], "test_123")
            self.assertEqual(data["status"], "completed")
    
    def test_get_ragas_endpoint(self):
        """Test ragas information endpoint."""
        response = self.client.get("/ragas")
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        
        # Check raga structure
        raga = data[0]
        self.assertIn("name", raga)
        self.assertIn("description", raga)
        self.assertIn("time_of_day", raga)
        self.assertIn("emotions", raga)
    
    def test_get_styles_endpoint(self):
        """Test styles information endpoint."""
        response = self.client.get("/styles")
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        
        # Check style structure
        style = data[0]
        self.assertIn("name", style)
        self.assertIn("description", style)
        self.assertIn("period", style)
        self.assertIn("region", style)
        self.assertIn("characteristics", style)


class TestAuthentication(unittest.TestCase):
    """Test suite for authentication middleware."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
    
    def test_valid_api_key(self):
        """Test valid API key authentication."""
        with patch('api.middleware.auth.verify_api_key') as mock_verify:
            mock_verify.return_value = {
                "user_id": 1,
                "username": "test_user",
                "role": "user",
                "permissions": ["api_access"]
            }
            
            headers = {"Authorization": "Bearer valid_api_key"}
            response = self.client.get("/health", headers=headers)
            
            self.assertEqual(response.status_code, status.HTTP_200_OK)
    
    def test_invalid_api_key(self):
        """Test invalid API key authentication."""
        with patch('api.middleware.auth.verify_api_key') as mock_verify:
            mock_verify.side_effect = Exception("Invalid API key")
            
            headers = {"Authorization": "Bearer invalid_api_key"}
            response = self.client.post(
                "/generate",
                json={"raga": "bhairav", "style": "rajput"},
                headers=headers
            )
            
            self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
    
    def test_missing_api_key(self):
        """Test missing API key."""
        response = self.client.post(
            "/generate",
            json={"raga": "bhairav", "style": "rajput"}
        )
        
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
    
    def test_malformed_authorization_header(self):
        """Test malformed authorization header."""
        headers = {"Authorization": "InvalidFormat api_key"}
        response = self.client.post(
            "/generate",
            json={"raga": "bhairav", "style": "rajput"},
            headers=headers
        )
        
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
    
    def test_jwt_token_creation(self):
        """Test JWT token creation."""
        user_data = {
            "sub": "test_user",
            "user_id": 1,
            "role": "user"
        }
        
        token = create_access_token(user_data)
        
        self.assertIsInstance(token, str)
        self.assertGreater(len(token), 0)
    
    def test_api_key_verification(self):
        """Test API key verification function."""
        # Mock API key data
        with patch('api.middleware.auth.api_keys_db') as mock_db:
            mock_db.get.return_value = {
                "user_id": 1,
                "username": "test_user",
                "role": "user",
                "permissions": ["api_access"],
                "is_active": True,
                "expires_at": None
            }
            
            from api.middleware.auth import verify_api_key as verify_func
            result = verify_func("valid_api_key")
            
            self.assertIsNotNone(result)
            self.assertEqual(result["user_id"], 1)
            self.assertEqual(result["username"], "test_user")


class TestRateLimiting(unittest.TestCase):
    """Test suite for rate limiting middleware."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
    
    @patch('api.middleware.auth.verify_api_key')
    def test_rate_limiting_within_limits(self, mock_verify):
        """Test requests within rate limits."""
        mock_verify.return_value = {"user_id": 1, "role": "user"}
        
        headers = {"Authorization": "Bearer test_api_key"}
        
        # Make requests within limit
        for _ in range(3):
            response = self.client.get("/health", headers=headers)
            self.assertEqual(response.status_code, status.HTTP_200_OK)
    
    @patch('api.middleware.auth.verify_api_key')
    def test_rate_limiting_exceeded(self, mock_verify):
        """Test rate limiting when limits are exceeded."""
        mock_verify.return_value = {"user_id": 1, "role": "user"}
        
        headers = {"Authorization": "Bearer test_api_key"}
        
        # Mock rate limiter to simulate limit exceeded
        with patch('api.middleware.rate_limiting.RateLimitingMiddleware.dispatch') as mock_dispatch:
            from fastapi.responses import JSONResponse
            
            mock_dispatch.return_value = JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"error": "Rate limit exceeded"}
            )
            
            response = self.client.get("/health", headers=headers)
            self.assertEqual(response.status_code, status.HTTP_429_TOO_MANY_REQUESTS)
    
    def test_rate_limit_headers(self):
        """Test rate limit headers in response."""
        with patch('api.middleware.auth.verify_api_key') as mock_verify:
            mock_verify.return_value = {"user_id": 1, "role": "user"}
            
            headers = {"Authorization": "Bearer test_api_key"}
            response = self.client.get("/health", headers=headers)
            
            # Check for rate limit headers (if implemented)
            # self.assertIn("X-RateLimit-Limit", response.headers)
            # self.assertIn("X-RateLimit-Remaining", response.headers)


class TestErrorHandling(unittest.TestCase):
    """Test suite for API error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
    
    def test_404_error(self):
        """Test 404 error handling."""
        response = self.client.get("/nonexistent-endpoint")
        
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
    
    def test_422_validation_error(self):
        """Test 422 validation error handling."""
        with patch('api.middleware.auth.verify_api_key') as mock_verify:
            mock_verify.return_value = {"user_id": 1, "role": "user"}
            
            headers = {"Authorization": "Bearer test_api_key"}
            
            # Send invalid data
            invalid_data = {
                "raga": "invalid_raga",
                "style": "invalid_style",
                "generation_params": {
                    "num_inference_steps": -1  # Invalid
                }
            }
            
            response = self.client.post(
                "/generate",
                json=invalid_data,
                headers=headers
            )
            
            self.assertEqual(response.status_code, status.HTTP_422_UNPROCESSABLE_ENTITY)
            data = response.json()
            self.assertIn("detail", data)
    
    @patch('api.middleware.auth.verify_api_key')
    @patch('api.routes.generate.generate_image_internal')
    def test_500_internal_error(self, mock_generate, mock_verify):
        """Test 500 internal server error handling."""
        mock_verify.return_value = {"user_id": 1, "role": "user"}
        mock_generate.side_effect = Exception("Internal error")
        
        headers = {"Authorization": "Bearer test_api_key"}
        request_data = {
            "raga": "bhairav",
            "style": "rajput"
        }
        
        response = self.client.post(
            "/generate",
            json=request_data,
            headers=headers
        )
        
        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def test_custom_exception_handler(self):
        """Test custom exception handling."""
        # This would test custom exception handlers if implemented
        pass


class TestAsyncEndpoints(unittest.TestCase):
    """Test suite for async API endpoints."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = app
    
    @pytest.mark.asyncio
    async def test_async_generation_endpoint(self):
        """Test async generation endpoint."""
        async with AsyncClient(app=self.app, base_url="http://test") as client:
            with patch('api.middleware.auth.verify_api_key') as mock_verify:
                mock_verify.return_value = {"user_id": 1, "role": "user"}
                
                with patch('api.routes.generate.generate_image_internal') as mock_generate:
                    mock_generate.return_value = {
                        'status': 'success',
                        'images': [{'image_id': 'test_123'}],
                        'generation_time': 10.0
                    }
                    
                    headers = {"Authorization": "Bearer test_api_key"}
                    request_data = {
                        "raga": "bhairav",
                        "style": "rajput"
                    }
                    
                    response = await client.post(
                        "/generate",
                        json=request_data,
                        headers=headers
                    )
                    
                    self.assertEqual(response.status_code, status.HTTP_200_OK)
    
    @pytest.mark.asyncio
    async def test_async_batch_generation(self):
        """Test async batch generation endpoint."""
        async with AsyncClient(app=self.app, base_url="http://test") as client:
            with patch('api.middleware.auth.verify_api_key') as mock_verify:
                mock_verify.return_value = {"user_id": 1, "role": "user"}
                
                with patch('api.routes.generate.process_single_generation') as mock_process:
                    mock_process.return_value = {
                        'request_id': 'test_123',
                        'status': 'completed',
                        'images': [{'image_id': 'img_1'}]
                    }
                    
                    headers = {"Authorization": "Bearer test_api_key"}
                    batch_data = {
                        "requests": [
                            {"raga": "bhairav", "style": "rajput"},
                            {"raga": "yaman", "style": "pahari"}
                        ]
                    }
                    
                    response = await client.post(
                        "/batch-generate",
                        json=batch_data,
                        headers=headers
                    )
                    
                    self.assertEqual(response.status_code, status.HTTP_200_OK)


class TestImageHandling(unittest.TestCase):
    """Test suite for image handling in API."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_base64_image_encoding(self):
        """Test base64 image encoding in responses."""
        # Create test image
        test_image = Image.new('RGB', (512, 512), color='red')
        
        # Convert to base64
        buffered = io.BytesIO()
        test_image.save(buffered, format="PNG")
        base64_data = base64.b64encode(buffered.getvalue()).decode()
        
        # Verify encoding
        self.assertIsInstance(base64_data, str)
        self.assertGreater(len(base64_data), 0)
        
        # Verify decoding
        decoded_data = base64.b64decode(base64_data)
        decoded_image = Image.open(io.BytesIO(decoded_data))
        self.assertEqual(decoded_image.size, (512, 512))
    
    @patch('api.middleware.auth.verify_api_key')
    def test_image_download_endpoint(self, mock_verify):
        """Test image download endpoint."""
        mock_verify.return_value = {"user_id": 1, "role": "user"}
        
        headers = {"Authorization": "Bearer test_api_key"}
        
        with patch('api.routes.generate.s3_manager') as mock_s3:
            mock_s3.generate_presigned_url.return_value = "https://example.com/download/image.png"
            
            response = self.client.get(
                "/download/test_image_123",
                headers=headers
            )
            
            self.assertEqual(response.status_code, status.HTTP_200_OK)
            data = response.json()
            self.assertIn("download_url", data)
    
    @patch('api.middleware.auth.verify_api_key')
    def test_image_upload_endpoint(self, mock_verify):
        """Test image upload for reference generation."""
        mock_verify.return_value = {"user_id": 1, "role": "user"}
        
        # Create test image file
        test_image = Image.new('RGB', (512, 512), color='blue')
        image_bytes = io.BytesIO()
        test_image.save(image_bytes, format='PNG')
        image_bytes.seek(0)
        
        headers = {"Authorization": "Bearer test_api_key"}
        
        with patch('api.routes.generate.generator') as mock_generator:
            mock_generator.generate_from_image.return_value = [test_image]
            
            files = {"reference_image": ("test.png", image_bytes, "image/png")}
            data = {
                "raga": "bhairav",
                "style": "rajput",
                "prompt": "A test painting"
            }
            
            response = self.client.post(
                "/generate-from-image",
                files=files,
                data=data,
                headers=headers
            )
            
            self.assertEqual(response.status_code, status.HTTP_200_OK)


class TestAPIIntegration(unittest.TestCase):
    """Integration tests for API components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
    
    def test_full_generation_workflow(self):
        """Test complete generation workflow."""
        with patch('api.middleware.auth.verify_api_key') as mock_verify:
            mock_verify.return_value = {"user_id": 1, "role": "user"}
            
            with patch('api.routes.generate.generator') as mock_generator:
                # Mock image generation
                mock_image = Mock(spec=Image.Image)
                mock_image.width = 1024
                mock_image.height = 1024
                mock_generator.generate.return_value = [mock_image]
                mock_generator.create_culturally_aware_prompt.return_value = "A beautiful painting"
                
                headers = {"Authorization": "Bearer test_api_key"}
                
                # Step 1: Check API health
                health_response = self.client.get("/health")
                self.assertEqual(health_response.status_code, status.HTTP_200_OK)
                
                # Step 2: Get available ragas
                ragas_response = self.client.get("/ragas")
                self.assertEqual(ragas_response.status_code, status.HTTP_200_OK)
                
                # Step 3: Generate image
                generation_request = {
                    "raga": "bhairav",
                    "style": "rajput",
                    "output_config": {
                        "return_base64": True,
                        "calculate_quality_metrics": True
                    }
                }
                
                with patch('api.routes.generate.calculate_quality_metrics') as mock_metrics:
                    mock_metrics.return_value = {
                        'overall_score': 0.85,
                        'sharpness': 0.9,
                        'color_harmony': 0.8
                    }
                    
                    generation_response = self.client.post(
                        "/generate",
                        json=generation_request,
                        headers=headers
                    )
                    
                    self.assertEqual(generation_response.status_code, status.HTTP_200_OK)
                    data = generation_response.json()
                    self.assertEqual(data["status"], "completed")
                    self.assertIn("images", data)
    
    def test_error_propagation(self):
        """Test error propagation through API layers."""
        with patch('api.middleware.auth.verify_api_key') as mock_verify:
            mock_verify.return_value = {"user_id": 1, "role": "user"}
            
            with patch('api.routes.generate.generate_image_internal') as mock_generate:
                # Simulate generation failure
                mock_generate.side_effect = Exception("Model loading failed")
                
                headers = {"Authorization": "Bearer test_api_key"}
                request_data = {
                    "raga": "bhairav",
                    "style": "rajput"
                }
                
                response = self.client.post(
                    "/generate",
                    json=request_data,
                    headers=headers
                )
                
                self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
                data = response.json()
                self.assertIn("detail", data)
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        import threading
        import time
        
        results = []
        
        def make_request():
            with patch('api.middleware.auth.verify_api_key') as mock_verify:
                mock_verify.return_value = {"user_id": 1, "role": "user"}
                
                headers = {"Authorization": "Bearer test_api_key"}
                response = self.client.get("/health", headers=headers)
                results.append(response.status_code)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        self.assertEqual(len(results), 5)
        self.assertTrue(all(status == 200 for status in results))


if __name__ == '__main__':
    # Configure test environment
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests
    
    # Run tests
    unittest.main(verbosity=2)
