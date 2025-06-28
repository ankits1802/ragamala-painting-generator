"""
Comprehensive test suite for model components in the Ragamala painting generation project.
Tests SDXL model implementation, LoRA fine-tuning, prompt encoding, and inference functionality.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

# Import modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.sdxl_lora import SDXLLoRAModel, LoRAConfig
from src.models.prompt_encoder import PromptEncoder, CulturalPromptEncoder
from src.models.scheduler import CustomDiffusionScheduler
from src.inference.generator import RagamalaGenerator, GenerationConfig
from src.training.trainer import RagamalaTrainer


class TestLoRAConfig(unittest.TestCase):
    """Test suite for LoRA configuration."""
    
    def test_lora_config_initialization(self):
        """Test LoRA configuration initialization."""
        config = LoRAConfig(
            rank=64,
            alpha=32,
            target_modules=["to_k", "to_q", "to_v"],
            dropout=0.1
        )
        
        self.assertEqual(config.rank, 64)
        self.assertEqual(config.alpha, 32)
        self.assertEqual(config.dropout, 0.1)
        self.assertIn("to_k", config.target_modules)
    
    def test_lora_config_validation(self):
        """Test LoRA configuration validation."""
        # Valid configuration
        valid_config = LoRAConfig(rank=32, alpha=16)
        self.assertTrue(valid_config.is_valid())
        
        # Invalid rank
        with self.assertRaises(ValueError):
            LoRAConfig(rank=0, alpha=16)
        
        # Invalid alpha
        with self.assertRaises(ValueError):
            LoRAConfig(rank=32, alpha=-1)
    
    def test_lora_config_serialization(self):
        """Test LoRA configuration serialization."""
        config = LoRAConfig(rank=64, alpha=32)
        
        # Test to_dict
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['rank'], 64)
        
        # Test from_dict
        restored_config = LoRAConfig.from_dict(config_dict)
        self.assertEqual(restored_config.rank, config.rank)
        self.assertEqual(restored_config.alpha, config.alpha)


class TestSDXLLoRAModel(unittest.TestCase):
    """Test suite for SDXL LoRA model implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")  # Use CPU for tests
        self.lora_config = LoRAConfig(
            rank=16,  # Small rank for testing
            alpha=8,
            target_modules=["to_k", "to_q", "to_v"],
            dropout=0.1
        )
        
        # Mock SDXL pipeline components
        self.mock_unet = Mock(spec=UNet2DConditionModel)
        self.mock_unet.config = Mock()
        self.mock_unet.config.in_channels = 4
        self.mock_unet.config.sample_size = 128
        
    def test_lora_layer_creation(self):
        """Test LoRA layer creation."""
        from src.models.sdxl_lora import LoRALayer
        
        # Create a simple linear layer for testing
        original_layer = nn.Linear(512, 512)
        lora_layer = LoRALayer(original_layer, rank=16, alpha=8)
        
        self.assertEqual(lora_layer.rank, 16)
        self.assertEqual(lora_layer.alpha, 8)
        self.assertEqual(lora_layer.lora_A.weight.shape, (16, 512))
        self.assertEqual(lora_layer.lora_B.weight.shape, (512, 16))
    
    def test_lora_layer_forward(self):
        """Test LoRA layer forward pass."""
        from src.models.sdxl_lora import LoRALayer
        
        original_layer = nn.Linear(512, 512)
        lora_layer = LoRALayer(original_layer, rank=16, alpha=8)
        
        # Test forward pass
        input_tensor = torch.randn(2, 512)
        output = lora_layer(input_tensor)
        
        self.assertEqual(output.shape, (2, 512))
        self.assertFalse(torch.allclose(output, original_layer(input_tensor)))
    
    @patch('diffusers.StableDiffusionXLPipeline.from_pretrained')
    def test_sdxl_lora_model_initialization(self, mock_from_pretrained):
        """Test SDXL LoRA model initialization."""
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.unet = self.mock_unet
        mock_pipeline.text_encoder = Mock()
        mock_pipeline.text_encoder_2 = Mock()
        mock_from_pretrained.return_value = mock_pipeline
        
        model = SDXLLoRAModel(
            model_name="stabilityai/stable-diffusion-xl-base-1.0",
            lora_config=self.lora_config,
            device=self.device
        )
        
        self.assertIsNotNone(model.pipeline)
        self.assertEqual(model.device, self.device)
        self.assertEqual(model.lora_config, self.lora_config)
    
    def test_lora_weight_initialization(self):
        """Test LoRA weight initialization."""
        from src.models.sdxl_lora import LoRALayer
        
        original_layer = nn.Linear(512, 512)
        lora_layer = LoRALayer(original_layer, rank=16, alpha=8)
        
        # Check that LoRA A weights are initialized with normal distribution
        self.assertFalse(torch.allclose(lora_layer.lora_A.weight, torch.zeros_like(lora_layer.lora_A.weight)))
        
        # Check that LoRA B weights are initialized to zero
        self.assertTrue(torch.allclose(lora_layer.lora_B.weight, torch.zeros_like(lora_layer.lora_B.weight)))
    
    def test_lora_scaling(self):
        """Test LoRA scaling factor."""
        from src.models.sdxl_lora import LoRALayer
        
        original_layer = nn.Linear(512, 512)
        lora_layer = LoRALayer(original_layer, rank=16, alpha=8)
        
        expected_scaling = 8 / 16  # alpha / rank
        self.assertEqual(lora_layer.scaling, expected_scaling)
    
    def test_lora_state_dict(self):
        """Test LoRA state dict operations."""
        from src.models.sdxl_lora import LoRALayer
        
        original_layer = nn.Linear(512, 512)
        lora_layer = LoRALayer(original_layer, rank=16, alpha=8)
        
        # Get state dict
        state_dict = lora_layer.state_dict()
        
        self.assertIn('lora_A.weight', state_dict)
        self.assertIn('lora_B.weight', state_dict)
        
        # Test loading state dict
        new_lora_layer = LoRALayer(nn.Linear(512, 512), rank=16, alpha=8)
        new_lora_layer.load_state_dict(state_dict)
        
        self.assertTrue(torch.allclose(
            lora_layer.lora_A.weight, 
            new_lora_layer.lora_A.weight
        ))


class TestPromptEncoder(unittest.TestCase):
    """Test suite for prompt encoder."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        
        # Mock tokenizer and text encoder
        self.mock_tokenizer = Mock(spec=CLIPTokenizer)
        self.mock_tokenizer.return_value = {
            'input_ids': torch.randint(0, 1000, (1, 77)),
            'attention_mask': torch.ones(1, 77)
        }
        
        self.mock_text_encoder = Mock(spec=CLIPTextModel)
        self.mock_text_encoder.return_value = Mock()
        self.mock_text_encoder.return_value.last_hidden_state = torch.randn(1, 77, 768)
        
        self.prompt_encoder = PromptEncoder(
            tokenizer=self.mock_tokenizer,
            text_encoder=self.mock_text_encoder,
            device=self.device
        )
    
    def test_prompt_encoding(self):
        """Test basic prompt encoding."""
        prompt = "A beautiful Ragamala painting"
        
        encoded = self.prompt_encoder.encode(prompt)
        
        self.assertIsInstance(encoded, torch.Tensor)
        self.assertEqual(encoded.shape[0], 1)  # Batch size
        self.assertEqual(encoded.shape[2], 768)  # Hidden size
    
    def test_batch_prompt_encoding(self):
        """Test batch prompt encoding."""
        prompts = [
            "A Rajput style Ragamala painting",
            "A Pahari style Ragamala painting",
            "A Deccan style Ragamala painting"
        ]
        
        # Mock batch tokenization
        self.mock_tokenizer.return_value = {
            'input_ids': torch.randint(0, 1000, (3, 77)),
            'attention_mask': torch.ones(3, 77)
        }
        self.mock_text_encoder.return_value.last_hidden_state = torch.randn(3, 77, 768)
        
        encoded = self.prompt_encoder.encode_batch(prompts)
        
        self.assertEqual(encoded.shape[0], 3)  # Batch size
        self.assertEqual(encoded.shape[2], 768)  # Hidden size
    
    def test_prompt_truncation(self):
        """Test prompt truncation for long prompts."""
        long_prompt = "A very " * 100 + "long Ragamala painting description"
        
        encoded = self.prompt_encoder.encode(long_prompt, max_length=77)
        
        # Should still work with truncation
        self.assertIsInstance(encoded, torch.Tensor)
        self.mock_tokenizer.assert_called()
    
    def test_empty_prompt_handling(self):
        """Test handling of empty prompts."""
        empty_prompt = ""
        
        encoded = self.prompt_encoder.encode(empty_prompt)
        
        self.assertIsInstance(encoded, torch.Tensor)
        self.assertEqual(encoded.shape[0], 1)


class TestCulturalPromptEncoder(unittest.TestCase):
    """Test suite for cultural prompt encoder."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        
        # Mock base prompt encoder
        self.mock_base_encoder = Mock()
        self.mock_base_encoder.encode.return_value = torch.randn(1, 77, 768)
        
        self.cultural_encoder = CulturalPromptEncoder(
            base_encoder=self.mock_base_encoder,
            device=self.device
        )
    
    def test_cultural_prompt_enhancement(self):
        """Test cultural prompt enhancement."""
        base_prompt = "A painting"
        raga = "bhairav"
        style = "rajput"
        
        enhanced_prompt = self.cultural_encoder.enhance_prompt(
            base_prompt, raga=raga, style=style
        )
        
        self.assertIn(raga, enhanced_prompt.lower())
        self.assertIn(style, enhanced_prompt.lower())
        self.assertIn("painting", enhanced_prompt.lower())
    
    def test_raga_specific_enhancement(self):
        """Test raga-specific prompt enhancement."""
        test_cases = [
            ("bhairav", ["dawn", "devotional", "shiva"]),
            ("yaman", ["evening", "romantic", "krishna"]),
            ("malkauns", ["midnight", "meditative", "mysterious"])
        ]
        
        for raga, expected_elements in test_cases:
            with self.subTest(raga=raga):
                enhanced = self.cultural_encoder.enhance_prompt(
                    "A painting", raga=raga, style="rajput"
                )
                
                # Check if at least one expected element is present
                has_element = any(elem in enhanced.lower() for elem in expected_elements)
                self.assertTrue(has_element, f"No expected elements found for {raga}")
    
    def test_style_specific_enhancement(self):
        """Test style-specific prompt enhancement."""
        test_cases = [
            ("rajput", ["bold", "geometric", "royal"]),
            ("pahari", ["soft", "naturalistic", "lyrical"]),
            ("deccan", ["persian", "architectural", "formal"]),
            ("mughal", ["elaborate", "naturalistic", "imperial"])
        ]
        
        for style, expected_elements in test_cases:
            with self.subTest(style=style):
                enhanced = self.cultural_encoder.enhance_prompt(
                    "A painting", raga="bhairav", style=style
                )
                
                # Check if at least one expected element is present
                has_element = any(elem in enhanced.lower() for elem in expected_elements)
                self.assertTrue(has_element, f"No expected elements found for {style}")
    
    def test_cultural_conditioning_encoding(self):
        """Test cultural conditioning in encoding."""
        prompt = "A Ragamala painting"
        
        encoded = self.cultural_encoder.encode_with_conditioning(
            prompt, raga="bhairav", style="rajput"
        )
        
        self.assertIsInstance(encoded, torch.Tensor)
        self.mock_base_encoder.encode.assert_called()


class TestCustomDiffusionScheduler(unittest.TestCase):
    """Test suite for custom diffusion scheduler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scheduler = CustomDiffusionScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear"
        )
    
    def test_scheduler_initialization(self):
        """Test scheduler initialization."""
        self.assertEqual(self.scheduler.num_train_timesteps, 1000)
        self.assertEqual(self.scheduler.beta_start, 0.00085)
        self.assertEqual(self.scheduler.beta_end, 0.012)
    
    def test_noise_schedule_generation(self):
        """Test noise schedule generation."""
        betas = self.scheduler.betas
        
        self.assertEqual(len(betas), 1000)
        self.assertGreaterEqual(betas[0], self.scheduler.beta_start)
        self.assertLessEqual(betas[-1], self.scheduler.beta_end)
        
        # Check monotonic increase for scaled_linear
        self.assertTrue(torch.all(betas[1:] >= betas[:-1]))
    
    def test_add_noise(self):
        """Test noise addition."""
        original_samples = torch.randn(2, 4, 64, 64)
        noise = torch.randn_like(original_samples)
        timesteps = torch.randint(0, 1000, (2,))
        
        noisy_samples = self.scheduler.add_noise(original_samples, noise, timesteps)
        
        self.assertEqual(noisy_samples.shape, original_samples.shape)
        self.assertFalse(torch.allclose(noisy_samples, original_samples))
    
    def test_step_prediction(self):
        """Test scheduler step prediction."""
        model_output = torch.randn(2, 4, 64, 64)
        timestep = torch.tensor([500])
        sample = torch.randn(2, 4, 64, 64)
        
        result = self.scheduler.step(model_output, timestep, sample)
        
        self.assertIn('prev_sample', result)
        self.assertEqual(result['prev_sample'].shape, sample.shape)
    
    def test_set_timesteps(self):
        """Test timestep setting for inference."""
        num_inference_steps = 50
        
        self.scheduler.set_timesteps(num_inference_steps)
        
        self.assertEqual(len(self.scheduler.timesteps), num_inference_steps)
        self.assertTrue(torch.all(self.scheduler.timesteps >= 0))
        self.assertTrue(torch.all(self.scheduler.timesteps < 1000))


class TestRagamalaGenerator(unittest.TestCase):
    """Test suite for Ragamala generator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        self.generation_config = GenerationConfig(
            model_path="stabilityai/stable-diffusion-xl-base-1.0",
            device="cpu",
            torch_dtype=torch.float32,
            enable_cpu_offload=True
        )
        
        # Mock the pipeline loading
        with patch('src.inference.generator.StableDiffusionXLPipeline.from_pretrained'):
            self.generator = RagamalaGenerator(self.generation_config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('src.inference.generator.StableDiffusionXLPipeline')
    def test_generator_initialization(self, mock_pipeline_class):
        """Test generator initialization."""
        mock_pipeline = Mock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        
        generator = RagamalaGenerator(self.generation_config)
        
        self.assertIsNotNone(generator.pipeline)
        self.assertEqual(generator.config, self.generation_config)
    
    def test_prompt_creation(self):
        """Test prompt creation for different ragas and styles."""
        test_cases = [
            ("bhairav", "rajput"),
            ("yaman", "pahari"),
            ("malkauns", "deccan")
        ]
        
        for raga, style in test_cases:
            with self.subTest(raga=raga, style=style):
                prompt = self.generator.create_prompt(raga, style)
                
                self.assertIsInstance(prompt, str)
                self.assertGreater(len(prompt), 0)
                self.assertIn(raga, prompt.lower())
                self.assertIn(style, prompt.lower())
    
    def test_culturally_aware_prompt_creation(self):
        """Test culturally aware prompt creation."""
        raga = "bhairav"
        style = "rajput"
        
        prompt = self.generator.create_culturally_aware_prompt(raga, style)
        
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 50)  # Should be detailed
        
        # Should contain cultural elements
        cultural_elements = ["devotional", "dawn", "temple", "traditional"]
        has_cultural_element = any(elem in prompt.lower() for elem in cultural_elements)
        self.assertTrue(has_cultural_element)
    
    @patch('PIL.Image.fromarray')
    def test_image_generation(self, mock_from_array):
        """Test image generation."""
        # Mock the pipeline call
        mock_image = Mock(spec=Image.Image)
        mock_image.size = (1024, 1024)
        mock_from_array.return_value = mock_image
        
        self.generator.pipeline = Mock()
        self.generator.pipeline.return_value = Mock()
        self.generator.pipeline.return_value.images = [mock_image]
        
        prompt = "A beautiful Ragamala painting"
        images = self.generator.generate(
            prompt=prompt,
            num_inference_steps=20,
            guidance_scale=7.5,
            width=1024,
            height=1024
        )
        
        self.assertEqual(len(images), 1)
        self.generator.pipeline.assert_called_once()
    
    def test_batch_generation(self):
        """Test batch image generation."""
        # Mock the pipeline
        mock_images = [Mock(spec=Image.Image) for _ in range(3)]
        for img in mock_images:
            img.size = (1024, 1024)
        
        self.generator.pipeline = Mock()
        self.generator.pipeline.return_value = Mock()
        self.generator.pipeline.return_value.images = mock_images
        
        prompts = [
            "A Rajput Ragamala painting",
            "A Pahari Ragamala painting",
            "A Deccan Ragamala painting"
        ]
        
        results = self.generator.generate_batch(prompts)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(self.generator.pipeline.call_count, 3)
    
    def test_seed_reproducibility(self):
        """Test seed-based reproducibility."""
        # Mock the pipeline to return deterministic results
        self.generator.pipeline = Mock()
        
        def mock_generate(*args, **kwargs):
            generator = kwargs.get('generator')
            if generator:
                # Simulate deterministic generation based on seed
                torch.manual_seed(generator.initial_seed())
                mock_result = Mock()
                mock_result.images = [Mock(spec=Image.Image)]
                return mock_result
            return Mock()
        
        self.generator.pipeline.side_effect = mock_generate
        
        prompt = "A test painting"
        seed = 42
        
        # Generate twice with same seed
        gen1 = torch.Generator().manual_seed(seed)
        gen2 = torch.Generator().manual_seed(seed)
        
        result1 = self.generator.generate(prompt=prompt, generator=gen1)
        result2 = self.generator.generate(prompt=prompt, generator=gen2)
        
        # Should have called pipeline twice
        self.assertEqual(self.generator.pipeline.call_count, 2)


class TestRagamalaTrainer(unittest.TestCase):
    """Test suite for Ragamala trainer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock training configuration
        self.training_config = {
            "learning_rate": 1e-4,
            "batch_size": 2,
            "num_epochs": 1,
            "save_steps": 100,
            "logging_steps": 10,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "output_dir": self.temp_dir
        }
        
        # Mock model and dataset
        self.mock_model = Mock()
        self.mock_dataset = Mock()
        self.mock_dataset.__len__ = Mock(return_value=10)
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        trainer = RagamalaTrainer(
            model=self.mock_model,
            config=self.training_config,
            train_dataset=self.mock_dataset
        )
        
        self.assertEqual(trainer.model, self.mock_model)
        self.assertEqual(trainer.config, self.training_config)
        self.assertEqual(trainer.train_dataset, self.mock_dataset)
    
    @patch('torch.optim.AdamW')
    @patch('torch.utils.data.DataLoader')
    def test_training_setup(self, mock_dataloader, mock_optimizer):
        """Test training setup."""
        trainer = RagamalaTrainer(
            model=self.mock_model,
            config=self.training_config,
            train_dataset=self.mock_dataset
        )
        
        trainer.setup_training()
        
        # Check that optimizer and dataloader were created
        mock_optimizer.assert_called()
        mock_dataloader.assert_called()
    
    def test_loss_calculation(self):
        """Test loss calculation."""
        trainer = RagamalaTrainer(
            model=self.mock_model,
            config=self.training_config,
            train_dataset=self.mock_dataset
        )
        
        # Mock batch data
        batch = {
            'pixel_values': torch.randn(2, 3, 512, 512),
            'input_ids': torch.randint(0, 1000, (2, 77)),
            'attention_mask': torch.ones(2, 77)
        }
        
        # Mock model output
        self.mock_model.return_value = Mock()
        self.mock_model.return_value.loss = torch.tensor(0.5)
        
        loss = trainer.compute_loss(batch)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.mock_model.assert_called()
    
    def test_checkpoint_saving(self):
        """Test checkpoint saving."""
        trainer = RagamalaTrainer(
            model=self.mock_model,
            config=self.training_config,
            train_dataset=self.mock_dataset
        )
        
        # Mock model state dict
        self.mock_model.state_dict.return_value = {'test_param': torch.tensor([1.0])}
        
        checkpoint_path = trainer.save_checkpoint(step=100, loss=0.5)
        
        self.assertTrue(os.path.exists(checkpoint_path))
        
        # Verify checkpoint content
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.assertIn('model_state_dict', checkpoint)
        self.assertIn('step', checkpoint)
        self.assertIn('loss', checkpoint)
    
    def test_checkpoint_loading(self):
        """Test checkpoint loading."""
        trainer = RagamalaTrainer(
            model=self.mock_model,
            config=self.training_config,
            train_dataset=self.mock_dataset
        )
        
        # Create a test checkpoint
        checkpoint_data = {
            'model_state_dict': {'test_param': torch.tensor([1.0])},
            'step': 100,
            'loss': 0.5
        }
        
        checkpoint_path = os.path.join(self.temp_dir, 'test_checkpoint.pt')
        torch.save(checkpoint_data, checkpoint_path)
        
        # Load checkpoint
        loaded_data = trainer.load_checkpoint(checkpoint_path)
        
        self.assertEqual(loaded_data['step'], 100)
        self.assertEqual(loaded_data['loss'], 0.5)
        self.mock_model.load_state_dict.assert_called()


class TestModelIntegration(unittest.TestCase):
    """Integration tests for model components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
    
    def test_end_to_end_model_pipeline(self):
        """Test complete model pipeline integration."""
        # 1. Create LoRA configuration
        lora_config = LoRAConfig(rank=16, alpha=8)
        
        # 2. Mock prompt encoder
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            'input_ids': torch.randint(0, 1000, (1, 77)),
            'attention_mask': torch.ones(1, 77)
        }
        
        mock_text_encoder = Mock()
        mock_text_encoder.return_value = Mock()
        mock_text_encoder.return_value.last_hidden_state = torch.randn(1, 77, 768)
        
        prompt_encoder = PromptEncoder(
            tokenizer=mock_tokenizer,
            text_encoder=mock_text_encoder,
            device=self.device
        )
        
        # 3. Test prompt encoding
        prompt = "A Rajput style Ragamala painting of Raga Bhairav"
        encoded_prompt = prompt_encoder.encode(prompt)
        
        # 4. Create scheduler
        scheduler = CustomDiffusionScheduler(num_train_timesteps=1000)
        
        # 5. Test noise scheduling
        sample = torch.randn(1, 4, 64, 64)
        noise = torch.randn_like(sample)
        timesteps = torch.randint(0, 1000, (1,))
        
        noisy_sample = scheduler.add_noise(sample, noise, timesteps)
        
        # Verify integration
        self.assertIsInstance(encoded_prompt, torch.Tensor)
        self.assertEqual(encoded_prompt.shape[0], 1)
        self.assertEqual(noisy_sample.shape, sample.shape)
        self.assertFalse(torch.allclose(noisy_sample, sample))
    
    def test_lora_integration_with_linear_layer(self):
        """Test LoRA integration with standard linear layers."""
        from src.models.sdxl_lora import LoRALayer
        
        # Create original layer
        original_layer = nn.Linear(512, 512)
        original_weight = original_layer.weight.clone()
        
        # Apply LoRA
        lora_layer = LoRALayer(original_layer, rank=16, alpha=8)
        
        # Test that original weights are preserved
        self.assertTrue(torch.allclose(
            lora_layer.original_layer.weight, 
            original_weight
        ))
        
        # Test forward pass difference
        input_tensor = torch.randn(2, 512)
        original_output = original_layer(input_tensor)
        lora_output = lora_layer(input_tensor)
        
        # Outputs should be different (LoRA adds adaptation)
        self.assertFalse(torch.allclose(original_output, lora_output))
    
    def test_cultural_prompt_integration(self):
        """Test integration of cultural prompt enhancement."""
        # Mock base encoder
        mock_base_encoder = Mock()
        mock_base_encoder.encode.return_value = torch.randn(1, 77, 768)
        
        cultural_encoder = CulturalPromptEncoder(
            base_encoder=mock_base_encoder,
            device=self.device
        )
        
        # Test cultural enhancement
        base_prompt = "A painting"
        enhanced_prompt = cultural_encoder.enhance_prompt(
            base_prompt, raga="bhairav", style="rajput"
        )
        
        # Test encoding
        encoded = cultural_encoder.encode_with_conditioning(
            enhanced_prompt, raga="bhairav", style="rajput"
        )
        
        # Verify integration
        self.assertIsInstance(enhanced_prompt, str)
        self.assertGreater(len(enhanced_prompt), len(base_prompt))
        self.assertIsInstance(encoded, torch.Tensor)
        mock_base_encoder.encode.assert_called()


if __name__ == '__main__':
    # Configure test environment
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run tests
    unittest.main(verbosity=2)
