"""
Comprehensive test suite for training components in the Ragamala painting generation project.
Tests training loops, loss functions, optimizers, schedulers, and LoRA fine-tuning functionality.
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
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from diffusers import DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

# Import modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.training.trainer import RagamalaTrainer, TrainingConfig
from src.training.losses import DiffusionLoss, CulturalLoss, PerceptualLoss
from src.training.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from src.training.utils import TrainingUtils, GradientClipping, MemoryOptimizer
from src.models.sdxl_lora import SDXLLoRAModel, LoRAConfig
from src.data.dataset import RagamalaDataset


class MockDataset(Dataset):
    """Mock dataset for testing."""
    
    def __init__(self, size=100):
        self.size = size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            'pixel_values': torch.randn(3, 512, 512),
            'input_ids': torch.randint(0, 1000, (77,)),
            'attention_mask': torch.ones(77),
            'raga': 'bhairav',
            'style': 'rajput',
            'prompt': f'A Ragamala painting {idx}'
        }


class TestTrainingConfig(unittest.TestCase):
    """Test suite for training configuration."""
    
    def test_training_config_initialization(self):
        """Test training configuration initialization."""
        config = TrainingConfig(
            learning_rate=1e-4,
            batch_size=4,
            num_epochs=10,
            gradient_accumulation_steps=2,
            max_grad_norm=1.0,
            warmup_steps=100,
            save_steps=500,
            logging_steps=50
        )
        
        self.assertEqual(config.learning_rate, 1e-4)
        self.assertEqual(config.batch_size, 4)
        self.assertEqual(config.num_epochs, 10)
        self.assertEqual(config.gradient_accumulation_steps, 2)
    
    def test_training_config_validation(self):
        """Test training configuration validation."""
        # Valid configuration
        valid_config = TrainingConfig(learning_rate=1e-4, batch_size=4)
        self.assertTrue(valid_config.is_valid())
        
        # Invalid learning rate
        with self.assertRaises(ValueError):
            TrainingConfig(learning_rate=0, batch_size=4)
        
        # Invalid batch size
        with self.assertRaises(ValueError):
            TrainingConfig(learning_rate=1e-4, batch_size=0)
    
    def test_training_config_serialization(self):
        """Test training configuration serialization."""
        config = TrainingConfig(learning_rate=1e-4, batch_size=4)
        
        # Test to_dict
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['learning_rate'], 1e-4)
        
        # Test from_dict
        restored_config = TrainingConfig.from_dict(config_dict)
        self.assertEqual(restored_config.learning_rate, config.learning_rate)
        self.assertEqual(restored_config.batch_size, config.batch_size)
    
    def test_training_config_update(self):
        """Test training configuration updates."""
        config = TrainingConfig(learning_rate=1e-4, batch_size=4)
        
        # Update configuration
        config.update({'learning_rate': 2e-4, 'batch_size': 8})
        
        self.assertEqual(config.learning_rate, 2e-4)
        self.assertEqual(config.batch_size, 8)


class TestRagamalaTrainer(unittest.TestCase):
    """Test suite for Ragamala trainer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock model components
        self.mock_unet = Mock(spec=UNet2DConditionModel)
        self.mock_unet.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
        self.mock_unet.train = Mock()
        self.mock_unet.eval = Mock()
        
        self.mock_text_encoder = Mock(spec=CLIPTextModel)
        self.mock_tokenizer = Mock(spec=CLIPTokenizer)
        
        # Training configuration
        self.training_config = TrainingConfig(
            learning_rate=1e-4,
            batch_size=2,
            num_epochs=1,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            save_steps=10,
            logging_steps=5,
            output_dir=self.temp_dir,
            mixed_precision='fp16',
            gradient_checkpointing=True
        )
        
        # Mock dataset
        self.mock_dataset = MockDataset(size=20)
        
        # Initialize trainer
        self.trainer = RagamalaTrainer(
            unet=self.mock_unet,
            text_encoder=self.mock_text_encoder,
            tokenizer=self.mock_tokenizer,
            config=self.training_config,
            train_dataset=self.mock_dataset
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        self.assertEqual(self.trainer.unet, self.mock_unet)
        self.assertEqual(self.trainer.text_encoder, self.mock_text_encoder)
        self.assertEqual(self.trainer.config, self.training_config)
        self.assertEqual(self.trainer.train_dataset, self.mock_dataset)
    
    def test_optimizer_setup(self):
        """Test optimizer setup."""
        optimizer = self.trainer.setup_optimizer()
        
        self.assertIsInstance(optimizer, torch.optim.Optimizer)
        self.assertEqual(optimizer.param_groups[0]['lr'], self.training_config.learning_rate)
    
    def test_scheduler_setup(self):
        """Test learning rate scheduler setup."""
        optimizer = self.trainer.setup_optimizer()
        scheduler = self.trainer.setup_scheduler(optimizer)
        
        self.assertIsNotNone(scheduler)
    
    def test_dataloader_setup(self):
        """Test dataloader setup."""
        dataloader = self.trainer.setup_dataloader()
        
        self.assertIsInstance(dataloader, DataLoader)
        self.assertEqual(dataloader.batch_size, self.training_config.batch_size)
    
    def test_loss_computation(self):
        """Test loss computation."""
        batch = {
            'pixel_values': torch.randn(2, 3, 512, 512),
            'input_ids': torch.randint(0, 1000, (2, 77)),
            'attention_mask': torch.ones(2, 77)
        }
        
        # Mock UNet forward pass
        self.mock_unet.return_value = Mock()
        self.mock_unet.return_value.sample = torch.randn(2, 4, 64, 64)
        
        loss = self.trainer.compute_loss(batch)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar loss
    
    def test_training_step(self):
        """Test single training step."""
        batch = {
            'pixel_values': torch.randn(2, 3, 512, 512),
            'input_ids': torch.randint(0, 1000, (2, 77)),
            'attention_mask': torch.ones(2, 77)
        }
        
        # Mock loss computation
        with patch.object(self.trainer, 'compute_loss') as mock_compute_loss:
            mock_compute_loss.return_value = torch.tensor(0.5, requires_grad=True)
            
            optimizer = self.trainer.setup_optimizer()
            loss = self.trainer.training_step(batch, optimizer)
            
            self.assertIsInstance(loss, torch.Tensor)
            mock_compute_loss.assert_called_once()
    
    def test_validation_step(self):
        """Test validation step."""
        batch = {
            'pixel_values': torch.randn(2, 3, 512, 512),
            'input_ids': torch.randint(0, 1000, (2, 77)),
            'attention_mask': torch.ones(2, 77)
        }
        
        with patch.object(self.trainer, 'compute_loss') as mock_compute_loss:
            mock_compute_loss.return_value = torch.tensor(0.3)
            
            loss = self.trainer.validation_step(batch)
            
            self.assertIsInstance(loss, torch.Tensor)
            mock_compute_loss.assert_called_once()
    
    def test_save_checkpoint(self):
        """Test checkpoint saving."""
        checkpoint_path = self.trainer.save_checkpoint(
            epoch=1,
            step=100,
            loss=0.5,
            optimizer=self.trainer.setup_optimizer()
        )
        
        self.assertTrue(os.path.exists(checkpoint_path))
        
        # Verify checkpoint content
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.assertIn('epoch', checkpoint)
        self.assertIn('step', checkpoint)
        self.assertIn('loss', checkpoint)
        self.assertIn('model_state_dict', checkpoint)
        self.assertIn('optimizer_state_dict', checkpoint)
    
    def test_load_checkpoint(self):
        """Test checkpoint loading."""
        # Create a test checkpoint
        checkpoint_data = {
            'epoch': 1,
            'step': 100,
            'loss': 0.5,
            'model_state_dict': {'test_param': torch.tensor([1.0])},
            'optimizer_state_dict': {'state': {}, 'param_groups': []}
        }
        
        checkpoint_path = os.path.join(self.temp_dir, 'test_checkpoint.pt')
        torch.save(checkpoint_data, checkpoint_path)
        
        # Load checkpoint
        optimizer = self.trainer.setup_optimizer()
        loaded_data = self.trainer.load_checkpoint(checkpoint_path, optimizer)
        
        self.assertEqual(loaded_data['epoch'], 1)
        self.assertEqual(loaded_data['step'], 100)
        self.assertEqual(loaded_data['loss'], 0.5)
    
    @patch('torch.cuda.is_available')
    def test_mixed_precision_training(self, mock_cuda_available):
        """Test mixed precision training setup."""
        mock_cuda_available.return_value = True
        
        # Test with mixed precision enabled
        config_fp16 = TrainingConfig(mixed_precision='fp16')
        trainer_fp16 = RagamalaTrainer(
            unet=self.mock_unet,
            text_encoder=self.mock_text_encoder,
            tokenizer=self.mock_tokenizer,
            config=config_fp16,
            train_dataset=self.mock_dataset
        )
        
        scaler = trainer_fp16.setup_mixed_precision()
        self.assertIsNotNone(scaler)
    
    def test_gradient_accumulation(self):
        """Test gradient accumulation."""
        config_with_accumulation = TrainingConfig(
            gradient_accumulation_steps=4,
            batch_size=2
        )
        
        trainer_with_accumulation = RagamalaTrainer(
            unet=self.mock_unet,
            text_encoder=self.mock_text_encoder,
            tokenizer=self.mock_tokenizer,
            config=config_with_accumulation,
            train_dataset=self.mock_dataset
        )
        
        effective_batch_size = trainer_with_accumulation.get_effective_batch_size()
        self.assertEqual(effective_batch_size, 8)  # 2 * 4


class TestDiffusionLoss(unittest.TestCase):
    """Test suite for diffusion loss functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loss_fn = DiffusionLoss()
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012
        )
    
    def test_mse_loss_computation(self):
        """Test MSE loss computation."""
        predicted_noise = torch.randn(2, 4, 64, 64)
        target_noise = torch.randn(2, 4, 64, 64)
        
        loss = self.loss_fn.mse_loss(predicted_noise, target_noise)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar loss
        self.assertGreaterEqual(loss.item(), 0)
    
    def test_huber_loss_computation(self):
        """Test Huber loss computation."""
        predicted_noise = torch.randn(2, 4, 64, 64)
        target_noise = torch.randn(2, 4, 64, 64)
        
        loss = self.loss_fn.huber_loss(predicted_noise, target_noise, delta=0.1)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)
        self.assertGreaterEqual(loss.item(), 0)
    
    def test_snr_weighted_loss(self):
        """Test SNR-weighted loss computation."""
        predicted_noise = torch.randn(2, 4, 64, 64)
        target_noise = torch.randn(2, 4, 64, 64)
        timesteps = torch.randint(0, 1000, (2,))
        
        loss = self.loss_fn.snr_weighted_loss(
            predicted_noise, target_noise, timesteps, self.noise_scheduler
        )
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)
        self.assertGreaterEqual(loss.item(), 0)
    
    def test_loss_with_mask(self):
        """Test loss computation with attention mask."""
        predicted_noise = torch.randn(2, 4, 64, 64)
        target_noise = torch.randn(2, 4, 64, 64)
        mask = torch.randint(0, 2, (2, 1, 64, 64)).float()
        
        loss = self.loss_fn.masked_loss(predicted_noise, target_noise, mask)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)
        self.assertGreaterEqual(loss.item(), 0)


class TestCulturalLoss(unittest.TestCase):
    """Test suite for cultural loss functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cultural_loss = CulturalLoss()
    
    def test_raga_consistency_loss(self):
        """Test raga consistency loss."""
        # Mock embeddings for different ragas
        bhairav_embedding = torch.randn(2, 768)
        yaman_embedding = torch.randn(2, 768)
        
        # Same raga should have lower loss
        same_raga_loss = self.cultural_loss.raga_consistency_loss(
            bhairav_embedding, bhairav_embedding
        )
        
        # Different ragas should have higher loss
        different_raga_loss = self.cultural_loss.raga_consistency_loss(
            bhairav_embedding, yaman_embedding
        )
        
        self.assertIsInstance(same_raga_loss, torch.Tensor)
        self.assertIsInstance(different_raga_loss, torch.Tensor)
        self.assertLess(same_raga_loss.item(), different_raga_loss.item())
    
    def test_style_consistency_loss(self):
        """Test style consistency loss."""
        # Mock style embeddings
        rajput_features = torch.randn(2, 512, 8, 8)
        pahari_features = torch.randn(2, 512, 8, 8)
        
        # Same style should have lower loss
        same_style_loss = self.cultural_loss.style_consistency_loss(
            rajput_features, rajput_features
        )
        
        # Different styles should have higher loss
        different_style_loss = self.cultural_loss.style_consistency_loss(
            rajput_features, pahari_features
        )
        
        self.assertIsInstance(same_style_loss, torch.Tensor)
        self.assertIsInstance(different_style_loss, torch.Tensor)
    
    def test_temporal_consistency_loss(self):
        """Test temporal consistency loss."""
        # Mock temporal features (dawn vs evening)
        dawn_features = torch.randn(2, 256)
        evening_features = torch.randn(2, 256)
        
        # Test temporal alignment
        temporal_loss = self.cultural_loss.temporal_consistency_loss(
            dawn_features, evening_features, 
            time_labels=['dawn', 'evening']
        )
        
        self.assertIsInstance(temporal_loss, torch.Tensor)
        self.assertGreaterEqual(temporal_loss.item(), 0)
    
    def test_iconographic_loss(self):
        """Test iconographic consistency loss."""
        # Mock iconographic features
        temple_features = torch.randn(2, 128)
        peacock_features = torch.randn(2, 128)
        
        iconographic_loss = self.cultural_loss.iconographic_loss(
            temple_features, peacock_features,
            raga='bhairav'  # Both temple and peacock are associated with Bhairav
        )
        
        self.assertIsInstance(iconographic_loss, torch.Tensor)
        self.assertGreaterEqual(iconographic_loss.item(), 0)


class TestPerceptualLoss(unittest.TestCase):
    """Test suite for perceptual loss functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.perceptual_loss = PerceptualLoss()
    
    def test_vgg_perceptual_loss(self):
        """Test VGG-based perceptual loss."""
        generated_images = torch.randn(2, 3, 256, 256)
        target_images = torch.randn(2, 3, 256, 256)
        
        with patch('torchvision.models.vgg19') as mock_vgg:
            mock_vgg.return_value = Mock()
            mock_vgg.return_value.features = Mock()
            mock_vgg.return_value.features.return_value = torch.randn(2, 512, 16, 16)
            
            loss = self.perceptual_loss.vgg_loss(generated_images, target_images)
            
            self.assertIsInstance(loss, torch.Tensor)
            self.assertGreaterEqual(loss.item(), 0)
    
    def test_lpips_loss(self):
        """Test LPIPS perceptual loss."""
        generated_images = torch.randn(2, 3, 256, 256)
        target_images = torch.randn(2, 3, 256, 256)
        
        # Mock LPIPS model
        with patch('lpips.LPIPS') as mock_lpips:
            mock_lpips_instance = Mock()
            mock_lpips_instance.return_value = torch.randn(2, 1, 1, 1)
            mock_lpips.return_value = mock_lpips_instance
            
            loss = self.perceptual_loss.lpips_loss(generated_images, target_images)
            
            self.assertIsInstance(loss, torch.Tensor)
            self.assertGreaterEqual(loss.item(), 0)
    
    def test_style_loss(self):
        """Test style loss computation."""
        generated_features = torch.randn(2, 512, 32, 32)
        target_features = torch.randn(2, 512, 32, 32)
        
        loss = self.perceptual_loss.style_loss(generated_features, target_features)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreaterEqual(loss.item(), 0)


class TestTrainingCallbacks(unittest.TestCase):
    """Test suite for training callbacks."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_model_checkpoint_callback(self):
        """Test model checkpoint callback."""
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.temp_dir,
            filename='model-{epoch:02d}-{loss:.2f}',
            save_top_k=3,
            monitor='loss',
            mode='min'
        )
        
        # Mock trainer and metrics
        mock_trainer = Mock()
        mock_trainer.current_epoch = 1
        
        metrics = {'loss': 0.5, 'val_loss': 0.6}
        
        # Test checkpoint saving
        checkpoint_callback.on_epoch_end(mock_trainer, metrics)
        
        # Check if checkpoint was saved
        checkpoint_files = list(Path(self.temp_dir).glob('*.ckpt'))
        self.assertGreater(len(checkpoint_files), 0)
    
    def test_early_stopping_callback(self):
        """Test early stopping callback."""
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            min_delta=0.01,
            mode='min'
        )
        
        mock_trainer = Mock()
        
        # Simulate improving loss
        early_stopping.on_epoch_end(mock_trainer, {'val_loss': 1.0})
        self.assertFalse(early_stopping.should_stop)
        
        early_stopping.on_epoch_end(mock_trainer, {'val_loss': 0.9})
        self.assertFalse(early_stopping.should_stop)
        
        # Simulate stagnating loss
        for _ in range(4):
            early_stopping.on_epoch_end(mock_trainer, {'val_loss': 0.9})
        
        self.assertTrue(early_stopping.should_stop)
    
    def test_learning_rate_scheduler_callback(self):
        """Test learning rate scheduler callback."""
        # Mock optimizer and scheduler
        mock_optimizer = Mock()
        mock_scheduler = Mock()
        
        lr_scheduler_callback = LearningRateScheduler(
            scheduler=mock_scheduler,
            monitor='loss'
        )
        
        mock_trainer = Mock()
        mock_trainer.optimizer = mock_optimizer
        
        metrics = {'loss': 0.5}
        
        # Test scheduler step
        lr_scheduler_callback.on_epoch_end(mock_trainer, metrics)
        
        mock_scheduler.step.assert_called()


class TestTrainingUtils(unittest.TestCase):
    """Test suite for training utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.training_utils = TrainingUtils()
    
    def test_gradient_clipping(self):
        """Test gradient clipping utility."""
        # Create a simple model
        model = nn.Linear(10, 1)
        
        # Create some gradients
        loss = model(torch.randn(5, 10)).sum()
        loss.backward()
        
        # Test gradient clipping
        grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        
        gradient_clipper = GradientClipping(max_norm=1.0)
        grad_norm_after = gradient_clipper.clip_gradients(model.parameters())
        
        self.assertIsInstance(grad_norm_before, torch.Tensor)
        self.assertIsInstance(grad_norm_after, torch.Tensor)
        self.assertLessEqual(grad_norm_after.item(), 1.0)
    
    def test_memory_optimizer(self):
        """Test memory optimization utilities."""
        memory_optimizer = MemoryOptimizer()
        
        # Test gradient checkpointing setup
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )
        
        optimized_model = memory_optimizer.enable_gradient_checkpointing(model)
        
        # Model should still be functional
        input_tensor = torch.randn(10, 100)
        output = optimized_model(input_tensor)
        self.assertEqual(output.shape, (10, 1))
    
    def test_learning_rate_finder(self):
        """Test learning rate finder utility."""
        # Mock model and dataset
        model = nn.Linear(10, 1)
        dataset = MockDataset(size=50)
        dataloader = DataLoader(dataset, batch_size=5)
        
        lr_finder = self.training_utils.find_learning_rate(
            model=model,
            dataloader=dataloader,
            start_lr=1e-7,
            end_lr=1e-1,
            num_iter=20
        )
        
        self.assertIsInstance(lr_finder, dict)
        self.assertIn('learning_rates', lr_finder)
        self.assertIn('losses', lr_finder)
        self.assertEqual(len(lr_finder['learning_rates']), len(lr_finder['losses']))
    
    def test_model_summary(self):
        """Test model summary utility."""
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.Softmax(dim=1)
        )
        
        summary = self.training_utils.get_model_summary(model, input_size=(1, 100))
        
        self.assertIsInstance(summary, dict)
        self.assertIn('total_params', summary)
        self.assertIn('trainable_params', summary)
        self.assertIn('model_size_mb', summary)
    
    def test_training_metrics_tracker(self):
        """Test training metrics tracking."""
        metrics_tracker = self.training_utils.create_metrics_tracker()
        
        # Add some metrics
        metrics_tracker.update({
            'epoch': 1,
            'train_loss': 0.5,
            'val_loss': 0.6,
            'learning_rate': 1e-4
        })
        
        metrics_tracker.update({
            'epoch': 2,
            'train_loss': 0.4,
            'val_loss': 0.55,
            'learning_rate': 9e-5
        })
        
        # Get metrics history
        history = metrics_tracker.get_history()
        
        self.assertEqual(len(history), 2)
        self.assertIn('train_loss', history[0])
        self.assertEqual(history[1]['epoch'], 2)


class TestLoRATraining(unittest.TestCase):
    """Test suite for LoRA-specific training functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # LoRA configuration
        self.lora_config = LoRAConfig(
            rank=16,
            alpha=8,
            target_modules=["to_k", "to_q", "to_v"],
            dropout=0.1
        )
        
        # Mock SDXL model
        self.mock_sdxl_model = Mock(spec=SDXLLoRAModel)
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_lora_parameter_counting(self):
        """Test LoRA parameter counting."""
        # Create a simple model with LoRA
        from src.models.sdxl_lora import LoRALayer
        
        original_layer = nn.Linear(512, 512)
        lora_layer = LoRALayer(original_layer, rank=16, alpha=8)
        
        # Count LoRA parameters
        lora_params = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
        original_params = sum(p.numel() for p in original_layer.parameters())
        
        # LoRA should add fewer parameters than original
        self.assertLess(lora_params, original_params)
        
        # LoRA parameters should be: rank * (input_dim + output_dim)
        expected_lora_params = 16 * (512 + 512)
        self.assertEqual(lora_params, expected_lora_params)
    
    def test_lora_weight_initialization(self):
        """Test LoRA weight initialization."""
        from src.models.sdxl_lora import LoRALayer
        
        original_layer = nn.Linear(512, 512)
        lora_layer = LoRALayer(original_layer, rank=16, alpha=8)
        
        # Check LoRA A initialization (should be non-zero)
        self.assertFalse(torch.allclose(
            lora_layer.lora_A.weight, 
            torch.zeros_like(lora_layer.lora_A.weight)
        ))
        
        # Check LoRA B initialization (should be zero)
        self.assertTrue(torch.allclose(
            lora_layer.lora_B.weight, 
            torch.zeros_like(lora_layer.lora_B.weight)
        ))
    
    def test_lora_training_mode(self):
        """Test LoRA training mode switching."""
        from src.models.sdxl_lora import LoRALayer
        
        original_layer = nn.Linear(512, 512)
        lora_layer = LoRALayer(original_layer, rank=16, alpha=8)
        
        # Test training mode
        lora_layer.train()
        self.assertTrue(lora_layer.training)
        self.assertTrue(lora_layer.lora_A.training)
        self.assertTrue(lora_layer.lora_B.training)
        
        # Test evaluation mode
        lora_layer.eval()
        self.assertFalse(lora_layer.training)
        self.assertFalse(lora_layer.lora_A.training)
        self.assertFalse(lora_layer.lora_B.training)
    
    def test_lora_state_dict_operations(self):
        """Test LoRA state dict save/load operations."""
        from src.models.sdxl_lora import LoRALayer
        
        original_layer = nn.Linear(512, 512)
        lora_layer = LoRALayer(original_layer, rank=16, alpha=8)
        
        # Get LoRA-only state dict
        lora_state_dict = lora_layer.get_lora_state_dict()
        
        self.assertIn('lora_A.weight', lora_state_dict)
        self.assertIn('lora_B.weight', lora_state_dict)
        self.assertNotIn('original_layer.weight', lora_state_dict)
        
        # Test loading LoRA state dict
        new_lora_layer = LoRALayer(nn.Linear(512, 512), rank=16, alpha=8)
        new_lora_layer.load_lora_state_dict(lora_state_dict)
        
        # Weights should match
        self.assertTrue(torch.allclose(
            lora_layer.lora_A.weight, 
            new_lora_layer.lora_A.weight
        ))


class TestTrainingIntegration(unittest.TestCase):
    """Integration tests for training components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.device = torch.device("cpu")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_training_setup(self):
        """Test complete training setup integration."""
        # 1. Create mock model components
        mock_unet = Mock()
        mock_text_encoder = Mock()
        mock_tokenizer = Mock()
        
        # 2. Create training configuration
        config = TrainingConfig(
            learning_rate=1e-4,
            batch_size=2,
            num_epochs=1,
            output_dir=self.temp_dir
        )
        
        # 3. Create dataset
        dataset = MockDataset(size=10)
        
        # 4. Initialize trainer
        trainer = RagamalaTrainer(
            unet=mock_unet,
            text_encoder=mock_text_encoder,
            tokenizer=mock_tokenizer,
            config=config,
            train_dataset=dataset
        )
        
        # 5. Setup training components
        optimizer = trainer.setup_optimizer()
        scheduler = trainer.setup_scheduler(optimizer)
        dataloader = trainer.setup_dataloader()
        
        # Verify integration
        self.assertIsInstance(optimizer, torch.optim.Optimizer)
        self.assertIsNotNone(scheduler)
        self.assertIsInstance(dataloader, DataLoader)
        self.assertEqual(len(dataloader), 5)  # 10 samples / batch_size 2
    
    def test_loss_function_integration(self):
        """Test integration of different loss functions."""
        # Create mock predictions and targets
        predicted_noise = torch.randn(2, 4, 64, 64, requires_grad=True)
        target_noise = torch.randn(2, 4, 64, 64)
        timesteps = torch.randint(0, 1000, (2,))
        
        # Test diffusion loss
        diffusion_loss = DiffusionLoss()
        diff_loss = diffusion_loss.mse_loss(predicted_noise, target_noise)
        
        # Test cultural loss
        cultural_loss = CulturalLoss()
        raga_embeddings = torch.randn(2, 768)
        cult_loss = cultural_loss.raga_consistency_loss(raga_embeddings, raga_embeddings)
        
        # Test perceptual loss
        perceptual_loss = PerceptualLoss()
        images = torch.randn(2, 3, 256, 256)
        
        with patch('torchvision.models.vgg19'):
            perc_loss = perceptual_loss.vgg_loss(images, images)
        
        # Combine losses
        total_loss = diff_loss + 0.1 * cult_loss + 0.05 * perc_loss
        
        # Test backward pass
        total_loss.backward()
        
        # Verify gradients
        self.assertIsNotNone(predicted_noise.grad)
        self.assertIsInstance(total_loss, torch.Tensor)
    
    def test_callback_integration(self):
        """Test integration of training callbacks."""
        # Create callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.temp_dir,
            save_top_k=2,
            monitor='loss'
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=2
        )
        
        # Mock trainer
        mock_trainer = Mock()
        mock_trainer.current_epoch = 1
        
        # Test callback execution
        metrics = {'loss': 0.5, 'val_loss': 0.6}
        
        checkpoint_callback.on_epoch_end(mock_trainer, metrics)
        early_stopping.on_epoch_end(mock_trainer, metrics)
        
        # Verify callback effects
        self.assertFalse(early_stopping.should_stop)
        
        # Check checkpoint was created
        checkpoint_files = list(Path(self.temp_dir).glob('*.ckpt'))
        self.assertGreater(len(checkpoint_files), 0)


if __name__ == '__main__':
    # Configure test environment
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run tests
    unittest.main(verbosity=2)
