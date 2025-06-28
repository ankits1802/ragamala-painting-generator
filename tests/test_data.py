"""
Comprehensive test suite for data processing components in the Ragamala painting generation project.
Tests data collection, preprocessing, annotation, and dataset functionality.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

# Import modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.collector import MuseumDataCollector, ImageCollector
from src.data.preprocessor import ImagePreprocessor, MetadataProcessor
from src.data.annotator import RagamalaAnnotator, CulturalAnnotator
from src.data.dataset import RagamalaDataset, RagamalaDataModule


class TestImageCollector(unittest.TestCase):
    """Test suite for ImageCollector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.collector = ImageCollector(output_dir=self.temp_dir)
        
        # Create test image
        self.test_image = Image.new('RGB', (512, 512), color='red')
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_collector_initialization(self):
        """Test ImageCollector initialization."""
        self.assertEqual(self.collector.output_dir, Path(self.temp_dir))
        self.assertTrue(self.collector.output_dir.exists())
    
    def test_save_image(self):
        """Test image saving functionality."""
        filename = "test_image.jpg"
        saved_path = self.collector.save_image(self.test_image, filename)
        
        self.assertTrue(saved_path.exists())
        self.assertEqual(saved_path.name, filename)
        
        # Verify image can be loaded
        loaded_image = Image.open(saved_path)
        self.assertEqual(loaded_image.size, (512, 512))
    
    def test_save_image_with_metadata(self):
        """Test saving image with metadata."""
        metadata = {
            "raga": "bhairav",
            "style": "rajput",
            "period": "18th century"
        }
        
        filename = "test_with_metadata.jpg"
        saved_path = self.collector.save_image(self.test_image, filename, metadata)
        
        # Check metadata file exists
        metadata_path = saved_path.with_suffix('.json')
        self.assertTrue(metadata_path.exists())
        
        # Verify metadata content
        with open(metadata_path, 'r') as f:
            saved_metadata = json.load(f)
        
        self.assertEqual(saved_metadata['raga'], 'bhairav')
        self.assertEqual(saved_metadata['style'], 'rajput')
    
    @patch('requests.get')
    def test_download_from_url(self, mock_get):
        """Test downloading image from URL."""
        # Mock response
        mock_response = Mock()
        mock_response.content = b'fake_image_data'
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        url = "https://example.com/test_image.jpg"
        filename = "downloaded_image.jpg"
        
        with patch.object(Image, 'open') as mock_open:
            mock_open.return_value = self.test_image
            
            saved_path = self.collector.download_from_url(url, filename)
            
            self.assertTrue(saved_path.exists())
            mock_get.assert_called_once_with(url, timeout=30)
    
    def test_download_from_url_failure(self):
        """Test handling of download failures."""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("Network error")
            
            url = "https://example.com/nonexistent.jpg"
            filename = "failed_download.jpg"
            
            with self.assertRaises(Exception):
                self.collector.download_from_url(url, filename)


class TestMuseumDataCollector(unittest.TestCase):
    """Test suite for MuseumDataCollector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.collector = MuseumDataCollector(output_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('requests.get')
    def test_search_met_museum(self, mock_get):
        """Test Metropolitan Museum API search."""
        # Mock search response
        search_response = Mock()
        search_response.json.return_value = {
            "total": 2,
            "objectIDs": [12345, 67890]
        }
        search_response.status_code = 200
        
        # Mock object detail responses
        object_response_1 = Mock()
        object_response_1.json.return_value = {
            "objectID": 12345,
            "title": "Ragamala Painting - Bhairav",
            "primaryImage": "https://example.com/image1.jpg",
            "culture": "Indian",
            "period": "18th century",
            "medium": "Opaque watercolor on paper"
        }
        object_response_1.status_code = 200
        
        object_response_2 = Mock()
        object_response_2.json.return_value = {
            "objectID": 67890,
            "title": "Ragamala Series - Yaman",
            "primaryImage": "https://example.com/image2.jpg",
            "culture": "Indian",
            "period": "17th century",
            "medium": "Watercolor and gold on paper"
        }
        object_response_2.status_code = 200
        
        mock_get.side_effect = [search_response, object_response_1, object_response_2]
        
        results = self.collector.search_met_museum("ragamala", max_results=2)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['objectID'], 12345)
        self.assertEqual(results[1]['objectID'], 67890)
        self.assertIn('Bhairav', results[0]['title'])
        self.assertIn('Yaman', results[1]['title'])
    
    def test_extract_raga_from_title(self):
        """Test raga extraction from artwork titles."""
        test_cases = [
            ("Ragamala: Raga Bhairav", "bhairav"),
            ("Yaman Raga Illustration", "yaman"),
            ("Malkauns from Ragamala Series", "malkauns"),
            ("Darbari Kanada Musical Mode", "darbari"),
            ("Unknown Painting", None)
        ]
        
        for title, expected_raga in test_cases:
            with self.subTest(title=title):
                extracted = self.collector.extract_raga_from_title(title)
                self.assertEqual(extracted, expected_raga)
    
    def test_extract_style_from_metadata(self):
        """Test painting style extraction from metadata."""
        test_cases = [
            ({"culture": "Rajasthani", "geography": "Rajasthan"}, "rajput"),
            ({"culture": "Pahari", "geography": "Punjab Hills"}, "pahari"),
            ({"culture": "Deccani", "geography": "Deccan"}, "deccan"),
            ({"culture": "Mughal", "geography": "Delhi"}, "mughal"),
            ({"culture": "Unknown", "geography": "India"}, None)
        ]
        
        for metadata, expected_style in test_cases:
            with self.subTest(metadata=metadata):
                extracted = self.collector.extract_style_from_metadata(metadata)
                self.assertEqual(extracted, expected_style)


class TestImagePreprocessor(unittest.TestCase):
    """Test suite for ImagePreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = ImagePreprocessor(
            target_size=(512, 512),
            normalize=True,
            augment=True
        )
        
        # Create test images
        self.test_image_rgb = Image.new('RGB', (1024, 768), color='blue')
        self.test_image_rgba = Image.new('RGBA', (800, 600), color='green')
        self.test_image_small = Image.new('RGB', (256, 256), color='yellow')
    
    def test_resize_image(self):
        """Test image resizing functionality."""
        resized = self.preprocessor.resize_image(self.test_image_rgb)
        
        self.assertEqual(resized.size, (512, 512))
        self.assertEqual(resized.mode, 'RGB')
    
    def test_resize_image_maintain_aspect_ratio(self):
        """Test resizing with aspect ratio preservation."""
        preprocessor = ImagePreprocessor(
            target_size=(512, 512),
            maintain_aspect_ratio=True
        )
        
        # Test with non-square image
        resized = preprocessor.resize_image(self.test_image_rgb)
        
        # Should be resized but maintain aspect ratio
        self.assertLessEqual(max(resized.size), 512)
        self.assertGreaterEqual(min(resized.size), 384)  # Reasonable minimum
    
    def test_convert_rgba_to_rgb(self):
        """Test RGBA to RGB conversion."""
        converted = self.preprocessor.convert_to_rgb(self.test_image_rgba)
        
        self.assertEqual(converted.mode, 'RGB')
        self.assertEqual(converted.size, self.test_image_rgba.size)
    
    def test_normalize_image(self):
        """Test image normalization."""
        # Convert PIL to tensor for testing
        import torchvision.transforms as transforms
        
        to_tensor = transforms.ToTensor()
        tensor_image = to_tensor(self.test_image_rgb)
        
        normalized = self.preprocessor.normalize_tensor(tensor_image)
        
        # Check normalization range
        self.assertGreaterEqual(normalized.min().item(), -1.0)
        self.assertLessEqual(normalized.max().item(), 1.0)
    
    def test_augment_image(self):
        """Test image augmentation."""
        augmented_images = []
        
        for _ in range(5):
            augmented = self.preprocessor.augment_image(self.test_image_rgb)
            augmented_images.append(augmented)
        
        # Check that augmentations produce different results
        self.assertEqual(len(set(img.tobytes() for img in augmented_images)), 5)
    
    def test_preprocess_pipeline(self):
        """Test complete preprocessing pipeline."""
        processed = self.preprocessor.preprocess(self.test_image_rgba)
        
        self.assertIsInstance(processed, torch.Tensor)
        self.assertEqual(processed.shape, (3, 512, 512))  # C, H, W
        self.assertGreaterEqual(processed.min().item(), -1.0)
        self.assertLessEqual(processed.max().item(), 1.0)
    
    def test_batch_preprocessing(self):
        """Test batch preprocessing."""
        images = [self.test_image_rgb, self.test_image_rgba, self.test_image_small]
        
        processed_batch = self.preprocessor.preprocess_batch(images)
        
        self.assertEqual(len(processed_batch), 3)
        for processed in processed_batch:
            self.assertIsInstance(processed, torch.Tensor)
            self.assertEqual(processed.shape, (3, 512, 512))


class TestMetadataProcessor(unittest.TestCase):
    """Test suite for MetadataProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = MetadataProcessor()
        
        self.sample_metadata = {
            "title": "Ragamala: Raga Bhairav at Dawn",
            "culture": "Rajasthani",
            "period": "18th century",
            "medium": "Opaque watercolor and gold on paper",
            "dimensions": "25.4 x 20.3 cm",
            "description": "A devotional scene showing Lord Shiva at dawn"
        }
    
    def test_extract_raga_information(self):
        """Test raga information extraction."""
        raga_info = self.processor.extract_raga_information(self.sample_metadata)
        
        self.assertEqual(raga_info['raga'], 'bhairav')
        self.assertEqual(raga_info['time_of_day'], 'dawn')
        self.assertEqual(raga_info['mood'], 'devotional')
    
    def test_extract_style_information(self):
        """Test style information extraction."""
        style_info = self.processor.extract_style_information(self.sample_metadata)
        
        self.assertEqual(style_info['style'], 'rajput')
        self.assertEqual(style_info['period'], '18th century')
        self.assertEqual(style_info['region'], 'rajasthan')
    
    def test_extract_technical_information(self):
        """Test technical information extraction."""
        tech_info = self.processor.extract_technical_information(self.sample_metadata)
        
        self.assertEqual(tech_info['medium'], 'watercolor')
        self.assertIn('gold', tech_info['materials'])
        self.assertEqual(tech_info['dimensions']['width'], 25.4)
        self.assertEqual(tech_info['dimensions']['height'], 20.3)
    
    def test_generate_prompt_from_metadata(self):
        """Test prompt generation from metadata."""
        prompt = self.processor.generate_prompt(self.sample_metadata)
        
        self.assertIn('bhairav', prompt.lower())
        self.assertIn('rajasthani', prompt.lower())
        self.assertIn('18th century', prompt.lower())
        self.assertIn('devotional', prompt.lower())
    
    def test_validate_metadata_completeness(self):
        """Test metadata completeness validation."""
        # Complete metadata
        is_complete, missing = self.processor.validate_metadata(self.sample_metadata)
        self.assertTrue(is_complete)
        self.assertEqual(len(missing), 0)
        
        # Incomplete metadata
        incomplete_metadata = {"title": "Some painting"}
        is_complete, missing = self.processor.validate_metadata(incomplete_metadata)
        self.assertFalse(is_complete)
        self.assertGreater(len(missing), 0)


class TestRagamalaAnnotator(unittest.TestCase):
    """Test suite for RagamalaAnnotator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.annotator = RagamalaAnnotator()
        
        # Create test image
        self.test_image = Image.new('RGB', (512, 512), color='red')
        
        self.sample_annotation = {
            "raga": "bhairav",
            "style": "rajput",
            "period": "18th century",
            "iconography": ["temple", "peacock", "sunrise"],
            "colors": ["white", "saffron", "gold"],
            "mood": "devotional"
        }
    
    def test_create_annotation(self):
        """Test annotation creation."""
        annotation = self.annotator.create_annotation(
            image_path="test_image.jpg",
            raga="bhairav",
            style="rajput",
            additional_metadata={"period": "18th century"}
        )
        
        self.assertEqual(annotation['raga'], 'bhairav')
        self.assertEqual(annotation['style'], 'rajput')
        self.assertEqual(annotation['period'], '18th century')
        self.assertIn('timestamp', annotation)
    
    def test_validate_annotation(self):
        """Test annotation validation."""
        # Valid annotation
        is_valid, errors = self.annotator.validate_annotation(self.sample_annotation)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
        # Invalid annotation - missing required fields
        invalid_annotation = {"raga": "bhairav"}
        is_valid, errors = self.annotator.validate_annotation(invalid_annotation)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
        
        # Invalid annotation - wrong raga
        invalid_raga = self.sample_annotation.copy()
        invalid_raga['raga'] = 'invalid_raga'
        is_valid, errors = self.annotator.validate_annotation(invalid_raga)
        self.assertFalse(is_valid)
        self.assertIn('Invalid raga', ' '.join(errors))
    
    def test_auto_annotate_from_filename(self):
        """Test automatic annotation from filename."""
        test_cases = [
            ("bhairav_rajput_18th_century.jpg", "bhairav", "rajput"),
            ("yaman_pahari_17th.png", "yaman", "pahari"),
            ("malkauns_deccan_painting.jpg", "malkauns", "deccan")
        ]
        
        for filename, expected_raga, expected_style in test_cases:
            with self.subTest(filename=filename):
                annotation = self.annotator.auto_annotate_from_filename(filename)
                self.assertEqual(annotation['raga'], expected_raga)
                self.assertEqual(annotation['style'], expected_style)
    
    def test_merge_annotations(self):
        """Test annotation merging."""
        base_annotation = {
            "raga": "bhairav",
            "style": "rajput",
            "confidence": 0.8
        }
        
        additional_annotation = {
            "period": "18th century",
            "iconography": ["temple", "peacock"],
            "confidence": 0.9
        }
        
        merged = self.annotator.merge_annotations(base_annotation, additional_annotation)
        
        self.assertEqual(merged['raga'], 'bhairav')
        self.assertEqual(merged['style'], 'rajput')
        self.assertEqual(merged['period'], '18th century')
        self.assertEqual(merged['confidence'], 0.9)  # Should take higher confidence


class TestCulturalAnnotator(unittest.TestCase):
    """Test suite for CulturalAnnotator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.annotator = CulturalAnnotator()
    
    def test_identify_iconographic_elements(self):
        """Test iconographic element identification."""
        # Mock image analysis
        with patch.object(self.annotator, '_analyze_image_content') as mock_analyze:
            mock_analyze.return_value = {
                "objects": ["temple", "peacock", "figure"],
                "colors": ["white", "gold", "blue"],
                "composition": "centered"
            }
            
            elements = self.annotator.identify_iconographic_elements(
                Image.new('RGB', (512, 512)), 
                raga="bhairav"
            )
            
            self.assertIn("temple", elements)
            self.assertIn("peacock", elements)
    
    def test_assess_cultural_accuracy(self):
        """Test cultural accuracy assessment."""
        annotation = {
            "raga": "bhairav",
            "style": "rajput",
            "iconography": ["temple", "peacock", "sunrise"],
            "colors": ["white", "saffron", "gold"],
            "time_of_day": "dawn"
        }
        
        accuracy_score = self.annotator.assess_cultural_accuracy(annotation)
        
        self.assertIsInstance(accuracy_score, float)
        self.assertGreaterEqual(accuracy_score, 0.0)
        self.assertLessEqual(accuracy_score, 1.0)
    
    def test_suggest_corrections(self):
        """Test cultural correction suggestions."""
        # Annotation with cultural inconsistencies
        inconsistent_annotation = {
            "raga": "bhairav",  # Dawn raga
            "style": "rajput",
            "iconography": ["moon", "night_scene"],  # Night iconography
            "time_of_day": "night"  # Wrong time
        }
        
        corrections = self.annotator.suggest_corrections(inconsistent_annotation)
        
        self.assertGreater(len(corrections), 0)
        self.assertTrue(any("time" in correction.lower() for correction in corrections))


class TestRagamalaDataset(unittest.TestCase):
    """Test suite for RagamalaDataset class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test dataset structure
        self.data_dir = Path(self.temp_dir) / "test_dataset"
        self.data_dir.mkdir()
        
        # Create test images and annotations
        self.create_test_dataset()
        
        self.dataset = RagamalaDataset(
            data_dir=str(self.data_dir),
            split="train",
            transform=None
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def create_test_dataset(self):
        """Create test dataset files."""
        # Create images
        for i in range(5):
            image = Image.new('RGB', (512, 512), color=(i*50, 100, 150))
            image_path = self.data_dir / f"image_{i:03d}.jpg"
            image.save(image_path)
            
            # Create corresponding annotation
            annotation = {
                "image_path": str(image_path),
                "raga": ["bhairav", "yaman", "malkauns", "darbari", "bageshri"][i],
                "style": ["rajput", "pahari", "deccan", "mughal", "rajput"][i],
                "period": "18th century",
                "prompt": f"A test painting of raga {['bhairav', 'yaman', 'malkauns', 'darbari', 'bageshri'][i]}"
            }
            
            annotation_path = self.data_dir / f"image_{i:03d}.json"
            with open(annotation_path, 'w') as f:
                json.dump(annotation, f)
        
        # Create split files
        train_split = self.data_dir / "train.txt"
        with open(train_split, 'w') as f:
            f.write("\n".join([f"image_{i:03d}" for i in range(3)]))
        
        val_split = self.data_dir / "val.txt"
        with open(val_split, 'w') as f:
            f.write("\n".join([f"image_{i:03d}" for i in range(3, 5)]))
    
    def test_dataset_initialization(self):
        """Test dataset initialization."""
        self.assertEqual(len(self.dataset), 3)  # Train split has 3 images
        self.assertEqual(self.dataset.split, "train")
    
    def test_dataset_getitem(self):
        """Test dataset item retrieval."""
        item = self.dataset[0]
        
        self.assertIn('image', item)
        self.assertIn('prompt', item)
        self.assertIn('raga', item)
        self.assertIn('style', item)
        
        # Check image format
        self.assertIsInstance(item['image'], (torch.Tensor, Image.Image))
        self.assertIsInstance(item['prompt'], str)
    
    def test_dataset_validation_split(self):
        """Test validation split loading."""
        val_dataset = RagamalaDataset(
            data_dir=str(self.data_dir),
            split="val",
            transform=None
        )
        
        self.assertEqual(len(val_dataset), 2)  # Val split has 2 images
    
    def test_dataset_with_transforms(self):
        """Test dataset with image transforms."""
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        dataset_with_transform = RagamalaDataset(
            data_dir=str(self.data_dir),
            split="train",
            transform=transform
        )
        
        item = dataset_with_transform[0]
        
        self.assertIsInstance(item['image'], torch.Tensor)
        self.assertEqual(item['image'].shape, (3, 256, 256))
    
    def test_dataset_statistics(self):
        """Test dataset statistics calculation."""
        stats = self.dataset.get_statistics()
        
        self.assertIn('total_images', stats)
        self.assertIn('raga_distribution', stats)
        self.assertIn('style_distribution', stats)
        
        self.assertEqual(stats['total_images'], 3)
        self.assertIsInstance(stats['raga_distribution'], dict)
        self.assertIsInstance(stats['style_distribution'], dict)


class TestRagamalaDataModule(unittest.TestCase):
    """Test suite for RagamalaDataModule class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test dataset
        self.data_dir = Path(self.temp_dir) / "test_dataset"
        self.data_dir.mkdir()
        self.create_test_dataset()
        
        self.data_module = RagamalaDataModule(
            data_dir=str(self.data_dir),
            batch_size=2,
            num_workers=0  # Avoid multiprocessing issues in tests
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def create_test_dataset(self):
        """Create test dataset files."""
        # Create more images for proper train/val/test splits
        for i in range(10):
            image = Image.new('RGB', (512, 512), color=(i*25, 100, 150))
            image_path = self.data_dir / f"image_{i:03d}.jpg"
            image.save(image_path)
            
            annotation = {
                "image_path": str(image_path),
                "raga": ["bhairav", "yaman", "malkauns", "darbari", "bageshri"][i % 5],
                "style": ["rajput", "pahari", "deccan", "mughal"][i % 4],
                "period": "18th century",
                "prompt": f"A test painting number {i}"
            }
            
            annotation_path = self.data_dir / f"image_{i:03d}.json"
            with open(annotation_path, 'w') as f:
                json.dump(annotation, f)
        
        # Create split files
        train_split = self.data_dir / "train.txt"
        with open(train_split, 'w') as f:
            f.write("\n".join([f"image_{i:03d}" for i in range(6)]))
        
        val_split = self.data_dir / "val.txt"
        with open(val_split, 'w') as f:
            f.write("\n".join([f"image_{i:03d}" for i in range(6, 8)]))
        
        test_split = self.data_dir / "test.txt"
        with open(test_split, 'w') as f:
            f.write("\n".join([f"image_{i:03d}" for i in range(8, 10)]))
    
    def test_data_module_setup(self):
        """Test data module setup."""
        self.data_module.setup()
        
        self.assertIsNotNone(self.data_module.train_dataset)
        self.assertIsNotNone(self.data_module.val_dataset)
        self.assertIsNotNone(self.data_module.test_dataset)
        
        self.assertEqual(len(self.data_module.train_dataset), 6)
        self.assertEqual(len(self.data_module.val_dataset), 2)
        self.assertEqual(len(self.data_module.test_dataset), 2)
    
    def test_data_loaders(self):
        """Test data loader creation."""
        self.data_module.setup()
        
        train_loader = self.data_module.train_dataloader()
        val_loader = self.data_module.val_dataloader()
        test_loader = self.data_module.test_dataloader()
        
        self.assertIsInstance(train_loader, DataLoader)
        self.assertIsInstance(val_loader, DataLoader)
        self.assertIsInstance(test_loader, DataLoader)
        
        # Test batch loading
        train_batch = next(iter(train_loader))
        self.assertEqual(len(train_batch['image']), 2)  # batch_size = 2
        self.assertEqual(len(train_batch['prompt']), 2)
    
    def test_data_module_statistics(self):
        """Test data module statistics."""
        self.data_module.setup()
        stats = self.data_module.get_statistics()
        
        self.assertIn('train_size', stats)
        self.assertIn('val_size', stats)
        self.assertIn('test_size', stats)
        self.assertIn('total_size', stats)
        
        self.assertEqual(stats['train_size'], 6)
        self.assertEqual(stats['val_size'], 2)
        self.assertEqual(stats['test_size'], 2)
        self.assertEqual(stats['total_size'], 10)


class TestDataIntegration(unittest.TestCase):
    """Integration tests for data processing pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_data_pipeline(self):
        """Test complete data processing pipeline."""
        # 1. Create collector
        collector = ImageCollector(output_dir=self.temp_dir)
        
        # 2. Create and save test image
        test_image = Image.new('RGB', (1024, 768), color='blue')
        metadata = {
            "title": "Ragamala: Raga Bhairav",
            "culture": "Rajasthani",
            "period": "18th century"
        }
        
        saved_path = collector.save_image(test_image, "test_ragamala.jpg", metadata)
        
        # 3. Process metadata
        metadata_processor = MetadataProcessor()
        processed_metadata = metadata_processor.process_metadata(metadata)
        
        # 4. Preprocess image
        preprocessor = ImagePreprocessor(target_size=(512, 512))
        processed_image = preprocessor.preprocess(test_image)
        
        # 5. Create annotation
        annotator = RagamalaAnnotator()
        annotation = annotator.create_annotation(
            image_path=str(saved_path),
            raga=processed_metadata.get('raga', 'bhairav'),
            style=processed_metadata.get('style', 'rajput'),
            additional_metadata=processed_metadata
        )
        
        # Verify pipeline results
        self.assertTrue(saved_path.exists())
        self.assertIsInstance(processed_image, torch.Tensor)
        self.assertEqual(processed_image.shape, (3, 512, 512))
        self.assertIn('raga', annotation)
        self.assertIn('style', annotation)
        self.assertIn('timestamp', annotation)
    
    def test_data_validation_pipeline(self):
        """Test data validation across components."""
        # Create test data
        test_metadata = {
            "title": "Invalid Ragamala Painting",
            "culture": "Unknown",
            "period": "Modern"
        }
        
        # Test metadata validation
        metadata_processor = MetadataProcessor()
        is_valid, missing_fields = metadata_processor.validate_metadata(test_metadata)
        
        # Test annotation validation
        annotator = RagamalaAnnotator()
        test_annotation = {
            "raga": "invalid_raga",
            "style": "unknown_style"
        }
        
        is_annotation_valid, annotation_errors = annotator.validate_annotation(test_annotation)
        
        # Verify validation results
        self.assertFalse(is_valid)
        self.assertGreater(len(missing_fields), 0)
        self.assertFalse(is_annotation_valid)
        self.assertGreater(len(annotation_errors), 0)


if __name__ == '__main__':
    # Configure test environment
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests
    
    # Run tests
    unittest.main(verbosity=2)
