"""
PyTorch Dataset Classes for Ragamala Paintings.

This module provides comprehensive dataset classes for SDXL fine-tuning on Ragamala paintings,
including support for multi-modal data (images, text, cultural metadata), data augmentation,
and cultural-aware sampling strategies.
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
import logging
from collections import defaultdict, Counter
import pickle

# PyTorch imports
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, WeightedRandomSampler
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pad_sequence

# Image processing imports
from PIL import Image, ImageOps, ImageEnhance
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Transformers imports
from transformers import CLIPTokenizer, AutoTokenizer

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logging_utils import setup_logger
from src.data.preprocessor import RagamalaImagePreprocessor, PreprocessingConfig

logger = setup_logger(__name__)

@dataclass
class DatasetConfig:
    """Configuration for dataset creation."""
    # Paths
    data_dir: str = "data/processed"
    metadata_file: str = "data/metadata/metadata.jsonl"
    splits_dir: str = "data/splits"
    
    # Image settings
    image_size: Tuple[int, int] = (1024, 1024)
    channels: int = 3
    
    # Text settings
    max_text_length: int = 77
    tokenizer_name: str = "openai/clip-vit-large-patch14"
    
    # Cultural settings
    enable_cultural_conditioning: bool = True
    raga_vocab_size: int = 50
    style_vocab_size: int = 20
    
    # Augmentation
    enable_augmentation: bool = True
    augmentation_probability: float = 0.8
    preserve_cultural_elements: bool = True
    
    # Sampling
    balance_by_raga: bool = True
    balance_by_style: bool = True
    min_samples_per_class: int = 5

class RagamalaDataset(Dataset):
    """Main dataset class for Ragamala paintings."""
    
    def __init__(self, 
                 config: DatasetConfig,
                 split: str = "train",
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        """
        Initialize Ragamala dataset.
        
        Args:
            config: Dataset configuration
            split: Dataset split ('train', 'val', 'test')
            transform: Image transformations
            target_transform: Target transformations
        """
        self.config = config
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # Initialize tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(config.tokenizer_name)
        
        # Load data
        self.metadata = self._load_metadata()
        self.image_paths = self._load_image_paths()
        self.samples = self._create_samples()
        
        # Create vocabularies
        self.raga_vocab = self._create_raga_vocabulary()
        self.style_vocab = self._create_style_vocabulary()
        
        # Setup cultural conditioning
        if config.enable_cultural_conditioning:
            self.cultural_embeddings = self._load_cultural_embeddings()
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_metadata(self) -> Dict[str, Dict]:
        """Load metadata from JSONL file."""
        metadata = {}
        
        try:
            with open(self.config.metadata_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    metadata[data['file_name']] = data
            
            logger.info(f"Loaded metadata for {len(metadata)} images")
            return metadata
            
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return {}
    
    def _load_image_paths(self) -> List[str]:
        """Load image paths for the current split."""
        split_file = Path(self.config.splits_dir) / f"{self.split}.txt"
        
        try:
            with open(split_file, 'r') as f:
                image_names = [line.strip() for line in f if line.strip()]
            
            # Convert to full paths
            image_paths = []
            for name in image_names:
                path = Path(self.config.data_dir) / name
                if path.exists():
                    image_paths.append(str(path))
                else:
                    logger.warning(f"Image not found: {path}")
            
            logger.info(f"Found {len(image_paths)} images for {self.split} split")
            return image_paths
            
        except Exception as e:
            logger.error(f"Error loading image paths: {e}")
            return []
    
    def _create_samples(self) -> List[Dict[str, Any]]:
        """Create samples with metadata."""
        samples = []
        
        for image_path in self.image_paths:
            image_name = Path(image_path).name
            
            if image_name in self.metadata:
                sample = {
                    'image_path': image_path,
                    'image_name': image_name,
                    'metadata': self.metadata[image_name]
                }
                samples.append(sample)
            else:
                logger.warning(f"No metadata found for {image_name}")
        
        return samples
    
    def _create_raga_vocabulary(self) -> Dict[str, int]:
        """Create vocabulary mapping for ragas."""
        ragas = set()
        for sample in self.samples:
            raga = sample['metadata'].get('raga', 'unknown')
            ragas.add(raga)
        
        # Sort for consistency
        raga_list = sorted(list(ragas))
        raga_vocab = {raga: idx for idx, raga in enumerate(raga_list)}
        
        logger.info(f"Created raga vocabulary with {len(raga_vocab)} entries")
        return raga_vocab
    
    def _create_style_vocabulary(self) -> Dict[str, int]:
        """Create vocabulary mapping for styles."""
        styles = set()
        for sample in self.samples:
            style = sample['metadata'].get('style', 'unknown')
            styles.add(style)
        
        # Sort for consistency
        style_list = sorted(list(styles))
        style_vocab = {style: idx for idx, style in enumerate(style_list)}
        
        logger.info(f"Created style vocabulary with {len(style_vocab)} entries")
        return style_vocab
    
    def _load_cultural_embeddings(self) -> Dict[str, torch.Tensor]:
        """Load pre-computed cultural embeddings."""
        embeddings_file = Path(self.config.data_dir).parent / "cultural_embeddings.pkl"
        
        if embeddings_file.exists():
            try:
                with open(embeddings_file, 'rb') as f:
                    embeddings = pickle.load(f)
                logger.info("Loaded cultural embeddings")
                return embeddings
            except Exception as e:
                logger.error(f"Error loading cultural embeddings: {e}")
        
        # Create default embeddings
        return self._create_default_embeddings()
    
    def _create_default_embeddings(self) -> Dict[str, torch.Tensor]:
        """Create default cultural embeddings."""
        embeddings = {}
        
        # Raga embeddings (random initialization)
        for raga in self.raga_vocab:
            embeddings[f"raga_{raga}"] = torch.randn(256)
        
        # Style embeddings
        for style in self.style_vocab:
            embeddings[f"style_{style}"] = torch.randn(128)
        
        return embeddings
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing image, text, and metadata tensors
        """
        try:
            sample = self.samples[idx]
            
            # Load and process image
            image = self._load_image(sample['image_path'])
            if self.transform:
                image = self.transform(image)
            
            # Process text
            text = sample['metadata'].get('text', '')
            text_tokens = self._tokenize_text(text)
            
            # Process cultural metadata
            raga = sample['metadata'].get('raga', 'unknown')
            style = sample['metadata'].get('style', 'unknown')
            
            raga_id = self.raga_vocab.get(raga, 0)
            style_id = self.style_vocab.get(style, 0)
            
            # Create return dictionary
            result = {
                'image': image,
                'text': text,
                'input_ids': text_tokens['input_ids'].squeeze(),
                'attention_mask': text_tokens['attention_mask'].squeeze(),
                'raga_id': torch.tensor(raga_id, dtype=torch.long),
                'style_id': torch.tensor(style_id, dtype=torch.long),
                'raga': raga,
                'style': style,
                'image_name': sample['image_name']
            }
            
            # Add cultural embeddings if enabled
            if self.config.enable_cultural_conditioning:
                raga_embedding = self.cultural_embeddings.get(f"raga_{raga}", torch.zeros(256))
                style_embedding = self.cultural_embeddings.get(f"style_{style}", torch.zeros(128))
                
                result['raga_embedding'] = raga_embedding
                result['style_embedding'] = style_embedding
            
            # Add additional metadata
            metadata = sample['metadata']
            result.update({
                'period': metadata.get('period', 'unknown'),
                'region': metadata.get('region', 'unknown'),
                'mood': metadata.get('mood', 'unknown'),
                'time_of_day': metadata.get('time_of_day', 'unknown'),
                'season': metadata.get('season', 'unknown'),
                'quality_score': torch.tensor(metadata.get('quality_score', 0.5), dtype=torch.float)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            # Return a default sample to avoid breaking training
            return self._get_default_sample()
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load and validate image."""
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Validate image size
            if image.size[0] < 256 or image.size[1] < 256:
                logger.warning(f"Small image: {image_path}")
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a default image
            return Image.new('RGB', self.config.image_size, color='white')
    
    def _tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text using CLIP tokenizer."""
        try:
            tokens = self.tokenizer(
                text,
                max_length=self.config.max_text_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return tokens
            
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            # Return default tokens
            return {
                'input_ids': torch.zeros(1, self.config.max_text_length, dtype=torch.long),
                'attention_mask': torch.zeros(1, self.config.max_text_length, dtype=torch.long)
            }
    
    def _get_default_sample(self) -> Dict[str, torch.Tensor]:
        """Get a default sample for error cases."""
        default_image = torch.zeros(3, *self.config.image_size)
        default_tokens = torch.zeros(self.config.max_text_length, dtype=torch.long)
        
        return {
            'image': default_image,
            'text': 'default ragamala painting',
            'input_ids': default_tokens,
            'attention_mask': default_tokens,
            'raga_id': torch.tensor(0, dtype=torch.long),
            'style_id': torch.tensor(0, dtype=torch.long),
            'raga': 'unknown',
            'style': 'unknown',
            'image_name': 'default.jpg',
            'quality_score': torch.tensor(0.5, dtype=torch.float)
        }
    
    def get_class_distribution(self) -> Dict[str, Dict[str, int]]:
        """Get distribution of classes in the dataset."""
        raga_counts = Counter()
        style_counts = Counter()
        
        for sample in self.samples:
            metadata = sample['metadata']
            raga_counts[metadata.get('raga', 'unknown')] += 1
            style_counts[metadata.get('style', 'unknown')] += 1
        
        return {
            'raga_distribution': dict(raga_counts),
            'style_distribution': dict(style_counts)
        }

class CulturallyBalancedSampler(Sampler):
    """Custom sampler for balanced sampling across ragas and styles."""
    
    def __init__(self, 
                 dataset: RagamalaDataset,
                 samples_per_epoch: Optional[int] = None,
                 replacement: bool = True):
        """
        Initialize culturally balanced sampler.
        
        Args:
            dataset: Ragamala dataset
            samples_per_epoch: Number of samples per epoch
            replacement: Whether to sample with replacement
        """
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch or len(dataset)
        self.replacement = replacement
        
        # Create class indices
        self.raga_indices = defaultdict(list)
        self.style_indices = defaultdict(list)
        
        for idx, sample in enumerate(dataset.samples):
            metadata = sample['metadata']
            raga = metadata.get('raga', 'unknown')
            style = metadata.get('style', 'unknown')
            
            self.raga_indices[raga].append(idx)
            self.style_indices[style].append(idx)
        
        # Calculate sampling weights
        self.weights = self._calculate_weights()
    
    def _calculate_weights(self) -> torch.Tensor:
        """Calculate sampling weights for balanced sampling."""
        weights = torch.zeros(len(self.dataset))
        
        # Calculate raga weights
        raga_counts = {raga: len(indices) for raga, indices in self.raga_indices.items()}
        max_raga_count = max(raga_counts.values())
        
        # Calculate style weights
        style_counts = {style: len(indices) for style, indices in self.style_indices.items()}
        max_style_count = max(style_counts.values())
        
        for idx, sample in enumerate(self.dataset.samples):
            metadata = sample['metadata']
            raga = metadata.get('raga', 'unknown')
            style = metadata.get('style', 'unknown')
            
            # Combine raga and style weights
            raga_weight = max_raga_count / raga_counts[raga]
            style_weight = max_style_count / style_counts[style]
            
            weights[idx] = (raga_weight + style_weight) / 2
        
        return weights
    
    def __iter__(self):
        """Generate sample indices."""
        if self.replacement:
            # Sample with replacement using weights
            indices = torch.multinomial(
                self.weights, 
                self.samples_per_epoch, 
                replacement=True
            ).tolist()
        else:
            # Sample without replacement
            indices = list(range(len(self.dataset)))
            random.shuffle(indices)
            indices = indices[:self.samples_per_epoch]
        
        return iter(indices)
    
    def __len__(self):
        """Return number of samples per epoch."""
        return self.samples_per_epoch

class RagamalaCollateFunction:
    """Custom collate function for batching Ragamala samples."""
    
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for creating batches.
        
        Args:
            batch: List of samples
            
        Returns:
            Batched tensors
        """
        # Stack images
        images = torch.stack([item['image'] for item in batch])
        
        # Pad text sequences
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_masks = torch.stack([item['attention_mask'] for item in batch])
        
        # Stack categorical data
        raga_ids = torch.stack([item['raga_id'] for item in batch])
        style_ids = torch.stack([item['style_id'] for item in batch])
        quality_scores = torch.stack([item['quality_score'] for item in batch])
        
        # Collect text and metadata
        texts = [item['text'] for item in batch]
        ragas = [item['raga'] for item in batch]
        styles = [item['style'] for item in batch]
        image_names = [item['image_name'] for item in batch]
        
        result = {
            'images': images,
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'raga_ids': raga_ids,
            'style_ids': style_ids,
            'quality_scores': quality_scores,
            'texts': texts,
            'ragas': ragas,
            'styles': styles,
            'image_names': image_names
        }
        
        # Add cultural embeddings if present
        if 'raga_embedding' in batch[0]:
            raga_embeddings = torch.stack([item['raga_embedding'] for item in batch])
            style_embeddings = torch.stack([item['style_embedding'] for item in batch])
            result['raga_embeddings'] = raga_embeddings
            result['style_embeddings'] = style_embeddings
        
        return result

class RagamalaDataModule:
    """Data module for managing Ragamala datasets and dataloaders."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Setup transforms
        self.train_transform = self._create_train_transform()
        self.val_transform = self._create_val_transform()
        
        # Setup collate function
        self.collate_fn = RagamalaCollateFunction()
    
    def _create_train_transform(self) -> Callable:
        """Create training transforms."""
        if self.config.enable_augmentation:
            transform = A.Compose([
                A.Resize(self.config.image_size[0], self.config.image_size[1]),
                A.HorizontalFlip(p=0.3),
                A.Rotate(limit=5, p=0.3),
                A.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.05,
                    p=0.4
                ),
                A.GaussNoise(var_limit=(5, 15), p=0.2),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2()
            ])
        else:
            transform = A.Compose([
                A.Resize(self.config.image_size[0], self.config.image_size[1]),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2()
            ])
        
        return lambda image: transform(image=np.array(image))['image']
    
    def _create_val_transform(self) -> Callable:
        """Create validation transforms."""
        transform = A.Compose([
            A.Resize(self.config.image_size[0], self.config.image_size[1]),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ])
        
        return lambda image: transform(image=np.array(image))['image']
    
    def setup(self):
        """Setup datasets."""
        # Create datasets
        self.train_dataset = RagamalaDataset(
            config=self.config,
            split='train',
            transform=self.train_transform
        )
        
        self.val_dataset = RagamalaDataset(
            config=self.config,
            split='val',
            transform=self.val_transform
        )
        
        self.test_dataset = RagamalaDataset(
            config=self.config,
            split='test',
            transform=self.val_transform
        )
        
        logger.info("Datasets created successfully")
    
    def train_dataloader(self, 
                        batch_size: int = 4,
                        num_workers: int = 4,
                        shuffle: bool = True,
                        use_balanced_sampler: bool = True) -> DataLoader:
        """Create training dataloader."""
        if use_balanced_sampler:
            sampler = CulturallyBalancedSampler(self.train_dataset)
            shuffle = False  # Don't shuffle when using custom sampler
        else:
            sampler = None
        
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self, 
                      batch_size: int = 4,
                      num_workers: int = 4) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True
        )
    
    def test_dataloader(self, 
                       batch_size: int = 4,
                       num_workers: int = 4) -> DataLoader:
        """Create test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True
        )
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics."""
        if not self.train_dataset:
            self.setup()
        
        train_dist = self.train_dataset.get_class_distribution()
        val_dist = self.val_dataset.get_class_distribution()
        
        return {
            'train_size': len(self.train_dataset),
            'val_size': len(self.val_dataset),
            'test_size': len(self.test_dataset),
            'train_distribution': train_dist,
            'val_distribution': val_dist,
            'raga_vocab_size': len(self.train_dataset.raga_vocab),
            'style_vocab_size': len(self.train_dataset.style_vocab),
            'image_size': self.config.image_size,
            'max_text_length': self.config.max_text_length
        }

def create_ragamala_datasets(config: DatasetConfig) -> RagamalaDataModule:
    """
    Factory function to create Ragamala datasets.
    
    Args:
        config: Dataset configuration
        
    Returns:
        Configured data module
    """
    data_module = RagamalaDataModule(config)
    data_module.setup()
    return data_module

def main():
    """Main function for testing dataset functionality."""
    # Create configuration
    config = DatasetConfig(
        data_dir="data/processed",
        metadata_file="data/metadata/metadata.jsonl",
        splits_dir="data/splits"
    )
    
    # Create data module
    data_module = create_ragamala_datasets(config)
    
    # Get statistics
    stats = data_module.get_dataset_statistics()
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test dataloader
    train_loader = data_module.train_dataloader(batch_size=2, num_workers=0)
    
    print(f"\nTesting dataloader with {len(train_loader)} batches...")
    for i, batch in enumerate(train_loader):
        print(f"Batch {i}:")
        print(f"  Images shape: {batch['images'].shape}")
        print(f"  Input IDs shape: {batch['input_ids'].shape}")
        print(f"  Ragas: {batch['ragas']}")
        print(f"  Styles: {batch['styles']}")
        
        if i >= 2:  # Test first 3 batches
            break
    
    print("Dataset testing completed successfully!")

if __name__ == "__main__":
    main()
