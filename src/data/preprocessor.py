"""
Image Preprocessing Module for Ragamala Paintings.

This module provides comprehensive image preprocessing functionality for SDXL fine-tuning
on Ragamala paintings, including resizing, normalization, augmentation, and cultural
preservation techniques.
"""

import os
import sys
import cv2
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from skimage import exposure, filters, morphology, segmentation
from skimage.feature import local_binary_pattern
from skimage.color import rgb2hsv, hsv2rgb, rgb2lab, lab2rgb
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logging_utils import setup_logger
from src.utils.visualization import plot_image_grid, save_comparison_plot

logger = setup_logger(__name__)

@dataclass
class PreprocessingConfig:
    """Configuration for image preprocessing."""
    target_size: Tuple[int, int] = (1024, 1024)
    maintain_aspect_ratio: bool = True
    padding_color: Tuple[int, int, int] = (255, 255, 255)
    
    # Normalization
    normalize: bool = True
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    
    # Quality filtering
    min_resolution: Tuple[int, int] = (512, 512)
    max_resolution: Tuple[int, int] = (4096, 4096)
    min_quality_score: float = 0.5
    
    # Augmentation
    enable_augmentation: bool = True
    augmentation_probability: float = 0.8
    preserve_cultural_elements: bool = True
    
    # Color correction
    enable_color_correction: bool = True
    histogram_equalization: bool = False
    gamma_correction: float = 1.0
    
    # Noise reduction
    enable_denoising: bool = True
    denoise_strength: float = 0.1
    
    # Output format
    output_format: str = "PNG"
    output_quality: int = 95

@dataclass
class ImageMetrics:
    """Metrics for image quality assessment."""
    resolution: Tuple[int, int]
    aspect_ratio: float
    file_size: int
    color_variance: float
    brightness: float
    contrast: float
    sharpness: float
    noise_level: float
    cultural_elements_score: float
    overall_quality: float

class CulturalElementDetector:
    """Detector for preserving cultural elements during preprocessing."""
    
    def __init__(self):
        self.color_palettes = {
            'rajput': [(255, 0, 0), (255, 215, 0), (255, 255, 255), (0, 128, 0)],
            'pahari': [(135, 206, 235), (0, 128, 0), (255, 192, 203), (255, 255, 255)],
            'deccan': [(0, 0, 139), (128, 0, 128), (255, 215, 0), (255, 255, 255)],
            'mughal': [(255, 215, 0), (255, 0, 0), (0, 128, 0), (128, 0, 128)]
        }
        
        self.iconographic_patterns = {
            'geometric': ['circles', 'squares', 'triangles', 'hexagons'],
            'floral': ['lotus', 'jasmine', 'roses', 'vines'],
            'architectural': ['arches', 'pillars', 'domes', 'minarets'],
            'figures': ['human_silhouettes', 'animal_shapes']
        }
    
    def detect_style_elements(self, image: np.ndarray, style: str) -> Dict[str, float]:
        """Detect style-specific elements in the image."""
        results = {}
        
        # Color palette analysis
        results['color_palette_match'] = self._analyze_color_palette(image, style)
        
        # Geometric pattern detection
        results['geometric_patterns'] = self._detect_geometric_patterns(image)
        
        # Edge density (architectural elements)
        results['architectural_elements'] = self._detect_architectural_elements(image)
        
        # Texture analysis
        results['texture_complexity'] = self._analyze_texture_complexity(image)
        
        return results
    
    def _analyze_color_palette(self, image: np.ndarray, style: str) -> float:
        """Analyze color palette match with traditional style."""
        if style not in self.color_palettes:
            return 0.5
        
        # Extract dominant colors
        pixels = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(pixels)
        dominant_colors = kmeans.cluster_centers_
        
        # Calculate similarity to traditional palette
        traditional_colors = np.array(self.color_palettes[style])
        similarity_scores = []
        
        for dom_color in dominant_colors:
            distances = np.linalg.norm(traditional_colors - dom_color, axis=1)
            min_distance = np.min(distances)
            similarity = max(0, 1 - min_distance / 255.0)
            similarity_scores.append(similarity)
        
        return np.mean(similarity_scores)
    
    def _detect_geometric_patterns(self, image: np.ndarray) -> float:
        """Detect geometric patterns using edge detection and Hough transforms."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        line_score = len(lines) / 1000.0 if lines is not None else 0
        
        # Detect circles
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                  param1=50, param2=30, minRadius=10, maxRadius=100)
        circle_score = len(circles[0]) / 50.0 if circles is not None else 0
        
        return min(1.0, (line_score + circle_score) / 2)
    
    def _detect_architectural_elements(self, image: np.ndarray) -> float:
        """Detect architectural elements like arches, pillars."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Vertical line detection (pillars)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
        vertical_lines = 0
        if lines is not None:
            for rho, theta in lines[:, 0]:
                if abs(theta - np.pi/2) < 0.1:  # Nearly vertical
                    vertical_lines += 1
        
        vertical_score = min(1.0, vertical_lines / 20.0)
        
        return (edge_density * 2 + vertical_score) / 3
    
    def _analyze_texture_complexity(self, image: np.ndarray) -> float:
        """Analyze texture complexity using Local Binary Patterns."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Local Binary Pattern
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        
        # Calculate texture complexity
        hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)
        
        # Entropy as complexity measure
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        normalized_entropy = entropy / np.log2(n_points + 2)
        
        return normalized_entropy

class ImageQualityAssessor:
    """Assess image quality for filtering low-quality images."""
    
    def __init__(self):
        self.cultural_detector = CulturalElementDetector()
    
    def assess_quality(self, image: np.ndarray, metadata: Dict = None) -> ImageMetrics:
        """Comprehensive image quality assessment."""
        h, w = image.shape[:2]
        
        # Basic metrics
        resolution = (w, h)
        aspect_ratio = w / h
        file_size = image.nbytes
        
        # Color analysis
        color_variance = self._calculate_color_variance(image)
        brightness = self._calculate_brightness(image)
        contrast = self._calculate_contrast(image)
        
        # Sharpness and noise
        sharpness = self._calculate_sharpness(image)
        noise_level = self._calculate_noise_level(image)
        
        # Cultural elements
        style = metadata.get('style', 'unknown') if metadata else 'unknown'
        cultural_elements = self.cultural_detector.detect_style_elements(image, style)
        cultural_score = np.mean(list(cultural_elements.values()))
        
        # Overall quality score
        quality_components = {
            'resolution': self._score_resolution(resolution),
            'aspect_ratio': self._score_aspect_ratio(aspect_ratio),
            'color_variance': min(1.0, color_variance / 50.0),
            'brightness': self._score_brightness(brightness),
            'contrast': min(1.0, contrast / 100.0),
            'sharpness': min(1.0, sharpness / 100.0),
            'noise': 1.0 - min(1.0, noise_level / 50.0),
            'cultural': cultural_score
        }
        
        overall_quality = np.mean(list(quality_components.values()))
        
        return ImageMetrics(
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            file_size=file_size,
            color_variance=color_variance,
            brightness=brightness,
            contrast=contrast,
            sharpness=sharpness,
            noise_level=noise_level,
            cultural_elements_score=cultural_score,
            overall_quality=overall_quality
        )
    
    def _calculate_color_variance(self, image: np.ndarray) -> float:
        """Calculate color variance across channels."""
        return np.mean([np.var(image[:, :, i]) for i in range(3)])
    
    def _calculate_brightness(self, image: np.ndarray) -> float:
        """Calculate average brightness."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return np.mean(gray)
    
    def _calculate_contrast(self, image: np.ndarray) -> float:
        """Calculate RMS contrast."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return np.std(gray)
    
    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """Calculate sharpness using Laplacian variance."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return np.var(laplacian)
    
    def _calculate_noise_level(self, image: np.ndarray) -> float:
        """Estimate noise level using high-frequency components."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur and calculate difference
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray.astype(float) - blurred.astype(float)
        
        return np.std(noise)
    
    def _score_resolution(self, resolution: Tuple[int, int]) -> float:
        """Score resolution quality."""
        w, h = resolution
        min_dim = min(w, h)
        
        if min_dim >= 1024:
            return 1.0
        elif min_dim >= 512:
            return 0.8
        elif min_dim >= 256:
            return 0.5
        else:
            return 0.2
    
    def _score_aspect_ratio(self, aspect_ratio: float) -> float:
        """Score aspect ratio (prefer square or traditional ratios)."""
        ideal_ratios = [1.0, 4/3, 3/2, 16/9]
        distances = [abs(aspect_ratio - ratio) for ratio in ideal_ratios]
        min_distance = min(distances)
        
        return max(0, 1 - min_distance)
    
    def _score_brightness(self, brightness: float) -> float:
        """Score brightness (prefer moderate values)."""
        ideal_brightness = 128
        distance = abs(brightness - ideal_brightness) / 128
        return max(0, 1 - distance)

class RagamalaImagePreprocessor:
    """Main preprocessing class for Ragamala paintings."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.quality_assessor = ImageQualityAssessor()
        self.cultural_detector = CulturalElementDetector()
        
        # Setup augmentation pipeline
        self.augmentation_pipeline = self._create_augmentation_pipeline()
        self.validation_pipeline = self._create_validation_pipeline()
        
    def _create_augmentation_pipeline(self) -> A.Compose:
        """Create culturally-aware augmentation pipeline."""
        transforms = []
        
        if self.config.enable_augmentation:
            # Geometric transformations (preserve cultural elements)
            transforms.extend([
                A.HorizontalFlip(p=0.3),  # Reduced probability for cultural preservation
                A.Rotate(limit=5, p=0.3),  # Small rotations only
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=3,
                    p=0.3
                ),
            ])
            
            # Color transformations (preserve traditional palettes)
            transforms.extend([
                A.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.05,
                    p=0.4
                ),
                A.RandomGamma(gamma_limit=(0.9, 1.1), p=0.3),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.3),
            ])
            
            # Noise and blur (minimal to preserve details)
            transforms.extend([
                A.GaussNoise(var_limit=(5, 15), p=0.2),
                A.OneOf([
                    A.MotionBlur(blur_limit=3, p=0.5),
                    A.MedianBlur(blur_limit=3, p=0.3),
                    A.GaussianBlur(blur_limit=3, p=0.2),
                ], p=0.2),
            ])
        
        # Always apply resizing and normalization
        transforms.extend([
            A.Resize(
                height=self.config.target_size[1],
                width=self.config.target_size[0],
                interpolation=cv2.INTER_LANCZOS4
            ),
            A.Normalize(
                mean=self.config.mean,
                std=self.config.std,
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])
        
        return A.Compose(transforms)
    
    def _create_validation_pipeline(self) -> A.Compose:
        """Create validation pipeline without augmentation."""
        return A.Compose([
            A.Resize(
                height=self.config.target_size[1],
                width=self.config.target_size[0],
                interpolation=cv2.INTER_LANCZOS4
            ),
            A.Normalize(
                mean=self.config.mean,
                std=self.config.std,
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])
    
    def preprocess_image(self, 
                        image_path: Union[str, Path], 
                        metadata: Dict = None,
                        is_training: bool = True) -> Dict[str, Any]:
        """Preprocess a single image."""
        try:
            # Load image
            image = self._load_image(image_path)
            if image is None:
                return None
            
            # Quality assessment
            quality_metrics = self.quality_assessor.assess_quality(image, metadata)
            
            # Filter low-quality images
            if quality_metrics.overall_quality < self.config.min_quality_score:
                logger.warning(f"Low quality image filtered: {image_path}")
                return None
            
            # Color correction
            if self.config.enable_color_correction:
                image = self._apply_color_correction(image, metadata)
            
            # Denoising
            if self.config.enable_denoising:
                image = self._apply_denoising(image)
            
            # Cultural element preservation
            if self.config.preserve_cultural_elements and metadata:
                image = self._preserve_cultural_elements(image, metadata)
            
            # Apply augmentation pipeline
            pipeline = self.augmentation_pipeline if is_training else self.validation_pipeline
            transformed = pipeline(image=image)
            
            return {
                'image': transformed['image'],
                'original_image': image,
                'quality_metrics': quality_metrics,
                'metadata': metadata,
                'preprocessing_applied': True
            }
            
        except Exception as e:
            logger.error(f"Error preprocessing {image_path}: {e}")
            return None
    
    def _load_image(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """Load and validate image."""
        try:
            # Load with PIL for better format support
            pil_image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to numpy array
            image = np.array(pil_image)
            
            # Validate resolution
            h, w = image.shape[:2]
            if (w < self.config.min_resolution[0] or 
                h < self.config.min_resolution[1]):
                logger.warning(f"Image too small: {w}x{h}")
                return None
            
            if (w > self.config.max_resolution[0] or 
                h > self.config.max_resolution[1]):
                logger.warning(f"Image too large: {w}x{h}")
                return None
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def _apply_color_correction(self, image: np.ndarray, metadata: Dict = None) -> np.ndarray:
        """Apply color correction based on painting style."""
        corrected = image.copy()
        
        # Histogram equalization
        if self.config.histogram_equalization:
            # Convert to LAB for better color preservation
            lab = cv2.cvtColor(corrected, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
            corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Gamma correction
        if self.config.gamma_correction != 1.0:
            corrected = exposure.adjust_gamma(corrected, self.config.gamma_correction)
        
        # Style-specific color enhancement
        if metadata and 'style' in metadata:
            corrected = self._apply_style_color_enhancement(corrected, metadata['style'])
        
        return corrected
    
    def _apply_style_color_enhancement(self, image: np.ndarray, style: str) -> np.ndarray:
        """Apply style-specific color enhancement."""
        enhanced = image.copy()
        
        if style == 'rajput':
            # Enhance reds and golds
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
            # Enhance red hues (0-10 and 170-180)
            mask1 = (hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 170)
            hsv[mask1, 1] = np.clip(hsv[mask1, 1] * 1.1, 0, 255)
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
        elif style == 'pahari':
            # Enhance blues and greens
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
            # Enhance blue-green hues (80-140)
            mask = (hsv[:, :, 0] >= 80) & (hsv[:, :, 0] <= 140)
            hsv[mask, 1] = np.clip(hsv[mask, 1] * 1.1, 0, 255)
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
        elif style == 'deccan':
            # Enhance deep blues and purples
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
            # Enhance blue-purple hues (120-160)
            mask = (hsv[:, :, 0] >= 120) & (hsv[:, :, 0] <= 160)
            hsv[mask, 1] = np.clip(hsv[mask, 1] * 1.15, 0, 255)
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
        elif style == 'mughal':
            # Enhance overall richness
            enhanced = exposure.adjust_gamma(enhanced, 0.95)  # Slight darkening
            # Increase saturation slightly
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.05, 0, 255)
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return enhanced
    
    def _apply_denoising(self, image: np.ndarray) -> np.ndarray:
        """Apply noise reduction while preserving details."""
        if self.config.denoise_strength <= 0:
            return image
        
        # Use Non-local Means denoising
        denoised = cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=self.config.denoise_strength * 10,
            hColor=self.config.denoise_strength * 10,
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        return denoised
    
    def _preserve_cultural_elements(self, image: np.ndarray, metadata: Dict) -> np.ndarray:
        """Preserve important cultural elements during preprocessing."""
        # This is a placeholder for advanced cultural preservation
        # In practice, this could involve:
        # 1. Detecting important cultural motifs
        # 2. Applying selective processing
        # 3. Preserving color palettes in specific regions
        
        style = metadata.get('style', '')
        raga = metadata.get('raga', '')
        
        # For now, return the image unchanged
        # Future implementations could include:
        # - Selective sharpening of architectural elements
        # - Color palette preservation in specific regions
        # - Protection of iconographic elements
        
        return image
    
    def process_dataset(self, 
                       input_dir: Union[str, Path],
                       output_dir: Union[str, Path],
                       metadata_file: Union[str, Path],
                       split: str = 'train') -> Dict[str, Any]:
        """Process entire dataset."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata_list = [json.loads(line) for line in f]
        
        metadata_dict = {item['file_name']: item for item in metadata_list}
        
        processed_count = 0
        failed_count = 0
        quality_scores = []
        
        # Process images
        for image_file in input_dir.glob('*.jpg'):
            if image_file.name not in metadata_dict:
                logger.warning(f"No metadata found for {image_file.name}")
                continue
            
            metadata = metadata_dict[image_file.name]
            
            # Preprocess image
            result = self.preprocess_image(
                image_file,
                metadata,
                is_training=(split == 'train')
            )
            
            if result is None:
                failed_count += 1
                continue
            
            # Save processed image
            output_file = output_dir / image_file.name
            self._save_processed_image(result['image'], output_file)
            
            quality_scores.append(result['quality_metrics'].overall_quality)
            processed_count += 1
            
            if processed_count % 100 == 0:
                logger.info(f"Processed {processed_count} images")
        
        # Generate processing report
        report = {
            'total_processed': processed_count,
            'total_failed': failed_count,
            'average_quality': np.mean(quality_scores) if quality_scores else 0,
            'quality_std': np.std(quality_scores) if quality_scores else 0,
            'min_quality': np.min(quality_scores) if quality_scores else 0,
            'max_quality': np.max(quality_scores) if quality_scores else 0
        }
        
        # Save report
        report_file = output_dir / f"preprocessing_report_{split}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Dataset processing complete: {report}")
        return report
    
    def _save_processed_image(self, tensor_image: torch.Tensor, output_path: Path):
        """Save processed tensor image."""
        # Denormalize
        image = tensor_image.clone()
        for i in range(3):
            image[i] = image[i] * self.config.std[i] + self.config.mean[i]
        
        # Convert to PIL and save
        image = torch.clamp(image, 0, 1)
        image = (image * 255).byte()
        image_np = image.permute(1, 2, 0).numpy()
        
        pil_image = Image.fromarray(image_np)
        pil_image.save(output_path, format=self.config.output_format, quality=self.config.output_quality)
    
    def generate_quality_report(self, 
                               image_dir: Union[str, Path],
                               metadata_file: Union[str, Path],
                               output_file: Union[str, Path]):
        """Generate comprehensive quality assessment report."""
        image_dir = Path(image_dir)
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata_list = [json.loads(line) for line in f]
        
        metadata_dict = {item['file_name']: item for item in metadata_list}
        
        quality_data = []
        
        for image_file in image_dir.glob('*.jpg'):
            if image_file.name not in metadata_dict:
                continue
            
            image = self._load_image(image_file)
            if image is None:
                continue
            
            metadata = metadata_dict[image_file.name]
            quality_metrics = self.quality_assessor.assess_quality(image, metadata)
            
            quality_data.append({
                'filename': image_file.name,
                'style': metadata.get('style', ''),
                'raga': metadata.get('raga', ''),
                **asdict(quality_metrics)
            })
        
        # Save detailed report
        import pandas as pd
        df = pd.DataFrame(quality_data)
        df.to_csv(output_file, index=False)
        
        # Generate summary statistics
        summary = {
            'total_images': len(quality_data),
            'average_quality': df['overall_quality'].mean(),
            'quality_by_style': df.groupby('style')['overall_quality'].mean().to_dict(),
            'quality_by_raga': df.groupby('raga')['overall_quality'].mean().to_dict(),
            'low_quality_count': len(df[df['overall_quality'] < self.config.min_quality_score])
        }
        
        summary_file = Path(output_file).with_suffix('.summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Quality report generated: {summary}")
        return summary

def main():
    """Main function for preprocessing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess Ragamala paintings")
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--metadata_file', type=str, required=True, help='Metadata JSONL file')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--config_file', type=str, help='Preprocessing config file')
    
    args = parser.parse_args()
    
    # Load config
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config_dict = json.load(f)
        config = PreprocessingConfig(**config_dict)
    else:
        config = PreprocessingConfig()
    
    # Initialize preprocessor
    preprocessor = RagamalaImagePreprocessor(config)
    
    # Process dataset
    report = preprocessor.process_dataset(
        args.input_dir,
        args.output_dir,
        args.metadata_file,
        args.split
    )
    
    print(f"Preprocessing complete: {report}")

if __name__ == "__main__":
    main()
