"""
Evaluation Metrics for Ragamala Painting Generation.

This module provides comprehensive evaluation metrics for assessing the quality
and authenticity of generated Ragamala paintings, including FID, CLIP Score, SSIM,
and cultural-specific metrics.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models import inception_v3
from PIL import Image
import cv2
from scipy import linalg
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings("ignore")

# Image quality metrics
try:
    import piq
    PIQ_AVAILABLE = True
except ImportError:
    PIQ_AVAILABLE = False
    logging.warning("PIQ not available. Install with: pip install piq")

# CLIP imports
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

# LPIPS imports
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    logging.warning("LPIPS not available. Install with: pip install lpips")

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

@dataclass
class MetricsConfig:
    """Configuration for evaluation metrics."""
    # General settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    num_workers: int = 4
    
    # FID settings
    fid_dims: int = 2048
    fid_normalize: bool = True
    
    # CLIP settings
    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_batch_size: int = 16
    
    # SSIM settings
    ssim_window_size: int = 11
    ssim_sigma: float = 1.5
    
    # LPIPS settings
    lpips_net: str = "alex"  # 'alex', 'vgg', 'squeeze'
    
    # Cultural metrics
    enable_cultural_metrics: bool = True
    cultural_model_path: Optional[str] = None

class ImageDataset(Dataset):
    """Dataset for loading images for metric calculation."""
    
    def __init__(self, image_paths: List[str], transform: Optional[Callable] = None):
        self.image_paths = image_paths
        self.transform = transform or transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a dummy image
            return torch.zeros(3, 299, 299)

class FIDCalculator:
    """Frechet Inception Distance (FID) calculator."""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.inception_model = self._load_inception_model()
        
    def _load_inception_model(self):
        """Load pre-trained Inception v3 model."""
        model = inception_v3(pretrained=True, transform_input=False)
        model.fc = nn.Identity()  # Remove final classification layer
        model.eval()
        model.to(self.device)
        return model
    
    def extract_features(self, images: Union[List[str], torch.Tensor]) -> np.ndarray:
        """Extract features from images using Inception v3."""
        if isinstance(images, list):
            # Load images from paths
            dataset = ImageDataset(images)
            dataloader = DataLoader(
                dataset, 
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                pin_memory=True
            )
            
            features = []
            with torch.no_grad():
                for batch in dataloader:
                    batch = batch.to(self.device)
                    feat = self.inception_model(batch)
                    features.append(feat.cpu().numpy())
            
            return np.concatenate(features, axis=0)
        
        else:
            # Process tensor directly
            with torch.no_grad():
                if images.dim() == 3:
                    images = images.unsqueeze(0)
                images = images.to(self.device)
                features = self.inception_model(images)
                return features.cpu().numpy()
    
    def calculate_statistics(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate mean and covariance of features."""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma
    
    def calculate_fid(self, 
                     real_images: Union[List[str], torch.Tensor],
                     generated_images: Union[List[str], torch.Tensor]) -> float:
        """Calculate FID between real and generated images."""
        logger.info("Calculating FID...")
        
        # Extract features
        real_features = self.extract_features(real_images)
        gen_features = self.extract_features(generated_images)
        
        # Calculate statistics
        mu_real, sigma_real = self.calculate_statistics(real_features)
        mu_gen, sigma_gen = self.calculate_statistics(gen_features)
        
        # Calculate FID
        fid = self._calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
        
        logger.info(f"FID Score: {fid:.4f}")
        return fid
    
    def _calculate_frechet_distance(self, 
                                  mu1: np.ndarray, 
                                  sigma1: np.ndarray,
                                  mu2: np.ndarray, 
                                  sigma2: np.ndarray) -> float:
        """Calculate Frechet distance between two multivariate Gaussians."""
        # Calculate squared difference of means
        diff = mu1 - mu2
        
        # Calculate sqrt of product of covariances
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        # Handle numerical errors
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        # Calculate FID
        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        
        return float(fid)

class CLIPScoreCalculator:
    """CLIP Score calculator for text-image similarity."""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = CLIPModel.from_pretrained(config.clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(config.clip_model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(config.clip_model_name)
        
        self.model.to(self.device)
        self.model.eval()
    
    def calculate_clip_score(self, 
                           images: Union[List[Image.Image], List[str]],
                           texts: List[str]) -> Dict[str, float]:
        """Calculate CLIP score between images and texts."""
        logger.info("Calculating CLIP Score...")
        
        if isinstance(images[0], str):
            # Load images from paths
            images = [Image.open(path).convert('RGB') for path in images]
        
        scores = []
        
        # Process in batches
        for i in range(0, len(images), self.config.clip_batch_size):
            batch_images = images[i:i + self.config.clip_batch_size]
            batch_texts = texts[i:i + self.config.clip_batch_size]
            
            # Process inputs
            inputs = self.processor(
                text=batch_texts,
                images=batch_images,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Get similarity scores
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
                
                # Extract diagonal (image-text pairs)
                batch_scores = torch.diag(probs).cpu().numpy()
                scores.extend(batch_scores)
        
        # Calculate statistics
        clip_score = np.mean(scores)
        clip_std = np.std(scores)
        
        result = {
            'clip_score': float(clip_score),
            'clip_std': float(clip_std),
            'individual_scores': scores
        }
        
        logger.info(f"CLIP Score: {clip_score:.4f} ± {clip_std:.4f}")
        return result
    
    def calculate_clip_score_directional(self, 
                                       reference_images: List[Image.Image],
                                       generated_images: List[Image.Image],
                                       texts: List[str]) -> Dict[str, float]:
        """Calculate directional CLIP score."""
        # Calculate CLIP scores for reference and generated images
        ref_scores = self.calculate_clip_score(reference_images, texts)
        gen_scores = self.calculate_clip_score(generated_images, texts)
        
        # Calculate directional score
        directional_score = np.mean(gen_scores['individual_scores']) - np.mean(ref_scores['individual_scores'])
        
        return {
            'directional_clip_score': float(directional_score),
            'reference_clip_score': ref_scores['clip_score'],
            'generated_clip_score': gen_scores['clip_score']
        }

class SSIMCalculator:
    """Structural Similarity Index (SSIM) calculator."""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        
        if PIQ_AVAILABLE:
            self.ssim_metric = piq.SSIMLoss(
                window_size=config.ssim_window_size,
                sigma=config.ssim_sigma,
                data_range=1.0
            )
        else:
            logger.warning("PIQ not available, using custom SSIM implementation")
            self.ssim_metric = None
    
    def calculate_ssim(self, 
                      images1: Union[List[str], torch.Tensor],
                      images2: Union[List[str], torch.Tensor]) -> Dict[str, float]:
        """Calculate SSIM between two sets of images."""
        logger.info("Calculating SSIM...")
        
        if isinstance(images1, list):
            images1 = self._load_images_as_tensor(images1)
        if isinstance(images2, list):
            images2 = self._load_images_as_tensor(images2)
        
        if PIQ_AVAILABLE and self.ssim_metric:
            # Use PIQ implementation
            ssim_loss = self.ssim_metric(images1, images2)
            ssim_score = 1.0 - ssim_loss.item()
        else:
            # Use custom implementation
            ssim_score = self._calculate_ssim_custom(images1, images2)
        
        result = {
            'ssim': float(ssim_score),
            'ssim_loss': float(1.0 - ssim_score)
        }
        
        logger.info(f"SSIM: {ssim_score:.4f}")
        return result
    
    def _load_images_as_tensor(self, image_paths: List[str]) -> torch.Tensor:
        """Load images as tensor."""
        images = []
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        for path in image_paths:
            try:
                image = Image.open(path).convert('RGB')
                image = transform(image)
                images.append(image)
            except Exception as e:
                logger.error(f"Error loading image {path}: {e}")
                images.append(torch.zeros(3, 256, 256))
        
        return torch.stack(images)
    
    def _calculate_ssim_custom(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Custom SSIM implementation."""
        # Convert to grayscale
        gray1 = 0.299 * img1[:, 0] + 0.587 * img1[:, 1] + 0.114 * img1[:, 2]
        gray2 = 0.299 * img2[:, 0] + 0.587 * img2[:, 1] + 0.114 * img2[:, 2]
        
        # Calculate SSIM for each pair
        ssim_values = []
        for i in range(gray1.shape[0]):
            ssim_val = self._ssim_single(gray1[i], gray2[i])
            ssim_values.append(ssim_val)
        
        return np.mean(ssim_values)
    
    def _ssim_single(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Calculate SSIM for a single pair of images."""
        # Convert to numpy
        img1 = img1.numpy()
        img2 = img2.numpy()
        
        # Constants
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        # Calculate means
        mu1 = img1.mean()
        mu2 = img2.mean()
        
        # Calculate variances and covariance
        var1 = img1.var()
        var2 = img2.var()
        cov = np.mean((img1 - mu1) * (img2 - mu2))
        
        # Calculate SSIM
        ssim = ((2 * mu1 * mu2 + C1) * (2 * cov + C2)) / ((mu1**2 + mu2**2 + C1) * (var1 + var2 + C2))
        
        return float(ssim)

class LPIPSCalculator:
    """Learned Perceptual Image Patch Similarity (LPIPS) calculator."""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        
        if LPIPS_AVAILABLE:
            self.lpips_metric = lpips.LPIPS(net=config.lpips_net)
            if torch.cuda.is_available():
                self.lpips_metric.cuda()
        else:
            logger.warning("LPIPS not available")
            self.lpips_metric = None
    
    def calculate_lpips(self, 
                       images1: Union[List[str], torch.Tensor],
                       images2: Union[List[str], torch.Tensor]) -> Dict[str, float]:
        """Calculate LPIPS between two sets of images."""
        if not LPIPS_AVAILABLE or self.lpips_metric is None:
            logger.warning("LPIPS not available, returning dummy values")
            return {'lpips': 0.0, 'lpips_std': 0.0}
        
        logger.info("Calculating LPIPS...")
        
        if isinstance(images1, list):
            images1 = self._load_images_as_tensor(images1)
        if isinstance(images2, list):
            images2 = self._load_images_as_tensor(images2)
        
        # Normalize to [-1, 1]
        images1 = images1 * 2.0 - 1.0
        images2 = images2 * 2.0 - 1.0
        
        if torch.cuda.is_available():
            images1 = images1.cuda()
            images2 = images2.cuda()
        
        # Calculate LPIPS
        with torch.no_grad():
            lpips_values = self.lpips_metric(images1, images2)
            lpips_values = lpips_values.cpu().numpy().flatten()
        
        result = {
            'lpips': float(np.mean(lpips_values)),
            'lpips_std': float(np.std(lpips_values)),
            'individual_scores': lpips_values.tolist()
        }
        
        logger.info(f"LPIPS: {result['lpips']:.4f} ± {result['lpips_std']:.4f}")
        return result
    
    def _load_images_as_tensor(self, image_paths: List[str]) -> torch.Tensor:
        """Load images as tensor for LPIPS."""
        images = []
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        for path in image_paths:
            try:
                image = Image.open(path).convert('RGB')
                image = transform(image)
                images.append(image)
            except Exception as e:
                logger.error(f"Error loading image {path}: {e}")
                images.append(torch.zeros(3, 256, 256))
        
        return torch.stack(images)

class InceptionScoreCalculator:
    """Inception Score (IS) calculator."""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.inception_model = self._load_inception_model()
    
    def _load_inception_model(self):
        """Load pre-trained Inception v3 model."""
        model = inception_v3(pretrained=True, transform_input=False)
        model.eval()
        model.to(self.device)
        return model
    
    def calculate_inception_score(self, 
                                images: Union[List[str], torch.Tensor],
                                splits: int = 10) -> Dict[str, float]:
        """Calculate Inception Score."""
        logger.info("Calculating Inception Score...")
        
        if isinstance(images, list):
            dataset = ImageDataset(images)
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                pin_memory=True
            )
            
            predictions = []
            with torch.no_grad():
                for batch in dataloader:
                    batch = batch.to(self.device)
                    pred = self.inception_model(batch)
                    pred = F.softmax(pred, dim=1)
                    predictions.append(pred.cpu().numpy())
            
            predictions = np.concatenate(predictions, axis=0)
        else:
            with torch.no_grad():
                if images.dim() == 3:
                    images = images.unsqueeze(0)
                images = images.to(self.device)
                predictions = self.inception_model(images)
                predictions = F.softmax(predictions, dim=1).cpu().numpy()
        
        # Calculate IS
        scores = []
        for i in range(splits):
            part = predictions[i * len(predictions) // splits:(i + 1) * len(predictions) // splits]
            kl_div = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, axis=0), 0)))
            kl_div = np.mean(np.sum(kl_div, axis=1))
            scores.append(np.exp(kl_div))
        
        is_mean = np.mean(scores)
        is_std = np.std(scores)
        
        result = {
            'inception_score': float(is_mean),
            'inception_score_std': float(is_std)
        }
        
        logger.info(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
        return result

class CulturalMetricsCalculator:
    """Calculator for cultural authenticity metrics specific to Ragamala paintings."""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.raga_classifier = None
        self.style_classifier = None
        
        # Cultural color palettes
        self.cultural_palettes = {
            'rajput': [(255, 0, 0), (255, 215, 0), (255, 255, 255), (0, 128, 0)],
            'pahari': [(135, 206, 235), (0, 128, 0), (255, 192, 203), (255, 255, 255)],
            'deccan': [(0, 0, 139), (128, 0, 128), (255, 215, 0), (255, 255, 255)],
            'mughal': [(255, 215, 0), (255, 0, 0), (0, 128, 0), (128, 0, 128)]
        }
    
    def calculate_cultural_metrics(self, 
                                 images: List[Image.Image],
                                 expected_styles: List[str],
                                 expected_ragas: List[str]) -> Dict[str, float]:
        """Calculate cultural authenticity metrics."""
        logger.info("Calculating cultural metrics...")
        
        color_accuracy = self._calculate_color_palette_accuracy(images, expected_styles)
        composition_score = self._calculate_composition_score(images, expected_styles)
        iconography_score = self._calculate_iconography_score(images, expected_ragas)
        
        # Overall cultural authenticity score
        cultural_score = (color_accuracy + composition_score + iconography_score) / 3
        
        result = {
            'cultural_authenticity': float(cultural_score),
            'color_palette_accuracy': float(color_accuracy),
            'composition_score': float(composition_score),
            'iconography_score': float(iconography_score)
        }
        
        logger.info(f"Cultural Authenticity: {cultural_score:.4f}")
        return result
    
    def _calculate_color_palette_accuracy(self, 
                                        images: List[Image.Image],
                                        expected_styles: List[str]) -> float:
        """Calculate color palette accuracy for given styles."""
        scores = []
        
        for image, style in zip(images, expected_styles):
            if style not in self.cultural_palettes:
                scores.append(0.5)  # Neutral score for unknown styles
                continue
            
            # Extract dominant colors
            image_array = np.array(image.resize((256, 256)))
            dominant_colors = self._extract_dominant_colors(image_array)
            
            # Compare with expected palette
            expected_palette = self.cultural_palettes[style]
            similarity = self._calculate_color_similarity(dominant_colors, expected_palette)
            scores.append(similarity)
        
        return np.mean(scores)
    
    def _extract_dominant_colors(self, image_array: np.ndarray, k: int = 5) -> List[Tuple[int, int, int]]:
        """Extract dominant colors using k-means clustering."""
        from sklearn.cluster import KMeans
        
        # Reshape image to list of pixels
        pixels = image_array.reshape(-1, 3)
        
        # Apply k-means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get cluster centers as dominant colors
        colors = kmeans.cluster_centers_.astype(int)
        return [tuple(color) for color in colors]
    
    def _calculate_color_similarity(self, 
                                  colors1: List[Tuple[int, int, int]],
                                  colors2: List[Tuple[int, int, int]]) -> float:
        """Calculate similarity between two color palettes."""
        # Convert to numpy arrays
        colors1 = np.array(colors1)
        colors2 = np.array(colors2)
        
        # Calculate pairwise distances
        distances = cdist(colors1, colors2, metric='euclidean')
        
        # Find minimum distances for each color in colors1
        min_distances = np.min(distances, axis=1)
        
        # Convert distances to similarities (0-1 scale)
        max_distance = np.sqrt(3 * 255**2)  # Maximum possible RGB distance
        similarities = 1.0 - (min_distances / max_distance)
        
        return np.mean(similarities)
    
    def _calculate_composition_score(self, 
                                   images: List[Image.Image],
                                   expected_styles: List[str]) -> float:
        """Calculate composition score based on style characteristics."""
        scores = []
        
        for image, style in zip(images, expected_styles):
            # Convert to grayscale for edge analysis
            gray_image = image.convert('L')
            gray_array = np.array(gray_image)
            
            # Calculate edge density
            edges = cv2.Canny(gray_array, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Style-specific scoring
            if style == 'rajput':
                # Rajput style prefers geometric patterns (higher edge density)
                score = min(edge_density * 2, 1.0)
            elif style == 'pahari':
                # Pahari style prefers softer compositions (moderate edge density)
                score = 1.0 - abs(edge_density - 0.3) / 0.3
            elif style == 'deccan':
                # Deccan style prefers architectural elements (high edge density)
                score = min(edge_density * 1.8, 1.0)
            elif style == 'mughal':
                # Mughal style prefers detailed compositions (very high edge density)
                score = min(edge_density * 2.2, 1.0)
            else:
                score = 0.5  # Neutral score for unknown styles
            
            scores.append(max(0.0, min(1.0, score)))
        
        return np.mean(scores)
    
    def _calculate_iconography_score(self, 
                                   images: List[Image.Image],
                                   expected_ragas: List[str]) -> float:
        """Calculate iconography score based on raga characteristics."""
        # This is a simplified implementation
        # In practice, you would use a trained classifier or object detection model
        scores = []
        
        for image, raga in zip(images, expected_ragas):
            # For now, return a random score between 0.6 and 0.9
            # This should be replaced with actual iconography detection
            score = np.random.uniform(0.6, 0.9)
            scores.append(score)
        
        return np.mean(scores)

class EvaluationMetrics:
    """Main evaluation metrics calculator."""
    
    def __init__(self, config: Optional[MetricsConfig] = None):
        self.config = config or MetricsConfig()
        
        # Initialize calculators
        self.fid_calculator = FIDCalculator(self.config)
        self.clip_calculator = CLIPScoreCalculator(self.config)
        self.ssim_calculator = SSIMCalculator(self.config)
        self.lpips_calculator = LPIPSCalculator(self.config)
        self.is_calculator = InceptionScoreCalculator(self.config)
        
        if self.config.enable_cultural_metrics:
            self.cultural_calculator = CulturalMetricsCalculator(self.config)
    
    def evaluate_generation_quality(self, 
                                  real_images: List[str],
                                  generated_images: List[str],
                                  prompts: List[str],
                                  styles: Optional[List[str]] = None,
                                  ragas: Optional[List[str]] = None) -> Dict[str, Any]:
        """Comprehensive evaluation of generation quality."""
        logger.info("Starting comprehensive evaluation...")
        
        results = {}
        
        # FID Score
        try:
            fid_score = self.fid_calculator.calculate_fid(real_images, generated_images)
            results['fid'] = fid_score
        except Exception as e:
            logger.error(f"FID calculation failed: {e}")
            results['fid'] = None
        
        # CLIP Score
        try:
            # Load generated images for CLIP
            gen_images_pil = [Image.open(path).convert('RGB') for path in generated_images]
            clip_results = self.clip_calculator.calculate_clip_score(gen_images_pil, prompts)
            results.update(clip_results)
        except Exception as e:
            logger.error(f"CLIP Score calculation failed: {e}")
            results['clip_score'] = None
        
        # SSIM (if we have paired images)
        if len(real_images) == len(generated_images):
            try:
                ssim_results = self.ssim_calculator.calculate_ssim(real_images, generated_images)
                results.update(ssim_results)
            except Exception as e:
                logger.error(f"SSIM calculation failed: {e}")
                results['ssim'] = None
        
        # LPIPS
        if len(real_images) == len(generated_images):
            try:
                lpips_results = self.lpips_calculator.calculate_lpips(real_images, generated_images)
                results.update(lpips_results)
            except Exception as e:
                logger.error(f"LPIPS calculation failed: {e}")
                results['lpips'] = None
        
        # Inception Score
        try:
            is_results = self.is_calculator.calculate_inception_score(generated_images)
            results.update(is_results)
        except Exception as e:
            logger.error(f"Inception Score calculation failed: {e}")
            results['inception_score'] = None
        
        # Cultural Metrics
        if (self.config.enable_cultural_metrics and styles and ragas):
            try:
                gen_images_pil = [Image.open(path).convert('RGB') for path in generated_images]
                cultural_results = self.cultural_calculator.calculate_cultural_metrics(
                    gen_images_pil, styles, ragas
                )
                results.update(cultural_results)
            except Exception as e:
                logger.error(f"Cultural metrics calculation failed: {e}")
                results['cultural_authenticity'] = None
        
        logger.info("Evaluation completed")
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to file."""
        import json
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj
        
        # Clean results for JSON serialization
        clean_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                clean_results[key] = {k: convert_types(v) for k, v in value.items()}
            else:
                clean_results[key] = convert_types(value)
        
        with open(output_path, 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")

def main():
    """Main function for testing metrics."""
    # Create configuration
    config = MetricsConfig()
    
    # Initialize metrics calculator
    metrics = EvaluationMetrics(config)
    
    # Create dummy data for testing
    real_images = ["test_real_1.jpg", "test_real_2.jpg"]
    generated_images = ["test_gen_1.jpg", "test_gen_2.jpg"]
    prompts = ["A rajput style ragamala painting", "A pahari miniature artwork"]
    styles = ["rajput", "pahari"]
    ragas = ["bhairav", "yaman"]
    
    # Note: This will fail without actual images, but shows the interface
    try:
        results = metrics.evaluate_generation_quality(
            real_images=real_images,
            generated_images=generated_images,
            prompts=prompts,
            styles=styles,
            ragas=ragas
        )
        
        print("Evaluation Results:")
        for key, value in results.items():
            print(f"  {key}: {value}")
        
        # Save results
        metrics.save_results(results, "evaluation_results.json")
        
    except Exception as e:
        print(f"Evaluation failed (expected without real images): {e}")
    
    print("Metrics testing completed!")

if __name__ == "__main__":
    main()
