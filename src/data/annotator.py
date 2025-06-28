"""
Metadata Annotation Module for Ragamala Paintings.

This module provides comprehensive annotation functionality for Ragamala paintings,
including automatic raga detection, style classification, cultural element identification,
and metadata enrichment for SDXL fine-tuning.
"""

import os
import sys
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
import logging
from datetime import datetime
import re
import pickle
from collections import defaultdict, Counter

# Machine Learning imports
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import (
    CLIPProcessor, CLIPModel, 
    BlipProcessor, BlipForConditionalGeneration,
    pipeline
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import spacy

# Image processing imports
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logging_utils import setup_logger
from src.utils.visualization import plot_annotation_results

logger = setup_logger(__name__)

@dataclass
class AnnotationConfig:
    """Configuration for metadata annotation."""
    # Model paths
    clip_model_name: str = "openai/clip-vit-large-patch14"
    blip_model_name: str = "Salesforce/blip-image-captioning-base"
    spacy_model: str = "en_core_web_sm"
    
    # Classification thresholds
    style_confidence_threshold: float = 0.7
    raga_confidence_threshold: float = 0.6
    cultural_element_threshold: float = 0.5
    
    # Text processing
    max_caption_length: int = 200
    min_description_length: int = 20
    
    # Cultural knowledge
    use_cultural_knowledge_base: bool = True
    knowledge_base_path: str = "data/metadata/cultural_knowledge.json"
    
    # Output settings
    save_intermediate_results: bool = True
    generate_visualizations: bool = True

@dataclass
class RagaAnnotation:
    """Annotation for a specific raga."""
    name: str
    confidence: float
    time_of_day: str
    season: str
    mood: str
    emotions: List[str]
    iconography: List[str]
    color_palette: List[str]
    musical_notes: List[str]

@dataclass
class StyleAnnotation:
    """Annotation for painting style."""
    name: str
    confidence: float
    period: str
    region: str
    characteristics: List[str]
    color_preferences: List[str]
    typical_motifs: List[str]
    brush_techniques: List[str]

@dataclass
class CulturalAnnotation:
    """Cultural elements annotation."""
    deities: List[str]
    architectural_elements: List[str]
    natural_elements: List[str]
    symbolic_objects: List[str]
    clothing_style: List[str]
    jewelry_type: List[str]
    musical_instruments: List[str]
    text_inscriptions: List[str]

@dataclass
class ComprehensiveAnnotation:
    """Complete annotation for a Ragamala painting."""
    filename: str
    raga: RagaAnnotation
    style: StyleAnnotation
    cultural_elements: CulturalAnnotation
    generated_caption: str
    quality_score: float
    annotation_confidence: float
    processing_time: float
    annotated_date: str

class CulturalKnowledgeBase:
    """Knowledge base for cultural elements in Ragamala paintings."""
    
    def __init__(self, knowledge_base_path: str):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.raga_knowledge = {}
        self.style_knowledge = {}
        self.iconography_knowledge = {}
        self.load_knowledge_base()
    
    def load_knowledge_base(self):
        """Load cultural knowledge from files."""
        try:
            # Load raga taxonomy
            raga_file = self.knowledge_base_path.parent / "raga_taxonomy.json"
            if raga_file.exists():
                with open(raga_file, 'r', encoding='utf-8') as f:
                    raga_data = json.load(f)
                    self.raga_knowledge = raga_data.get('primary_ragas', {})
                    self.raga_knowledge.update(raga_data.get('secondary_ragas', {}))
            
            # Load style taxonomy
            style_file = self.knowledge_base_path.parent / "style_taxonomy.json"
            if style_file.exists():
                with open(style_file, 'r', encoding='utf-8') as f:
                    style_data = json.load(f)
                    self.style_knowledge = style_data.get('primary_styles', {})
                    self.style_knowledge.update(style_data.get('sub_styles', {}))
            
            # Load iconography mapping
            self._build_iconography_knowledge()
            
            logger.info("Cultural knowledge base loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            self._create_default_knowledge()
    
    def _build_iconography_knowledge(self):
        """Build iconography knowledge from raga and style data."""
        self.iconography_knowledge = {
            'deities': {
                'shiva': ['bhairav', 'malkauns', 'kedar'],
                'krishna': ['yaman', 'hindola', 'vasanta'],
                'vishnu': ['bhimpalasi', 'sarang'],
                'devi': ['bageshri', 'puriya']
            },
            'natural_elements': {
                'lotus': ['devotional_ragas', 'water_themes'],
                'peacock': ['monsoon_ragas', 'royal_themes'],
                'moon': ['night_ragas', 'romantic_themes'],
                'sun': ['dawn_ragas', 'heroic_themes']
            },
            'architectural': {
                'temple': ['devotional_context', 'spiritual_ragas'],
                'palace': ['royal_context', 'court_ragas'],
                'garden': ['romantic_context', 'seasonal_ragas'],
                'pavilion': ['leisure_context', 'entertainment_ragas']
            }
        }
    
    def _create_default_knowledge(self):
        """Create default knowledge base if files are missing."""
        self.raga_knowledge = {
            'bhairav': {
                'time_of_day': 'dawn',
                'mood': 'devotional',
                'iconography': ['temple', 'ascetic', 'peacocks'],
                'color_palette': ['white', 'saffron', 'gold']
            },
            'yaman': {
                'time_of_day': 'evening',
                'mood': 'romantic',
                'iconography': ['garden', 'lovers', 'moon'],
                'color_palette': ['blue', 'white', 'pink']
            }
        }
        
        self.style_knowledge = {
            'rajput': {
                'characteristics': ['bold_colors', 'geometric_patterns'],
                'color_preferences': ['red', 'gold', 'white'],
                'period': '16th-18th_century'
            },
            'pahari': {
                'characteristics': ['soft_colors', 'natural_settings'],
                'color_preferences': ['soft_blue', 'green', 'pink'],
                'period': '17th-19th_century'
            }
        }
    
    def get_raga_info(self, raga_name: str) -> Dict[str, Any]:
        """Get information about a specific raga."""
        return self.raga_knowledge.get(raga_name.lower(), {})
    
    def get_style_info(self, style_name: str) -> Dict[str, Any]:
        """Get information about a specific style."""
        return self.style_knowledge.get(style_name.lower(), {})
    
    def find_similar_ragas(self, features: List[str]) -> List[Tuple[str, float]]:
        """Find ragas similar to given features."""
        similarities = []
        
        for raga_name, raga_info in self.raga_knowledge.items():
            raga_features = []
            raga_features.extend(raga_info.get('iconography', []))
            raga_features.extend(raga_info.get('color_palette', []))
            raga_features.append(raga_info.get('mood', ''))
            raga_features.append(raga_info.get('time_of_day', ''))
            
            # Calculate similarity
            common_features = set(features) & set(raga_features)
            similarity = len(common_features) / max(len(features), len(raga_features), 1)
            
            if similarity > 0:
                similarities.append((raga_name, similarity))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)

class ImageCaptionGenerator:
    """Generate descriptive captions for Ragamala paintings."""
    
    def __init__(self, config: AnnotationConfig):
        self.config = config
        self.blip_processor = None
        self.blip_model = None
        self.clip_processor = None
        self.clip_model = None
        self.load_models()
    
    def load_models(self):
        """Load caption generation models."""
        try:
            # Load BLIP for image captioning
            self.blip_processor = BlipProcessor.from_pretrained(self.config.blip_model_name)
            self.blip_model = BlipForConditionalGeneration.from_pretrained(self.config.blip_model_name)
            
            # Load CLIP for image-text matching
            self.clip_processor = CLIPProcessor.from_pretrained(self.config.clip_model_name)
            self.clip_model = CLIPModel.from_pretrained(self.config.clip_model_name)
            
            logger.info("Caption generation models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def generate_caption(self, image: Union[Image.Image, np.ndarray]) -> str:
        """Generate descriptive caption for the image."""
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Generate basic caption with BLIP
            inputs = self.blip_processor(image, return_tensors="pt")
            out = self.blip_model.generate(**inputs, max_length=self.config.max_caption_length)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            return caption
            
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return "Traditional Indian miniature painting"
    
    def enhance_caption_with_cultural_context(self, 
                                            caption: str, 
                                            raga_info: Dict, 
                                            style_info: Dict) -> str:
        """Enhance caption with cultural context."""
        enhanced_parts = [caption]
        
        # Add style information
        if style_info:
            style_name = style_info.get('name', '')
            period = style_info.get('period', '')
            if style_name and period:
                enhanced_parts.append(f"painted in {style_name} style from {period}")
        
        # Add raga information
        if raga_info:
            raga_name = raga_info.get('name', '')
            mood = raga_info.get('mood', '')
            time_of_day = raga_info.get('time_of_day', '')
            
            if raga_name:
                enhanced_parts.append(f"depicting raga {raga_name}")
            if mood and time_of_day:
                enhanced_parts.append(f"with {mood} mood suitable for {time_of_day}")
        
        return ", ".join(enhanced_parts)

class StyleClassifier:
    """Classify painting styles using visual features and cultural knowledge."""
    
    def __init__(self, config: AnnotationConfig, knowledge_base: CulturalKnowledgeBase):
        self.config = config
        self.knowledge_base = knowledge_base
        self.style_templates = self._create_style_templates()
        self.clip_processor = CLIPProcessor.from_pretrained(config.clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(config.clip_model_name)
    
    def _create_style_templates(self) -> Dict[str, List[str]]:
        """Create text templates for style classification."""
        return {
            'rajput': [
                "a rajput style miniature painting with bold colors and geometric patterns",
                "traditional rajasthani painting with red and gold colors",
                "mewar school painting with royal themes and vibrant colors"
            ],
            'pahari': [
                "a pahari style painting with soft colors and natural settings",
                "kangra school miniature with delicate features and romantic themes",
                "hill painting with atmospheric landscapes and lyrical quality"
            ],
            'deccan': [
                "a deccan style painting with persian influence and formal composition",
                "golconda school artwork with architectural elements and rich colors",
                "southern indian miniature with geometric precision and courtly themes"
            ],
            'mughal': [
                "a mughal style painting with elaborate details and court scenes",
                "imperial mughal artwork with naturalistic portraiture and rich decoration",
                "persian influenced painting with hierarchical composition and fine details"
            ]
        }
    
    def classify_style(self, image: Union[Image.Image, np.ndarray]) -> StyleAnnotation:
        """Classify the painting style."""
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Prepare image
            inputs = self.clip_processor(images=image, return_tensors="pt")
            image_features = self.clip_model.get_image_features(**inputs)
            
            style_scores = {}
            
            # Compare with style templates
            for style_name, templates in self.style_templates.items():
                template_scores = []
                
                for template in templates:
                    text_inputs = self.clip_processor(text=template, return_tensors="pt")
                    text_features = self.clip_model.get_text_features(**text_inputs)
                    
                    # Calculate similarity
                    similarity = torch.cosine_similarity(image_features, text_features).item()
                    template_scores.append(similarity)
                
                style_scores[style_name] = max(template_scores)
            
            # Find best matching style
            best_style = max(style_scores, key=style_scores.get)
            confidence = style_scores[best_style]
            
            # Get style information from knowledge base
            style_info = self.knowledge_base.get_style_info(best_style)
            
            return StyleAnnotation(
                name=best_style,
                confidence=confidence,
                period=style_info.get('period', 'unknown'),
                region=style_info.get('region', 'unknown'),
                characteristics=style_info.get('characteristics', []),
                color_preferences=style_info.get('color_preferences', []),
                typical_motifs=style_info.get('typical_motifs', []),
                brush_techniques=style_info.get('brush_techniques', [])
            )
            
        except Exception as e:
            logger.error(f"Error classifying style: {e}")
            return StyleAnnotation(
                name="unknown",
                confidence=0.0,
                period="unknown",
                region="unknown",
                characteristics=[],
                color_preferences=[],
                typical_motifs=[],
                brush_techniques=[]
            )

class RagaClassifier:
    """Classify ragas based on visual elements and cultural context."""
    
    def __init__(self, config: AnnotationConfig, knowledge_base: CulturalKnowledgeBase):
        self.config = config
        self.knowledge_base = knowledge_base
        self.raga_templates = self._create_raga_templates()
        self.clip_processor = CLIPProcessor.from_pretrained(config.clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(config.clip_model_name)
        self.color_analyzer = ColorAnalyzer()
    
    def _create_raga_templates(self) -> Dict[str, List[str]]:
        """Create text templates for raga classification."""
        templates = {}
        
        for raga_name, raga_info in self.knowledge_base.raga_knowledge.items():
            raga_templates = []
            
            # Create templates based on raga characteristics
            mood = raga_info.get('mood', '')
            time_of_day = raga_info.get('time_of_day', '')
            iconography = raga_info.get('iconography', [])
            
            if mood and time_of_day:
                raga_templates.append(f"a {mood} painting suitable for {time_of_day}")
            
            for icon in iconography[:3]:  # Limit to top 3 iconographic elements
                raga_templates.append(f"a painting featuring {icon}")
            
            if raga_templates:
                templates[raga_name] = raga_templates
        
        return templates
    
    def classify_raga(self, image: Union[Image.Image, np.ndarray]) -> RagaAnnotation:
        """Classify the raga depicted in the painting."""
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Analyze visual features
            color_features = self.color_analyzer.analyze_colors(image)
            
            # CLIP-based classification
            inputs = self.clip_processor(images=image, return_tensors="pt")
            image_features = self.clip_model.get_image_features(**inputs)
            
            raga_scores = {}
            
            # Compare with raga templates
            for raga_name, templates in self.raga_templates.items():
                template_scores = []
                
                for template in templates:
                    text_inputs = self.clip_processor(text=template, return_tensors="pt")
                    text_features = self.clip_model.get_text_features(**text_inputs)
                    
                    similarity = torch.cosine_similarity(image_features, text_features).item()
                    template_scores.append(similarity)
                
                # Combine with color similarity
                raga_info = self.knowledge_base.get_raga_info(raga_name)
                color_similarity = self._calculate_color_similarity(
                    color_features, 
                    raga_info.get('color_palette', [])
                )
                
                # Weighted combination
                combined_score = (max(template_scores) * 0.7 + color_similarity * 0.3)
                raga_scores[raga_name] = combined_score
            
            # Find best matching raga
            best_raga = max(raga_scores, key=raga_scores.get)
            confidence = raga_scores[best_raga]
            
            # Get raga information
            raga_info = self.knowledge_base.get_raga_info(best_raga)
            
            return RagaAnnotation(
                name=best_raga,
                confidence=confidence,
                time_of_day=raga_info.get('time_of_day', 'unknown'),
                season=raga_info.get('season', 'unknown'),
                mood=raga_info.get('mood', 'unknown'),
                emotions=raga_info.get('emotions', []),
                iconography=raga_info.get('iconography', []),
                color_palette=raga_info.get('color_palette', []),
                musical_notes=raga_info.get('notes', {}).get('aroha', [])
            )
            
        except Exception as e:
            logger.error(f"Error classifying raga: {e}")
            return RagaAnnotation(
                name="unknown",
                confidence=0.0,
                time_of_day="unknown",
                season="unknown",
                mood="unknown",
                emotions=[],
                iconography=[],
                color_palette=[],
                musical_notes=[]
            )
    
    def _calculate_color_similarity(self, image_colors: List[str], raga_colors: List[str]) -> float:
        """Calculate similarity between image colors and raga color palette."""
        if not image_colors or not raga_colors:
            return 0.0
        
        # Simple color name matching
        common_colors = set(image_colors) & set(raga_colors)
        similarity = len(common_colors) / max(len(image_colors), len(raga_colors))
        
        return similarity

class ColorAnalyzer:
    """Analyze color composition of paintings."""
    
    def __init__(self):
        self.color_names = {
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'orange': (255, 165, 0),
            'purple': (128, 0, 128),
            'pink': (255, 192, 203),
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'brown': (165, 42, 42),
            'gold': (255, 215, 0),
            'silver': (192, 192, 192),
            'saffron': (244, 196, 48),
            'crimson': (220, 20, 60)
        }
    
    def analyze_colors(self, image: Image.Image) -> List[str]:
        """Analyze dominant colors in the image."""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Reshape for clustering
            pixels = img_array.reshape(-1, 3)
            
            # K-means clustering to find dominant colors
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans.fit(pixels)
            dominant_colors = kmeans.cluster_centers_
            
            # Map to color names
            color_names = []
            for color in dominant_colors:
                closest_color = self._find_closest_color(color)
                color_names.append(closest_color)
            
            return color_names
            
        except Exception as e:
            logger.error(f"Error analyzing colors: {e}")
            return []
    
    def _find_closest_color(self, rgb: np.ndarray) -> str:
        """Find the closest named color to an RGB value."""
        min_distance = float('inf')
        closest_color = 'unknown'
        
        for color_name, color_rgb in self.color_names.items():
            distance = np.linalg.norm(rgb - np.array(color_rgb))
            if distance < min_distance:
                min_distance = distance
                closest_color = color_name
        
        return closest_color

class CulturalElementDetector:
    """Detect cultural elements in Ragamala paintings."""
    
    def __init__(self, config: AnnotationConfig, knowledge_base: CulturalKnowledgeBase):
        self.config = config
        self.knowledge_base = knowledge_base
        self.element_templates = self._create_element_templates()
        self.clip_processor = CLIPProcessor.from_pretrained(config.clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(config.clip_model_name)
    
    def _create_element_templates(self) -> Dict[str, List[str]]:
        """Create templates for cultural element detection."""
        return {
            'deities': [
                "hindu deity", "god", "goddess", "divine figure",
                "shiva", "krishna", "vishnu", "devi", "ganesha"
            ],
            'architectural': [
                "temple", "palace", "pavilion", "arch", "pillar",
                "dome", "minaret", "courtyard", "garden", "fountain"
            ],
            'natural': [
                "lotus", "peacock", "elephant", "horse", "deer",
                "tree", "flower", "river", "mountain", "moon", "sun"
            ],
            'objects': [
                "musical instrument", "veena", "tabla", "flute",
                "jewelry", "crown", "sword", "book", "lamp"
            ],
            'clothing': [
                "sari", "dhoti", "turban", "jewelry", "ornaments"
            ]
        }
    
    def detect_elements(self, image: Union[Image.Image, np.ndarray]) -> CulturalAnnotation:
        """Detect cultural elements in the painting."""
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Prepare image
            inputs = self.clip_processor(images=image, return_tensors="pt")
            image_features = self.clip_model.get_image_features(**inputs)
            
            detected_elements = defaultdict(list)
            
            # Check each category of elements
            for category, templates in self.element_templates.items():
                for template in templates:
                    text_inputs = self.clip_processor(text=f"a painting with {template}", return_tensors="pt")
                    text_features = self.clip_model.get_text_features(**text_inputs)
                    
                    similarity = torch.cosine_similarity(image_features, text_features).item()
                    
                    if similarity > self.config.cultural_element_threshold:
                        detected_elements[category].append({
                            'element': template,
                            'confidence': similarity
                        })
            
            # Sort by confidence and take top elements
            for category in detected_elements:
                detected_elements[category] = sorted(
                    detected_elements[category], 
                    key=lambda x: x['confidence'], 
                    reverse=True
                )[:5]  # Top 5 per category
            
            return CulturalAnnotation(
                deities=[elem['element'] for elem in detected_elements.get('deities', [])],
                architectural_elements=[elem['element'] for elem in detected_elements.get('architectural', [])],
                natural_elements=[elem['element'] for elem in detected_elements.get('natural', [])],
                symbolic_objects=[elem['element'] for elem in detected_elements.get('objects', [])],
                clothing_style=[elem['element'] for elem in detected_elements.get('clothing', [])],
                jewelry_type=[],  # Could be enhanced with specific jewelry detection
                musical_instruments=[],  # Could be enhanced with instrument detection
                text_inscriptions=[]  # Could be enhanced with OCR
            )
            
        except Exception as e:
            logger.error(f"Error detecting cultural elements: {e}")
            return CulturalAnnotation(
                deities=[],
                architectural_elements=[],
                natural_elements=[],
                symbolic_objects=[],
                clothing_style=[],
                jewelry_type=[],
                musical_instruments=[],
                text_inscriptions=[]
            )

class RagamalaAnnotator:
    """Main annotator class for Ragamala paintings."""
    
    def __init__(self, config: AnnotationConfig):
        self.config = config
        self.knowledge_base = CulturalKnowledgeBase(config.knowledge_base_path)
        self.caption_generator = ImageCaptionGenerator(config)
        self.style_classifier = StyleClassifier(config, self.knowledge_base)
        self.raga_classifier = RagaClassifier(config, self.knowledge_base)
        self.cultural_detector = CulturalElementDetector(config, self.knowledge_base)
        
    def annotate_image(self, 
                      image_path: Union[str, Path], 
                      existing_metadata: Dict = None) -> ComprehensiveAnnotation:
        """Annotate a single Ragamala painting."""
        start_time = datetime.now()
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            filename = Path(image_path).name
            
            # Generate caption
            caption = self.caption_generator.generate_caption(image)
            
            # Classify style
            style_annotation = self.style_classifier.classify_style(image)
            
            # Classify raga
            raga_annotation = self.raga_classifier.classify_raga(image)
            
            # Detect cultural elements
            cultural_annotation = self.cultural_detector.detect_elements(image)
            
            # Enhance caption with cultural context
            enhanced_caption = self.caption_generator.enhance_caption_with_cultural_context(
                caption,
                asdict(raga_annotation),
                asdict(style_annotation)
            )
            
            # Calculate overall confidence
            annotation_confidence = (
                style_annotation.confidence * 0.4 +
                raga_annotation.confidence * 0.4 +
                0.2  # Base confidence for cultural elements
            )
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(
                style_annotation,
                raga_annotation,
                cultural_annotation
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ComprehensiveAnnotation(
                filename=filename,
                raga=raga_annotation,
                style=style_annotation,
                cultural_elements=cultural_annotation,
                generated_caption=enhanced_caption,
                quality_score=quality_score,
                annotation_confidence=annotation_confidence,
                processing_time=processing_time,
                annotated_date=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error annotating {image_path}: {e}")
            return None
    
    def _calculate_quality_score(self, 
                                style: StyleAnnotation,
                                raga: RagaAnnotation,
                                cultural: CulturalAnnotation) -> float:
        """Calculate overall quality score for the annotation."""
        scores = []
        
        # Style confidence
        scores.append(style.confidence)
        
        # Raga confidence
        scores.append(raga.confidence)
        
        # Cultural element richness
        cultural_richness = (
            len(cultural.deities) * 0.2 +
            len(cultural.architectural_elements) * 0.2 +
            len(cultural.natural_elements) * 0.2 +
            len(cultural.symbolic_objects) * 0.2 +
            len(cultural.clothing_style) * 0.2
        ) / 5.0  # Normalize
        
        scores.append(min(cultural_richness, 1.0))
        
        return np.mean(scores)
    
    def annotate_dataset(self, 
                        image_dir: Union[str, Path],
                        output_file: Union[str, Path],
                        existing_metadata_file: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Annotate an entire dataset of Ragamala paintings."""
        image_dir = Path(image_dir)
        output_file = Path(output_file)
        
        # Load existing metadata if available
        existing_metadata = {}
        if existing_metadata_file and Path(existing_metadata_file).exists():
            with open(existing_metadata_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    existing_metadata[data['file_name']] = data
        
        annotations = []
        failed_annotations = []
        
        # Process all images
        image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        
        for i, image_file in enumerate(image_files):
            logger.info(f"Annotating {image_file.name} ({i+1}/{len(image_files)})")
            
            existing_meta = existing_metadata.get(image_file.name, {})
            annotation = self.annotate_image(image_file, existing_meta)
            
            if annotation:
                annotations.append(annotation)
            else:
                failed_annotations.append(image_file.name)
            
            # Save intermediate results
            if self.config.save_intermediate_results and (i + 1) % 50 == 0:
                self._save_annotations(annotations, output_file.with_suffix('.temp.jsonl'))
        
        # Save final annotations
        self._save_annotations(annotations, output_file)
        
        # Generate report
        report = {
            'total_images': len(image_files),
            'successful_annotations': len(annotations),
            'failed_annotations': len(failed_annotations),
            'average_confidence': np.mean([ann.annotation_confidence for ann in annotations]),
            'average_quality': np.mean([ann.quality_score for ann in annotations]),
            'style_distribution': self._get_style_distribution(annotations),
            'raga_distribution': self._get_raga_distribution(annotations)
        }
        
        # Save report
        report_file = output_file.with_suffix('.report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Annotation complete: {report}")
        return report
    
    def _save_annotations(self, annotations: List[ComprehensiveAnnotation], output_file: Path):
        """Save annotations to JSONL file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for annotation in annotations:
                # Convert to training format
                training_record = {
                    'file_name': annotation.filename,
                    'text': annotation.generated_caption,
                    'raga': annotation.raga.name,
                    'style': annotation.style.name,
                    'period': annotation.style.period,
                    'region': annotation.style.region,
                    'mood': annotation.raga.mood,
                    'time_of_day': annotation.raga.time_of_day,
                    'season': annotation.raga.season,
                    'color_palette': annotation.raga.color_palette,
                    'iconography': annotation.raga.iconography,
                    'cultural_elements': asdict(annotation.cultural_elements),
                    'quality_score': annotation.quality_score,
                    'annotation_confidence': annotation.annotation_confidence,
                    'annotated_date': annotation.annotated_date
                }
                
                f.write(json.dumps(training_record, ensure_ascii=False) + '\n')
    
    def _get_style_distribution(self, annotations: List[ComprehensiveAnnotation]) -> Dict[str, int]:
        """Get distribution of styles in annotations."""
        style_counts = Counter([ann.style.name for ann in annotations])
        return dict(style_counts)
    
    def _get_raga_distribution(self, annotations: List[ComprehensiveAnnotation]) -> Dict[str, int]:
        """Get distribution of ragas in annotations."""
        raga_counts = Counter([ann.raga.name for ann in annotations])
        return dict(raga_counts)

def main():
    """Main function for annotation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Annotate Ragamala paintings")
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--output_file', type=str, required=True, help='Output JSONL file')
    parser.add_argument('--existing_metadata', type=str, help='Existing metadata file')
    parser.add_argument('--config_file', type=str, help='Configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config_dict = json.load(f)
        config = AnnotationConfig(**config_dict)
    else:
        config = AnnotationConfig()
    
    # Initialize annotator
    annotator = RagamalaAnnotator(config)
    
    # Annotate dataset
    report = annotator.annotate_dataset(
        args.image_dir,
        args.output_file,
        args.existing_metadata
    )
    
    print(f"Annotation complete: {report}")

if __name__ == "__main__":
    main()
