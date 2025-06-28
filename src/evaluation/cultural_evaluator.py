"""
Cultural Accuracy Assessment for Ragamala Painting Generation.

This module provides comprehensive cultural evaluation functionality for assessing
the authenticity and accuracy of generated Ragamala paintings based on traditional
Indian art forms, iconography, and cultural representation principles.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, asdict
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings("ignore")

# CLIP imports
from transformers import CLIPProcessor, CLIPModel

# Computer vision imports
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

@dataclass
class CulturalEvaluationConfig:
    """Configuration for cultural evaluation."""
    # Model settings
    clip_model_name: str = "openai/clip-vit-base-patch32"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Evaluation thresholds
    authenticity_threshold: float = 0.7
    iconography_threshold: float = 0.6
    color_palette_threshold: float = 0.8
    composition_threshold: float = 0.7
    
    # Cultural knowledge
    enable_expert_validation: bool = False
    expert_api_endpoint: Optional[str] = None
    
    # Evaluation dimensions
    evaluate_iconography: bool = True
    evaluate_color_palette: bool = True
    evaluate_composition: bool = True
    evaluate_temporal_consistency: bool = True
    evaluate_regional_specificity: bool = True

@dataclass
class CulturalEvaluationResult:
    """Result of cultural evaluation."""
    overall_authenticity_score: float
    iconography_score: float
    color_palette_score: float
    composition_score: float
    temporal_consistency_score: float
    regional_specificity_score: float
    detailed_feedback: Dict[str, Any]
    cultural_violations: List[str]
    recommendations: List[str]

class CulturalKnowledgeBase:
    """Comprehensive knowledge base for Ragamala cultural elements."""
    
    def __init__(self):
        self.raga_iconography = self._load_raga_iconography()
        self.style_characteristics = self._load_style_characteristics()
        self.color_symbolism = self._load_color_symbolism()
        self.temporal_elements = self._load_temporal_elements()
        self.regional_specifics = self._load_regional_specifics()
        self.cultural_violations = self._load_cultural_violations()
    
    def _load_raga_iconography(self) -> Dict[str, Dict[str, Any]]:
        """Load iconographic elements for each raga."""
        return {
            'bhairav': {
                'deities': ['shiva', 'bhairava'],
                'animals': ['peacock', 'snake', 'bull'],
                'objects': ['trident', 'damaru', 'rudraksha', 'lotus'],
                'settings': ['temple', 'cremation_ground', 'mountain'],
                'time_indicators': ['sunrise', 'dawn_sky', 'morning_light'],
                'human_figures': ['ascetic', 'devotee', 'sage'],
                'architectural': ['temple_spire', 'pillars', 'sacred_geometry'],
                'natural': ['sacred_fire', 'flowing_water', 'mountain_peaks'],
                'forbidden': ['modern_objects', 'western_clothing', 'inappropriate_poses']
            },
            'yaman': {
                'deities': ['krishna', 'radha', 'vishnu'],
                'animals': ['cow', 'peacock', 'swan'],
                'objects': ['flute', 'lotus', 'crown', 'jewelry'],
                'settings': ['garden', 'palace', 'vrindavan'],
                'time_indicators': ['moon', 'evening_sky', 'stars'],
                'human_figures': ['lovers', 'gopis', 'royal_couple'],
                'architectural': ['pavilion', 'garden_walls', 'ornate_pillars'],
                'natural': ['flowering_trees', 'river', 'moonlight'],
                'forbidden': ['harsh_lighting', 'aggressive_poses', 'inappropriate_intimacy']
            },
            'malkauns': {
                'deities': ['shiva', 'kali', 'bhairava'],
                'animals': ['tiger', 'serpent', 'owl'],
                'objects': ['meditation_beads', 'water_pot', 'sacred_ash'],
                'settings': ['forest', 'cave', 'riverside'],
                'time_indicators': ['midnight', 'dark_sky', 'stars'],
                'human_figures': ['meditating_sage', 'hermit', 'yogi'],
                'architectural': ['simple_hut', 'cave_entrance', 'natural_formations'],
                'natural': ['flowing_river', 'dense_forest', 'rocky_terrain'],
                'forbidden': ['bright_colors', 'festive_elements', 'crowded_scenes']
            },
            'darbari': {
                'deities': ['indra', 'vishnu', 'royal_deities'],
                'animals': ['elephant', 'horse', 'peacock'],
                'objects': ['throne', 'crown', 'royal_regalia', 'ceremonial_weapons'],
                'settings': ['royal_court', 'palace', 'durbar_hall'],
                'time_indicators': ['evening', 'ceremonial_lighting', 'torches'],
                'human_figures': ['king', 'courtiers', 'ministers', 'guards'],
                'architectural': ['grand_pillars', 'ornate_ceiling', 'royal_chambers'],
                'natural': ['formal_gardens', 'fountains', 'decorative_pools'],
                'forbidden': ['informal_settings', 'common_people', 'rustic_elements']
            }
        }
    
    def _load_style_characteristics(self) -> Dict[str, Dict[str, Any]]:
        """Load characteristics for each painting style."""
        return {
            'rajput': {
                'color_palette': ['red', 'gold', 'white', 'green', 'saffron'],
                'composition': 'hierarchical_and_symmetrical',
                'brushwork': 'bold_and_precise',
                'perspective': 'flat_with_multiple_viewpoints',
                'facial_features': 'stylized_with_large_eyes',
                'clothing': ['dhoti', 'turban', 'jewelry', 'royal_attire'],
                'patterns': ['geometric', 'floral', 'architectural'],
                'background': ['decorative_patterns', 'architectural_elements'],
                'period_markers': ['16th_to_18th_century', 'rajasthani_influence'],
                'forbidden_elements': ['realistic_perspective', 'muted_colors', 'western_influence']
            },
            'pahari': {
                'color_palette': ['soft_blue', 'green', 'pink', 'white', 'yellow'],
                'composition': 'naturalistic_and_flowing',
                'brushwork': 'delicate_and_refined',
                'perspective': 'atmospheric_depth',
                'facial_features': 'delicate_with_soft_expressions',
                'clothing': ['fine_textiles', 'flowing_garments', 'subtle_jewelry'],
                'patterns': ['natural_motifs', 'floral_designs', 'paisley'],
                'background': ['landscape_integration', 'natural_settings'],
                'period_markers': ['17th_to_19th_century', 'himalayan_influence'],
                'forbidden_elements': ['harsh_colors', 'geometric_rigidity', 'urban_settings']
            },
            'deccan': {
                'color_palette': ['deep_blue', 'purple', 'gold', 'white', 'crimson'],
                'composition': 'formal_and_structured',
                'brushwork': 'precise_and_elaborate',
                'perspective': 'architectural_accuracy',
                'facial_features': 'persian_influenced_profiles',
                'clothing': ['persian_robes', 'turbans', 'rich_fabrics'],
                'patterns': ['geometric_precision', 'calligraphic_elements', 'persian_motifs'],
                'background': ['architectural_integration', 'formal_gardens'],
                'period_markers': ['16th_to_18th_century', 'persian_influence'],
                'forbidden_elements': ['informal_composition', 'folk_elements', 'crude_execution']
            },
            'mughal': {
                'color_palette': ['rich_colors', 'gold', 'jewel_tones', 'earth_colors'],
                'composition': 'balanced_and_hierarchical',
                'brushwork': 'highly_refined_and_detailed',
                'perspective': 'european_influenced_realism',
                'facial_features': 'naturalistic_portraits',
                'clothing': ['court_dress', 'elaborate_textiles', 'precious_jewelry'],
                'patterns': ['intricate_details', 'floral_arabesques', 'geometric_borders'],
                'background': ['realistic_landscapes', 'architectural_monuments'],
                'period_markers': ['16th_to_18th_century', 'imperial_grandeur'],
                'forbidden_elements': ['simple_details', 'flat_composition', 'folk_style']
            }
        }
    
    def _load_color_symbolism(self) -> Dict[str, Dict[str, str]]:
        """Load color symbolism in Indian art."""
        return {
            'red': {
                'symbolism': 'power_passion_fertility',
                'appropriate_contexts': ['royal_scenes', 'romantic_themes', 'festive_occasions'],
                'cultural_significance': 'auspicious_color_in_hindu_tradition'
            },
            'saffron': {
                'symbolism': 'spirituality_renunciation_sacred',
                'appropriate_contexts': ['religious_scenes', 'ascetic_figures', 'temple_settings'],
                'cultural_significance': 'color_of_hindu_and_buddhist_monks'
            },
            'blue': {
                'symbolism': 'divine_infinite_peaceful',
                'appropriate_contexts': ['krishna_depictions', 'water_elements', 'sky_representations'],
                'cultural_significance': 'color_associated_with_vishnu_and_krishna'
            },
            'white': {
                'symbolism': 'purity_peace_knowledge',
                'appropriate_contexts': ['saraswati_depictions', 'spiritual_scenes', 'dawn_settings'],
                'cultural_significance': 'color_of_purity_and_wisdom'
            },
            'green': {
                'symbolism': 'nature_fertility_harmony',
                'appropriate_contexts': ['natural_settings', 'spring_scenes', 'vegetation'],
                'cultural_significance': 'color_of_nature_and_new_beginnings'
            },
            'gold': {
                'symbolism': 'wealth_prosperity_divine',
                'appropriate_contexts': ['royal_scenes', 'divine_figures', 'precious_objects'],
                'cultural_significance': 'color_of_wealth_and_divinity'
            }
        }
    
    def _load_temporal_elements(self) -> Dict[str, Dict[str, List[str]]]:
        """Load temporal consistency elements."""
        return {
            'historical_periods': {
                '16th_century': ['early_mughal_influence', 'persian_elements', 'court_patronage'],
                '17th_century': ['mature_styles', 'regional_variations', 'technical_refinement'],
                '18th_century': ['peak_development', 'diverse_schools', 'cultural_synthesis'],
                '19th_century': ['late_traditions', 'colonial_influence', 'revival_movements']
            },
            'anachronistic_elements': {
                'modern_objects': ['cars', 'phones', 'modern_buildings', 'contemporary_clothing'],
                'wrong_period_elements': ['british_colonial_architecture', 'modern_hairstyles'],
                'inappropriate_technology': ['electric_lights', 'modern_weapons', 'industrial_elements']
            }
        }
    
    def _load_regional_specifics(self) -> Dict[str, Dict[str, Any]]:
        """Load region-specific cultural elements."""
        return {
            'rajasthan': {
                'architectural_style': ['haveli_architecture', 'jharokhas', 'chhatris'],
                'clothing_style': ['rajasthani_turbans', 'lehenga_choli', 'bandhani_patterns'],
                'landscape': ['desert_setting', 'arid_vegetation', 'fort_architecture'],
                'cultural_markers': ['rajput_valor', 'royal_traditions', 'warrior_culture']
            },
            'himachal_pradesh': {
                'architectural_style': ['hill_architecture', 'wooden_structures', 'sloped_roofs'],
                'clothing_style': ['pahari_caps', 'woolen_garments', 'hill_jewelry'],
                'landscape': ['mountain_setting', 'pine_forests', 'terraced_fields'],
                'cultural_markers': ['hill_traditions', 'pastoral_life', 'nature_worship']
            },
            'deccan_plateau': {
                'architectural_style': ['indo_islamic_architecture', 'domes', 'minarets'],
                'clothing_style': ['deccani_dress', 'persian_influence', 'court_attire'],
                'landscape': ['plateau_terrain', 'formal_gardens', 'water_features'],
                'cultural_markers': ['sultanate_culture', 'persian_influence', 'court_traditions']
            }
        }
    
    def _load_cultural_violations(self) -> Dict[str, List[str]]:
        """Load common cultural violations to detect."""
        return {
            'iconographic_violations': [
                'inappropriate_deity_combinations',
                'wrong_animal_associations',
                'misplaced_sacred_objects',
                'incorrect_symbolic_elements'
            ],
            'temporal_violations': [
                'anachronistic_elements',
                'wrong_period_clothing',
                'modern_objects_in_traditional_settings',
                'inappropriate_technology'
            ],
            'cultural_misrepresentation': [
                'stereotypical_portrayals',
                'oversimplified_cultural_elements',
                'mixing_incompatible_traditions',
                'inappropriate_religious_symbols'
            ],
            'compositional_violations': [
                'wrong_hierarchical_arrangements',
                'inappropriate_spatial_relationships',
                'incorrect_scale_proportions',
                'misaligned_cultural_elements'
            ]
        }

class IconographyEvaluator:
    """Evaluator for iconographic accuracy in Ragamala paintings."""
    
    def __init__(self, knowledge_base: CulturalKnowledgeBase, config: CulturalEvaluationConfig):
        self.knowledge_base = knowledge_base
        self.config = config
        self.clip_model = CLIPModel.from_pretrained(config.clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(config.clip_model_name)
        self.clip_model.to(config.device)
        self.clip_model.eval()
    
    def evaluate_iconography(self, 
                           image: Image.Image,
                           raga: str,
                           style: str) -> Dict[str, Any]:
        """Evaluate iconographic accuracy for given raga and style."""
        logger.info(f"Evaluating iconography for raga: {raga}, style: {style}")
        
        # Get expected iconographic elements
        raga_iconography = self.knowledge_base.raga_iconography.get(raga.lower(), {})
        
        # Detect present elements
        detected_elements = self._detect_iconographic_elements(image, raga_iconography)
        
        # Calculate accuracy scores
        accuracy_scores = self._calculate_iconographic_accuracy(detected_elements, raga_iconography)
        
        # Detect violations
        violations = self._detect_iconographic_violations(detected_elements, raga_iconography)
        
        return {
            'iconography_score': accuracy_scores['overall_score'],
            'detected_elements': detected_elements,
            'expected_elements': raga_iconography,
            'accuracy_breakdown': accuracy_scores,
            'violations': violations,
            'recommendations': self._generate_iconographic_recommendations(
                detected_elements, raga_iconography, violations
            )
        }
    
    def _detect_iconographic_elements(self, 
                                    image: Image.Image,
                                    expected_iconography: Dict[str, List[str]]) -> Dict[str, List[Tuple[str, float]]]:
        """Detect iconographic elements in the image using CLIP."""
        detected_elements = defaultdict(list)
        
        # Prepare image
        inputs = self.clip_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            image_features = F.normalize(image_features, dim=-1)
        
        # Check each category of elements
        for category, elements in expected_iconography.items():
            if category == 'forbidden':
                continue  # Handle forbidden elements separately
            
            for element in elements:
                # Create text queries for element detection
                queries = [
                    f"a painting with {element}",
                    f"traditional indian art featuring {element}",
                    f"ragamala painting showing {element}"
                ]
                
                max_similarity = 0.0
                for query in queries:
                    text_inputs = self.clip_processor(text=query, return_tensors="pt")
                    text_inputs = {k: v.to(self.config.device) for k, v in text_inputs.items()}
                    
                    with torch.no_grad():
                        text_features = self.clip_model.get_text_features(**text_inputs)
                        text_features = F.normalize(text_features, dim=-1)
                        
                        similarity = torch.cosine_similarity(image_features, text_features).item()
                        max_similarity = max(max_similarity, similarity)
                
                if max_similarity > self.config.iconography_threshold:
                    detected_elements[category].append((element, max_similarity))
        
        return dict(detected_elements)
    
    def _calculate_iconographic_accuracy(self, 
                                       detected_elements: Dict[str, List[Tuple[str, float]]],
                                       expected_iconography: Dict[str, List[str]]) -> Dict[str, float]:
        """Calculate iconographic accuracy scores."""
        category_scores = {}
        
        for category, expected_items in expected_iconography.items():
            if category == 'forbidden':
                continue
            
            detected_items = [item[0] for item in detected_elements.get(category, [])]
            
            if not expected_items:
                category_scores[category] = 1.0
                continue
            
            # Calculate recall (how many expected items were detected)
            detected_count = len(set(detected_items) & set(expected_items))
            recall = detected_count / len(expected_items)
            
            # Weight by confidence scores
            if detected_items:
                confidence_weights = [conf for item, conf in detected_elements.get(category, []) 
                                    if item in expected_items]
                weighted_score = np.mean(confidence_weights) if confidence_weights else 0.0
                category_scores[category] = (recall + weighted_score) / 2
            else:
                category_scores[category] = recall
        
        # Calculate overall score
        overall_score = np.mean(list(category_scores.values())) if category_scores else 0.0
        
        return {
            'overall_score': overall_score,
            'category_scores': category_scores
        }
    
    def _detect_iconographic_violations(self, 
                                      detected_elements: Dict[str, List[Tuple[str, float]]],
                                      expected_iconography: Dict[str, List[str]]) -> List[str]:
        """Detect iconographic violations."""
        violations = []
        
        # Check for forbidden elements
        forbidden_elements = expected_iconography.get('forbidden', [])
        all_detected = [item for sublist in detected_elements.values() for item, _ in sublist]
        
        for forbidden in forbidden_elements:
            if forbidden in all_detected:
                violations.append(f"Forbidden element detected: {forbidden}")
        
        # Check for missing critical elements
        critical_categories = ['deities', 'time_indicators']
        for category in critical_categories:
            if category in expected_iconography and category not in detected_elements:
                violations.append(f"Missing critical {category} elements")
        
        return violations
    
    def _generate_iconographic_recommendations(self, 
                                             detected_elements: Dict[str, List[Tuple[str, float]]],
                                             expected_iconography: Dict[str, List[str]],
                                             violations: List[str]) -> List[str]:
        """Generate recommendations for improving iconographic accuracy."""
        recommendations = []
        
        # Recommend missing elements
        for category, expected_items in expected_iconography.items():
            if category == 'forbidden':
                continue
            
            detected_items = [item[0] for item in detected_elements.get(category, [])]
            missing_items = set(expected_items) - set(detected_items)
            
            if missing_items:
                recommendations.append(
                    f"Consider adding {category}: {', '.join(list(missing_items)[:3])}"
                )
        
        # Address violations
        if violations:
            recommendations.append("Remove inappropriate or forbidden elements")
        
        return recommendations

class ColorPaletteEvaluator:
    """Evaluator for color palette authenticity."""
    
    def __init__(self, knowledge_base: CulturalKnowledgeBase, config: CulturalEvaluationConfig):
        self.knowledge_base = knowledge_base
        self.config = config
    
    def evaluate_color_palette(self, 
                              image: Image.Image,
                              style: str,
                              raga: str) -> Dict[str, Any]:
        """Evaluate color palette authenticity."""
        logger.info(f"Evaluating color palette for style: {style}")
        
        # Extract dominant colors
        dominant_colors = self._extract_dominant_colors(image)
        
        # Get expected color palette
        style_characteristics = self.knowledge_base.style_characteristics.get(style.lower(), {})
        expected_palette = style_characteristics.get('color_palette', [])
        
        # Calculate color accuracy
        color_accuracy = self._calculate_color_accuracy(dominant_colors, expected_palette)
        
        # Evaluate color symbolism
        symbolism_score = self._evaluate_color_symbolism(dominant_colors, raga)
        
        # Detect color violations
        violations = self._detect_color_violations(dominant_colors, style)
        
        return {
            'color_palette_score': (color_accuracy + symbolism_score) / 2,
            'dominant_colors': dominant_colors,
            'expected_palette': expected_palette,
            'color_accuracy': color_accuracy,
            'symbolism_score': symbolism_score,
            'violations': violations,
            'recommendations': self._generate_color_recommendations(
                dominant_colors, expected_palette, violations
            )
        }
    
    def _extract_dominant_colors(self, image: Image.Image, k: int = 8) -> List[Dict[str, Any]]:
        """Extract dominant colors using k-means clustering."""
        # Resize image for faster processing
        img_resized = image.resize((150, 150))
        img_array = np.array(img_resized)
        
        # Reshape to list of pixels
        pixels = img_array.reshape(-1, 3)
        
        # Apply k-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get colors and their frequencies
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        
        # Calculate color frequencies
        color_info = []
        for i, color in enumerate(colors):
            frequency = np.sum(labels == i) / len(labels)
            color_name = self._rgb_to_color_name(color)
            
            color_info.append({
                'rgb': tuple(color),
                'color_name': color_name,
                'frequency': frequency
            })
        
        # Sort by frequency
        color_info.sort(key=lambda x: x['frequency'], reverse=True)
        
        return color_info
    
    def _rgb_to_color_name(self, rgb: np.ndarray) -> str:
        """Convert RGB values to approximate color names."""
        color_map = {
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
            'saffron': (244, 196, 48),
            'crimson': (220, 20, 60)
        }
        
        # Find closest color
        min_distance = float('inf')
        closest_color = 'unknown'
        
        for color_name, color_rgb in color_map.items():
            distance = np.linalg.norm(rgb - np.array(color_rgb))
            if distance < min_distance:
                min_distance = distance
                closest_color = color_name
        
        return closest_color
    
    def _calculate_color_accuracy(self, 
                                dominant_colors: List[Dict[str, Any]],
                                expected_palette: List[str]) -> float:
        """Calculate color palette accuracy."""
        if not expected_palette:
            return 1.0
        
        detected_color_names = [color['color_name'] for color in dominant_colors]
        
        # Calculate how many expected colors are present
        matches = 0
        for expected_color in expected_palette:
            if expected_color in detected_color_names:
                matches += 1
        
        accuracy = matches / len(expected_palette)
        return accuracy
    
    def _evaluate_color_symbolism(self, 
                                dominant_colors: List[Dict[str, Any]],
                                raga: str) -> float:
        """Evaluate appropriateness of colors for the given raga."""
        # Get raga iconography to understand context
        raga_iconography = self.knowledge_base.raga_iconography.get(raga.lower(), {})
        
        # Simple scoring based on color appropriateness
        symbolism_scores = []
        
        for color_info in dominant_colors[:5]:  # Top 5 colors
            color_name = color_info['color_name']
            frequency = color_info['frequency']
            
            # Get color symbolism
            color_symbolism = self.knowledge_base.color_symbolism.get(color_name, {})
            
            # Score based on appropriateness (simplified)
            if color_symbolism:
                score = 0.8  # Base score for recognized colors
            else:
                score = 0.5  # Neutral score for unrecognized colors
            
            # Weight by frequency
            weighted_score = score * frequency
            symbolism_scores.append(weighted_score)
        
        return np.mean(symbolism_scores) if symbolism_scores else 0.5
    
    def _detect_color_violations(self, 
                               dominant_colors: List[Dict[str, Any]],
                               style: str) -> List[str]:
        """Detect color palette violations."""
        violations = []
        
        style_characteristics = self.knowledge_base.style_characteristics.get(style.lower(), {})
        forbidden_elements = style_characteristics.get('forbidden_elements', [])
        
        detected_color_names = [color['color_name'] for color in dominant_colors]
        
        # Check for style-inappropriate colors
        if style == 'rajput' and 'muted_colors' in str(detected_color_names):
            violations.append("Muted colors inappropriate for Rajput style")
        
        if style == 'pahari' and 'harsh_colors' in str(detected_color_names):
            violations.append("Harsh colors inappropriate for Pahari style")
        
        return violations
    
    def _generate_color_recommendations(self, 
                                      dominant_colors: List[Dict[str, Any]],
                                      expected_palette: List[str],
                                      violations: List[str]) -> List[str]:
        """Generate color palette recommendations."""
        recommendations = []
        
        detected_color_names = [color['color_name'] for color in dominant_colors]
        missing_colors = set(expected_palette) - set(detected_color_names)
        
        if missing_colors:
            recommendations.append(f"Consider incorporating: {', '.join(list(missing_colors))}")
        
        if violations:
            recommendations.append("Adjust color palette to match traditional style")
        
        return recommendations

class CompositionEvaluator:
    """Evaluator for compositional authenticity."""
    
    def __init__(self, knowledge_base: CulturalKnowledgeBase, config: CulturalEvaluationConfig):
        self.knowledge_base = knowledge_base
        self.config = config
    
    def evaluate_composition(self, 
                           image: Image.Image,
                           style: str) -> Dict[str, Any]:
        """Evaluate compositional authenticity."""
        logger.info(f"Evaluating composition for style: {style}")
        
        # Analyze composition elements
        composition_analysis = self._analyze_composition(image)
        
        # Get expected composition characteristics
        style_characteristics = self.knowledge_base.style_characteristics.get(style.lower(), {})
        expected_composition = style_characteristics.get('composition', '')
        
        # Calculate composition score
        composition_score = self._calculate_composition_score(
            composition_analysis, expected_composition, style
        )
        
        # Detect composition violations
        violations = self._detect_composition_violations(composition_analysis, style)
        
        return {
            'composition_score': composition_score,
            'composition_analysis': composition_analysis,
            'expected_characteristics': expected_composition,
            'violations': violations,
            'recommendations': self._generate_composition_recommendations(
                composition_analysis, style, violations
            )
        }
    
    def _analyze_composition(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze compositional elements of the image."""
        # Convert to numpy array
        img_array = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Edge analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Symmetry analysis
        height, width = gray.shape
        left_half = gray[:, :width//2]
        right_half = np.fliplr(gray[:, width//2:])
        
        # Resize to same size for comparison
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        symmetry_score = 1.0 - np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255.0
        
        # Color distribution analysis
        color_variance = np.var(img_array, axis=(0, 1))
        color_uniformity = 1.0 / (1.0 + np.mean(color_variance) / 1000.0)
        
        # Spatial frequency analysis
        spatial_freq = np.mean(np.abs(np.gradient(gray.astype(float))))
        
        return {
            'edge_density': edge_density,
            'symmetry_score': symmetry_score,
            'color_uniformity': color_uniformity,
            'spatial_frequency': spatial_freq,
            'aspect_ratio': image.width / image.height
        }
    
    def _calculate_composition_score(self, 
                                   composition_analysis: Dict[str, Any],
                                   expected_composition: str,
                                   style: str) -> float:
        """Calculate composition authenticity score."""
        score_components = []
        
        # Style-specific scoring
        if style == 'rajput':
            # Rajput style prefers hierarchical and symmetrical composition
            symmetry_score = composition_analysis['symmetry_score']
            edge_score = min(composition_analysis['edge_density'] * 2, 1.0)  # Bold outlines
            score_components.extend([symmetry_score, edge_score])
            
        elif style == 'pahari':
            # Pahari style prefers naturalistic and flowing composition
            flow_score = 1.0 - abs(composition_analysis['spatial_frequency'] - 0.3) / 0.3
            naturalness_score = composition_analysis['color_uniformity']
            score_components.extend([flow_score, naturalness_score])
            
        elif style == 'deccan':
            # Deccan style prefers formal and structured composition
            structure_score = composition_analysis['edge_density']
            formal_score = composition_analysis['symmetry_score']
            score_components.extend([structure_score, formal_score])
            
        elif style == 'mughal':
            # Mughal style prefers balanced and hierarchical composition
            balance_score = composition_analysis['symmetry_score']
            detail_score = composition_analysis['edge_density']
            score_components.extend([balance_score, detail_score])
        
        # General composition quality
        aspect_ratio_score = 1.0 - abs(composition_analysis['aspect_ratio'] - 1.0)  # Prefer square format
        score_components.append(aspect_ratio_score)
        
        return np.mean(score_components) if score_components else 0.5
    
    def _detect_composition_violations(self, 
                                     composition_analysis: Dict[str, Any],
                                     style: str) -> List[str]:
        """Detect compositional violations."""
        violations = []
        
        # Check for style-inappropriate composition
        if style == 'rajput' and composition_analysis['symmetry_score'] < 0.3:
            violations.append("Asymmetrical composition inappropriate for Rajput style")
        
        if style == 'pahari' and composition_analysis['edge_density'] > 0.8:
            violations.append("Overly rigid composition inappropriate for Pahari style")
        
        # Check aspect ratio
        if abs(composition_analysis['aspect_ratio'] - 1.0) > 0.5:
            violations.append("Non-traditional aspect ratio")
        
        return violations
    
    def _generate_composition_recommendations(self, 
                                            composition_analysis: Dict[str, Any],
                                            style: str,
                                            violations: List[str]) -> List[str]:
        """Generate composition recommendations."""
        recommendations = []
        
        if style == 'rajput':
            if composition_analysis['symmetry_score'] < 0.5:
                recommendations.append("Increase symmetrical balance in composition")
            if composition_analysis['edge_density'] < 0.4:
                recommendations.append("Add more defined outlines and geometric elements")
        
        elif style == 'pahari':
            if composition_analysis['spatial_frequency'] > 0.6:
                recommendations.append("Soften composition for more naturalistic flow")
        
        if violations:
            recommendations.append("Address compositional violations for style authenticity")
        
        return recommendations

class CulturalAccuracyEvaluator:
    """Main cultural accuracy evaluator for Ragamala paintings."""
    
    def __init__(self, config: Optional[CulturalEvaluationConfig] = None):
        self.config = config or CulturalEvaluationConfig()
        self.knowledge_base = CulturalKnowledgeBase()
        
        # Initialize evaluators
        self.iconography_evaluator = IconographyEvaluator(self.knowledge_base, self.config)
        self.color_evaluator = ColorPaletteEvaluator(self.knowledge_base, self.config)
        self.composition_evaluator = CompositionEvaluator(self.knowledge_base, self.config)
    
    def evaluate_cultural_accuracy(self, 
                                 image: Image.Image,
                                 raga: str,
                                 style: str,
                                 additional_context: Optional[Dict[str, Any]] = None) -> CulturalEvaluationResult:
        """Comprehensive cultural accuracy evaluation."""
        logger.info(f"Evaluating cultural accuracy for {raga} raga in {style} style")
        
        evaluation_results = {}
        
        # Iconography evaluation
        if self.config.evaluate_iconography:
            iconography_result = self.iconography_evaluator.evaluate_iconography(image, raga, style)
            evaluation_results['iconography'] = iconography_result
        
        # Color palette evaluation
        if self.config.evaluate_color_palette:
            color_result = self.color_evaluator.evaluate_color_palette(image, style, raga)
            evaluation_results['color_palette'] = color_result
        
        # Composition evaluation
        if self.config.evaluate_composition:
            composition_result = self.composition_evaluator.evaluate_composition(image, style)
            evaluation_results['composition'] = composition_result
        
        # Temporal consistency evaluation
        temporal_score = self._evaluate_temporal_consistency(image, style)
        evaluation_results['temporal_consistency'] = {'score': temporal_score}
        
        # Regional specificity evaluation
        regional_score = self._evaluate_regional_specificity(image, style)
        evaluation_results['regional_specificity'] = {'score': regional_score}
        
        # Calculate overall scores
        overall_scores = self._calculate_overall_scores(evaluation_results)
        
        # Collect violations and recommendations
        all_violations = self._collect_violations(evaluation_results)
        all_recommendations = self._collect_recommendations(evaluation_results)
        
        return CulturalEvaluationResult(
            overall_authenticity_score=overall_scores['overall'],
            iconography_score=overall_scores.get('iconography', 0.0),
            color_palette_score=overall_scores.get('color_palette', 0.0),
            composition_score=overall_scores.get('composition', 0.0),
            temporal_consistency_score=overall_scores.get('temporal_consistency', 0.0),
            regional_specificity_score=overall_scores.get('regional_specificity', 0.0),
            detailed_feedback=evaluation_results,
            cultural_violations=all_violations,
            recommendations=all_recommendations
        )
    
    def _evaluate_temporal_consistency(self, image: Image.Image, style: str) -> float:
        """Evaluate temporal consistency (no anachronisms)."""
        # This is a simplified implementation
        # In practice, you would use object detection to identify anachronistic elements
        return 0.8  # Placeholder score
    
    def _evaluate_regional_specificity(self, image: Image.Image, style: str) -> float:
        """Evaluate regional specificity."""
        # This is a simplified implementation
        # In practice, you would analyze region-specific architectural and cultural elements
        return 0.7  # Placeholder score
    
    def _calculate_overall_scores(self, evaluation_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall scores from individual evaluations."""
        scores = {}
        
        # Extract individual scores
        if 'iconography' in evaluation_results:
            scores['iconography'] = evaluation_results['iconography']['iconography_score']
        
        if 'color_palette' in evaluation_results:
            scores['color_palette'] = evaluation_results['color_palette']['color_palette_score']
        
        if 'composition' in evaluation_results:
            scores['composition'] = evaluation_results['composition']['composition_score']
        
        if 'temporal_consistency' in evaluation_results:
            scores['temporal_consistency'] = evaluation_results['temporal_consistency']['score']
        
        if 'regional_specificity' in evaluation_results:
            scores['regional_specificity'] = evaluation_results['regional_specificity']['score']
        
        # Calculate weighted overall score
        weights = {
            'iconography': 0.3,
            'color_palette': 0.25,
            'composition': 0.25,
            'temporal_consistency': 0.1,
            'regional_specificity': 0.1
        }
        
        overall_score = 0.0
        total_weight = 0.0
        
        for component, score in scores.items():
            weight = weights.get(component, 0.0)
            overall_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            overall_score /= total_weight
        
        scores['overall'] = overall_score
        return scores
    
    def _collect_violations(self, evaluation_results: Dict[str, Any]) -> List[str]:
        """Collect all cultural violations."""
        violations = []
        
        for component, result in evaluation_results.items():
            if isinstance(result, dict) and 'violations' in result:
                violations.extend(result['violations'])
        
        return violations
    
    def _collect_recommendations(self, evaluation_results: Dict[str, Any]) -> List[str]:
        """Collect all recommendations."""
        recommendations = []
        
        for component, result in evaluation_results.items():
            if isinstance(result, dict) and 'recommendations' in result:
                recommendations.extend(result['recommendations'])
        
        return recommendations
    
    def evaluate_batch(self, 
                      images: List[Image.Image],
                      ragas: List[str],
                      styles: List[str]) -> List[CulturalEvaluationResult]:
        """Evaluate a batch of images."""
        results = []
        
        for image, raga, style in zip(images, ragas, styles):
            try:
                result = self.evaluate_cultural_accuracy(image, raga, style)
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating image: {e}")
                # Return a default result with low scores
                results.append(CulturalEvaluationResult(
                    overall_authenticity_score=0.0,
                    iconography_score=0.0,
                    color_palette_score=0.0,
                    composition_score=0.0,
                    temporal_consistency_score=0.0,
                    regional_specificity_score=0.0,
                    detailed_feedback={},
                    cultural_violations=[f"Evaluation error: {str(e)}"],
                    recommendations=["Review image for technical issues"]
                ))
        
        return results
    
    def save_evaluation_report(self, 
                             results: Union[CulturalEvaluationResult, List[CulturalEvaluationResult]],
                             output_path: str):
        """Save evaluation report to file."""
        if isinstance(results, CulturalEvaluationResult):
            results = [results]
        
        report = {
            'evaluation_summary': {
                'total_images': len(results),
                'average_authenticity_score': np.mean([r.overall_authenticity_score for r in results]),
                'average_iconography_score': np.mean([r.iconography_score for r in results]),
                'average_color_score': np.mean([r.color_palette_score for r in results]),
                'average_composition_score': np.mean([r.composition_score for r in results])
            },
            'detailed_results': [asdict(result) for result in results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved to {output_path}")

def main():
    """Main function for testing cultural evaluation."""
    # Create configuration
    config = CulturalEvaluationConfig()
    
    # Initialize evaluator
    evaluator = CulturalAccuracyEvaluator(config)
    
    # Create test image
    test_image = Image.new('RGB', (512, 512), color='red')
    
    # Evaluate cultural accuracy
    result = evaluator.evaluate_cultural_accuracy(
        image=test_image,
        raga="bhairav",
        style="rajput"
    )
    
    print(f"Overall Authenticity Score: {result.overall_authenticity_score:.3f}")
    print(f"Iconography Score: {result.iconography_score:.3f}")
    print(f"Color Palette Score: {result.color_palette_score:.3f}")
    print(f"Composition Score: {result.composition_score:.3f}")
    print(f"Violations: {result.cultural_violations}")
    print(f"Recommendations: {result.recommendations}")
    
    # Save report
    evaluator.save_evaluation_report(result, "cultural_evaluation_report.json")
    
    print("Cultural evaluation testing completed!")

if __name__ == "__main__":
    main()
