"""
Prompt Engineering and Template Management for Ragamala Paintings.

This module provides comprehensive prompt engineering functionality for generating
authentic Ragamala paintings with cultural conditioning, including template management,
prompt enhancement, and cultural context integration.
"""

import os
import sys
import json
import re
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
import logging
from collections import defaultdict
from enum import Enum

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

class PromptStyle(Enum):
    """Enumeration of prompt styles."""
    BASIC = "basic"
    DETAILED = "detailed"
    CULTURAL = "cultural"
    ATMOSPHERIC = "atmospheric"
    COMPOSITIONAL = "compositional"
    ARTISTIC = "artistic"
    NARRATIVE = "narrative"

class TimeOfDay(Enum):
    """Enumeration of time periods."""
    DAWN = "dawn"
    MORNING = "morning"
    MIDDAY = "midday"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    TWILIGHT = "twilight"
    NIGHT = "night"
    MIDNIGHT = "midnight"

class Season(Enum):
    """Enumeration of seasons."""
    SPRING = "spring"
    SUMMER = "summer"
    MONSOON = "monsoon"
    AUTUMN = "autumn"
    WINTER = "winter"
    LATE_WINTER = "late_winter"

@dataclass
class PromptTemplate:
    """Template for generating prompts."""
    name: str
    template: str
    variables: List[str]
    style: PromptStyle
    description: str
    examples: List[str] = None
    
    def __post_init__(self):
        if self.examples is None:
            self.examples = []

@dataclass
class CulturalContext:
    """Cultural context for prompt enhancement."""
    raga: Optional[str] = None
    style: Optional[str] = None
    period: Optional[str] = None
    region: Optional[str] = None
    time_of_day: Optional[str] = None
    season: Optional[str] = None
    mood: Optional[str] = None
    deity: Optional[str] = None
    iconography: List[str] = None
    color_palette: List[str] = None
    
    def __post_init__(self):
        if self.iconography is None:
            self.iconography = []
        if self.color_palette is None:
            self.color_palette = []

class RagamalaCulturalDatabase:
    """Database of cultural information for Ragamala paintings."""
    
    def __init__(self):
        self.raga_data = self._load_raga_data()
        self.style_data = self._load_style_data()
        self.iconography_data = self._load_iconography_data()
        self.color_data = self._load_color_data()
    
    def _load_raga_data(self) -> Dict[str, Dict[str, Any]]:
        """Load raga characteristics data."""
        return {
            'bhairav': {
                'time': TimeOfDay.DAWN,
                'season': Season.WINTER,
                'mood': 'devotional and solemn',
                'deity': 'shiva',
                'emotions': ['reverence', 'spirituality', 'awakening'],
                'iconography': ['temple', 'ascetic', 'peacocks', 'sunrise'],
                'colors': ['white', 'saffron', 'gold', 'pale_blue'],
                'setting': 'temple courtyard at dawn',
                'musical_notes': ['Sa', 're', 'Ga', 'Ma', 'Pa', 'dha', 'Ni'],
                'description': 'The morning raga that awakens spiritual consciousness'
            },
            'yaman': {
                'time': TimeOfDay.EVENING,
                'season': Season.SPRING,
                'mood': 'romantic and serene',
                'deity': 'krishna',
                'emotions': ['love', 'beauty', 'longing'],
                'iconography': ['garden', 'lovers', 'moon', 'flowers'],
                'colors': ['blue', 'white', 'pink', 'silver'],
                'setting': 'moonlit garden pavilion',
                'musical_notes': ['Sa', 'Re', 'Ga', 'Ma#', 'Pa', 'Dha', 'Ni'],
                'description': 'The evening raga of love and beauty'
            },
            'malkauns': {
                'time': TimeOfDay.MIDNIGHT,
                'season': Season.MONSOON,
                'mood': 'meditative and mysterious',
                'deity': 'shiva',
                'emotions': ['contemplation', 'introspection', 'depth'],
                'iconography': ['river', 'meditation', 'stars', 'solitude'],
                'colors': ['deep_blue', 'purple', 'black', 'silver'],
                'setting': 'riverside under starlight',
                'musical_notes': ['Sa', 'ga', 'ma', 'dha', 'ni'],
                'description': 'The midnight raga of deep meditation'
            },
            'darbari': {
                'time': TimeOfDay.NIGHT,
                'season': Season.AUTUMN,
                'mood': 'regal and dignified',
                'deity': 'indra',
                'emotions': ['majesty', 'grandeur', 'nobility'],
                'iconography': ['court', 'throne', 'courtiers', 'ceremony'],
                'colors': ['purple', 'gold', 'red', 'blue'],
                'setting': 'royal court in evening',
                'musical_notes': ['Sa', 'Re', 'ga', 'ma', 'Pa', 'dha', 'ni'],
                'description': 'The royal raga of courts and ceremonies'
            },
            'bageshri': {
                'time': TimeOfDay.NIGHT,
                'season': Season.WINTER,
                'mood': 'romantic and yearning',
                'deity': 'krishna',
                'emotions': ['longing', 'devotion', 'romantic_yearning'],
                'iconography': ['waiting_woman', 'lotus', 'moonlight', 'swans'],
                'colors': ['white', 'blue', 'silver', 'pink'],
                'setting': 'moonlit lotus pond',
                'musical_notes': ['Sa', 'ga', 'ma', 'Pa', 'ni', 'Dha'],
                'description': 'The night raga of romantic devotion'
            },
            'todi': {
                'time': TimeOfDay.MORNING,
                'season': Season.SPRING,
                'mood': 'enchanting and charming',
                'deity': 'saraswati',
                'emotions': ['charm', 'allure', 'musical_magic'],
                'iconography': ['musician', 'veena', 'animals', 'forest'],
                'colors': ['yellow', 'green', 'brown', 'gold'],
                'setting': 'forest clearing with enchanted animals',
                'musical_notes': ['Sa', 're', 'ga', 'Ma#', 'Pa', 'dha', 'Ni'],
                'description': 'The morning raga of musical enchantment'
            }
        }
    
    def _load_style_data(self) -> Dict[str, Dict[str, Any]]:
        """Load painting style characteristics."""
        return {
            'rajput': {
                'period': '16th-18th century',
                'region': 'Rajasthan',
                'characteristics': ['bold colors', 'geometric patterns', 'royal themes'],
                'techniques': ['flat perspective', 'decorative borders', 'precise outlines'],
                'color_palette': ['red', 'gold', 'white', 'green'],
                'typical_subjects': ['royal courts', 'hunting scenes', 'religious themes'],
                'composition': 'symmetrical and hierarchical',
                'brushwork': 'precise and controlled'
            },
            'pahari': {
                'period': '17th-19th century',
                'region': 'Himalayan foothills',
                'characteristics': ['soft colors', 'natural settings', 'romantic themes'],
                'techniques': ['atmospheric depth', 'delicate brushwork', 'lyrical quality'],
                'color_palette': ['soft blue', 'green', 'pink', 'white'],
                'typical_subjects': ['krishna leela', 'romantic couples', 'natural landscapes'],
                'composition': 'naturalistic and flowing',
                'brushwork': 'delicate and refined'
            },
            'deccan': {
                'period': '16th-18th century',
                'region': 'Deccan plateau',
                'characteristics': ['persian influence', 'architectural elements', 'formal composition'],
                'techniques': ['geometric precision', 'detailed architecture', 'rich colors'],
                'color_palette': ['deep blue', 'purple', 'gold', 'white'],
                'typical_subjects': ['court scenes', 'architectural settings', 'persian motifs'],
                'composition': 'formal and structured',
                'brushwork': 'precise and elaborate'
            },
            'mughal': {
                'period': '16th-18th century',
                'region': 'Northern India',
                'characteristics': ['elaborate details', 'court scenes', 'naturalistic portraiture'],
                'techniques': ['fine details', 'realistic perspective', 'hierarchical composition'],
                'color_palette': ['rich colors', 'gold', 'jewel tones'],
                'typical_subjects': ['emperor portraits', 'court ceremonies', 'hunting scenes'],
                'composition': 'balanced and hierarchical',
                'brushwork': 'highly refined'
            }
        }
    
    def _load_iconography_data(self) -> Dict[str, List[str]]:
        """Load iconographic elements."""
        return {
            'divine_figures': ['shiva', 'krishna', 'radha', 'vishnu', 'devi', 'ganesha'],
            'human_figures': ['ascetic', 'musician', 'dancer', 'lover', 'king', 'queen', 'courtier'],
            'animals': ['peacock', 'elephant', 'horse', 'deer', 'swan', 'cow', 'tiger'],
            'birds': ['peacock', 'swan', 'crane', 'parrot', 'dove'],
            'architecture': ['temple', 'palace', 'pavilion', 'arch', 'pillar', 'dome', 'courtyard'],
            'natural_elements': ['lotus', 'tree', 'flower', 'river', 'mountain', 'moon', 'sun'],
            'objects': ['veena', 'tabla', 'flute', 'crown', 'jewelry', 'sword', 'book'],
            'textiles': ['sari', 'dhoti', 'turban', 'shawl', 'carpet']
        }
    
    def _load_color_data(self) -> Dict[str, List[str]]:
        """Load color associations."""
        return {
            'traditional_colors': ['vermillion', 'ultramarine', 'gold_leaf', 'ivory', 'lamp_black'],
            'sacred_colors': ['saffron', 'vermillion', 'turmeric_yellow', 'sacred_white'],
            'royal_colors': ['deep_purple', 'royal_blue', 'gold', 'crimson'],
            'natural_colors': ['earth_brown', 'forest_green', 'sky_blue', 'sunset_orange'],
            'seasonal_colors': {
                'spring': ['fresh_green', 'pink', 'yellow', 'light_blue'],
                'summer': ['bright_yellow', 'orange', 'red', 'white'],
                'monsoon': ['dark_blue', 'grey', 'green', 'black'],
                'autumn': ['gold', 'brown', 'orange', 'red'],
                'winter': ['white', 'blue', 'grey', 'silver']
            }
        }
    
    def get_raga_context(self, raga: str) -> Optional[Dict[str, Any]]:
        """Get cultural context for a specific raga."""
        return self.raga_data.get(raga.lower())
    
    def get_style_context(self, style: str) -> Optional[Dict[str, Any]]:
        """Get cultural context for a specific style."""
        return self.style_data.get(style.lower())

class PromptTemplateManager:
    """Manager for prompt templates and cultural enhancement."""
    
    def __init__(self):
        self.cultural_db = RagamalaCulturalDatabase()
        self.templates = self._load_templates()
        self.negative_prompts = self._load_negative_prompts()
        self.enhancement_modifiers = self._load_enhancement_modifiers()
    
    def _load_templates(self) -> Dict[str, PromptTemplate]:
        """Load prompt templates."""
        templates = {}
        
        # Basic template
        templates['basic'] = PromptTemplate(
            name="basic",
            template="A {style} style ragamala painting depicting raga {raga}",
            variables=['style', 'raga'],
            style=PromptStyle.BASIC,
            description="Simple template for basic generation",
            examples=[
                "A rajput style ragamala painting depicting raga bhairav",
                "A pahari style ragamala painting depicting raga yaman"
            ]
        )
        
        # Detailed template
        templates['detailed'] = PromptTemplate(
            name="detailed",
            template="An exquisite {style} miniature painting from {period} illustrating Raga {raga}, {mood} mood suitable for {time_of_day}, featuring {elements} with {colors} palette",
            variables=['style', 'period', 'raga', 'mood', 'time_of_day', 'elements', 'colors'],
            style=PromptStyle.DETAILED,
            description="Detailed template with cultural context",
            examples=[
                "An exquisite rajput miniature painting from 17th century illustrating Raga bhairav, devotional mood suitable for dawn, featuring temple courtyard with peacocks with white and saffron palette"
            ]
        )
        
        # Cultural template
        templates['cultural'] = PromptTemplate(
            name="cultural",
            template="Traditional Indian {style} school ragamala artwork representing {raga} raga, painted with {characteristics} and featuring {iconography}, embodying {emotions}",
            variables=['style', 'raga', 'characteristics', 'iconography', 'emotions'],
            style=PromptStyle.CULTURAL,
            description="Template emphasizing cultural authenticity",
            examples=[
                "Traditional Indian rajput school ragamala artwork representing bhairav raga, painted with bold colors and geometric patterns and featuring temple and peacocks, embodying reverence and spirituality"
            ]
        )
        
        # Atmospheric template
        templates['atmospheric'] = PromptTemplate(
            name="atmospheric",
            template="A {style} ragamala painting of raga {raga} capturing {emotions}, set during {time_of_day} in {season} with {setting} atmosphere, rendered in {colors} tones",
            variables=['style', 'raga', 'emotions', 'time_of_day', 'season', 'setting', 'colors'],
            style=PromptStyle.ATMOSPHERIC,
            description="Template focusing on mood and atmosphere",
            examples=[
                "A pahari ragamala painting of raga yaman capturing love and beauty, set during evening in spring with moonlit garden atmosphere, rendered in blue and silver tones"
            ]
        )
        
        # Compositional template
        templates['compositional'] = PromptTemplate(
            name="compositional",
            template="A masterful {style} style ragamala depicting raga {raga}, composed with {techniques} and featuring {subjects}, painted using {brushwork} with {composition} arrangement",
            variables=['style', 'raga', 'techniques', 'subjects', 'brushwork', 'composition'],
            style=PromptStyle.COMPOSITIONAL,
            description="Template emphasizing artistic technique",
            examples=[
                "A masterful mughal style ragamala depicting raga darbari, composed with fine details and realistic perspective and featuring court ceremony, painted using highly refined brushwork with hierarchical arrangement"
            ]
        )
        
        # Artistic template
        templates['artistic'] = PromptTemplate(
            name="artistic",
            template="An authentic {period} {style} ragamala miniature of raga {raga}, executed in traditional {techniques} with {color_palette} pigments, depicting {narrative} in {composition} style",
            variables=['period', 'style', 'raga', 'techniques', 'color_palette', 'narrative', 'composition'],
            style=PromptStyle.ARTISTIC,
            description="Template for artistic authenticity",
            examples=[
                "An authentic 18th century pahari ragamala miniature of raga yaman, executed in traditional delicate brushwork with soft blue and pink pigments, depicting romantic garden scene in naturalistic style"
            ]
        )
        
        # Narrative template
        templates['narrative'] = PromptTemplate(
            name="narrative",
            template="A {style} ragamala painting narrating the story of raga {raga}, showing {protagonist} in {setting} during {time_of_day}, with {supporting_elements} and {symbolic_objects}",
            variables=['style', 'raga', 'protagonist', 'setting', 'time_of_day', 'supporting_elements', 'symbolic_objects'],
            style=PromptStyle.NARRATIVE,
            description="Template for storytelling approach",
            examples=[
                "A rajput ragamala painting narrating the story of raga bhairav, showing devotee in temple courtyard during dawn, with peacocks and sunrise and sacred symbols"
            ]
        )
        
        return templates
    
    def _load_negative_prompts(self) -> Dict[str, str]:
        """Load negative prompt templates."""
        return {
            'basic': "blurry, low quality, distorted, modern, western art",
            'comprehensive': "blurry, low quality, distorted, modern, western art, cartoon, anime, photography, 3d render, digital art, watermark, signature, text, cropped, out of frame",
            'style_specific': {
                'rajput': "muted colors, realistic perspective, informal composition",
                'pahari': "harsh colors, geometric rigidity, urban settings",
                'deccan': "informal composition, folk art style, crude execution",
                'mughal': "simple details, flat composition, poor craftsmanship"
            },
            'cultural': "inappropriate cultural elements, modern clothing, contemporary objects, western architecture, non-traditional colors"
        }
    
    def _load_enhancement_modifiers(self) -> Dict[str, List[str]]:
        """Load enhancement modifiers for quality improvement."""
        return {
            'quality_boosters': [
                "masterpiece", "best quality", "highly detailed", "intricate details",
                "fine art", "museum quality", "traditional art", "authentic",
                "historically accurate", "culturally authentic"
            ],
            'artistic_styles': [
                "miniature painting", "traditional indian art", "classical painting",
                "manuscript illumination", "court painting", "devotional art"
            ],
            'technical_quality': [
                "sharp focus", "high resolution", "detailed brushwork",
                "rich colors", "perfect composition", "balanced lighting"
            ],
            'cultural_authenticity': [
                "traditional iconography", "classical indian aesthetics",
                "authentic cultural elements", "period appropriate",
                "historically accurate details"
            ]
        }
    
    def generate_prompt(self, 
                       template_name: str,
                       cultural_context: CulturalContext,
                       base_prompt: str = "",
                       enhance_quality: bool = True,
                       add_negative: bool = True) -> Dict[str, str]:
        """Generate a complete prompt with cultural enhancement."""
        
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        template = self.templates[template_name]
        
        # Get cultural data
        raga_data = self.cultural_db.get_raga_context(cultural_context.raga) if cultural_context.raga else {}
        style_data = self.cultural_db.get_style_context(cultural_context.style) if cultural_context.style else {}
        
        # Prepare template variables
        variables = self._prepare_template_variables(cultural_context, raga_data, style_data)
        
        # Generate main prompt
        try:
            main_prompt = template.template.format(**variables)
        except KeyError as e:
            logger.warning(f"Missing variable {e} for template {template_name}")
            main_prompt = template.template
        
        # Add base prompt if provided
        if base_prompt:
            main_prompt = f"{main_prompt}, {base_prompt}"
        
        # Enhance with quality modifiers
        if enhance_quality:
            main_prompt = self._enhance_prompt_quality(main_prompt)
        
        # Generate negative prompt
        negative_prompt = ""
        if add_negative:
            negative_prompt = self._generate_negative_prompt(cultural_context.style)
        
        return {
            'prompt': main_prompt,
            'negative_prompt': negative_prompt,
            'template_used': template_name,
            'variables_used': variables
        }
    
    def _prepare_template_variables(self, 
                                  context: CulturalContext,
                                  raga_data: Dict[str, Any],
                                  style_data: Dict[str, Any]) -> Dict[str, str]:
        """Prepare variables for template formatting."""
        variables = {}
        
        # Basic context
        variables['raga'] = context.raga or 'classical'
        variables['style'] = context.style or 'traditional'
        variables['period'] = context.period or style_data.get('period', 'classical period')
        variables['region'] = context.region or style_data.get('region', 'India')
        
        # Raga-specific
        if raga_data:
            variables['mood'] = raga_data.get('mood', 'serene')
            variables['time_of_day'] = raga_data.get('time', TimeOfDay.EVENING).value
            variables['season'] = raga_data.get('season', Season.SPRING).value
            variables['deity'] = raga_data.get('deity', 'divine')
            variables['emotions'] = ', '.join(raga_data.get('emotions', ['peaceful']))
            variables['setting'] = raga_data.get('setting', 'traditional setting')
            
            # Iconography
            iconography = raga_data.get('iconography', [])
            variables['elements'] = ', '.join(iconography[:3]) if iconography else 'traditional elements'
            variables['iconography'] = ', '.join(iconography) if iconography else 'classical motifs'
            
            # Colors
            colors = raga_data.get('colors', [])
            variables['colors'] = ' and '.join(colors) if colors else 'traditional colors'
            variables['color_palette'] = ', '.join(colors) if colors else 'classical palette'
        
        # Style-specific
        if style_data:
            characteristics = style_data.get('characteristics', [])
            variables['characteristics'] = ', '.join(characteristics) if characteristics else 'traditional characteristics'
            
            techniques = style_data.get('techniques', [])
            variables['techniques'] = ', '.join(techniques) if techniques else 'traditional techniques'
            
            subjects = style_data.get('typical_subjects', [])
            variables['subjects'] = ', '.join(subjects[:2]) if subjects else 'traditional subjects'
            
            variables['brushwork'] = style_data.get('brushwork', 'traditional brushwork')
            variables['composition'] = style_data.get('composition', 'balanced composition')
        
        # Narrative elements
        variables['protagonist'] = self._select_protagonist(raga_data, style_data)
        variables['supporting_elements'] = self._select_supporting_elements(raga_data)
        variables['symbolic_objects'] = self._select_symbolic_objects(raga_data)
        variables['narrative'] = self._create_narrative(raga_data)
        
        return variables
    
    def _select_protagonist(self, raga_data: Dict[str, Any], style_data: Dict[str, Any]) -> str:
        """Select appropriate protagonist for the scene."""
        if raga_data.get('deity'):
            return raga_data['deity']
        
        mood = raga_data.get('mood', '')
        if 'romantic' in mood:
            return 'lovers'
        elif 'devotional' in mood:
            return 'devotee'
        elif 'regal' in mood:
            return 'royal figure'
        else:
            return 'classical figure'
    
    def _select_supporting_elements(self, raga_data: Dict[str, Any]) -> str:
        """Select supporting visual elements."""
        iconography = raga_data.get('iconography', [])
        if len(iconography) >= 2:
            return ' and '.join(iconography[1:3])
        return 'traditional elements'
    
    def _select_symbolic_objects(self, raga_data: Dict[str, Any]) -> str:
        """Select symbolic objects for the scene."""
        iconography = raga_data.get('iconography', [])
        objects = [item for item in iconography if item in ['lotus', 'veena', 'crown', 'book']]
        return ' and '.join(objects) if objects else 'sacred symbols'
    
    def _create_narrative(self, raga_data: Dict[str, Any]) -> str:
        """Create a narrative description."""
        mood = raga_data.get('mood', 'peaceful')
        setting = raga_data.get('setting', 'traditional setting')
        return f"{mood} scene in {setting}"
    
    def _enhance_prompt_quality(self, prompt: str) -> str:
        """Enhance prompt with quality modifiers."""
        quality_modifiers = random.sample(self.enhancement_modifiers['quality_boosters'], 2)
        artistic_modifiers = random.sample(self.enhancement_modifiers['artistic_styles'], 1)
        cultural_modifiers = random.sample(self.enhancement_modifiers['cultural_authenticity'], 1)
        
        enhanced_prompt = f"{prompt}, {', '.join(quality_modifiers + artistic_modifiers + cultural_modifiers)}"
        return enhanced_prompt
    
    def _generate_negative_prompt(self, style: Optional[str] = None) -> str:
        """Generate appropriate negative prompt."""
        base_negative = self.negative_prompts['comprehensive']
        cultural_negative = self.negative_prompts['cultural']
        
        negative_parts = [base_negative, cultural_negative]
        
        if style and style in self.negative_prompts['style_specific']:
            style_negative = self.negative_prompts['style_specific'][style]
            negative_parts.append(style_negative)
        
        return ', '.join(negative_parts)
    
    def get_template_suggestions(self, 
                               raga: Optional[str] = None,
                               style: Optional[str] = None) -> List[str]:
        """Get template suggestions based on cultural context."""
        suggestions = []
        
        # Default suggestions
        suggestions.extend(['detailed', 'cultural', 'atmospheric'])
        
        # Raga-specific suggestions
        if raga:
            raga_data = self.cultural_db.get_raga_context(raga)
            if raga_data:
                mood = raga_data.get('mood', '')
                if 'romantic' in mood:
                    suggestions.append('narrative')
                elif 'devotional' in mood:
                    suggestions.append('cultural')
                elif 'regal' in mood:
                    suggestions.append('compositional')
        
        # Style-specific suggestions
        if style:
            style_data = self.cultural_db.get_style_context(style)
            if style_data:
                if 'elaborate' in str(style_data.get('characteristics', [])):
                    suggestions.append('artistic')
                if 'court' in str(style_data.get('typical_subjects', [])):
                    suggestions.append('compositional')
        
        return list(set(suggestions))
    
    def validate_cultural_consistency(self, 
                                    context: CulturalContext) -> Dict[str, Any]:
        """Validate cultural consistency of the context."""
        issues = []
        score = 1.0
        
        if context.raga and context.time_of_day:
            raga_data = self.cultural_db.get_raga_context(context.raga)
            if raga_data:
                expected_time = raga_data.get('time')
                if expected_time and expected_time.value != context.time_of_day:
                    issues.append(f"Time mismatch: {context.raga} should be performed at {expected_time.value}")
                    score -= 0.2
        
        if context.raga and context.season:
            raga_data = self.cultural_db.get_raga_context(context.raga)
            if raga_data:
                expected_season = raga_data.get('season')
                if expected_season and expected_season.value != context.season:
                    issues.append(f"Season mismatch: {context.raga} is associated with {expected_season.value}")
                    score -= 0.2
        
        return {
            'consistency_score': max(0.0, score),
            'issues': issues,
            'is_consistent': len(issues) == 0
        }
    
    def create_prompt_variations(self, 
                               base_context: CulturalContext,
                               num_variations: int = 3) -> List[Dict[str, str]]:
        """Create variations of a prompt with different templates."""
        variations = []
        
        # Get template suggestions
        suggested_templates = self.get_template_suggestions(
            base_context.raga, 
            base_context.style
        )
        
        # Select templates for variations
        templates_to_use = suggested_templates[:num_variations]
        if len(templates_to_use) < num_variations:
            all_templates = list(self.templates.keys())
            remaining = [t for t in all_templates if t not in templates_to_use]
            templates_to_use.extend(remaining[:num_variations - len(templates_to_use)])
        
        # Generate variations
        for template_name in templates_to_use:
            try:
                prompt_data = self.generate_prompt(template_name, base_context)
                prompt_data['variation_type'] = template_name
                variations.append(prompt_data)
            except Exception as e:
                logger.warning(f"Failed to generate variation with template {template_name}: {e}")
        
        return variations

def create_cultural_context(raga: str = None,
                          style: str = None,
                          **kwargs) -> CulturalContext:
    """Factory function to create cultural context."""
    return CulturalContext(
        raga=raga,
        style=style,
        **kwargs
    )

def main():
    """Main function for testing prompt templates."""
    # Initialize template manager
    template_manager = PromptTemplateManager()
    
    # Test cultural contexts
    test_contexts = [
        create_cultural_context(raga="bhairav", style="rajput"),
        create_cultural_context(raga="yaman", style="pahari"),
        create_cultural_context(raga="malkauns", style="deccan"),
        create_cultural_context(raga="darbari", style="mughal")
    ]
    
    # Test prompt generation
    for context in test_contexts:
        print(f"\nTesting {context.raga} + {context.style}:")
        
        # Validate cultural consistency
        validation = template_manager.validate_cultural_consistency(context)
        print(f"Cultural consistency: {validation['consistency_score']:.2f}")
        
        # Get template suggestions
        suggestions = template_manager.get_template_suggestions(context.raga, context.style)
        print(f"Suggested templates: {suggestions}")
        
        # Generate prompt with detailed template
        prompt_data = template_manager.generate_prompt('detailed', context)
        print(f"Generated prompt: {prompt_data['prompt']}")
        print(f"Negative prompt: {prompt_data['negative_prompt']}")
        
        # Create variations
        variations = template_manager.create_prompt_variations(context, 2)
        print(f"Generated {len(variations)} variations")

if __name__ == "__main__":
    main()
