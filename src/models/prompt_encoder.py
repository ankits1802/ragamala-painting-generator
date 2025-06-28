"""
Text Conditioning and Prompt Encoding Module for Ragamala Paintings.

This module provides comprehensive text conditioning functionality for SDXL fine-tuning
on Ragamala paintings, including dual text encoder support, cultural prompt engineering,
and raga-style specific conditioning.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import json
import numpy as np
import re
from collections import defaultdict

# Transformers imports
from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    AutoTokenizer,
    T5EncoderModel,
    T5Tokenizer
)

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

@dataclass
class PromptEncodingConfig:
    """Configuration for prompt encoding."""
    # Model paths
    clip_model_name: str = "openai/clip-vit-large-patch14"
    clip_model_name_2: str = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    t5_model_name: str = "google/t5-v1_1-xxl"
    
    # Text processing
    max_length: int = 77
    max_length_2: int = 77
    truncation: bool = True
    padding: str = "max_length"
    
    # Cultural conditioning
    enable_cultural_conditioning: bool = True
    cultural_weight: float = 0.3
    
    # Prompt engineering
    enable_prompt_weighting: bool = True
    enable_negative_prompting: bool = True
    
    # Output settings
    output_hidden_states: bool = True
    return_dict: bool = True

class CulturalPromptTemplates:
    """Templates for cultural prompt engineering."""
    
    def __init__(self):
        self.raga_descriptors = {
            'bhairav': {
                'time': 'dawn',
                'mood': 'devotional and solemn',
                'colors': 'white, saffron, gold',
                'elements': 'temple, ascetic figure, peacocks, morning light',
                'emotions': 'reverence, spirituality, awakening'
            },
            'yaman': {
                'time': 'evening',
                'mood': 'romantic and serene',
                'colors': 'blue, white, pink, silver',
                'elements': 'garden, lovers, moonlight, flowers',
                'emotions': 'love, beauty, longing'
            },
            'malkauns': {
                'time': 'midnight',
                'mood': 'meditative and mysterious',
                'colors': 'deep blue, purple, black, silver',
                'elements': 'river, meditation, stars, solitude',
                'emotions': 'contemplation, introspection, depth'
            },
            'darbari': {
                'time': 'late evening',
                'mood': 'regal and dignified',
                'colors': 'purple, gold, red, blue',
                'elements': 'royal court, throne, courtiers, ceremony',
                'emotions': 'majesty, grandeur, nobility'
            },
            'bageshri': {
                'time': 'late night',
                'mood': 'romantic and yearning',
                'colors': 'white, blue, silver, pink',
                'elements': 'waiting woman, lotus pond, moonlight, swans',
                'emotions': 'longing, devotion, romantic yearning'
            },
            'todi': {
                'time': 'morning',
                'mood': 'enchanting and charming',
                'colors': 'yellow, green, brown, gold',
                'elements': 'musician with veena, forest animals, enchantment',
                'emotions': 'charm, allure, musical magic'
            }
        }
        
        self.style_descriptors = {
            'rajput': {
                'characteristics': 'bold colors, geometric patterns, royal themes',
                'techniques': 'flat perspective, decorative borders, precise outlines',
                'palette': 'red, gold, white, green, vibrant colors',
                'period': '16th-18th century',
                'region': 'Rajasthan, Mewar, Marwar'
            },
            'pahari': {
                'characteristics': 'soft colors, natural settings, romantic themes',
                'techniques': 'atmospheric depth, delicate brushwork, lyrical quality',
                'palette': 'soft blues, greens, pinks, pastels',
                'period': '17th-19th century',
                'region': 'Himalayan foothills, Kangra, Basohli'
            },
            'deccan': {
                'characteristics': 'Persian influence, formal composition, architectural elements',
                'techniques': 'geometric precision, rich colors, detailed architecture',
                'palette': 'deep blue, purple, gold, white',
                'period': '16th-18th century',
                'region': 'Deccan plateau, Golconda, Bijapur'
            },
            'mughal': {
                'characteristics': 'elaborate details, court scenes, naturalistic portraiture',
                'techniques': 'fine details, realistic perspective, hierarchical composition',
                'palette': 'rich colors, gold, jewel tones, intricate patterns',
                'period': '16th-18th century',
                'region': 'Northern India, Delhi, Agra'
            }
        }
        
        self.base_templates = {
            'basic': "A {style} style ragamala painting depicting raga {raga}",
            'detailed': "An exquisite {style} miniature painting from {period} illustrating Raga {raga}, {mood} mood suitable for {time}, featuring {elements}",
            'cultural': "Traditional Indian {style} school ragamala artwork representing {raga} raga, painted with {colors} palette and {characteristics}",
            'atmospheric': "A {style} ragamala painting of raga {raga} capturing {emotions}, set during {time} with {elements} in {colors} tones",
            'compositional': "A masterful {style} style ragamala depicting raga {raga}, composed with {techniques} and featuring {elements}"
        }
    
    def get_enhanced_prompt(self, 
                          base_prompt: str,
                          raga: Optional[str] = None,
                          style: Optional[str] = None,
                          template_type: str = 'detailed') -> str:
        """Generate enhanced prompt with cultural context."""
        if not raga and not style:
            return base_prompt
        
        # Get descriptors
        raga_desc = self.raga_descriptors.get(raga.lower(), {}) if raga else {}
        style_desc = self.style_descriptors.get(style.lower(), {}) if style else {}
        
        # Select template
        template = self.base_templates.get(template_type, self.base_templates['detailed'])
        
        # Fill template
        enhanced_prompt = template.format(
            style=style or 'traditional',
            raga=raga or 'classical',
            period=style_desc.get('period', 'classical period'),
            mood=raga_desc.get('mood', 'serene'),
            time=raga_desc.get('time', 'appropriate time'),
            elements=raga_desc.get('elements', 'traditional elements'),
            colors=raga_desc.get('colors', style_desc.get('palette', 'traditional colors')),
            emotions=raga_desc.get('emotions', 'peaceful emotions'),
            characteristics=style_desc.get('characteristics', 'traditional characteristics'),
            techniques=style_desc.get('techniques', 'traditional techniques')
        )
        
        # Append base prompt if provided
        if base_prompt and base_prompt.strip():
            enhanced_prompt = f"{enhanced_prompt}, {base_prompt}"
        
        return enhanced_prompt

class PromptWeightingParser:
    """Parser for prompt weighting syntax."""
    
    def __init__(self):
        self.weight_pattern = re.compile(r'\(([^)]+):(\d*\.?\d+)\)')
        self.emphasis_pattern = re.compile(r'\(([^)]+)\)')
        self.de_emphasis_pattern = re.compile(r'\[([^\]]+)\]')
    
    def parse_weighted_prompt(self, prompt: str) -> List[Tuple[str, float]]:
        """Parse prompt with weight syntax (text:weight)."""
        tokens = []
        remaining = prompt
        
        # Find weighted tokens
        for match in self.weight_pattern.finditer(prompt):
            text = match.group(1)
            weight = float(match.group(2))
            tokens.append((text, weight))
            remaining = remaining.replace(match.group(0), '')
        
        # Find emphasized tokens ()
        for match in self.emphasis_pattern.finditer(remaining):
            text = match.group(1)
            tokens.append((text, 1.1))
            remaining = remaining.replace(match.group(0), '')
        
        # Find de-emphasized tokens []
        for match in self.de_emphasis_pattern.finditer(remaining):
            text = match.group(1)
            tokens.append((text, 0.9))
            remaining = remaining.replace(match.group(0), '')
        
        # Add remaining text with default weight
        if remaining.strip():
            tokens.append((remaining.strip(), 1.0))
        
        return tokens

class DualTextEncoder(nn.Module):
    """Dual text encoder for SDXL with cultural conditioning."""
    
    def __init__(self, config: PromptEncodingConfig):
        super().__init__()
        self.config = config
        
        # Load tokenizers
        self.tokenizer = CLIPTokenizer.from_pretrained(config.clip_model_name)
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(config.clip_model_name_2)
        
        # Load text encoders
        self.text_encoder = CLIPTextModel.from_pretrained(config.clip_model_name)
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(config.clip_model_name_2)
        
        # Cultural conditioning components
        if config.enable_cultural_conditioning:
            self.cultural_projection = nn.Linear(
                self.text_encoder.config.hidden_size + self.text_encoder_2.config.hidden_size,
                self.text_encoder.config.hidden_size + self.text_encoder_2.config.hidden_size
            )
            self.cultural_attention = nn.MultiheadAttention(
                embed_dim=self.text_encoder.config.hidden_size + self.text_encoder_2.config.hidden_size,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
        
        # Prompt weighting components
        if config.enable_prompt_weighting:
            self.weight_parser = PromptWeightingParser()
        
        # Cultural templates
        self.cultural_templates = CulturalPromptTemplates()
        
        # Freeze text encoders initially
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
    
    def encode_prompt(self,
                     prompt: Union[str, List[str]],
                     prompt_2: Optional[Union[str, List[str]]] = None,
                     device: torch.device = None,
                     num_images_per_prompt: int = 1,
                     do_classifier_free_guidance: bool = True,
                     negative_prompt: Optional[Union[str, List[str]]] = None,
                     negative_prompt_2: Optional[Union[str, List[str]]] = None,
                     prompt_embeds: Optional[torch.FloatTensor] = None,
                     negative_prompt_embeds: Optional[torch.FloatTensor] = None,
                     pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
                     negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
                     lora_scale: Optional[float] = None,
                     clip_skip: Optional[int] = None,
                     raga: Optional[str] = None,
                     style: Optional[str] = None) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Encode prompts using dual text encoders with cultural conditioning.
        
        Args:
            prompt: Primary prompt for CLIP-G
            prompt_2: Secondary prompt for CLIP-L (style prompt)
            device: Target device
            num_images_per_prompt: Number of images per prompt
            do_classifier_free_guidance: Whether to use CFG
            negative_prompt: Negative prompt for CLIP-G
            negative_prompt_2: Negative prompt for CLIP-L
            prompt_embeds: Pre-computed prompt embeddings
            negative_prompt_embeds: Pre-computed negative prompt embeddings
            pooled_prompt_embeds: Pre-computed pooled prompt embeddings
            negative_pooled_prompt_embeds: Pre-computed negative pooled prompt embeddings
            lora_scale: LoRA scale for text encoders
            clip_skip: Number of layers to skip in CLIP
            raga: Raga name for cultural conditioning
            style: Style name for cultural conditioning
            
        Returns:
            Tuple of (prompt_embeds, pooled_prompt_embeds)
        """
        device = device or self.text_encoder.device
        
        # Handle batch inputs
        if isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)
        
        # Enhance prompts with cultural context
        if raga or style:
            if isinstance(prompt, str):
                enhanced_prompt = self.cultural_templates.get_enhanced_prompt(prompt, raga, style)
                prompt = [enhanced_prompt]
            else:
                prompt = [self.cultural_templates.get_enhanced_prompt(p, raga, style) for p in prompt]
        
        # Set secondary prompt if not provided
        if prompt_2 is None:
            prompt_2 = prompt
        elif isinstance(prompt_2, str):
            prompt_2 = [prompt_2]
        
        # Encode with first text encoder (CLIP-G)
        text_inputs = self.tokenizer(
            prompt,
            padding=self.config.padding,
            max_length=self.config.max_length,
            truncation=self.config.truncation,
            return_tensors="pt"
        )
        
        text_input_ids = text_inputs.input_ids.to(device)
        
        # Handle prompt weighting for first encoder
        if self.config.enable_prompt_weighting and hasattr(self, 'weight_parser'):
            prompt_embeds_1 = self._encode_with_weighting(
                text_input_ids, self.text_encoder, self.tokenizer
            )
        else:
            prompt_embeds_1 = self.text_encoder(
                text_input_ids,
                output_hidden_states=self.config.output_hidden_states
            )
        
        # Extract hidden states and pooled output
        if self.config.output_hidden_states:
            if clip_skip is None:
                prompt_embeds_1 = prompt_embeds_1.hidden_states[-2]
            else:
                prompt_embeds_1 = prompt_embeds_1.hidden_states[-(clip_skip + 1)]
        else:
            prompt_embeds_1 = prompt_embeds_1.last_hidden_state
        
        # Encode with second text encoder (CLIP-L)
        text_inputs_2 = self.tokenizer_2(
            prompt_2,
            padding=self.config.padding,
            max_length=self.config.max_length_2,
            truncation=self.config.truncation,
            return_tensors="pt"
        )
        
        text_input_ids_2 = text_inputs_2.input_ids.to(device)
        
        # Handle prompt weighting for second encoder
        if self.config.enable_prompt_weighting and hasattr(self, 'weight_parser'):
            prompt_embeds_2 = self._encode_with_weighting(
                text_input_ids_2, self.text_encoder_2, self.tokenizer_2
            )
        else:
            prompt_embeds_2 = self.text_encoder_2(
                text_input_ids_2,
                output_hidden_states=self.config.output_hidden_states
            )
        
        # Extract hidden states and pooled output
        pooled_prompt_embeds = prompt_embeds_2[0]
        
        if self.config.output_hidden_states:
            if clip_skip is None:
                prompt_embeds_2 = prompt_embeds_2.hidden_states[-2]
            else:
                prompt_embeds_2 = prompt_embeds_2.hidden_states[-(clip_skip + 1)]
        else:
            prompt_embeds_2 = prompt_embeds_2.last_hidden_state
        
        # Concatenate embeddings from both encoders
        prompt_embeds = torch.concat([prompt_embeds_1, prompt_embeds_2], dim=-1)
        
        # Apply cultural conditioning
        if self.config.enable_cultural_conditioning and (raga or style):
            prompt_embeds = self._apply_cultural_conditioning(
                prompt_embeds, raga, style
            )
        
        # Duplicate for multiple images per prompt
        if num_images_per_prompt > 1:
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
            
            pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed * num_images_per_prompt, -1)
        
        # Handle negative prompts for classifier-free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt
            
            # Encode negative prompts
            negative_prompt_embeds, negative_pooled_prompt_embeds = self._encode_negative_prompts(
                negative_prompt, negative_prompt_2, device, num_images_per_prompt, clip_skip
            )
            
            # Concatenate positive and negative embeddings
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])
        
        return prompt_embeds, pooled_prompt_embeds
    
    def _encode_with_weighting(self,
                              input_ids: torch.Tensor,
                              text_encoder: nn.Module,
                              tokenizer) -> torch.FloatTensor:
        """Encode text with attention weighting."""
        # This is a simplified implementation
        # In practice, you would implement attention weighting here
        return text_encoder(input_ids, output_hidden_states=self.config.output_hidden_states)
    
    def _apply_cultural_conditioning(self,
                                   prompt_embeds: torch.FloatTensor,
                                   raga: Optional[str],
                                   style: Optional[str]) -> torch.FloatTensor:
        """Apply cultural conditioning to prompt embeddings."""
        if not hasattr(self, 'cultural_projection'):
            return prompt_embeds
        
        # Project embeddings
        conditioned_embeds = self.cultural_projection(prompt_embeds)
        
        # Apply attention-based conditioning
        attended_embeds, _ = self.cultural_attention(
            conditioned_embeds, conditioned_embeds, conditioned_embeds
        )
        
        # Weighted combination
        cultural_weight = self.config.cultural_weight
        final_embeds = (1 - cultural_weight) * prompt_embeds + cultural_weight * attended_embeds
        
        return final_embeds
    
    def _encode_negative_prompts(self,
                                negative_prompt: str,
                                negative_prompt_2: str,
                                device: torch.device,
                                num_images_per_prompt: int,
                                clip_skip: Optional[int]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Encode negative prompts for classifier-free guidance."""
        # Tokenize negative prompts
        uncond_tokens = self.tokenizer(
            [negative_prompt],
            padding=self.config.padding,
            max_length=self.config.max_length,
            truncation=self.config.truncation,
            return_tensors="pt"
        )
        
        uncond_tokens_2 = self.tokenizer_2(
            [negative_prompt_2],
            padding=self.config.padding,
            max_length=self.config.max_length_2,
            truncation=self.config.truncation,
            return_tensors="pt"
        )
        
        # Encode negative prompts
        negative_prompt_embeds_1 = self.text_encoder(
            uncond_tokens.input_ids.to(device),
            output_hidden_states=self.config.output_hidden_states
        )
        
        negative_prompt_embeds_2 = self.text_encoder_2(
            uncond_tokens_2.input_ids.to(device),
            output_hidden_states=self.config.output_hidden_states
        )
        
        # Extract embeddings
        if self.config.output_hidden_states:
            if clip_skip is None:
                negative_prompt_embeds_1 = negative_prompt_embeds_1.hidden_states[-2]
                negative_prompt_embeds_2 = negative_prompt_embeds_2.hidden_states[-2]
            else:
                negative_prompt_embeds_1 = negative_prompt_embeds_1.hidden_states[-(clip_skip + 1)]
                negative_prompt_embeds_2 = negative_prompt_embeds_2.hidden_states[-(clip_skip + 1)]
        else:
            negative_prompt_embeds_1 = negative_prompt_embeds_1.last_hidden_state
            negative_prompt_embeds_2 = negative_prompt_embeds_2.last_hidden_state
        
        # Get pooled embeddings
        negative_pooled_prompt_embeds = negative_prompt_embeds_2[0] if hasattr(negative_prompt_embeds_2, '__getitem__') else negative_prompt_embeds_2.pooler_output
        
        # Concatenate embeddings
        negative_prompt_embeds = torch.concat([negative_prompt_embeds_1, negative_prompt_embeds_2], dim=-1)
        
        # Duplicate for multiple images per prompt
        if num_images_per_prompt > 1:
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.repeat(num_images_per_prompt, 1, 1)
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(num_images_per_prompt, 1)
        
        return negative_prompt_embeds, negative_pooled_prompt_embeds

class PromptEngineeringUtils:
    """Utilities for prompt engineering and optimization."""
    
    @staticmethod
    def create_style_prompt(base_prompt: str, style: str) -> str:
        """Create style-specific prompt for CLIP-L."""
        style_modifiers = {
            'rajput': 'bold colors, geometric patterns, royal themes, traditional rajasthani style',
            'pahari': 'soft colors, delicate brushwork, natural landscapes, romantic themes',
            'deccan': 'persian influence, architectural elements, formal composition, rich colors',
            'mughal': 'elaborate details, court scenes, naturalistic style, fine miniature painting'
        }
        
        modifier = style_modifiers.get(style.lower(), f'{style} style')
        return f"{base_prompt}, {modifier}"
    
    @staticmethod
    def create_negative_prompt(style: Optional[str] = None) -> str:
        """Create comprehensive negative prompt."""
        base_negative = [
            "blurry", "low quality", "distorted", "modern", "western art",
            "cartoon", "anime", "photography", "3d render", "digital art",
            "watermark", "signature", "text", "cropped", "out of frame"
        ]
        
        # Add style-specific negative terms
        if style:
            style_negative = {
                'rajput': ['muted colors', 'realistic perspective'],
                'pahari': ['harsh colors', 'geometric rigidity'],
                'deccan': ['informal composition', 'folk art style'],
                'mughal': ['simple details', 'flat composition']
            }
            base_negative.extend(style_negative.get(style.lower(), []))
        
        return ", ".join(base_negative)
    
    @staticmethod
    def optimize_prompt_length(prompt: str, max_length: int = 77) -> str:
        """Optimize prompt length for tokenizer limits."""
        # Simple truncation with preservation of important terms
        words = prompt.split()
        if len(words) <= max_length:
            return prompt
        
        # Preserve important cultural terms
        important_terms = [
            'ragamala', 'raga', 'rajput', 'pahari', 'deccan', 'mughal',
            'miniature', 'painting', 'traditional', 'indian'
        ]
        
        preserved_words = []
        other_words = []
        
        for word in words:
            if any(term in word.lower() for term in important_terms):
                preserved_words.append(word)
            else:
                other_words.append(word)
        
        # Combine preserved words with truncated other words
        available_length = max_length - len(preserved_words)
        final_words = preserved_words + other_words[:available_length]
        
        return " ".join(final_words)

def create_prompt_encoder(config: PromptEncodingConfig) -> DualTextEncoder:
    """Factory function to create prompt encoder."""
    return DualTextEncoder(config)

def main():
    """Main function for testing prompt encoding."""
    # Create configuration
    config = PromptEncodingConfig()
    
    # Create encoder
    encoder = create_prompt_encoder(config)
    
    # Test encoding
    test_prompts = [
        "A beautiful ragamala painting",
        "Traditional Indian miniature art"
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    
    for prompt in test_prompts:
        print(f"Testing prompt: {prompt}")
        
        # Encode prompt
        prompt_embeds, pooled_embeds = encoder.encode_prompt(
            prompt=prompt,
            device=device,
            raga="bhairav",
            style="rajput"
        )
        
        print(f"Prompt embeddings shape: {prompt_embeds.shape}")
        print(f"Pooled embeddings shape: {pooled_embeds.shape}")
        print()
    
    print("Prompt encoding testing completed successfully!")

if __name__ == "__main__":
    main()
