"""
Gradio Web Interface for Ragamala Painting Generation.
Provides an intuitive web interface for generating authentic Ragamala paintings
using the fine-tuned SDXL 1.0 model with cultural conditioning.
"""

import asyncio
import base64
import io
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import gradio as gr
import numpy as np
import pandas as pd
import requests
from PIL import Image
import torch

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.inference.generator import RagamalaGenerator, GenerationConfig
from src.evaluation.metrics import EvaluationMetrics
from src.evaluation.cultural_evaluator import CulturalAccuracyEvaluator
from src.utils.logging_utils import setup_logger

# Setup logging
logger = setup_logger(__name__)

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "raga_demo_key")

# Global variables
generator = None
evaluation_metrics = None
cultural_evaluator = None

# Raga and Style definitions
RAGAS = {
    "bhairav": {
        "name": "Bhairav",
        "time": "Dawn",
        "mood": "Devotional, Solemn",
        "description": "A morning raga that evokes the feeling of dawn and spiritual awakening",
        "colors": ["White", "Saffron", "Gold", "Pale Blue"],
        "iconography": ["Temple", "Peacocks", "Sunrise", "Ascetic", "Trident"]
    },
    "yaman": {
        "name": "Yaman",
        "time": "Evening",
        "mood": "Romantic, Serene",
        "description": "An evening raga expressing beauty, romance, and tranquility",
        "colors": ["Blue", "White", "Pink", "Silver"],
        "iconography": ["Garden", "Lovers", "Moon", "Flowers", "Pavilion"]
    },
    "malkauns": {
        "name": "Malkauns",
        "time": "Midnight",
        "mood": "Meditative, Mysterious",
        "description": "A deep night raga that evokes contemplation and mystery",
        "colors": ["Deep Blue", "Purple", "Black", "Silver"],
        "iconography": ["River", "Meditation", "Stars", "Solitude", "Caves"]
    },
    "darbari": {
        "name": "Darbari",
        "time": "Late Evening",
        "mood": "Regal, Dignified",
        "description": "A court raga expressing majesty and royal grandeur",
        "colors": ["Purple", "Gold", "Red", "Royal Blue"],
        "iconography": ["Court", "Throne", "Courtiers", "Ceremony", "Elephants"]
    },
    "bageshri": {
        "name": "Bageshri",
        "time": "Night",
        "mood": "Yearning, Devotional",
        "description": "A night raga expressing longing and patient devotion",
        "colors": ["White", "Blue", "Silver", "Pink"],
        "iconography": ["Waiting Woman", "Lotus Pond", "Moonlight", "Swans"]
    },
    "todi": {
        "name": "Todi",
        "time": "Morning",
        "mood": "Enchanting, Charming",
        "description": "A morning raga that captivates with its musical charm",
        "colors": ["Yellow", "Green", "Brown", "Gold"],
        "iconography": ["Musician", "Veena", "Animals", "Forest", "Birds"]
    }
}

STYLES = {
    "rajput": {
        "name": "Rajput",
        "period": "16th-18th Century",
        "region": "Rajasthan",
        "description": "Bold colors, geometric patterns, and royal themes",
        "characteristics": ["Bold Colors", "Geometric Patterns", "Flat Perspective", "Royal Themes"]
    },
    "pahari": {
        "name": "Pahari",
        "period": "17th-19th Century", 
        "region": "Himalayan Foothills",
        "description": "Soft colors, naturalistic style, and lyrical quality",
        "characteristics": ["Soft Colors", "Naturalistic", "Lyrical", "Delicate Brushwork"]
    },
    "deccan": {
        "name": "Deccan",
        "period": "16th-18th Century",
        "region": "Deccan Plateau",
        "description": "Persian influence with architectural elements",
        "characteristics": ["Persian Influence", "Architectural", "Formal", "Geometric Precision"]
    },
    "mughal": {
        "name": "Mughal",
        "period": "16th-18th Century",
        "region": "Northern India",
        "description": "Elaborate details and naturalistic portraiture",
        "characteristics": ["Elaborate Details", "Naturalistic", "Imperial", "Fine Miniature Work"]
    }
}

PROMPT_TEMPLATES = {
    "basic": "A {style} style ragamala painting of raga {raga}",
    "detailed": "An exquisite {style} miniature painting depicting raga {raga}, featuring traditional iconography and {mood} atmosphere",
    "cultural": "A traditional Indian {style} school ragamala artwork representing raga {raga}, painted in {period} style with authentic cultural elements",
    "atmospheric": "A {style} ragamala painting of raga {raga} set during {time}, capturing the {mood} mood with appropriate colors and symbols",
    "narrative": "A {style} style ragamala painting illustrating the story of raga {raga}, showing {iconography} in a {period} artistic tradition"
}

# Utility functions
def load_model():
    """Load the Ragamala generation model."""
    global generator, evaluation_metrics, cultural_evaluator
    
    try:
        # Initialize generator
        config = GenerationConfig(
            model_path="models/sdxl-ragamala",
            device="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        generator = RagamalaGenerator(config)
        
        # Initialize evaluation components
        evaluation_metrics = EvaluationMetrics()
        cultural_evaluator = CulturalAccuracyEvaluator()
        
        logger.info("Model loaded successfully")
        return "Model loaded successfully"
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return f"Failed to load model: {str(e)}"

def call_api(endpoint: str, data: Dict) -> Dict:
    """Call the API endpoint."""
    try:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            f"{API_BASE_URL}/{endpoint}",
            json=data,
            headers=headers,
            timeout=300
        )
        
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API call failed: {e}")
        raise gr.Error(f"API call failed: {str(e)}")

def create_prompt(raga: str, style: str, template: str, custom_elements: str = "") -> str:
    """Create a culturally appropriate prompt."""
    raga_info = RAGAS.get(raga, {})
    style_info = STYLES.get(style, {})
    
    template_str = PROMPT_TEMPLATES.get(template, PROMPT_TEMPLATES["detailed"])
    
    prompt = template_str.format(
        raga=raga_info.get("name", raga),
        style=style_info.get("name", style),
        mood=raga_info.get("mood", "serene"),
        time=raga_info.get("time", "day"),
        period=style_info.get("period", "traditional"),
        iconography=", ".join(raga_info.get("iconography", [])[:3])
    )
    
    if custom_elements:
        prompt += f", {custom_elements}"
    
    return prompt

def generate_image_local(
    raga: str,
    style: str,
    prompt_template: str,
    custom_elements: str,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    width: int,
    height: int,
    use_cultural_conditioning: bool,
    calculate_metrics: bool
) -> Tuple[Image.Image, str, str, str]:
    """Generate image using local model."""
    global generator, evaluation_metrics, cultural_evaluator
    
    if not generator:
        raise gr.Error("Model not loaded. Please load the model first.")
    
    try:
        # Create prompt
        prompt = create_prompt(raga, style, prompt_template, custom_elements)
        
        # Generate image
        start_time = time.time()
        
        generation_params = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "generator": torch.Generator().manual_seed(seed) if seed > 0 else None
        }
        
        if use_cultural_conditioning:
            generation_params["cultural_conditioning"] = {
                "raga": raga,
                "style": style,
                "strict_authenticity": True
            }
        
        images = generator.generate(**generation_params)
        generation_time = time.time() - start_time
        
        image = images[0]
        
        # Calculate metrics if requested
        metrics_text = ""
        if calculate_metrics and evaluation_metrics:
            try:
                metrics = evaluation_metrics.calculate_comprehensive_metrics(image)
                metrics_text = f"""
Quality Metrics:
- Overall Score: {metrics.get('overall_score', 0):.3f}
- Sharpness: {metrics.get('sharpness', 0):.3f}
- Color Harmony: {metrics.get('color_harmony', 0):.3f}
- Composition: {metrics.get('composition_balance', 0):.3f}
"""
            except Exception as e:
                metrics_text = f"Metrics calculation failed: {str(e)}"
        
        # Cultural authenticity assessment
        authenticity_text = ""
        if cultural_evaluator:
            try:
                auth_result = cultural_evaluator.assess_cultural_authenticity(
                    image, raga, style
                )
                authenticity_text = f"""
Cultural Authenticity:
- Overall Score: {auth_result.get('overall_authenticity', 0):.3f}
- Iconographic Accuracy: {auth_result.get('iconographic_accuracy', 0):.3f}
- Style Consistency: {auth_result.get('style_consistency', 0):.3f}
- Temporal Accuracy: {auth_result.get('temporal_consistency', 0):.3f}
"""
            except Exception as e:
                authenticity_text = f"Authenticity assessment failed: {str(e)}"
        
        # Generation info
        info_text = f"""
Generation Information:
- Raga: {RAGAS[raga]['name']} ({RAGAS[raga]['time']})
- Style: {STYLES[style]['name']} ({STYLES[style]['period']})
- Prompt: {prompt}
- Generation Time: {generation_time:.2f} seconds
- Steps: {num_inference_steps}
- Guidance Scale: {guidance_scale}
- Seed: {seed if seed > 0 else 'Random'}
- Resolution: {width}x{height}
"""
        
        return image, info_text, metrics_text, authenticity_text
        
    except Exception as e:
        logger.error(f"Local generation failed: {e}")
        raise gr.Error(f"Generation failed: {str(e)}")

def generate_image_api(
    raga: str,
    style: str,
    prompt_template: str,
    custom_elements: str,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    width: int,
    height: int,
    use_cultural_conditioning: bool,
    calculate_metrics: bool
) -> Tuple[Image.Image, str, str, str]:
    """Generate image using API."""
    try:
        # Create prompt
        prompt = create_prompt(raga, style, prompt_template, custom_elements)
        
        # Prepare API request
        request_data = {
            "raga": raga,
            "style": style,
            "generation_params": {
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "width": width,
                "height": height,
                "seed": seed if seed > 0 else None
            },
            "prompt_config": {
                "template": prompt_template,
                "custom_prompt": prompt if custom_elements else None
            },
            "cultural_config": {
                "strict_authenticity": use_cultural_conditioning,
                "include_iconography": True,
                "temporal_accuracy": True
            },
            "output_config": {
                "return_base64": True,
                "calculate_quality_metrics": calculate_metrics
            }
        }
        
        # Call API
        response = call_api("generate", request_data)
        
        # Process response
        if response["status"] == "completed" and response["images"]:
            image_data = response["images"][0]
            
            # Decode base64 image
            if image_data.get("image_data"):
                image_bytes = base64.b64decode(image_data["image_data"])
                image = Image.open(io.BytesIO(image_bytes))
            else:
                raise gr.Error("No image data received from API")
            
            # Extract information
            metadata = image_data.get("metadata", {})
            quality_metrics = image_data.get("quality_metrics", {})
            cultural_auth = image_data.get("cultural_authenticity", {})
            
            # Format information
            info_text = f"""
Generation Information:
- Raga: {RAGAS[raga]['name']} ({RAGAS[raga]['time']})
- Style: {STYLES[style]['name']} ({STYLES[style]['period']})
- Prompt: {metadata.get('prompt_used', prompt)}
- Generation Time: {response.get('generation_time', 0):.2f} seconds
- Image ID: {metadata.get('image_id', 'N/A')}
- Model Version: {metadata.get('model_version', 'N/A')}
"""
            
            metrics_text = ""
            if quality_metrics:
                metrics_text = f"""
Quality Metrics:
- Overall Score: {quality_metrics.get('overall_score', 0):.3f}
- Sharpness: {quality_metrics.get('sharpness', 0):.3f}
- Color Harmony: {quality_metrics.get('color_harmony', 0):.3f}
- Composition: {quality_metrics.get('composition_balance', 0):.3f}
- Quality Level: {quality_metrics.get('quality_level', 'N/A')}
"""
            
            authenticity_text = ""
            if cultural_auth:
                authenticity_text = f"""
Cultural Authenticity:
- Overall Score: {cultural_auth.get('overall_authenticity', 0):.3f}
- Iconographic Accuracy: {cultural_auth.get('iconographic_accuracy', 0):.3f}
- Style Consistency: {cultural_auth.get('style_consistency', 0):.3f}
- Temporal Accuracy: {cultural_auth.get('temporal_consistency', 0):.3f}
- Authenticity Level: {cultural_auth.get('authenticity_level', 'N/A')}
"""
            
            return image, info_text, metrics_text, authenticity_text
        else:
            raise gr.Error(f"API generation failed: {response.get('message', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"API generation failed: {e}")
        raise gr.Error(f"API generation failed: {str(e)}")

def update_raga_info(raga: str) -> str:
    """Update raga information display."""
    raga_info = RAGAS.get(raga, {})
    
    info_text = f"""
**{raga_info.get('name', raga)}**

**Time of Day:** {raga_info.get('time', 'N/A')}
**Mood:** {raga_info.get('mood', 'N/A')}

**Description:** {raga_info.get('description', 'No description available')}

**Traditional Colors:** {', '.join(raga_info.get('colors', []))}

**Iconographic Elements:** {', '.join(raga_info.get('iconography', []))}
"""
    
    return info_text

def update_style_info(style: str) -> str:
    """Update style information display."""
    style_info = STYLES.get(style, {})
    
    info_text = f"""
**{style_info.get('name', style)} School**

**Period:** {style_info.get('period', 'N/A')}
**Region:** {style_info.get('region', 'N/A')}

**Description:** {style_info.get('description', 'No description available')}

**Characteristics:** {', '.join(style_info.get('characteristics', []))}
"""
    
    return info_text

def batch_generate(
    raga_list: str,
    style_list: str,
    prompt_template: str,
    num_inference_steps: int,
    guidance_scale: float,
    use_api: bool
) -> Tuple[List[Image.Image], str]:
    """Generate multiple images in batch."""
    try:
        # Parse input lists
        ragas = [r.strip() for r in raga_list.split(',') if r.strip()]
        styles = [s.strip() for s in style_list.split(',') if s.strip()]
        
        if not ragas or not styles:
            raise gr.Error("Please provide at least one raga and one style")
        
        images = []
        results_info = []
        
        # Generate for each combination
        for raga in ragas:
            for style in styles:
                if raga not in RAGAS or style not in STYLES:
                    continue
                
                try:
                    if use_api:
                        image, info, _, _ = generate_image_api(
                            raga, style, prompt_template, "",
                            num_inference_steps, guidance_scale,
                            -1, 1024, 1024, True, False
                        )
                    else:
                        image, info, _, _ = generate_image_local(
                            raga, style, prompt_template, "",
                            num_inference_steps, guidance_scale,
                            -1, 1024, 1024, True, False
                        )
                    
                    images.append(image)
                    results_info.append(f"{RAGAS[raga]['name']} - {STYLES[style]['name']}")
                    
                except Exception as e:
                    logger.error(f"Failed to generate {raga}-{style}: {e}")
                    continue
        
        info_text = f"""
Batch Generation Results:
- Total Combinations: {len(ragas)} ragas × {len(styles)} styles = {len(ragas) * len(styles)}
- Successfully Generated: {len(images)}
- Failed: {len(ragas) * len(styles) - len(images)}

Generated Combinations:
{chr(10).join(f"- {info}" for info in results_info)}
"""
        
        return images, info_text
        
    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        raise gr.Error(f"Batch generation failed: {str(e)}")

def save_image(image: Image.Image, raga: str, style: str) -> str:
    """Save generated image to disk."""
    try:
        if image is None:
            return "No image to save"
        
        # Create output directory
        output_dir = Path("outputs/gradio_generated")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{raga}_{style}_{timestamp}.png"
        filepath = output_dir / filename
        
        # Save image
        image.save(filepath, "PNG")
        
        return f"Image saved as: {filepath}"
        
    except Exception as e:
        logger.error(f"Failed to save image: {e}")
        return f"Failed to save image: {str(e)}"

# Create Gradio interface
def create_interface():
    """Create the main Gradio interface."""
    
    with gr.Blocks(
        title="Ragamala Painting Generator",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            font-family: 'Georgia', serif;
        }
        .raga-info, .style-info {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #007bff;
        }
        """
    ) as interface:
        
        gr.Markdown("""
        # Ragamala Painting Generator
        
        Generate authentic Ragamala paintings using AI fine-tuned on traditional Indian miniature paintings.
        Ragamala paintings are visual representations of ragas (musical modes) in Indian classical music,
        combining artistic beauty with musical and spiritual significance.
        """)
        
        with gr.Tabs():
            # Main Generation Tab
            with gr.Tab("Generate Painting"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Configuration")
                        
                        # Model selection
                        use_api = gr.Checkbox(
                            label="Use API (recommended)",
                            value=True,
                            info="Use the production API for better results"
                        )
                        
                        # Basic parameters
                        raga_dropdown = gr.Dropdown(
                            choices=list(RAGAS.keys()),
                            value="bhairav",
                            label="Raga",
                            info="Select the musical raga to depict"
                        )
                        
                        style_dropdown = gr.Dropdown(
                            choices=list(STYLES.keys()),
                            value="rajput",
                            label="Painting Style",
                            info="Select the regional painting style"
                        )
                        
                        prompt_template_dropdown = gr.Dropdown(
                            choices=list(PROMPT_TEMPLATES.keys()),
                            value="detailed",
                            label="Prompt Template",
                            info="Choose the prompt complexity level"
                        )
                        
                        custom_elements = gr.Textbox(
                            label="Custom Elements",
                            placeholder="Add custom elements to the painting...",
                            lines=2,
                            info="Optional: Add specific elements or descriptions"
                        )
                        
                        # Advanced parameters
                        with gr.Accordion("Advanced Settings", open=False):
                            num_inference_steps = gr.Slider(
                                minimum=10,
                                maximum=100,
                                value=30,
                                step=5,
                                label="Inference Steps",
                                info="More steps = higher quality but slower generation"
                            )
                            
                            guidance_scale = gr.Slider(
                                minimum=1.0,
                                maximum=20.0,
                                value=7.5,
                                step=0.5,
                                label="Guidance Scale",
                                info="How closely to follow the prompt"
                            )
                            
                            seed = gr.Number(
                                label="Seed",
                                value=-1,
                                precision=0,
                                info="Random seed (-1 for random)"
                            )
                            
                            width = gr.Slider(
                                minimum=512,
                                maximum=1024,
                                value=1024,
                                step=64,
                                label="Width"
                            )
                            
                            height = gr.Slider(
                                minimum=512,
                                maximum=1024,
                                value=1024,
                                step=64,
                                label="Height"
                            )
                            
                            use_cultural_conditioning = gr.Checkbox(
                                label="Cultural Conditioning",
                                value=True,
                                info="Apply cultural authenticity constraints"
                            )
                            
                            calculate_metrics = gr.Checkbox(
                                label="Calculate Quality Metrics",
                                value=False,
                                info="Compute image quality assessment (slower)"
                            )
                        
                        # Generate button
                        generate_btn = gr.Button(
                            "Generate Ragamala Painting",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=2):
                        # Output image
                        output_image = gr.Image(
                            label="Generated Ragamala Painting",
                            type="pil",
                            height=600
                        )
                        
                        # Save button
                        save_btn = gr.Button("Save Image", variant="secondary")
                        save_status = gr.Textbox(label="Save Status", interactive=False)
                
                # Information panels
                with gr.Row():
                    with gr.Column():
                        raga_info = gr.Markdown(
                            value=update_raga_info("bhairav"),
                            elem_classes=["raga-info"]
                        )
                    
                    with gr.Column():
                        style_info = gr.Markdown(
                            value=update_style_info("rajput"),
                            elem_classes=["style-info"]
                        )
                
                # Results information
                with gr.Row():
                    with gr.Column():
                        generation_info = gr.Textbox(
                            label="Generation Information",
                            lines=8,
                            interactive=False
                        )
                    
                    with gr.Column():
                        quality_metrics = gr.Textbox(
                            label="Quality Metrics",
                            lines=8,
                            interactive=False
                        )
                    
                    with gr.Column():
                        authenticity_info = gr.Textbox(
                            label="Cultural Authenticity",
                            lines=8,
                            interactive=False
                        )
            
            # Batch Generation Tab
            with gr.Tab("Batch Generation"):
                gr.Markdown("### Generate Multiple Paintings")
                
                with gr.Row():
                    with gr.Column():
                        batch_ragas = gr.Textbox(
                            label="Ragas (comma-separated)",
                            value="bhairav,yaman,malkauns",
                            info="Enter raga names separated by commas"
                        )
                        
                        batch_styles = gr.Textbox(
                            label="Styles (comma-separated)",
                            value="rajput,pahari",
                            info="Enter style names separated by commas"
                        )
                        
                        batch_template = gr.Dropdown(
                            choices=list(PROMPT_TEMPLATES.keys()),
                            value="detailed",
                            label="Prompt Template"
                        )
                        
                        batch_steps = gr.Slider(
                            minimum=10,
                            maximum=50,
                            value=20,
                            step=5,
                            label="Inference Steps"
                        )
                        
                        batch_guidance = gr.Slider(
                            minimum=1.0,
                            maximum=15.0,
                            value=7.5,
                            step=0.5,
                            label="Guidance Scale"
                        )
                        
                        batch_use_api = gr.Checkbox(
                            label="Use API",
                            value=True
                        )
                        
                        batch_generate_btn = gr.Button(
                            "Generate Batch",
                            variant="primary"
                        )
                    
                    with gr.Column():
                        batch_output = gr.Gallery(
                            label="Generated Paintings",
                            columns=3,
                            rows=2,
                            height=600
                        )
                        
                        batch_info = gr.Textbox(
                            label="Batch Results",
                            lines=10,
                            interactive=False
                        )
            
            # Cultural Guide Tab
            with gr.Tab("Cultural Guide"):
                gr.Markdown("### Understanding Ragas and Styles")
                
                with gr.Tabs():
                    with gr.Tab("Ragas"):
                        gr.Markdown("""
                        Ragas are melodic frameworks in Indian classical music, each with specific emotional and temporal associations.
                        In Ragamala paintings, these musical concepts are translated into visual narratives.
                        """)
                        
                        for raga_key, raga_data in RAGAS.items():
                            with gr.Accordion(f"{raga_data['name']} - {raga_data['time']}", open=False):
                                gr.Markdown(f"""
                                **Mood:** {raga_data['mood']}
                                
                                **Description:** {raga_data['description']}
                                
                                **Traditional Colors:** {', '.join(raga_data['colors'])}
                                
                                **Iconographic Elements:** {', '.join(raga_data['iconography'])}
                                """)
                    
                    with gr.Tab("Painting Styles"):
                        gr.Markdown("""
                        Different regional schools of Indian miniature painting developed distinct characteristics
                        in terms of color palette, composition, and artistic techniques.
                        """)
                        
                        for style_key, style_data in STYLES.items():
                            with gr.Accordion(f"{style_data['name']} School", open=False):
                                gr.Markdown(f"""
                                **Period:** {style_data['period']}
                                
                                **Region:** {style_data['region']}
                                
                                **Description:** {style_data['description']}
                                
                                **Characteristics:** {', '.join(style_data['characteristics'])}
                                """)
            
            # Model Management Tab
            with gr.Tab("Model Management"):
                gr.Markdown("### Model Status and Controls")
                
                with gr.Row():
                    with gr.Column():
                        model_status = gr.Textbox(
                            label="Model Status",
                            value="Not loaded",
                            interactive=False
                        )
                        
                        load_model_btn = gr.Button("Load Model", variant="primary")
                        
                        gr.Markdown("""
                        ### Model Information
                        
                        This interface uses a fine-tuned SDXL 1.0 model specifically trained on Ragamala paintings.
                        The model has been conditioned to understand the cultural and artistic nuances of different
                        ragas and painting styles.
                        
                        **Features:**
                        - Cultural conditioning for authenticity
                        - Style-specific adaptations
                        - Raga-aware iconography
                        - Traditional color palette adherence
                        """)
                    
                    with gr.Column():
                        api_status = gr.HTML("""
                        <div style="padding: 20px; background-color: #f8f9fa; border-radius: 8px;">
                            <h4>API Status</h4>
                            <p>API URL: <code>""" + API_BASE_URL + """</code></p>
                            <p>Status: <span id="api-status">Checking...</span></p>
                        </div>
                        """)
        
        # Event handlers
        def generate_wrapper(*args):
            """Wrapper for generation function selection."""
            use_api_flag = args[0]
            if use_api_flag:
                return generate_image_api(*args[1:])
            else:
                return generate_image_local(*args[1:])
        
        # Connect event handlers
        generate_btn.click(
            fn=generate_wrapper,
            inputs=[
                use_api, raga_dropdown, style_dropdown, prompt_template_dropdown,
                custom_elements, num_inference_steps, guidance_scale, seed,
                width, height, use_cultural_conditioning, calculate_metrics
            ],
            outputs=[output_image, generation_info, quality_metrics, authenticity_info]
        )
        
        save_btn.click(
            fn=save_image,
            inputs=[output_image, raga_dropdown, style_dropdown],
            outputs=[save_status]
        )
        
        batch_generate_btn.click(
            fn=batch_generate,
            inputs=[
                batch_ragas, batch_styles, batch_template,
                batch_steps, batch_guidance, batch_use_api
            ],
            outputs=[batch_output, batch_info]
        )
        
        raga_dropdown.change(
            fn=update_raga_info,
            inputs=[raga_dropdown],
            outputs=[raga_info]
        )
        
        style_dropdown.change(
            fn=update_style_info,
            inputs=[style_dropdown],
            outputs=[style_info]
        )
        
        load_model_btn.click(
            fn=load_model,
            outputs=[model_status]
        )
    
    return interface

# Main execution
if __name__ == "__main__":
    # Create and launch interface
    demo = create_interface()
    
    # Launch configuration
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True,
        quiet=False,
        favicon_path=None,
        ssl_keyfile=None,
        ssl_certfile=None,
        ssl_keyfile_password=None,
        max_threads=40,
        auth=None,  # Add authentication if needed
        auth_message="Enter credentials to access Ragamala Generator",
        prevent_thread_lock=False,
        height=800,
        width="100%",
        encrypt=False,
        show_tips=True,
        enable_queue=True,
        max_size=20,
        api_open=True
    )
