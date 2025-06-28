"""
Streamlit Dashboard for Ragamala Painting Generation.
Provides a comprehensive web interface for generating, analyzing, and managing
Ragamala paintings using the fine-tuned SDXL 1.0 model.
"""

import asyncio
import base64
import io
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
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

# Page configuration
st.set_page_config(
    page_title="Ragamala Painting Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/ragamala-generator',
        'Report a bug': 'https://github.com/your-repo/ragamala-generator/issues',
        'About': "Ragamala Painting Generator using SDXL 1.0"
    }
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-family: 'Georgia', serif;
    }
    
    .raga-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .style-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    
    .generation-info {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #28a745;
    }
    
    .error-info {
        background: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dc3545;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "raga_demo_key")

# Initialize session state
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []
if 'generation_history' not in st.session_state:
    st.session_state.generation_history = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'api_status' not in st.session_state:
    st.session_state.api_status = "Unknown"

# Raga and Style definitions
RAGAS = {
    "bhairav": {
        "name": "Bhairav",
        "time": "Dawn (5-7 AM)",
        "mood": "Devotional, Solemn, Spiritual",
        "description": "A morning raga that evokes the feeling of dawn and spiritual awakening. Associated with Lord Shiva.",
        "colors": ["White", "Saffron", "Gold", "Pale Blue"],
        "iconography": ["Temple", "Peacocks", "Sunrise", "Ascetic", "Trident", "Meditation"],
        "emotions": ["Reverence", "Devotion", "Awakening", "Purity"],
        "season": "All seasons",
        "deity": "Shiva"
    },
    "yaman": {
        "name": "Yaman",
        "time": "Evening (6-9 PM)",
        "mood": "Romantic, Serene, Beautiful",
        "description": "An evening raga expressing beauty, romance, and tranquility. Often called the king of ragas.",
        "colors": ["Blue", "White", "Pink", "Silver"],
        "iconography": ["Garden", "Lovers", "Moon", "Flowers", "Pavilion", "Swans"],
        "emotions": ["Romance", "Beauty", "Serenity", "Love"],
        "season": "Spring",
        "deity": "Krishna"
    },
    "malkauns": {
        "name": "Malkauns",
        "time": "Midnight (12-3 AM)",
        "mood": "Meditative, Mysterious, Deep",
        "description": "A deep night raga that evokes contemplation, mystery, and introspection.",
        "colors": ["Deep Blue", "Purple", "Black", "Silver"],
        "iconography": ["River", "Meditation", "Stars", "Solitude", "Caves", "Sage"],
        "emotions": ["Contemplation", "Mystery", "Depth", "Introspection"],
        "season": "Monsoon",
        "deity": "Shiva"
    },
    "darbari": {
        "name": "Darbari",
        "time": "Late Evening (9-12 PM)",
        "mood": "Regal, Dignified, Majestic",
        "description": "A court raga expressing majesty, dignity, and royal grandeur.",
        "colors": ["Purple", "Gold", "Red", "Royal Blue"],
        "iconography": ["Court", "Throne", "Courtiers", "Ceremony", "Elephants", "Palace"],
        "emotions": ["Majesty", "Dignity", "Power", "Grandeur"],
        "season": "Winter",
        "deity": "Indra"
    },
    "bageshri": {
        "name": "Bageshri",
        "time": "Night (9 PM-12 AM)",
        "mood": "Yearning, Devotional, Patient",
        "description": "A night raga expressing longing, patient devotion, and faithful waiting.",
        "colors": ["White", "Blue", "Silver", "Pink"],
        "iconography": ["Waiting Woman", "Lotus Pond", "Moonlight", "Swans", "Garden"],
        "emotions": ["Yearning", "Devotion", "Patience", "Faith"],
        "season": "Spring",
        "deity": "Krishna"
    },
    "todi": {
        "name": "Todi",
        "time": "Morning (9-12 PM)",
        "mood": "Enchanting, Charming, Musical",
        "description": "A morning raga that captivates with its musical charm and enchanting quality.",
        "colors": ["Yellow", "Green", "Brown", "Gold"],
        "iconography": ["Musician", "Veena", "Animals", "Forest", "Birds", "Nature"],
        "emotions": ["Enchantment", "Charm", "Musical Joy", "Harmony"],
        "season": "Spring",
        "deity": "Saraswati"
    }
}

STYLES = {
    "rajput": {
        "name": "Rajput",
        "period": "16th-18th Century",
        "region": "Rajasthan, Western India",
        "description": "Bold colors, geometric patterns, and royal themes with flat perspective",
        "characteristics": ["Bold Colors", "Geometric Patterns", "Flat Perspective", "Royal Themes", "Gold Detailing"],
        "typical_colors": ["Red", "Gold", "White", "Green", "Blue"],
        "subjects": ["Royal Courts", "Hunting Scenes", "Religious Themes", "Portraits"],
        "techniques": ["Flat Modeling", "Bold Outlines", "Decorative Patterns"]
    },
    "pahari": {
        "name": "Pahari",
        "period": "17th-19th Century",
        "region": "Himalayan Foothills",
        "description": "Soft colors, naturalistic style, and lyrical quality with delicate brushwork",
        "characteristics": ["Soft Colors", "Naturalistic", "Lyrical", "Delicate Brushwork", "Landscape Integration"],
        "typical_colors": ["Soft Blue", "Green", "Pink", "White", "Yellow"],
        "subjects": ["Krishna Legends", "Ragamala", "Portraits", "Nature Scenes"],
        "techniques": ["Soft Modeling", "Fine Details", "Atmospheric Perspective"]
    },
    "deccan": {
        "name": "Deccan",
        "period": "16th-18th Century",
        "region": "Deccan Plateau, Southern India",
        "description": "Persian influence with architectural elements and formal composition",
        "characteristics": ["Persian Influence", "Architectural", "Formal", "Geometric Precision", "Rich Textures"],
        "typical_colors": ["Deep Blue", "Purple", "Gold", "White", "Crimson"],
        "subjects": ["Court Scenes", "Portraits", "Literary Themes", "Architecture"],
        "techniques": ["Precise Lines", "Formal Composition", "Rich Detailing"]
    },
    "mughal": {
        "name": "Mughal",
        "period": "16th-18th Century",
        "region": "Northern India",
        "description": "Elaborate details, naturalistic portraiture, and imperial grandeur",
        "characteristics": ["Elaborate Details", "Naturalistic", "Imperial", "Fine Miniature Work", "European Influence"],
        "typical_colors": ["Rich Colors", "Gold", "Jewel Tones", "Earth Colors"],
        "subjects": ["Imperial Portraits", "Court Scenes", "Historical Events", "Nature Studies"],
        "techniques": ["Realistic Modeling", "Fine Brushwork", "Detailed Backgrounds"]
    }
}

PROMPT_TEMPLATES = {
    "basic": "A {style} style ragamala painting of raga {raga}",
    "detailed": "An exquisite {style} miniature painting from {period} depicting raga {raga}, featuring {iconography} with {mood} atmosphere and traditional {colors} palette",
    "cultural": "A traditional Indian {style} school ragamala artwork representing raga {raga}, painted in {period} style with authentic cultural elements including {iconography} and {emotions}",
    "atmospheric": "A {style} ragamala painting of raga {raga} set during {time}, capturing the {mood} mood with {colors} colors and {iconography} in {season} setting",
    "narrative": "A {style} style ragamala painting illustrating the story of raga {raga}, showing {iconography} with {deity} presence in a {period} artistic tradition",
    "technical": "A masterpiece {style} miniature painting of raga {raga}, highly detailed, intricate, traditional iconography, authentic colors, museum quality, {period} style"
}

# Utility functions
@st.cache_data
def load_sample_images():
    """Load sample images for reference."""
    sample_dir = Path("assets/samples")
    if sample_dir.exists():
        return list(sample_dir.glob("*.png"))
    return []

def check_api_status():
    """Check API health status."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            st.session_state.api_status = "Online"
            return True
        else:
            st.session_state.api_status = "Error"
            return False
    except Exception as e:
        st.session_state.api_status = "Offline"
        return False

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
        st.error(f"API call failed: {str(e)}")
        return {"error": str(e)}

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
        iconography=", ".join(raga_info.get("iconography", [])[:3]),
        colors=", ".join(raga_info.get("colors", [])[:3]),
        emotions=", ".join(raga_info.get("emotions", [])[:2]),
        deity=raga_info.get("deity", "divine"),
        season=raga_info.get("season", "appropriate season")
    )
    
    if custom_elements:
        prompt += f", {custom_elements}"
    
    return prompt

def save_generation_to_history(generation_data: Dict):
    """Save generation data to history."""
    generation_data['timestamp'] = datetime.now().isoformat()
    st.session_state.generation_history.append(generation_data)
    
    # Keep only last 50 generations
    if len(st.session_state.generation_history) > 50:
        st.session_state.generation_history = st.session_state.generation_history[-50:]

def export_generation_history():
    """Export generation history as JSON."""
    if st.session_state.generation_history:
        history_json = json.dumps(st.session_state.generation_history, indent=2)
        return history_json
    return None

# Main application functions
def render_sidebar():
    """Render the sidebar with navigation and settings."""
    with st.sidebar:
        st.markdown('<div class="sidebar-header">', unsafe_allow_html=True)
        st.title("Ragamala Generator")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # API Status
        api_online = check_api_status()
        status_color = "green" if api_online else "red"
        st.markdown(f"**API Status:** <span style='color: {status_color}'>{st.session_state.api_status}</span>", 
                   unsafe_allow_html=True)
        
        # Navigation
        st.markdown("---")
        page = st.selectbox(
            "Navigate to:",
            ["🎨 Generate", "📊 Analytics", "📚 Cultural Guide", "⚙️ Settings", "📜 History"]
        )
        
        # Quick stats
        st.markdown("---")
        st.markdown("**Quick Stats**")
        st.metric("Generated Images", len(st.session_state.generated_images))
        st.metric("Total Generations", len(st.session_state.generation_history))
        
        if st.session_state.generation_history:
            avg_time = np.mean([g.get('generation_time', 0) for g in st.session_state.generation_history])
            st.metric("Avg Generation Time", f"{avg_time:.1f}s")
        
        return page

def render_raga_info_card(raga_key: str):
    """Render raga information card."""
    raga = RAGAS[raga_key]
    
    st.markdown(f"""
    <div class="raga-card">
        <h3>{raga['name']}</h3>
        <p><strong>Time:</strong> {raga['time']}</p>
        <p><strong>Mood:</strong> {raga['mood']}</p>
        <p><strong>Deity:</strong> {raga['deity']}</p>
        <p>{raga['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander(f"More about {raga['name']}"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Traditional Colors:**")
            for color in raga['colors']:
                st.markdown(f"- {color}")
            
            st.markdown("**Emotions:**")
            for emotion in raga['emotions']:
                st.markdown(f"- {emotion}")
        
        with col2:
            st.markdown("**Iconographic Elements:**")
            for icon in raga['iconography']:
                st.markdown(f"- {icon}")
            
            st.markdown(f"**Season:** {raga['season']}")

def render_style_info_card(style_key: str):
    """Render style information card."""
    style = STYLES[style_key]
    
    st.markdown(f"""
    <div class="style-card">
        <h3>{style['name']} School</h3>
        <p><strong>Period:</strong> {style['period']}</p>
        <p><strong>Region:</strong> {style['region']}</p>
        <p>{style['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander(f"More about {style['name']} Style"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Characteristics:**")
            for char in style['characteristics']:
                st.markdown(f"- {char}")
            
            st.markdown("**Typical Colors:**")
            for color in style['typical_colors']:
                st.markdown(f"- {color}")
        
        with col2:
            st.markdown("**Common Subjects:**")
            for subject in style['subjects']:
                st.markdown(f"- {subject}")
            
            st.markdown("**Techniques:**")
            for technique in style['techniques']:
                st.markdown(f"- {technique}")

def generate_page():
    """Main generation page."""
    st.markdown('<h1 class="main-header">Generate Ragamala Painting</h1>', unsafe_allow_html=True)
    
    # Main generation interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Configuration")
        
        # Basic parameters
        raga = st.selectbox(
            "Select Raga:",
            options=list(RAGAS.keys()),
            format_func=lambda x: f"{RAGAS[x]['name']} ({RAGAS[x]['time']})"
        )
        
        style = st.selectbox(
            "Select Style:",
            options=list(STYLES.keys()),
            format_func=lambda x: f"{STYLES[x]['name']} ({STYLES[x]['period']})"
        )
        
        template = st.selectbox(
            "Prompt Template:",
            options=list(PROMPT_TEMPLATES.keys()),
            index=1
        )
        
        custom_elements = st.text_area(
            "Custom Elements:",
            placeholder="Add specific elements, colors, or details...",
            height=100
        )
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            num_images = st.slider("Number of Images:", 1, 4, 1)
            
            col_a, col_b = st.columns(2)
            with col_a:
                steps = st.slider("Inference Steps:", 10, 100, 30, 5)
                guidance = st.slider("Guidance Scale:", 1.0, 20.0, 7.5, 0.5)
            
            with col_b:
                width = st.selectbox("Width:", [512, 768, 1024], index=2)
                height = st.selectbox("Height:", [512, 768, 1024], index=2)
            
            seed = st.number_input("Seed (-1 for random):", -1, 999999, -1)
            
            cultural_conditioning = st.checkbox("Cultural Conditioning", True)
            calculate_metrics = st.checkbox("Calculate Quality Metrics", False)
        
        # Generate button
        if st.button("🎨 Generate Ragamala Painting", type="primary", use_container_width=True):
            generate_image(raga, style, template, custom_elements, num_images, 
                         steps, guidance, width, height, seed, cultural_conditioning, calculate_metrics)
    
    with col2:
        st.markdown("### Preview & Results")
        
        # Show generated images
        if st.session_state.generated_images:
            latest_generation = st.session_state.generated_images[-1]
            
            if 'images' in latest_generation:
                for i, img_data in enumerate(latest_generation['images']):
                    st.image(img_data['image'], caption=f"Generated Image {i+1}", use_column_width=True)
                    
                    # Download button
                    img_bytes = io.BytesIO()
                    img_data['image'].save(img_bytes, format='PNG')
                    st.download_button(
                        f"Download Image {i+1}",
                        img_bytes.getvalue(),
                        f"ragamala_{raga}_{style}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        "image/png"
                    )
            
            # Show generation info
            if 'generation_info' in latest_generation:
                with st.expander("Generation Information"):
                    st.json(latest_generation['generation_info'])
        
        else:
            st.info("Configure settings and click 'Generate' to create your Ragamala painting")
    
    # Information cards
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        render_raga_info_card(raga)
    
    with col2:
        render_style_info_card(style)

def generate_image(raga, style, template, custom_elements, num_images, 
                  steps, guidance, width, height, seed, cultural_conditioning, calculate_metrics):
    """Generate image using API."""
    
    # Create prompt
    prompt = create_prompt(raga, style, template, custom_elements)
    
    # Show prompt preview
    with st.expander("Generated Prompt Preview"):
        st.code(prompt)
    
    # Prepare API request
    request_data = {
        "raga": raga,
        "style": style,
        "num_images": num_images,
        "generation_params": {
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "width": width,
            "height": height,
            "seed": seed if seed > 0 else None
        },
        "prompt_config": {
            "template": template,
            "custom_prompt": prompt if custom_elements else None
        },
        "cultural_config": {
            "strict_authenticity": cultural_conditioning,
            "include_iconography": True,
            "temporal_accuracy": True
        },
        "output_config": {
            "return_base64": True,
            "calculate_quality_metrics": calculate_metrics
        }
    }
    
    # Show loading
    with st.spinner("Generating Ragamala painting... This may take a few minutes."):
        start_time = time.time()
        
        # Call API
        response = call_api("generate", request_data)
        
        generation_time = time.time() - start_time
        
        if "error" not in response and response.get("status") == "completed":
            # Process successful response
            images = []
            
            for img_data in response.get("images", []):
                if img_data.get("image_data"):
                    # Decode base64 image
                    image_bytes = base64.b64decode(img_data["image_data"])
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    images.append({
                        "image": image,
                        "metadata": img_data.get("metadata", {}),
                        "quality_metrics": img_data.get("quality_metrics", {}),
                        "cultural_authenticity": img_data.get("cultural_authenticity", {})
                    })
            
            # Save to session state
            generation_data = {
                "raga": raga,
                "style": style,
                "prompt": prompt,
                "images": images,
                "generation_time": generation_time,
                "generation_info": {
                    "request_id": response.get("request_id"),
                    "parameters": request_data,
                    "response": response
                }
            }
            
            st.session_state.generated_images.append(generation_data)
            save_generation_to_history(generation_data)
            
            st.success(f"Successfully generated {len(images)} image(s) in {generation_time:.1f} seconds!")
            st.rerun()
            
        else:
            error_msg = response.get("detail", "Unknown error occurred")
            st.error(f"Generation failed: {error_msg}")

def analytics_page():
    """Analytics and statistics page."""
    st.markdown('<h1 class="main-header">Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    if not st.session_state.generation_history:
        st.info("No generation data available. Generate some images first!")
        return
    
    # Create analytics data
    df = pd.DataFrame(st.session_state.generation_history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Generations", len(df))
    
    with col2:
        avg_time = df['generation_time'].mean()
        st.metric("Avg Generation Time", f"{avg_time:.1f}s")
    
    with col3:
        total_images = df['images'].apply(len).sum()
        st.metric("Total Images", total_images)
    
    with col4:
        if len(df) > 1:
            recent_avg = df.tail(10)['generation_time'].mean()
            previous_avg = df.head(-10)['generation_time'].mean() if len(df) > 10 else avg_time
            delta = recent_avg - previous_avg
            st.metric("Recent Avg Time", f"{recent_avg:.1f}s", f"{delta:+.1f}s")
        else:
            st.metric("Recent Avg Time", f"{avg_time:.1f}s")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Raga distribution
        raga_counts = df['raga'].value_counts()
        fig_raga = px.pie(
            values=raga_counts.values,
            names=raga_counts.index,
            title="Raga Distribution"
        )
        st.plotly_chart(fig_raga, use_container_width=True)
        
        # Generation time trend
        df_sorted = df.sort_values('timestamp')
        fig_time = px.line(
            df_sorted,
            x='timestamp',
            y='generation_time',
            title="Generation Time Trend"
        )
        st.plotly_chart(fig_time, use_container_width=True)
    
    with col2:
        # Style distribution
        style_counts = df['style'].value_counts()
        fig_style = px.bar(
            x=style_counts.index,
            y=style_counts.values,
            title="Style Distribution"
        )
        st.plotly_chart(fig_style, use_container_width=True)
        
        # Raga-Style heatmap
        raga_style_matrix = pd.crosstab(df['raga'], df['style'])
        fig_heatmap = px.imshow(
            raga_style_matrix.values,
            x=raga_style_matrix.columns,
            y=raga_style_matrix.index,
            title="Raga-Style Combination Heatmap",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Detailed statistics
    st.markdown("### Detailed Statistics")
    
    # Performance by raga
    raga_stats = df.groupby('raga')['generation_time'].agg(['mean', 'std', 'count']).round(2)
    raga_stats.columns = ['Avg Time (s)', 'Std Dev', 'Count']
    st.subheader("Performance by Raga")
    st.dataframe(raga_stats)
    
    # Performance by style
    style_stats = df.groupby('style')['generation_time'].agg(['mean', 'std', 'count']).round(2)
    style_stats.columns = ['Avg Time (s)', 'Std Dev', 'Count']
    st.subheader("Performance by Style")
    st.dataframe(style_stats)

def cultural_guide_page():
    """Cultural guide and educational content."""
    st.markdown('<h1 class="main-header">Cultural Guide</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🎵 Ragas", "🎨 Painting Styles", "📖 About Ragamala"])
    
    with tab1:
        st.markdown("## Understanding Ragas")
        st.markdown("""
        Ragas are melodic frameworks in Indian classical music, each with specific emotional and temporal associations.
        In Ragamala paintings, these musical concepts are translated into visual narratives.
        """)
        
        # Raga comparison
        selected_ragas = st.multiselect(
            "Compare Ragas:",
            options=list(RAGAS.keys()),
            default=["bhairav", "yaman"],
            format_func=lambda x: RAGAS[x]['name']
        )
        
        if selected_ragas:
            cols = st.columns(len(selected_ragas))
            for i, raga_key in enumerate(selected_ragas):
                with cols[i]:
                    render_raga_info_card(raga_key)
        
        # Raga timeline
        st.markdown("### Raga Timeline (24-hour cycle)")
        
        raga_times = []
        for raga_key, raga_data in RAGAS.items():
            time_str = raga_data['time']
            # Extract hour from time string (simplified)
            if 'Dawn' in time_str or '5-7' in time_str:
                hour = 6
            elif 'Morning' in time_str or '9-12' in time_str:
                hour = 10
            elif 'Evening' in time_str or '6-9' in time_str:
                hour = 19
            elif 'Late Evening' in time_str or '9-12' in time_str:
                hour = 22
            elif 'Night' in time_str or '9 PM-12' in time_str:
                hour = 21
            elif 'Midnight' in time_str or '12-3' in time_str:
                hour = 1
            else:
                hour = 12
            
            raga_times.append({
                'raga': raga_data['name'],
                'hour': hour,
                'time_desc': time_str,
                'mood': raga_data['mood']
            })
        
        timeline_df = pd.DataFrame(raga_times)
        timeline_df = timeline_df.sort_values('hour')
        
        fig_timeline = px.scatter(
            timeline_df,
            x='hour',
            y='raga',
            size=[20]*len(timeline_df),
            color='mood',
            title="Raga Performance Times",
            labels={'hour': 'Hour of Day', 'raga': 'Raga'}
        )
        fig_timeline.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=2))
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    with tab2:
        st.markdown("## Painting Styles")
        st.markdown("""
        Different regional schools of Indian miniature painting developed distinct characteristics
        in terms of color palette, composition, and artistic techniques.
        """)
        
        # Style comparison
        selected_styles = st.multiselect(
            "Compare Styles:",
            options=list(STYLES.keys()),
            default=["rajput", "pahari"],
            format_func=lambda x: STYLES[x]['name']
        )
        
        if selected_styles:
            cols = st.columns(len(selected_styles))
            for i, style_key in enumerate(selected_styles):
                with cols[i]:
                    render_style_info_card(style_key)
        
        # Style characteristics comparison
        if len(selected_styles) > 1:
            st.markdown("### Characteristic Comparison")
            
            comparison_data = []
            for style_key in selected_styles:
                style_data = STYLES[style_key]
                comparison_data.append({
                    'Style': style_data['name'],
                    'Period': style_data['period'],
                    'Region': style_data['region'],
                    'Key Colors': ', '.join(style_data['typical_colors'][:3]),
                    'Main Characteristics': ', '.join(style_data['characteristics'][:3])
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.table(comparison_df)
    
    with tab3:
        st.markdown("## About Ragamala Paintings")
        st.markdown("""
        ### What are Ragamala Paintings?
        
        Ragamala (literally "garland of ragas") paintings are a series of illustrative paintings from Indian art
        that depict various Indian musical modes called ragas. They stand as a classical example of the
        amalgamation of art, poetry and classical music in medieval India.
        
        ### Historical Context
        
        - **Origins**: Emerged in the 15th-16th centuries
        - **Purpose**: Visual representation of musical ragas and their emotional content
        - **Cultural Significance**: Bridge between auditory and visual arts
        - **Regional Variations**: Different painting schools developed unique interpretations
        
        ### Key Elements
        
        1. **Raga Personification**: Each raga is depicted as a human figure (often divine)
        2. **Temporal Association**: Time of day/season when the raga is traditionally performed
        3. **Emotional Content**: Visual representation of the raga's mood and sentiment
        4. **Iconographic Symbols**: Specific objects, animals, and settings associated with each raga
        5. **Color Symbolism**: Traditional color palettes that enhance the raga's emotional impact
        
        ### Artistic Techniques
        
        - **Miniature Format**: Small-scale detailed paintings
        - **Natural Pigments**: Traditional colors derived from minerals and plants
        - **Symbolic Composition**: Every element has meaning and purpose
        - **Narrative Structure**: Often tells a story related to the raga's character
        
        ### Modern Relevance
        
        Today, Ragamala paintings serve as:
        - Cultural preservation tools
        - Educational resources for understanding Indian classical music
        - Inspiration for contemporary artists
        - Bridge between traditional and modern artistic expression
        """)
        
        # Interactive raga-style matrix
        st.markdown("### Interactive Raga-Style Matrix")
        st.markdown("Explore how different ragas might be interpreted in various painting styles:")
        
        matrix_data = []
        for raga_key, raga_data in RAGAS.items():
            for style_key, style_data in STYLES.items():
                matrix_data.append({
                    'Raga': raga_data['name'],
                    'Style': style_data['name'],
                    'Time': raga_data['time'],
                    'Mood': raga_data['mood'],
                    'Period': style_data['period'],
                    'Region': style_data['region']
                })
        
        matrix_df = pd.DataFrame(matrix_data)
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            selected_time = st.selectbox(
                "Filter by Time:",
                ["All"] + list(set([raga['time'] for raga in RAGAS.values()]))
            )
        
        with col2:
            selected_period = st.selectbox(
                "Filter by Period:",
                ["All"] + list(set([style['period'] for style in STYLES.values()]))
            )
        
        # Apply filters
        filtered_df = matrix_df.copy()
        if selected_time != "All":
            filtered_df = filtered_df[filtered_df['Time'] == selected_time]
        if selected_period != "All":
            filtered_df = filtered_df[filtered_df['Period'] == selected_period]
        
        st.dataframe(filtered_df, use_container_width=True)

def settings_page():
    """Settings and configuration page."""
    st.markdown('<h1 class="main-header">Settings</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🔧 API Settings", "🎨 Default Parameters", "💾 Data Management"])
    
    with tab1:
        st.markdown("### API Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            api_url = st.text_input("API Base URL:", value=API_BASE_URL)
            api_key = st.text_input("API Key:", value=API_KEY, type="password")
            
            if st.button("Test Connection"):
                # Test API connection
                try:
                    response = requests.get(f"{api_url}/health", timeout=5)
                    if response.status_code == 200:
                        st.success("API connection successful!")
                        st.json(response.json())
                    else:
                        st.error(f"API returned status code: {response.status_code}")
                except Exception as e:
                    st.error(f"Connection failed: {str(e)}")
        
        with col2:
            st.markdown("### API Status")
            api_online = check_api_status()
            
            if api_online:
                st.success("API is online and responding")
            else:
                st.error("API is not responding")
            
            # API health details
            try:
                response = requests.get(f"{API_BASE_URL}/health", timeout=5)
                if response.status_code == 200:
                    health_data = response.json()
                    st.json(health_data)
            except:
                st.warning("Could not fetch detailed health information")
    
    with tab2:
        st.markdown("### Default Generation Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            default_steps = st.slider("Default Inference Steps:", 10, 100, 30)
            default_guidance = st.slider("Default Guidance Scale:", 1.0, 20.0, 7.5)
            default_width = st.selectbox("Default Width:", [512, 768, 1024], index=2)
            default_height = st.selectbox("Default Height:", [512, 768, 1024], index=2)
        
        with col2:
            default_template = st.selectbox("Default Template:", list(PROMPT_TEMPLATES.keys()), index=1)
            default_cultural = st.checkbox("Default Cultural Conditioning", True)
            default_metrics = st.checkbox("Default Quality Metrics", False)
            auto_save = st.checkbox("Auto-save Generated Images", True)
        
        if st.button("Save Default Settings"):
            # Save to session state or local storage
            st.session_state.default_settings = {
                'steps': default_steps,
                'guidance': default_guidance,
                'width': default_width,
                'height': default_height,
                'template': default_template,
                'cultural': default_cultural,
                'metrics': default_metrics,
                'auto_save': auto_save
            }
            st.success("Default settings saved!")
    
    with tab3:
        st.markdown("### Data Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Export Data")
            
            if st.button("Export Generation History"):
                history_json = export_generation_history()
                if history_json:
                    st.download_button(
                        "Download History JSON",
                        history_json,
                        f"ragamala_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        "application/json"
                    )
                else:
                    st.warning("No history data to export")
            
            if st.button("Export Generated Images"):
                if st.session_state.generated_images:
                    # Create a zip file with all images
                    import zipfile
                    zip_buffer = io.BytesIO()
                    
                    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                        for i, gen_data in enumerate(st.session_state.generated_images):
                            for j, img_data in enumerate(gen_data['images']):
                                img_bytes = io.BytesIO()
                                img_data['image'].save(img_bytes, format='PNG')
                                zip_file.writestr(
                                    f"image_{i}_{j}_{gen_data['raga']}_{gen_data['style']}.png",
                                    img_bytes.getvalue()
                                )
                    
                    st.download_button(
                        "Download Images ZIP",
                        zip_buffer.getvalue(),
                        f"ragamala_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        "application/zip"
                    )
                else:
                    st.warning("No images to export")
        
        with col2:
            st.markdown("#### Clear Data")
            
            if st.button("Clear Generation History", type="secondary"):
                if st.session_state.generation_history:
                    st.session_state.generation_history = []
                    st.success("Generation history cleared!")
                else:
                    st.info("No history to clear")
            
            if st.button("Clear Generated Images", type="secondary"):
                if st.session_state.generated_images:
                    st.session_state.generated_images = []
                    st.success("Generated images cleared!")
                else:
                    st.info("No images to clear")
            
            if st.button("Clear All Data", type="secondary"):
                st.session_state.generation_history = []
                st.session_state.generated_images = []
                st.success("All data cleared!")

def history_page():
    """Generation history page."""
    st.markdown('<h1 class="main-header">Generation History</h1>', unsafe_allow_html=True)
    
    if not st.session_state.generation_history:
        st.info("No generation history available. Generate some images first!")
        return
    
    # History overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Generations", len(st.session_state.generation_history))
    
    with col2:
        total_images = sum(len(gen['images']) for gen in st.session_state.generation_history)
        st.metric("Total Images", total_images)
    
    with col3:
        if st.session_state.generation_history:
            latest = max(st.session_state.generation_history, key=lambda x: x['timestamp'])
            latest_time = datetime.fromisoformat(latest['timestamp'])
            time_diff = datetime.now() - latest_time
            st.metric("Last Generation", f"{time_diff.seconds // 60}m ago")
    
    # History table
    st.markdown("### Recent Generations")
    
    # Create history dataframe
    history_data = []
    for i, gen in enumerate(reversed(st.session_state.generation_history[-20:])):  # Last 20
        history_data.append({
            'Index': len(st.session_state.generation_history) - i,
            'Timestamp': datetime.fromisoformat(gen['timestamp']).strftime('%Y-%m-%d %H:%M:%S'),
            'Raga': gen['raga'],
            'Style': gen['style'],
            'Images': len(gen['images']),
            'Generation Time': f"{gen['generation_time']:.1f}s",
            'Prompt': gen['prompt'][:50] + "..." if len(gen['prompt']) > 50 else gen['prompt']
        })
    
    history_df = pd.DataFrame(history_data)
    
    # Interactive table
    selected_indices = st.multiselect(
        "Select generations to view:",
        options=history_df['Index'].tolist(),
        default=history_df['Index'].tolist()[:3] if len(history_df) > 0 else []
    )
    
    if selected_indices:
        filtered_df = history_df[history_df['Index'].isin(selected_indices)]
        st.dataframe(filtered_df, use_container_width=True)
        
        # Show selected generations
        st.markdown("### Selected Generations")
        
        for idx in selected_indices:
            # Find the generation data
            gen_data = None
            for gen in st.session_state.generation_history:
                if st.session_state.generation_history.index(gen) + 1 == idx:
                    gen_data = gen
                    break
            
            if gen_data:
                with st.expander(f"Generation {idx}: {gen_data['raga']} - {gen_data['style']}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Show images
                        if gen_data['images']:
                            for i, img_data in enumerate(gen_data['images']):
                                st.image(
                                    img_data['image'], 
                                    caption=f"Image {i+1}",
                                    width=300
                                )
                    
                    with col2:
                        # Show metadata
                        st.markdown("**Generation Details:**")
                        st.markdown(f"- **Raga:** {gen_data['raga']}")
                        st.markdown(f"- **Style:** {gen_data['style']}")
                        st.markdown(f"- **Time:** {gen_data['generation_time']:.1f}s")
                        st.markdown(f"- **Images:** {len(gen_data['images'])}")
                        
                        st.markdown("**Prompt:**")
                        st.code(gen_data['prompt'])
                        
                        # Download options
                        if gen_data['images']:
                            for i, img_data in enumerate(gen_data['images']):
                                img_bytes = io.BytesIO()
                                img_data['image'].save(img_bytes, format='PNG')
                                st.download_button(
                                    f"Download Image {i+1}",
                                    img_bytes.getvalue(),
                                    f"ragamala_{gen_data['raga']}_{gen_data['style']}_{idx}_{i}.png",
                                    "image/png",
                                    key=f"download_{idx}_{i}"
                                )

# Main application
def main():
    """Main application function."""
    
    # Render sidebar and get current page
    current_page = render_sidebar()
    
    # Route to appropriate page
    if current_page == "🎨 Generate":
        generate_page()
    elif current_page == "📊 Analytics":
        analytics_page()
    elif current_page == "📚 Cultural Guide":
        cultural_guide_page()
    elif current_page == "⚙️ Settings":
        settings_page()
    elif current_page == "📜 History":
        history_page()

if __name__ == "__main__":
    main()
