"""
Result Visualization Utilities for Ragamala Painting Generation.

This module provides comprehensive visualization functionality for analyzing
training results, generated images, cultural authenticity metrics, and
evaluation data for the Ragamala painting generation project.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# Optional imports for advanced visualization
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

class RagamalaVisualizer:
    """Main visualization class for Ragamala painting analysis."""
    
    def __init__(self, style: str = "seaborn-v0_8", figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        self.color_palette = self._setup_color_palette()
        self._setup_matplotlib()
    
    def _setup_matplotlib(self):
        """Setup matplotlib configuration."""
        plt.style.use(self.style)
        plt.rcParams['figure.figsize'] = self.figsize
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
    
    def _setup_color_palette(self) -> Dict[str, str]:
        """Setup color palette for different styles and ragas."""
        return {
            'rajput': '#DC143C',      # Crimson
            'pahari': '#4682B4',      # Steel Blue
            'deccan': '#800080',      # Purple
            'mughal': '#FFD700',      # Gold
            'bhairav': '#FF6347',     # Tomato
            'yaman': '#87CEEB',       # Sky Blue
            'malkauns': '#2F4F4F',    # Dark Slate Gray
            'darbari': '#9932CC',     # Dark Orchid
            'bageshri': '#F0E68C',    # Khaki
            'todi': '#32CD32'         # Lime Green
        }
    
    def plot_training_metrics(self, 
                             metrics_data: Dict[str, List[float]],
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot training metrics over time.
        
        Args:
            metrics_data: Dictionary containing metric names and values
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Metrics Over Time', fontsize=16, fontweight='bold')
        
        # Loss curves
        if 'train_loss' in metrics_data:
            axes[0, 0].plot(metrics_data['train_loss'], label='Training Loss', color='blue')
        if 'val_loss' in metrics_data:
            axes[0, 0].plot(metrics_data['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Learning rate
        if 'learning_rate' in metrics_data:
            axes[0, 1].plot(metrics_data['learning_rate'], color='green')
            axes[0, 1].set_title('Learning Rate Schedule')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)
        
        # FID Score
        if 'fid_score' in metrics_data:
            axes[1, 0].plot(metrics_data['fid_score'], color='purple')
            axes[1, 0].set_title('FID Score')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('FID Score')
            axes[1, 0].grid(True, alpha=0.3)
        
        # CLIP Score
        if 'clip_score' in metrics_data:
            axes[1, 1].plot(metrics_data['clip_score'], color='orange')
            axes[1, 1].set_title('CLIP Score')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('CLIP Score')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training metrics plot saved to {save_path}")
        
        return fig
    
    def plot_image_grid(self, 
                       images: List[Image.Image],
                       titles: Optional[List[str]] = None,
                       grid_size: Optional[Tuple[int, int]] = None,
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot a grid of images.
        
        Args:
            images: List of PIL images
            titles: Optional titles for each image
            grid_size: Grid dimensions (rows, cols)
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        n_images = len(images)
        
        if grid_size is None:
            cols = int(np.ceil(np.sqrt(n_images)))
            rows = int(np.ceil(n_images / cols))
        else:
            rows, cols = grid_size
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i in range(rows * cols):
            if i < n_images:
                axes[i].imshow(images[i])
                axes[i].axis('off')
                
                if titles and i < len(titles):
                    axes[i].set_title(titles[i], fontsize=10, pad=10)
            else:
                axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Image grid saved to {save_path}")
        
        return fig
    
    def plot_style_distribution(self, 
                               style_data: Dict[str, int],
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot distribution of painting styles.
        
        Args:
            style_data: Dictionary with style names and counts
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        styles = list(style_data.keys())
        counts = list(style_data.values())
        colors = [self.color_palette.get(style, '#808080') for style in styles]
        
        # Bar plot
        bars = ax1.bar(styles, counts, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title('Distribution of Painting Styles', fontweight='bold')
        ax1.set_xlabel('Style')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count}', ha='center', va='bottom')
        
        # Pie chart
        ax2.pie(counts, labels=styles, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Style Distribution (Percentage)', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Style distribution plot saved to {save_path}")
        
        return fig
    
    def plot_raga_distribution(self, 
                              raga_data: Dict[str, int],
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot distribution of ragas.
        
        Args:
            raga_data: Dictionary with raga names and counts
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ragas = list(raga_data.keys())
        counts = list(raga_data.values())
        colors = [self.color_palette.get(raga, '#808080') for raga in ragas]
        
        # Horizontal bar plot for better readability
        bars = ax.barh(ragas, counts, color=colors, alpha=0.7, edgecolor='black')
        ax.set_title('Distribution of Ragas', fontweight='bold', fontsize=14)
        ax.set_xlabel('Count')
        ax.set_ylabel('Raga')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                   f'{count}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Raga distribution plot saved to {save_path}")
        
        return fig
    
    def plot_evaluation_metrics(self, 
                               metrics_data: Dict[str, float],
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot evaluation metrics comparison.
        
        Args:
            metrics_data: Dictionary with metric names and values
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        metrics = list(metrics_data.keys())
        values = list(metrics_data.values())
        
        # Bar plot
        bars = ax1.bar(metrics, values, color='skyblue', alpha=0.7, edgecolor='navy')
        ax1.set_title('Evaluation Metrics', fontweight='bold')
        ax1.set_ylabel('Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values_radar = values + [values[0]]  # Complete the circle
        angles += angles[:1]
        
        ax2 = plt.subplot(122, projection='polar')
        ax2.plot(angles, values_radar, 'o-', linewidth=2, color='red')
        ax2.fill(angles, values_radar, alpha=0.25, color='red')
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(metrics)
        ax2.set_title('Metrics Radar Chart', fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Evaluation metrics plot saved to {save_path}")
        
        return fig
    
    def plot_cultural_authenticity_scores(self, 
                                         scores_data: Dict[str, Dict[str, float]],
                                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot cultural authenticity scores by style and raga.
        
        Args:
            scores_data: Nested dictionary with styles/ragas and their scores
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Cultural Authenticity Analysis', fontsize=16, fontweight='bold')
        
        # Overall authenticity by style
        if 'by_style' in scores_data:
            styles = list(scores_data['by_style'].keys())
            scores = list(scores_data['by_style'].values())
            colors = [self.color_palette.get(style, '#808080') for style in styles]
            
            axes[0, 0].bar(styles, scores, color=colors, alpha=0.7)
            axes[0, 0].set_title('Authenticity by Style')
            axes[0, 0].set_ylabel('Authenticity Score')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Authenticity by raga
        if 'by_raga' in scores_data:
            ragas = list(scores_data['by_raga'].keys())
            scores = list(scores_data['by_raga'].values())
            colors = [self.color_palette.get(raga, '#808080') for raga in ragas]
            
            axes[0, 1].bar(ragas, scores, color=colors, alpha=0.7)
            axes[0, 1].set_title('Authenticity by Raga')
            axes[0, 1].set_ylabel('Authenticity Score')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Heatmap of style-raga combinations
        if 'style_raga_matrix' in scores_data:
            matrix_data = scores_data['style_raga_matrix']
            df = pd.DataFrame(matrix_data)
            
            sns.heatmap(df, annot=True, cmap='YlOrRd', ax=axes[1, 0], 
                       cbar_kws={'label': 'Authenticity Score'})
            axes[1, 0].set_title('Style-Raga Authenticity Matrix')
        
        # Component scores breakdown
        if 'components' in scores_data:
            components = list(scores_data['components'].keys())
            scores = list(scores_data['components'].values())
            
            axes[1, 1].barh(components, scores, color='lightcoral', alpha=0.7)
            axes[1, 1].set_title('Authenticity Components')
            axes[1, 1].set_xlabel('Score')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Cultural authenticity plot saved to {save_path}")
        
        return fig
    
    def plot_embedding_visualization(self, 
                                   embeddings: np.ndarray,
                                   labels: List[str],
                                   method: str = 'tsne',
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot 2D visualization of embeddings.
        
        Args:
            embeddings: High-dimensional embeddings
            labels: Labels for each embedding
            method: Dimensionality reduction method ('tsne', 'pca', 'umap')
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Reduce dimensionality
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        elif method == 'umap' and UMAP_AVAILABLE:
            reducer = umap.UMAP(n_components=2, random_state=42)
        else:
            logger.warning(f"Method {method} not available, using PCA")
            reducer = PCA(n_components=2, random_state=42)
        
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get unique labels and assign colors
        unique_labels = list(set(labels))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        label_to_color = dict(zip(unique_labels, colors))
        
        # Plot points
        for label in unique_labels:
            mask = np.array(labels) == label
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                      c=[label_to_color[label]], label=label, alpha=0.7, s=50)
        
        ax.set_title(f'Embedding Visualization ({method.upper()})', fontweight='bold')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Embedding visualization saved to {save_path}")
        
        return fig
    
    def plot_generation_comparison(self, 
                                  original_images: List[Image.Image],
                                  generated_images: List[Image.Image],
                                  prompts: List[str],
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison between original and generated images.
        
        Args:
            original_images: List of original images
            generated_images: List of generated images
            prompts: List of prompts used for generation
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        n_pairs = min(len(original_images), len(generated_images))
        
        fig, axes = plt.subplots(n_pairs, 2, figsize=(12, 4 * n_pairs))
        fig.suptitle('Original vs Generated Images', fontsize=16, fontweight='bold')
        
        if n_pairs == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_pairs):
            # Original image
            axes[i, 0].imshow(original_images[i])
            axes[i, 0].set_title('Original', fontweight='bold')
            axes[i, 0].axis('off')
            
            # Generated image
            axes[i, 1].imshow(generated_images[i])
            axes[i, 1].set_title('Generated', fontweight='bold')
            axes[i, 1].axis('off')
            
            # Add prompt as subtitle
            if i < len(prompts):
                fig.text(0.5, 0.95 - (i * 0.9 / n_pairs), prompts[i], 
                        ha='center', va='top', fontsize=10, style='italic')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Generation comparison saved to {save_path}")
        
        return fig
    
    def create_interactive_dashboard(self, 
                                   data: Dict[str, Any],
                                   save_path: Optional[str] = None) -> go.Figure:
        """
        Create interactive dashboard using Plotly.
        
        Args:
            data: Dictionary containing various data for visualization
            save_path: Path to save the HTML file
            
        Returns:
            Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Loss', 'Style Distribution', 
                          'Evaluation Metrics', 'Cultural Scores'),
            specs=[[{"secondary_y": False}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Training loss
        if 'training_loss' in data:
            fig.add_trace(
                go.Scatter(y=data['training_loss'], name='Training Loss', 
                          line=dict(color='blue')),
                row=1, col=1
            )
        
        # Style distribution pie chart
        if 'style_distribution' in data:
            styles = list(data['style_distribution'].keys())
            counts = list(data['style_distribution'].values())
            
            fig.add_trace(
                go.Pie(labels=styles, values=counts, name="Style Distribution"),
                row=1, col=2
            )
        
        # Evaluation metrics
        if 'evaluation_metrics' in data:
            metrics = list(data['evaluation_metrics'].keys())
            values = list(data['evaluation_metrics'].values())
            
            fig.add_trace(
                go.Bar(x=metrics, y=values, name='Evaluation Metrics',
                      marker_color='lightblue'),
                row=2, col=1
            )
        
        # Cultural scores
        if 'cultural_scores' in data:
            components = list(data['cultural_scores'].keys())
            scores = list(data['cultural_scores'].values())
            
            fig.add_trace(
                go.Bar(x=components, y=scores, name='Cultural Scores',
                      marker_color='lightcoral'),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Ragamala Generation Dashboard",
            title_x=0.5,
            showlegend=False,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive dashboard saved to {save_path}")
        
        return fig
    
    def generate_word_cloud(self, 
                           text_data: List[str],
                           save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """
        Generate word cloud from text data.
        
        Args:
            text_data: List of text strings
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure or None if WordCloud not available
        """
        if not WORDCLOUD_AVAILABLE:
            logger.warning("WordCloud not available. Install with: pip install wordcloud")
            return None
        
        # Combine all text
        combined_text = ' '.join(text_data)
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate(combined_text)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Prompt Word Cloud', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Word cloud saved to {save_path}")
        
        return fig

def save_training_samples(images: List[Image.Image], 
                         prompts: List[str],
                         epoch: int,
                         output_dir: str = "outputs/training_samples"):
    """
    Save training samples with metadata.
    
    Args:
        images: List of generated images
        prompts: List of prompts used
        epoch: Current epoch number
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create grid
    visualizer = RagamalaVisualizer()
    fig = visualizer.plot_image_grid(images, prompts)
    
    # Save grid
    grid_path = output_dir / f"epoch_{epoch:04d}_samples.png"
    fig.savefig(grid_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Save individual images
    for i, (image, prompt) in enumerate(zip(images, prompts)):
        image_path = output_dir / f"epoch_{epoch:04d}_sample_{i:03d}.png"
        image.save(image_path)
        
        # Save metadata
        metadata = {
            'epoch': epoch,
            'sample_id': i,
            'prompt': prompt,
            'image_path': str(image_path)
        }
        
        metadata_path = output_dir / f"epoch_{epoch:04d}_sample_{i:03d}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved {len(images)} training samples for epoch {epoch}")

def create_loss_plot(loss_history: Dict[str, List[float]], 
                    save_path: str):
    """
    Create and save loss plot.
    
    Args:
        loss_history: Dictionary with loss values
        save_path: Path to save the plot
    """
    visualizer = RagamalaVisualizer()
    fig = visualizer.plot_training_metrics(loss_history, save_path)
    plt.close(fig)

def plot_annotation_results(annotations: List[Dict[str, Any]], 
                           save_path: str):
    """
    Plot annotation results analysis.
    
    Args:
        annotations: List of annotation dictionaries
        save_path: Path to save the plot
    """
    # Extract data
    styles = [ann.get('style', 'unknown') for ann in annotations]
    ragas = [ann.get('raga', 'unknown') for ann in annotations]
    quality_scores = [ann.get('quality_score', 0) for ann in annotations]
    
    # Create visualizations
    visualizer = RagamalaVisualizer()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Annotation Analysis', fontsize=16, fontweight='bold')
    
    # Style distribution
    style_counts = pd.Series(styles).value_counts()
    colors = [visualizer.color_palette.get(style, '#808080') for style in style_counts.index]
    
    axes[0, 0].bar(style_counts.index, style_counts.values, color=colors, alpha=0.7)
    axes[0, 0].set_title('Style Distribution')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Raga distribution
    raga_counts = pd.Series(ragas).value_counts()
    colors = [visualizer.color_palette.get(raga, '#808080') for raga in raga_counts.index]
    
    axes[0, 1].bar(raga_counts.index, raga_counts.values, color=colors, alpha=0.7)
    axes[0, 1].set_title('Raga Distribution')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Quality score distribution
    axes[1, 0].hist(quality_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 0].set_title('Quality Score Distribution')
    axes[1, 0].set_xlabel('Quality Score')
    axes[1, 0].set_ylabel('Frequency')
    
    # Quality by style
    df = pd.DataFrame({'style': styles, 'quality': quality_scores})
    style_quality = df.groupby('style')['quality'].mean()
    
    axes[1, 1].bar(style_quality.index, style_quality.values, 
                   color='lightcoral', alpha=0.7)
    axes[1, 1].set_title('Average Quality by Style')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Annotation results plot saved to {save_path}")

def main():
    """Main function for testing visualization utilities."""
    # Create test data
    visualizer = RagamalaVisualizer()
    
    # Test training metrics plot
    metrics_data = {
        'train_loss': np.random.exponential(0.5, 100).cumsum() * -0.01 + 2,
        'val_loss': np.random.exponential(0.5, 100).cumsum() * -0.01 + 2.2,
        'learning_rate': [1e-4 * (0.95 ** i) for i in range(100)],
        'fid_score': np.random.exponential(0.1, 100).cumsum() * -0.5 + 50,
        'clip_score': np.random.exponential(0.01, 100).cumsum() * 0.001 + 0.7
    }
    
    fig = visualizer.plot_training_metrics(metrics_data)
    plt.show()
    
    # Test style distribution
    style_data = {
        'rajput': 45,
        'pahari': 32,
        'deccan': 28,
        'mughal': 35
    }
    
    fig = visualizer.plot_style_distribution(style_data)
    plt.show()
    
    # Test evaluation metrics
    eval_metrics = {
        'FID': 25.3,
        'CLIP Score': 0.82,
        'SSIM': 0.75,
        'Cultural Auth': 0.88
    }
    
    fig = visualizer.plot_evaluation_metrics(eval_metrics)
    plt.show()
    
    print("Visualization utilities testing completed!")

if __name__ == "__main__":
    main()
