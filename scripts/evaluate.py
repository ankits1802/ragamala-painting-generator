"""
Evaluation Script for Ragamala Painting Generation using SDXL + LoRA.

This script provides comprehensive evaluation functionality for assessing
the quality, cultural authenticity, and artistic merit of generated Ragamala
paintings using various metrics and human evaluation protocols.
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import warnings
warnings.filterwarnings("ignore")

# Core ML imports
import torch
import numpy as np
import pandas as pd
from PIL import Image

# Diffusers and transformers
from diffusers import StableDiffusionXLPipeline
from transformers import CLIPProcessor, CLIPModel

# Evaluation utilities
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.metrics import EvaluationMetrics, MetricsConfig
from src.evaluation.cultural_evaluator import CulturalAccuracyEvaluator, CulturalEvaluationConfig
from src.evaluation.human_eval import HumanEvaluationManager, CallbackConfig as HumanEvalConfig
from src.inference.generator import RagamalaGenerator, GenerationConfig, GenerationRequest
from src.data.dataset import RagamalaDataModule, DatasetConfig
from src.utils.logging_utils import setup_logger, create_training_logger
from src.utils.visualization import RagamalaVisualizer, save_training_samples
from src.utils.aws_utils import AWSUtilities, create_aws_config_from_env

logger = setup_logger(__name__)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate SDXL Ragamala painting generation model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model or checkpoint"
    )
    model_group.add_argument(
        "--lora_weights_path",
        type=str,
        default=None,
        help="Path to LoRA weights if separate from model"
    )
    model_group.add_argument(
        "--base_model_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Path to base SDXL model"
    )
    model_group.add_argument(
        "--vae_model_path",
        type=str,
        default="madebyollin/sdxl-vae-fp16-fix",
        help="Path to VAE model"
    )
    
    # Evaluation configuration
    eval_group = parser.add_argument_group('Evaluation Configuration')
    eval_group.add_argument(
        "--evaluation_type",
        type=str,
        choices=["quantitative", "cultural", "human", "comprehensive"],
        default="comprehensive",
        help="Type of evaluation to perform"
    )
    eval_group.add_argument(
        "--test_data_dir",
        type=str,
        default="data/test",
        help="Directory containing test images"
    )
    eval_group.add_argument(
        "--test_metadata_file",
        type=str,
        default="data/metadata/test_metadata.jsonl",
        help="Path to test metadata file"
    )
    eval_group.add_argument(
        "--reference_data_dir",
        type=str,
        default="data/reference",
        help="Directory containing reference images for comparison"
    )
    eval_group.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate"
    )
    eval_group.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation"
    )
    
    # Generation configuration
    gen_group = parser.add_argument_group('Generation Configuration')
    gen_group.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="Number of denoising steps"
    )
    gen_group.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale for generation"
    )
    gen_group.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Resolution for generated images"
    )
    gen_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation"
    )
    gen_group.add_argument(
        "--use_refiner",
        action="store_true",
        help="Use SDXL refiner for post-processing"
    )
    
    # Output configuration
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument(
        "--output_dir",
        type=str,
        default="outputs/evaluation",
        help="Directory to save evaluation results"
    )
    output_group.add_argument(
        "--save_generated_images",
        action="store_true",
        help="Save generated images during evaluation"
    )
    output_group.add_argument(
        "--save_visualizations",
        action="store_true",
        help="Save evaluation visualizations"
    )
    output_group.add_argument(
        "--report_format",
        type=str,
        choices=["json", "html", "pdf"],
        default="json",
        help="Format for evaluation report"
    )
    
    # Metrics configuration
    metrics_group = parser.add_argument_group('Metrics Configuration')
    metrics_group.add_argument(
        "--enable_fid",
        action="store_true",
        help="Calculate FID score"
    )
    metrics_group.add_argument(
        "--enable_clip_score",
        action="store_true",
        help="Calculate CLIP score"
    )
    metrics_group.add_argument(
        "--enable_ssim",
        action="store_true",
        help="Calculate SSIM score"
    )
    metrics_group.add_argument(
        "--enable_lpips",
        action="store_true",
        help="Calculate LPIPS score"
    )
    metrics_group.add_argument(
        "--enable_cultural_metrics",
        action="store_true",
        help="Calculate cultural authenticity metrics"
    )
    
    # Cultural evaluation
    cultural_group = parser.add_argument_group('Cultural Evaluation')
    cultural_group.add_argument(
        "--cultural_evaluation_config",
        type=str,
        default=None,
        help="Path to cultural evaluation configuration file"
    )
    cultural_group.add_argument(
        "--expert_evaluation",
        action="store_true",
        help="Enable expert cultural evaluation"
    )
    
    # Human evaluation
    human_group = parser.add_argument_group('Human Evaluation')
    human_group.add_argument(
        "--enable_human_eval",
        action="store_true",
        help="Enable human evaluation interface"
    )
    human_group.add_argument(
        "--human_eval_port",
        type=int,
        default=7860,
        help="Port for human evaluation interface"
    )
    human_group.add_argument(
        "--human_eval_samples",
        type=int,
        default=50,
        help="Number of samples for human evaluation"
    )
    
    # Cloud configuration
    cloud_group = parser.add_argument_group('Cloud Configuration')
    cloud_group.add_argument(
        "--enable_s3_backup",
        action="store_true",
        help="Enable S3 backup of evaluation results"
    )
    cloud_group.add_argument(
        "--s3_bucket",
        type=str,
        default=None,
        help="S3 bucket for backup"
    )
    cloud_group.add_argument(
        "--s3_prefix",
        type=str,
        default="ragamala/evaluation/",
        help="S3 prefix for backup"
    )
    
    return parser.parse_args()

def setup_logging_and_monitoring(args: argparse.Namespace) -> logging.Logger:
    """Setup logging for evaluation."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    eval_logger = create_training_logger(
        experiment_name=f"evaluation_{args.evaluation_type}",
        run_id=f"eval_{timestamp}",
        model_name="sdxl_ragamala"
    )
    
    eval_logger.info("Starting Ragamala model evaluation")
    eval_logger.info(f"Evaluation type: {args.evaluation_type}")
    eval_logger.info(f"Model path: {args.model_path}")
    
    return eval_logger

def load_model_and_generator(args: argparse.Namespace) -> RagamalaGenerator:
    """Load the fine-tuned model and create generator."""
    logger.info("Loading model and creating generator...")
    
    # Create generation configuration
    gen_config = GenerationConfig(
        model_path=args.base_model_path,
        lora_weights_path=args.lora_weights_path or args.model_path,
        vae_model_path=args.vae_model_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        width=args.resolution,
        height=args.resolution,
        use_refiner=args.use_refiner,
        generator_seed=args.seed
    )
    
    # Create generator
    generator = RagamalaGenerator(gen_config)
    
    logger.info("Model and generator loaded successfully")
    return generator

def load_test_data(args: argparse.Namespace) -> Tuple[List[Dict], List[str]]:
    """Load test data and prompts."""
    logger.info("Loading test data...")
    
    test_data = []
    prompts = []
    
    # Load metadata
    if Path(args.test_metadata_file).exists():
        with open(args.test_metadata_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                test_data.append(data)
                prompts.append(data.get('prompt', data.get('caption', '')))
    
    # Limit to specified number of samples
    if args.num_samples and len(test_data) > args.num_samples:
        test_data = test_data[:args.num_samples]
        prompts = prompts[:args.num_samples]
    
    logger.info(f"Loaded {len(test_data)} test samples")
    return test_data, prompts

def generate_test_images(generator: RagamalaGenerator, 
                        test_data: List[Dict], 
                        prompts: List[str],
                        args: argparse.Namespace) -> List[Image.Image]:
    """Generate images for evaluation."""
    logger.info("Generating test images...")
    
    generated_images = []
    
    for i, (data, prompt) in enumerate(tqdm(zip(test_data, prompts), desc="Generating images")):
        # Create generation request
        request = GenerationRequest(
            prompt=prompt,
            raga=data.get('raga'),
            style=data.get('style'),
            width=args.resolution,
            height=args.resolution,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed + i if args.seed else None
        )
        
        # Generate image
        try:
            result = generator.generate(request)
            generated_images.extend(result.images)
            
            # Save image if requested
            if args.save_generated_images:
                output_dir = Path(args.output_dir) / "generated_images"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                for j, img in enumerate(result.images):
                    img_path = output_dir / f"generated_{i:04d}_{j:02d}.png"
                    img.save(img_path)
                    
                    # Save metadata
                    metadata = {
                        'prompt': prompt,
                        'raga': data.get('raga'),
                        'style': data.get('style'),
                        'generation_time': result.generation_time,
                        'seed': request.seed
                    }
                    
                    metadata_path = output_dir / f"generated_{i:04d}_{j:02d}_metadata.json"
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to generate image {i}: {e}")
            # Create placeholder image
            placeholder = Image.new('RGB', (args.resolution, args.resolution), color='gray')
            generated_images.append(placeholder)
    
    logger.info(f"Generated {len(generated_images)} images")
    return generated_images

def load_reference_images(args: argparse.Namespace, num_samples: int) -> List[Image.Image]:
    """Load reference images for comparison."""
    logger.info("Loading reference images...")
    
    reference_images = []
    reference_dir = Path(args.reference_data_dir)
    
    if reference_dir.exists():
        image_files = list(reference_dir.glob("*.jpg")) + list(reference_dir.glob("*.png"))
        
        for img_path in image_files[:num_samples]:
            try:
                img = Image.open(img_path).convert('RGB')
                reference_images.append(img)
            except Exception as e:
                logger.warning(f"Failed to load reference image {img_path}: {e}")
    
    logger.info(f"Loaded {len(reference_images)} reference images")
    return reference_images

def run_quantitative_evaluation(generated_images: List[Image.Image],
                               reference_images: List[Image.Image],
                               prompts: List[str],
                               args: argparse.Namespace) -> Dict[str, Any]:
    """Run quantitative evaluation metrics."""
    logger.info("Running quantitative evaluation...")
    
    # Create metrics configuration
    metrics_config = MetricsConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=args.batch_size
    )
    
    # Initialize metrics calculator
    metrics_calculator = EvaluationMetrics(metrics_config)
    
    results = {}
    
    try:
        # Convert images to paths for metrics calculation
        gen_img_paths = []
        ref_img_paths = []
        
        # Save temporary images for metrics calculation
        temp_dir = Path(args.output_dir) / "temp_images"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        for i, img in enumerate(generated_images):
            img_path = temp_dir / f"gen_{i:04d}.png"
            img.save(img_path)
            gen_img_paths.append(str(img_path))
        
        for i, img in enumerate(reference_images):
            img_path = temp_dir / f"ref_{i:04d}.png"
            img.save(img_path)
            ref_img_paths.append(str(img_path))
        
        # Calculate metrics
        evaluation_results = metrics_calculator.evaluate_generation_quality(
            real_images=ref_img_paths,
            generated_images=gen_img_paths,
            prompts=prompts
        )
        
        results.update(evaluation_results)
        
        # Cleanup temporary files
        import shutil
        shutil.rmtree(temp_dir)
        
    except Exception as e:
        logger.error(f"Quantitative evaluation failed: {e}")
        results['error'] = str(e)
    
    logger.info("Quantitative evaluation completed")
    return results

def run_cultural_evaluation(generated_images: List[Image.Image],
                           test_data: List[Dict],
                           args: argparse.Namespace) -> Dict[str, Any]:
    """Run cultural authenticity evaluation."""
    logger.info("Running cultural evaluation...")
    
    # Create cultural evaluation configuration
    if args.cultural_evaluation_config and Path(args.cultural_evaluation_config).exists():
        with open(args.cultural_evaluation_config, 'r') as f:
            config_data = json.load(f)
        cultural_config = CulturalEvaluationConfig(**config_data)
    else:
        cultural_config = CulturalEvaluationConfig()
    
    # Initialize cultural evaluator
    cultural_evaluator = CulturalAccuracyEvaluator(cultural_config)
    
    results = []
    
    try:
        for i, (img, data) in enumerate(tqdm(zip(generated_images, test_data), desc="Cultural evaluation")):
            raga = data.get('raga', 'unknown')
            style = data.get('style', 'unknown')
            
            # Evaluate cultural accuracy
            eval_result = cultural_evaluator.evaluate_cultural_accuracy(
                image=img,
                raga=raga,
                style=style
            )
            
            results.append({
                'image_id': i,
                'raga': raga,
                'style': style,
                'overall_authenticity_score': eval_result.overall_authenticity_score,
                'iconography_score': eval_result.iconography_score,
                'color_palette_score': eval_result.color_palette_score,
                'composition_score': eval_result.composition_score,
                'cultural_violations': eval_result.cultural_violations,
                'recommendations': eval_result.recommendations
            })
        
        # Calculate aggregate statistics
        if results:
            aggregate_results = {
                'total_images': len(results),
                'average_authenticity_score': np.mean([r['overall_authenticity_score'] for r in results]),
                'average_iconography_score': np.mean([r['iconography_score'] for r in results]),
                'average_color_score': np.mean([r['color_palette_score'] for r in results]),
                'average_composition_score': np.mean([r['composition_score'] for r in results]),
                'detailed_results': results
            }
        else:
            aggregate_results = {'error': 'No results generated'}
        
    except Exception as e:
        logger.error(f"Cultural evaluation failed: {e}")
        aggregate_results = {'error': str(e)}
    
    logger.info("Cultural evaluation completed")
    return aggregate_results

def run_human_evaluation(generated_images: List[Image.Image],
                        test_data: List[Dict],
                        args: argparse.Namespace) -> Dict[str, Any]:
    """Setup and run human evaluation interface."""
    logger.info("Setting up human evaluation...")
    
    try:
        # Create human evaluation configuration
        human_config = HumanEvalConfig(
            log_every_n_steps=1,
            enable_cultural_validation=True
        )
        
        # Initialize human evaluation manager
        human_eval_manager = HumanEvaluationManager()
        
        # Select subset of images for human evaluation
        eval_indices = np.random.choice(
            len(generated_images), 
            min(args.human_eval_samples, len(generated_images)), 
            replace=False
        )
        
        eval_images = [generated_images[i] for i in eval_indices]
        eval_data = [test_data[i] for i in eval_indices]
        
        # Save images for human evaluation
        human_eval_dir = Path(args.output_dir) / "human_evaluation"
        human_eval_dir.mkdir(parents=True, exist_ok=True)
        
        for i, (img, data) in enumerate(zip(eval_images, eval_data)):
            img_path = human_eval_dir / f"eval_image_{i:03d}.png"
            img.save(img_path)
            
            # Save metadata
            metadata_path = human_eval_dir / f"eval_image_{i:03d}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        logger.info(f"Human evaluation setup completed. {len(eval_images)} images prepared.")
        logger.info(f"Launch human evaluation interface on port {args.human_eval_port}")
        
        # Launch Gradio interface
        human_eval_manager.launch_gradio_interface(share=True)
        
        return {
            'status': 'interface_launched',
            'num_images': len(eval_images),
            'evaluation_dir': str(human_eval_dir)
        }
        
    except Exception as e:
        logger.error(f"Human evaluation setup failed: {e}")
        return {'error': str(e)}

def create_evaluation_visualizations(results: Dict[str, Any],
                                   generated_images: List[Image.Image],
                                   test_data: List[Dict],
                                   args: argparse.Namespace):
    """Create evaluation visualizations."""
    logger.info("Creating evaluation visualizations...")
    
    try:
        visualizer = RagamalaVisualizer()
        viz_dir = Path(args.output_dir) / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot quantitative metrics
        if 'quantitative' in results and isinstance(results['quantitative'], dict):
            metrics_data = results['quantitative']
            
            # Filter out non-numeric values
            numeric_metrics = {k: v for k, v in metrics_data.items() 
                             if isinstance(v, (int, float)) and not k.endswith('_std')}
            
            if numeric_metrics:
                fig = visualizer.plot_evaluation_metrics(
                    numeric_metrics,
                    save_path=str(viz_dir / "quantitative_metrics.png")
                )
                plt.close(fig)
        
        # Plot cultural evaluation results
        if 'cultural' in results and isinstance(results['cultural'], dict):
            cultural_data = results['cultural']
            
            if 'detailed_results' in cultural_data:
                detailed_results = cultural_data['detailed_results']
                
                # Create style and raga distributions
                styles = [r.get('style', 'unknown') for r in detailed_results]
                ragas = [r.get('raga', 'unknown') for r in detailed_results]
                
                style_counts = pd.Series(styles).value_counts().to_dict()
                raga_counts = pd.Series(ragas).value_counts().to_dict()
                
                # Plot distributions
                if style_counts:
                    fig = visualizer.plot_style_distribution(
                        style_counts,
                        save_path=str(viz_dir / "style_distribution.png")
                    )
                    plt.close(fig)
                
                if raga_counts:
                    fig = visualizer.plot_raga_distribution(
                        raga_counts,
                        save_path=str(viz_dir / "raga_distribution.png")
                    )
                    plt.close(fig)
                
                # Plot cultural authenticity scores
                scores_data = {
                    'by_style': {},
                    'by_raga': {},
                    'components': {
                        'iconography': cultural_data.get('average_iconography_score', 0),
                        'color_palette': cultural_data.get('average_color_score', 0),
                        'composition': cultural_data.get('average_composition_score', 0)
                    }
                }
                
                # Calculate scores by style and raga
                style_scores = {}
                raga_scores = {}
                
                for result in detailed_results:
                    style = result.get('style', 'unknown')
                    raga = result.get('raga', 'unknown')
                    score = result.get('overall_authenticity_score', 0)
                    
                    if style not in style_scores:
                        style_scores[style] = []
                    style_scores[style].append(score)
                    
                    if raga not in raga_scores:
                        raga_scores[raga] = []
                    raga_scores[raga].append(score)
                
                scores_data['by_style'] = {k: np.mean(v) for k, v in style_scores.items()}
                scores_data['by_raga'] = {k: np.mean(v) for k, v in raga_scores.items()}
                
                fig = visualizer.plot_cultural_authenticity_scores(
                    scores_data,
                    save_path=str(viz_dir / "cultural_authenticity.png")
                )
                plt.close(fig)
        
        # Create sample grid
        if generated_images:
            sample_indices = np.random.choice(
                len(generated_images), 
                min(16, len(generated_images)), 
                replace=False
            )
            
            sample_images = [generated_images[i] for i in sample_indices]
            sample_titles = [
                f"{test_data[i].get('raga', 'unknown')} - {test_data[i].get('style', 'unknown')}"
                for i in sample_indices
            ]
            
            fig = visualizer.plot_image_grid(
                sample_images,
                sample_titles,
                grid_size=(4, 4),
                save_path=str(viz_dir / "sample_grid.png")
            )
            plt.close(fig)
        
        logger.info(f"Visualizations saved to {viz_dir}")
        
    except Exception as e:
        logger.error(f"Visualization creation failed: {e}")

def save_evaluation_report(results: Dict[str, Any], args: argparse.Namespace):
    """Save comprehensive evaluation report."""
    logger.info("Saving evaluation report...")
    
    try:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Add metadata to results
        results['metadata'] = {
            'evaluation_date': time.strftime("%Y-%m-%d %H:%M:%S"),
            'model_path': args.model_path,
            'evaluation_type': args.evaluation_type,
            'num_samples': args.num_samples,
            'resolution': args.resolution,
            'inference_steps': args.num_inference_steps,
            'guidance_scale': args.guidance_scale
        }
        
        # Save JSON report
        if args.report_format in ['json', 'all']:
            json_path = output_dir / "evaluation_report.json"
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"JSON report saved to {json_path}")
        
        # Save summary statistics
        summary_path = output_dir / "evaluation_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("Ragamala Painting Generation - Evaluation Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Evaluation Date: {results['metadata']['evaluation_date']}\n")
            f.write(f"Model Path: {results['metadata']['model_path']}\n")
            f.write(f"Evaluation Type: {results['metadata']['evaluation_type']}\n")
            f.write(f"Number of Samples: {results['metadata']['num_samples']}\n\n")
            
            # Quantitative results
            if 'quantitative' in results:
                f.write("Quantitative Metrics:\n")
                f.write("-" * 20 + "\n")
                quant_results = results['quantitative']
                for metric, value in quant_results.items():
                    if isinstance(value, (int, float)):
                        f.write(f"{metric}: {value:.4f}\n")
                f.write("\n")
            
            # Cultural results
            if 'cultural' in results:
                f.write("Cultural Authenticity:\n")
                f.write("-" * 20 + "\n")
                cultural_results = results['cultural']
                if 'average_authenticity_score' in cultural_results:
                    f.write(f"Overall Authenticity: {cultural_results['average_authenticity_score']:.4f}\n")
                if 'average_iconography_score' in cultural_results:
                    f.write(f"Iconography Score: {cultural_results['average_iconography_score']:.4f}\n")
                if 'average_color_score' in cultural_results:
                    f.write(f"Color Palette Score: {cultural_results['average_color_score']:.4f}\n")
                if 'average_composition_score' in cultural_results:
                    f.write(f"Composition Score: {cultural_results['average_composition_score']:.4f}\n")
                f.write("\n")
        
        logger.info(f"Evaluation report saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Failed to save evaluation report: {e}")

def backup_to_s3(args: argparse.Namespace):
    """Backup evaluation results to S3."""
    if not args.enable_s3_backup:
        return
    
    try:
        logger.info("Backing up evaluation results to S3...")
        
        aws_config = create_aws_config_from_env()
        if args.s3_bucket:
            aws_config.s3_bucket_name = args.s3_bucket
        aws_config.s3_prefix = args.s3_prefix
        
        aws_utils = AWSUtilities(aws_config)
        
        # Upload evaluation directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        s3_key = f"{args.s3_prefix}evaluation_{timestamp}/"
        
        aws_utils.s3.upload_directory(
            args.output_dir,
            s3_key
        )
        
        logger.info(f"Evaluation results backed up to S3: {s3_key}")
        
    except Exception as e:
        logger.warning(f"S3 backup failed: {e}")

def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    eval_logger = setup_logging_and_monitoring(args)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and generator
    generator = load_model_and_generator(args)
    
    # Load test data
    test_data, prompts = load_test_data(args)
    
    if not test_data:
        eval_logger.error("No test data found")
        return
    
    # Initialize results dictionary
    results = {}
    
    # Generate test images
    generated_images = generate_test_images(generator, test_data, prompts, args)
    
    # Run evaluations based on type
    if args.evaluation_type in ["quantitative", "comprehensive"]:
        eval_logger.info("Running quantitative evaluation...")
        reference_images = load_reference_images(args, len(generated_images))
        
        if reference_images:
            quant_results = run_quantitative_evaluation(
                generated_images, reference_images, prompts, args
            )
            results['quantitative'] = quant_results
        else:
            eval_logger.warning("No reference images found for quantitative evaluation")
    
    if args.evaluation_type in ["cultural", "comprehensive"]:
        eval_logger.info("Running cultural evaluation...")
        cultural_results = run_cultural_evaluation(generated_images, test_data, args)
        results['cultural'] = cultural_results
    
    if args.evaluation_type in ["human", "comprehensive"] and args.enable_human_eval:
        eval_logger.info("Setting up human evaluation...")
        human_results = run_human_evaluation(generated_images, test_data, args)
        results['human'] = human_results
    
    # Create visualizations
    if args.save_visualizations:
        create_evaluation_visualizations(results, generated_images, test_data, args)
    
    # Save evaluation report
    save_evaluation_report(results, args)
    
    # Backup to S3
    backup_to_s3(args)
    
    # Print summary
    eval_logger.info("Evaluation completed successfully!")
    eval_logger.info(f"Results saved to: {args.output_dir}")
    
    # Print key metrics
    if 'quantitative' in results:
        quant = results['quantitative']
        if 'fid' in quant:
            eval_logger.info(f"FID Score: {quant['fid']:.4f}")
        if 'clip_score' in quant:
            eval_logger.info(f"CLIP Score: {quant['clip_score']:.4f}")
    
    if 'cultural' in results:
        cultural = results['cultural']
        if 'average_authenticity_score' in cultural:
            eval_logger.info(f"Cultural Authenticity: {cultural['average_authenticity_score']:.4f}")

if __name__ == "__main__":
    main()
