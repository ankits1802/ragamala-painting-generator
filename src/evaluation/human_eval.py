"""
Human Evaluation Interface for Ragamala Painting Generation.

This module provides comprehensive human evaluation functionality for assessing
the quality, cultural authenticity, and artistic merit of generated Ragamala paintings
through structured evaluation interfaces and expert assessment tools.
"""

import os
import sys
import json
import time
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, asdict
import numpy as np
from PIL import Image
import streamlit as st
import gradio as gr
from datetime import datetime
import sqlite3
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

@dataclass
class EvaluationCriteria:
    """Criteria for human evaluation."""
    name: str
    description: str
    scale: Tuple[int, int]
    labels: List[str]
    weight: float = 1.0

@dataclass
class EvaluationSession:
    """Human evaluation session data."""
    session_id: str
    evaluator_id: str
    start_time: datetime
    end_time: Optional[datetime]
    images_evaluated: int
    total_images: int
    evaluation_type: str
    metadata: Dict[str, Any]

@dataclass
class ImageEvaluation:
    """Individual image evaluation result."""
    image_id: str
    session_id: str
    evaluator_id: str
    scores: Dict[str, float]
    comments: str
    time_spent: float
    timestamp: datetime
    image_metadata: Dict[str, Any]

class EvaluationDatabase:
    """Database manager for human evaluation data."""
    
    def __init__(self, db_path: str = "evaluation_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the evaluation database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluators (
                evaluator_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                expertise_level TEXT,
                background TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_sessions (
                session_id TEXT PRIMARY KEY,
                evaluator_id TEXT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                images_evaluated INTEGER,
                total_images INTEGER,
                evaluation_type TEXT,
                metadata TEXT,
                FOREIGN KEY (evaluator_id) REFERENCES evaluators (evaluator_id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_evaluations (
                evaluation_id TEXT PRIMARY KEY,
                image_id TEXT,
                session_id TEXT,
                evaluator_id TEXT,
                scores TEXT,
                comments TEXT,
                time_spent REAL,
                timestamp TIMESTAMP,
                image_metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES evaluation_sessions (session_id),
                FOREIGN KEY (evaluator_id) REFERENCES evaluators (evaluator_id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS comparison_evaluations (
                comparison_id TEXT PRIMARY KEY,
                session_id TEXT,
                evaluator_id TEXT,
                image_a_id TEXT,
                image_b_id TEXT,
                preference TEXT,
                confidence REAL,
                reasoning TEXT,
                timestamp TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES evaluation_sessions (session_id),
                FOREIGN KEY (evaluator_id) REFERENCES evaluators (evaluator_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def add_evaluator(self, evaluator_id: str, name: str, expertise_level: str, background: str):
        """Add a new evaluator to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO evaluators 
            (evaluator_id, name, expertise_level, background)
            VALUES (?, ?, ?, ?)
        """, (evaluator_id, name, expertise_level, background))
        
        conn.commit()
        conn.close()
    
    def create_session(self, session: EvaluationSession):
        """Create a new evaluation session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO evaluation_sessions 
            (session_id, evaluator_id, start_time, end_time, images_evaluated, 
             total_images, evaluation_type, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session.session_id,
            session.evaluator_id,
            session.start_time,
            session.end_time,
            session.images_evaluated,
            session.total_images,
            session.evaluation_type,
            json.dumps(session.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def save_image_evaluation(self, evaluation: ImageEvaluation):
        """Save an image evaluation."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        evaluation_id = str(uuid.uuid4())
        
        cursor.execute("""
            INSERT INTO image_evaluations 
            (evaluation_id, image_id, session_id, evaluator_id, scores, 
             comments, time_spent, timestamp, image_metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            evaluation_id,
            evaluation.image_id,
            evaluation.session_id,
            evaluation.evaluator_id,
            json.dumps(evaluation.scores),
            evaluation.comments,
            evaluation.time_spent,
            evaluation.timestamp,
            json.dumps(evaluation.image_metadata)
        ))
        
        conn.commit()
        conn.close()
        
        return evaluation_id
    
    def save_comparison_evaluation(self, 
                                 session_id: str,
                                 evaluator_id: str,
                                 image_a_id: str,
                                 image_b_id: str,
                                 preference: str,
                                 confidence: float,
                                 reasoning: str):
        """Save a comparison evaluation."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        comparison_id = str(uuid.uuid4())
        
        cursor.execute("""
            INSERT INTO comparison_evaluations 
            (comparison_id, session_id, evaluator_id, image_a_id, image_b_id,
             preference, confidence, reasoning, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            comparison_id,
            session_id,
            evaluator_id,
            image_a_id,
            image_b_id,
            preference,
            confidence,
            reasoning,
            datetime.now()
        ))
        
        conn.commit()
        conn.close()
        
        return comparison_id
    
    def get_evaluation_results(self, session_id: Optional[str] = None) -> pd.DataFrame:
        """Get evaluation results as DataFrame."""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT ie.*, es.evaluation_type, ev.name as evaluator_name, ev.expertise_level
            FROM image_evaluations ie
            JOIN evaluation_sessions es ON ie.session_id = es.session_id
            JOIN evaluators ev ON ie.evaluator_id = ev.evaluator_id
        """
        
        if session_id:
            query += f" WHERE ie.session_id = '{session_id}'"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df

class RagamalaEvaluationCriteria:
    """Predefined evaluation criteria for Ragamala paintings."""
    
    @staticmethod
    def get_quality_criteria() -> List[EvaluationCriteria]:
        """Get quality assessment criteria."""
        return [
            EvaluationCriteria(
                name="overall_quality",
                description="Overall artistic quality and visual appeal",
                scale=(1, 10),
                labels=["Poor", "Fair", "Good", "Very Good", "Excellent"],
                weight=1.0
            ),
            EvaluationCriteria(
                name="technical_execution",
                description="Technical quality of the painting (composition, color, detail)",
                scale=(1, 10),
                labels=["Poor", "Fair", "Good", "Very Good", "Excellent"],
                weight=0.8
            ),
            EvaluationCriteria(
                name="visual_coherence",
                description="How well the elements work together visually",
                scale=(1, 10),
                labels=["Incoherent", "Somewhat coherent", "Coherent", "Very coherent", "Perfectly coherent"],
                weight=0.7
            )
        ]
    
    @staticmethod
    def get_cultural_criteria() -> List[EvaluationCriteria]:
        """Get cultural authenticity criteria."""
        return [
            EvaluationCriteria(
                name="cultural_authenticity",
                description="Authenticity to traditional Ragamala painting conventions",
                scale=(1, 10),
                labels=["Not authentic", "Somewhat authentic", "Authentic", "Very authentic", "Perfectly authentic"],
                weight=1.0
            ),
            EvaluationCriteria(
                name="iconographic_accuracy",
                description="Accuracy of iconographic elements for the specific raga",
                scale=(1, 10),
                labels=["Inaccurate", "Somewhat accurate", "Accurate", "Very accurate", "Perfectly accurate"],
                weight=0.9
            ),
            EvaluationCriteria(
                name="style_consistency",
                description="Consistency with the specified painting style (Rajput, Pahari, etc.)",
                scale=(1, 10),
                labels=["Inconsistent", "Somewhat consistent", "Consistent", "Very consistent", "Perfectly consistent"],
                weight=0.8
            ),
            EvaluationCriteria(
                name="color_appropriateness",
                description="Appropriateness of color palette for the raga and style",
                scale=(1, 10),
                labels=["Inappropriate", "Somewhat appropriate", "Appropriate", "Very appropriate", "Perfectly appropriate"],
                weight=0.7
            )
        ]
    
    @staticmethod
    def get_prompt_adherence_criteria() -> List[EvaluationCriteria]:
        """Get prompt adherence criteria."""
        return [
            EvaluationCriteria(
                name="prompt_adherence",
                description="How well the image matches the given text prompt",
                scale=(1, 10),
                labels=["No match", "Poor match", "Fair match", "Good match", "Perfect match"],
                weight=1.0
            ),
            EvaluationCriteria(
                name="raga_representation",
                description="How well the image represents the specified raga",
                scale=(1, 10),
                labels=["Poor representation", "Fair representation", "Good representation", "Very good representation", "Excellent representation"],
                weight=0.9
            )
        ]

class StreamlitEvaluationInterface:
    """Streamlit-based evaluation interface."""
    
    def __init__(self, db: EvaluationDatabase):
        self.db = db
        self.criteria_sets = {
            "Quality Assessment": RagamalaEvaluationCriteria.get_quality_criteria(),
            "Cultural Authenticity": RagamalaEvaluationCriteria.get_cultural_criteria(),
            "Prompt Adherence": RagamalaEvaluationCriteria.get_prompt_adherence_criteria()
        }
    
    def run_evaluation_interface(self):
        """Run the Streamlit evaluation interface."""
        st.set_page_config(
            page_title="Ragamala Painting Evaluation",
            page_icon="🎨",
            layout="wide"
        )
        
        st.title("Ragamala Painting Human Evaluation")
        st.markdown("---")
        
        # Sidebar for evaluator information
        with st.sidebar:
            st.header("Evaluator Information")
            evaluator_name = st.text_input("Your Name")
            expertise_level = st.selectbox(
                "Expertise Level",
                ["Beginner", "Intermediate", "Advanced", "Expert"]
            )
            background = st.text_area("Background/Expertise")
            
            if st.button("Register Evaluator"):
                if evaluator_name:
                    evaluator_id = str(uuid.uuid4())
                    self.db.add_evaluator(evaluator_id, evaluator_name, expertise_level, background)
                    st.session_state.evaluator_id = evaluator_id
                    st.success("Evaluator registered!")
        
        # Main evaluation interface
        if 'evaluator_id' in st.session_state:
            self._render_evaluation_interface()
        else:
            st.info("Please register as an evaluator to begin evaluation.")
    
    def _render_evaluation_interface(self):
        """Render the main evaluation interface."""
        # Evaluation type selection
        eval_type = st.selectbox(
            "Evaluation Type",
            ["Single Image Evaluation", "Comparative Evaluation", "Batch Evaluation"]
        )
        
        if eval_type == "Single Image Evaluation":
            self._render_single_image_evaluation()
        elif eval_type == "Comparative Evaluation":
            self._render_comparative_evaluation()
        elif eval_type == "Batch Evaluation":
            self._render_batch_evaluation()
    
    def _render_single_image_evaluation(self):
        """Render single image evaluation interface."""
        st.header("Single Image Evaluation")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Upload image to evaluate",
            type=['png', 'jpg', 'jpeg']
        )
        
        if uploaded_file:
            # Display image
            image = Image.open(uploaded_file)
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Image to Evaluate", use_column_width=True)
                
                # Image metadata
                st.subheader("Image Information")
                raga = st.text_input("Raga")
                style = st.text_input("Style")
                prompt = st.text_area("Original Prompt")
            
            with col2:
                st.subheader("Evaluation Criteria")
                
                # Criteria selection
                selected_criteria_set = st.selectbox(
                    "Select Criteria Set",
                    list(self.criteria_sets.keys())
                )
                
                criteria = self.criteria_sets[selected_criteria_set]
                scores = {}
                
                # Render evaluation criteria
                for criterion in criteria:
                    st.markdown(f"**{criterion.name.replace('_', ' ').title()}**")
                    st.caption(criterion.description)
                    
                    score = st.slider(
                        f"Score ({criterion.scale[0]}-{criterion.scale[1]})",
                        min_value=criterion.scale[0],
                        max_value=criterion.scale[1],
                        value=criterion.scale[1] // 2,
                        key=f"score_{criterion.name}"
                    )
                    scores[criterion.name] = score
                
                # Comments
                comments = st.text_area("Additional Comments")
                
                # Submit evaluation
                if st.button("Submit Evaluation"):
                    self._save_single_evaluation(
                        uploaded_file.name,
                        scores,
                        comments,
                        {"raga": raga, "style": style, "prompt": prompt}
                    )
                    st.success("Evaluation submitted!")
    
    def _render_comparative_evaluation(self):
        """Render comparative evaluation interface."""
        st.header("Comparative Evaluation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Image A")
            image_a = st.file_uploader("Upload Image A", type=['png', 'jpg', 'jpeg'], key="image_a")
            if image_a:
                st.image(Image.open(image_a), use_column_width=True)
        
        with col2:
            st.subheader("Image B")
            image_b = st.file_uploader("Upload Image B", type=['png', 'jpg', 'jpeg'], key="image_b")
            if image_b:
                st.image(Image.open(image_b), use_column_width=True)
        
        if image_a and image_b:
            st.subheader("Comparison")
            
            preference = st.radio(
                "Which image is better?",
                ["Image A", "Image B", "Equal"]
            )
            
            confidence = st.slider(
                "How confident are you in this choice?",
                min_value=1,
                max_value=10,
                value=5
            )
            
            reasoning = st.text_area("Explain your reasoning")
            
            if st.button("Submit Comparison"):
                self._save_comparison_evaluation(
                    image_a.name,
                    image_b.name,
                    preference,
                    confidence,
                    reasoning
                )
                st.success("Comparison submitted!")
    
    def _render_batch_evaluation(self):
        """Render batch evaluation interface."""
        st.header("Batch Evaluation")
        
        # Upload multiple images
        uploaded_files = st.file_uploader(
            "Upload images to evaluate",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.write(f"Uploaded {len(uploaded_files)} images")
            
            # Criteria selection
            selected_criteria_set = st.selectbox(
                "Select Criteria Set",
                list(self.criteria_sets.keys()),
                key="batch_criteria"
            )
            
            criteria = self.criteria_sets[selected_criteria_set]
            
            # Evaluate each image
            for i, uploaded_file in enumerate(uploaded_files):
                with st.expander(f"Evaluate {uploaded_file.name}"):
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        image = Image.open(uploaded_file)
                        st.image(image, use_column_width=True)
                    
                    with col2:
                        scores = {}
                        for criterion in criteria:
                            score = st.slider(
                                criterion.name.replace('_', ' ').title(),
                                min_value=criterion.scale[0],
                                max_value=criterion.scale[1],
                                value=criterion.scale[1] // 2,
                                key=f"batch_score_{i}_{criterion.name}"
                            )
                            scores[criterion.name] = score
                        
                        comments = st.text_area(
                            "Comments",
                            key=f"batch_comments_{i}"
                        )
                        
                        if st.button(f"Save Evaluation {i+1}", key=f"save_{i}"):
                            self._save_single_evaluation(
                                uploaded_file.name,
                                scores,
                                comments,
                                {}
                            )
                            st.success(f"Evaluation {i+1} saved!")
    
    def _save_single_evaluation(self, 
                              image_name: str,
                              scores: Dict[str, float],
                              comments: str,
                              metadata: Dict[str, Any]):
        """Save a single image evaluation."""
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
            
            # Create session
            session = EvaluationSession(
                session_id=st.session_state.session_id,
                evaluator_id=st.session_state.evaluator_id,
                start_time=datetime.now(),
                end_time=None,
                images_evaluated=0,
                total_images=1,
                evaluation_type="single_image",
                metadata={}
            )
            self.db.create_session(session)
        
        evaluation = ImageEvaluation(
            image_id=image_name,
            session_id=st.session_state.session_id,
            evaluator_id=st.session_state.evaluator_id,
            scores=scores,
            comments=comments,
            time_spent=0.0,  # Could be tracked with timer
            timestamp=datetime.now(),
            image_metadata=metadata
        )
        
        self.db.save_image_evaluation(evaluation)
    
    def _save_comparison_evaluation(self,
                                  image_a_name: str,
                                  image_b_name: str,
                                  preference: str,
                                  confidence: float,
                                  reasoning: str):
        """Save a comparison evaluation."""
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        
        self.db.save_comparison_evaluation(
            st.session_state.session_id,
            st.session_state.evaluator_id,
            image_a_name,
            image_b_name,
            preference,
            confidence,
            reasoning
        )

class GradioEvaluationInterface:
    """Gradio-based evaluation interface."""
    
    def __init__(self, db: EvaluationDatabase):
        self.db = db
        self.criteria = RagamalaEvaluationCriteria.get_quality_criteria()
    
    def create_interface(self):
        """Create Gradio interface."""
        with gr.Blocks(title="Ragamala Painting Evaluation") as interface:
            gr.Markdown("# Ragamala Painting Human Evaluation")
            
            with gr.Tab("Single Image Evaluation"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(type="pil", label="Upload Image")
                        raga_input = gr.Textbox(label="Raga")
                        style_input = gr.Textbox(label="Style")
                        prompt_input = gr.Textbox(label="Original Prompt", lines=3)
                    
                    with gr.Column():
                        # Create sliders for each criterion
                        score_inputs = []
                        for criterion in self.criteria:
                            slider = gr.Slider(
                                minimum=criterion.scale[0],
                                maximum=criterion.scale[1],
                                value=criterion.scale[1] // 2,
                                label=criterion.name.replace('_', ' ').title(),
                                info=criterion.description
                            )
                            score_inputs.append(slider)
                        
                        comments_input = gr.Textbox(label="Comments", lines=3)
                        evaluator_input = gr.Textbox(label="Evaluator Name")
                        
                        submit_btn = gr.Button("Submit Evaluation")
                        output_text = gr.Textbox(label="Result")
                
                submit_btn.click(
                    fn=self._process_evaluation,
                    inputs=[image_input, raga_input, style_input, prompt_input,
                           evaluator_input, comments_input] + score_inputs,
                    outputs=output_text
                )
            
            with gr.Tab("Comparison Evaluation"):
                with gr.Row():
                    image_a = gr.Image(type="pil", label="Image A")
                    image_b = gr.Image(type="pil", label="Image B")
                
                preference = gr.Radio(
                    choices=["Image A", "Image B", "Equal"],
                    label="Which image is better?"
                )
                confidence = gr.Slider(1, 10, value=5, label="Confidence")
                reasoning = gr.Textbox(label="Reasoning", lines=3)
                evaluator_comp = gr.Textbox(label="Evaluator Name")
                
                submit_comp = gr.Button("Submit Comparison")
                output_comp = gr.Textbox(label="Result")
                
                submit_comp.click(
                    fn=self._process_comparison,
                    inputs=[image_a, image_b, preference, confidence, reasoning, evaluator_comp],
                    outputs=output_comp
                )
        
        return interface
    
    def _process_evaluation(self, image, raga, style, prompt, evaluator_name, comments, *scores):
        """Process single image evaluation."""
        try:
            # Register evaluator if not exists
            evaluator_id = str(uuid.uuid4())
            self.db.add_evaluator(evaluator_id, evaluator_name, "Unknown", "")
            
            # Create scores dictionary
            score_dict = {}
            for i, criterion in enumerate(self.criteria):
                score_dict[criterion.name] = scores[i]
            
            # Save evaluation
            evaluation = ImageEvaluation(
                image_id=str(uuid.uuid4()),
                session_id=str(uuid.uuid4()),
                evaluator_id=evaluator_id,
                scores=score_dict,
                comments=comments,
                time_spent=0.0,
                timestamp=datetime.now(),
                image_metadata={"raga": raga, "style": style, "prompt": prompt}
            )
            
            self.db.save_image_evaluation(evaluation)
            
            return "Evaluation submitted successfully!"
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _process_comparison(self, image_a, image_b, preference, confidence, reasoning, evaluator_name):
        """Process comparison evaluation."""
        try:
            # Register evaluator if not exists
            evaluator_id = str(uuid.uuid4())
            self.db.add_evaluator(evaluator_id, evaluator_name, "Unknown", "")
            
            # Save comparison
            session_id = str(uuid.uuid4())
            self.db.save_comparison_evaluation(
                session_id,
                evaluator_id,
                "image_a",
                "image_b",
                preference,
                confidence,
                reasoning
            )
            
            return "Comparison submitted successfully!"
            
        except Exception as e:
            return f"Error: {str(e)}"

class EvaluationAnalyzer:
    """Analyzer for human evaluation results."""
    
    def __init__(self, db: EvaluationDatabase):
        self.db = db
    
    def generate_evaluation_report(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        df = self.db.get_evaluation_results(session_id)
        
        if df.empty:
            return {"error": "No evaluation data found"}
        
        # Parse scores
        df['scores_parsed'] = df['scores'].apply(json.loads)
        
        # Calculate statistics
        report = {
            "summary": {
                "total_evaluations": len(df),
                "unique_evaluators": df['evaluator_id'].nunique(),
                "evaluation_period": {
                    "start": df['timestamp'].min(),
                    "end": df['timestamp'].max()
                }
            },
            "score_statistics": {},
            "evaluator_agreement": {},
            "recommendations": []
        }
        
        # Score statistics
        all_scores = defaultdict(list)
        for scores in df['scores_parsed']:
            for criterion, score in scores.items():
                all_scores[criterion].append(score)
        
        for criterion, scores in all_scores.items():
            report["score_statistics"][criterion] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores),
                "median": np.median(scores)
            }
        
        # Inter-evaluator agreement (simplified)
        if df['evaluator_id'].nunique() > 1:
            report["evaluator_agreement"] = self._calculate_agreement(df)
        
        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(report["score_statistics"])
        
        return report
    
    def _calculate_agreement(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate inter-evaluator agreement."""
        # Simplified agreement calculation
        agreement_scores = {}
        
        # Group by image and calculate variance across evaluators
        grouped = df.groupby('image_id')
        
        for criterion in ['overall_quality', 'cultural_authenticity']:
            variances = []
            for image_id, group in grouped:
                if len(group) > 1:
                    scores = [json.loads(row['scores']).get(criterion, 0) 
                             for _, row in group.iterrows()]
                    if scores:
                        variances.append(np.var(scores))
            
            if variances:
                # Lower variance indicates higher agreement
                agreement_scores[criterion] = 1.0 / (1.0 + np.mean(variances))
            else:
                agreement_scores[criterion] = 0.0
        
        return agreement_scores
    
    def _generate_recommendations(self, score_stats: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        for criterion, stats in score_stats.items():
            if stats['mean'] < 5.0:
                recommendations.append(
                    f"Low scores in {criterion} (avg: {stats['mean']:.2f}) - "
                    f"consider improving this aspect"
                )
            
            if stats['std'] > 2.0:
                recommendations.append(
                    f"High variance in {criterion} scores (std: {stats['std']:.2f}) - "
                    f"may indicate inconsistent quality or evaluator disagreement"
                )
        
        return recommendations
    
    def export_results(self, output_path: str, format: str = "csv"):
        """Export evaluation results."""
        df = self.db.get_evaluation_results()
        
        if format == "csv":
            df.to_csv(output_path, index=False)
        elif format == "json":
            df.to_json(output_path, orient="records", indent=2)
        elif format == "excel":
            df.to_excel(output_path, index=False)
        
        logger.info(f"Results exported to {output_path}")

class HumanEvaluationManager:
    """Main manager for human evaluation system."""
    
    def __init__(self, db_path: str = "evaluation_data.db"):
        self.db = EvaluationDatabase(db_path)
        self.streamlit_interface = StreamlitEvaluationInterface(self.db)
        self.gradio_interface = GradioEvaluationInterface(self.db)
        self.analyzer = EvaluationAnalyzer(self.db)
    
    def launch_streamlit_interface(self):
        """Launch Streamlit evaluation interface."""
        self.streamlit_interface.run_evaluation_interface()
    
    def launch_gradio_interface(self, share: bool = False):
        """Launch Gradio evaluation interface."""
        interface = self.gradio_interface.create_interface()
        interface.launch(share=share)
    
    def generate_report(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate evaluation report."""
        return self.analyzer.generate_evaluation_report(session_id)
    
    def export_data(self, output_path: str, format: str = "csv"):
        """Export evaluation data."""
        self.analyzer.export_results(output_path, format)

def main():
    """Main function for testing human evaluation interface."""
    # Initialize evaluation manager
    manager = HumanEvaluationManager()
    
    # Test database functionality
    print("Testing database functionality...")
    
    # Add test evaluator
    manager.db.add_evaluator(
        "test_evaluator_1",
        "Test Evaluator",
        "Expert",
        "Art historian specializing in Indian miniature paintings"
    )
    
    # Create test evaluation
    test_evaluation = ImageEvaluation(
        image_id="test_image_1",
        session_id="test_session_1",
        evaluator_id="test_evaluator_1",
        scores={"overall_quality": 8, "cultural_authenticity": 7},
        comments="Well executed painting with good cultural elements",
        time_spent=120.0,
        timestamp=datetime.now(),
        image_metadata={"raga": "bhairav", "style": "rajput"}
    )
    
    manager.db.save_image_evaluation(test_evaluation)
    
    # Generate report
    report = manager.generate_report()
    print("Evaluation Report:")
    print(json.dumps(report, indent=2, default=str))
    
    # Launch interface (uncomment to test)
    # manager.launch_gradio_interface()
    
    print("Human evaluation system testing completed!")

if __name__ == "__main__":
    main()
