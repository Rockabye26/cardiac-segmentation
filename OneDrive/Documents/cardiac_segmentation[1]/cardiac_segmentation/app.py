import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import io
import time
import tempfile
import os
from typing import Optional, Tuple, Dict, List
import json
import pandas as pd

# Import our modules
from models.unet import create_unet_model
from visualization.visualizer import CardiacVisualization
from evaluation.evaluator import ModelEvaluator
from training.metrics import SegmentationMetrics

# Set page configuration
st.set_page_config(
    page_title="Cardiac MRI Segmentation Demo",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .section-header {
        color: #A23B72;
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class CardiacSegmentationApp:
    """Main Streamlit application for cardiac MRI segmentation demo"""
    
    def __init__(self):
        self.class_names = ['Background', 'RV', 'Myocardium', 'LV']
        self.class_colors = ['#000000', '#0000FF', '#00FF00', '#FF0000']
        self.visualizer = CardiacVisualization(self.class_names, self.class_colors)
        
        # Initialize session state
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'processing_results' not in st.session_state:
            st.session_state.processing_results = None
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load the U-Net model"""
        try:
            with st.spinner("Loading U-Net model..."):
                # Create model
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = create_unet_model('standard', device=device)
                
                # Load weights if path provided
                if model_path and os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    st.success(f"Model loaded from: {model_path}")
                else:
                    # Initialize with random weights for demo
                    st.warning("Using randomly initialized model for demo purposes")
                
                model.eval()
                st.session_state.model = model
                st.session_state.model_loaded = True
                
                return True
                
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def load_nifti_file(self, uploaded_file) -> Optional[np.ndarray]:
        """Load and process NIfTI file"""
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            
            # Load NIfTI file
            nii = nib.load(tmp_file_path)
            data = nii.get_fdata()
            
            # Handle 3D volumes by selecting middle slice
            if len(data.shape) == 3:
                middle_slice = data.shape[2] // 2
                data = data[:, :, middle_slice]
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            return data.astype(np.float32)
            
        except Exception as e:
            st.error(f"Error loading NIfTI file: {str(e)}")
            return None
    
    def preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> torch.Tensor:
        """Preprocess image for model input"""
        # Normalize to [0, 1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Resize to target size
        image_pil = Image.fromarray((image * 255).astype(np.uint8))
        image_pil = image_pil.resize(target_size, Image.LANCZOS)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(np.array(image_pil)).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        
        return image_tensor
    
    def run_inference(self, image_tensor: torch.Tensor) -> Tuple[torch.Tensor, np.ndarray, float]:
        """Run model inference"""
        device = next(st.session_state.model.parameters()).device
        image_tensor = image_tensor.to(device)
        
        start_time = time.time()
        
        with torch.no_grad():
            logits = st.session_state.model(image_tensor)
            probabilities = F.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1)
        
        inference_time = time.time() - start_time
        
        # Convert to numpy
        prediction_np = prediction.cpu().numpy().squeeze()
        probabilities_np = probabilities.cpu().numpy().squeeze()
        
        return probabilities, prediction_np, inference_time
    
    def calculate_metrics(self, prediction: np.ndarray, ground_truth: Optional[np.ndarray] = None) -> Dict:
        """Calculate segmentation metrics"""
        metrics = {}
        
        # Class distribution
        unique, counts = np.unique(prediction, return_counts=True)
        total_pixels = prediction.size
        
        for class_id, count in zip(unique, counts):
            class_name = self.class_names[class_id]
            metrics[f'{class_name}_pixels'] = int(count)
            metrics[f'{class_name}_percentage'] = (count / total_pixels) * 100
        
        # If ground truth available, calculate Dice scores
        if ground_truth is not None:
            dice_scores = []
            for class_id in range(len(self.class_names)):
                pred_mask = (prediction == class_id)
                gt_mask = (ground_truth == class_id)
                
                intersection = (pred_mask & gt_mask).sum()
                union = pred_mask.sum() + gt_mask.sum()
                
                if union == 0:
                    dice = 1.0
                else:
                    dice = 2.0 * intersection / union
                
                dice_scores.append(dice)
                metrics[f'dice_{self.class_names[class_id].lower()}'] = dice
            
            metrics['dice_mean'] = np.mean(dice_scores[1:])  # Exclude background
        
        return metrics
    
    def render_header(self):
        """Render application header"""
        st.markdown('<h1 class="main-header">🫀 Cardiac MRI Segmentation Demo</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-box">
        <h3>🎯 Demo Overview</h3>
        This demonstration showcases an end-to-end cardiac MRI segmentation system using U-Net architecture. 
        Upload a cardiac MRI image (NIfTI format) to see real-time segmentation results with detailed analysis.
        <br><br>
        <strong>Segmentation Classes:</strong>
        <ul>
            <li>🔵 <span style="color: #0000FF;">Right Ventricle (RV)</span></li>
            <li>🟢 <span style="color: #00FF00;">Myocardium</span></li>
            <li>🔴 <span style="color: #FF0000;">Left Ventricle (LV)</span></li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.markdown("## 🔧 Configuration")
        
        # Model loading section
        st.sidebar.markdown("### Model Settings")
        
        model_path = st.sidebar.text_input(
            "Model Path (optional)",
            placeholder="path/to/best_model.pth",
            help="Leave empty to use randomly initialized model for demo"
        )
        
        if st.sidebar.button("Load Model", type="primary"):
            if self.load_model(model_path if model_path else None):
                st.sidebar.success("✅ Model loaded successfully!")
            else:
                st.sidebar.error("❌ Failed to load model")
        
        # Model status
        if st.session_state.model_loaded:
            st.sidebar.success("🤖 Model Ready")
            
            # Model info
            if st.session_state.model:
                info = st.session_state.model.get_model_info()
                st.sidebar.markdown("**Model Info:**")
                st.sidebar.write(f"Parameters: {info['total_parameters']:,}")
                st.sidebar.write(f"Size: {info['model_size_mb']:.1f} MB")
        else:
            st.sidebar.warning("⚠️ Model not loaded")
        
        # Visualization settings
        st.sidebar.markdown("### Visualization Settings")
        
        overlay_alpha = st.sidebar.slider(
            "Overlay Opacity",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Adjust transparency of segmentation overlay"
        )
        
        show_probabilities = st.sidebar.checkbox(
            "Show Class Probabilities",
            value=True,
            help="Display probability maps for each class"
        )
        
        return overlay_alpha, show_probabilities
    
    def render_file_upload(self):
        """Render file upload section"""
        st.markdown('<h2 class="section-header">📁 Upload Cardiac MRI</h2>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a NIfTI file (.nii, .nii.gz)",
            type=['nii', 'gz'],
            help="Upload a cardiac MRI scan in NIfTI format"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # Display file info
            file_info = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.1f} KB",
                "File type": uploaded_file.type
            }
            
            info_df = pd.DataFrame(list(file_info.items()), columns=['Property', 'Value'])
            st.table(info_df)
            
            return uploaded_file
        
        # Show example data info if no file uploaded
        else:
            st.info("💡 **No file uploaded.** Please upload a cardiac MRI NIfTI file to proceed.")
            
            st.markdown("""
            <div class="warning-box">
            <h4>📋 Expected File Format</h4>
            <ul>
                <li><strong>Format:</strong> NIfTI (.nii or .nii.gz)</li>
                <li><strong>Content:</strong> Cardiac MRI scan (grayscale)</li>
                <li><strong>Dimensions:</strong> 2D or 3D (middle slice will be used for 3D)</li>
                <li><strong>Size:</strong> Automatically resized to 256×256 for processing</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        return None
    
    def render_processing_section(self, uploaded_file, overlay_alpha: float, show_probabilities: bool):
        """Render image processing and results section"""
        if uploaded_file is None or not st.session_state.model_loaded:
            return
        
        st.markdown('<h2 class="section-header">⚡ Processing & Results</h2>', unsafe_allow_html=True)
        
        # Process button
        if st.button("🚀 Process Image", type="primary", use_container_width=True):
            
            with st.spinner("Processing image..."):
                # Load and preprocess image
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Load image
                status_text.text("Loading NIfTI file...")
                image = self.load_nifti_file(uploaded_file)
                progress_bar.progress(25)
                
                if image is None:
                    st.error("❌ Failed to load image")
                    return
                
                # Step 2: Preprocess
                status_text.text("Preprocessing image...")
                image_tensor = self.preprocess_image(image)
                progress_bar.progress(50)
                
                # Step 3: Run inference
                status_text.text("Running segmentation...")
                probabilities, prediction, inference_time = self.run_inference(image_tensor)
                progress_bar.progress(75)
                
                # Step 4: Calculate metrics
                status_text.text("Calculating metrics...")
                metrics = self.calculate_metrics(prediction)
                progress_bar.progress(100)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Store results
                st.session_state.processing_results = {
                    'original_image': image,
                    'prediction': prediction,
                    'probabilities': probabilities,
                    'metrics': metrics,
                    'inference_time': inference_time
                }
        
        # Display results if available
        if st.session_state.processing_results:
            self.render_results(overlay_alpha, show_probabilities)
    
    def render_results(self, overlay_alpha: float, show_probabilities: bool):
        """Render processing results"""
        results = st.session_state.processing_results
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>⏱️ Processing Time</h3>
                <h2>{results['inference_time']:.2f}s</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            rv_percentage = results['metrics'].get('RV_percentage', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h3>🔵 RV Coverage</h3>
                <h2>{rv_percentage:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            myo_percentage = results['metrics'].get('Myocardium_percentage', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h3>🟢 Myocardium</h3>
                <h2>{myo_percentage:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            lv_percentage = results['metrics'].get('LV_percentage', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h3>🔴 LV Coverage</h3>
                <h2>{lv_percentage:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Visualization section
        st.markdown("### 🖼️ Segmentation Results")
        
        # Create visualizations
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Original Image**")
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(results['original_image'], cmap='gray')
            ax.set_title('Original Cardiac MRI')
            ax.axis('off')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("**Segmentation Mask**")
            fig, ax = plt.subplots(figsize=(6, 6))
            cmap = ListedColormap(self.class_colors)
            im = ax.imshow(results['prediction'], cmap=cmap, vmin=0, vmax=3)
            ax.set_title('Predicted Segmentation')
            ax.axis('off')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_ticks(range(len(self.class_names)))
            cbar.set_ticklabels(self.class_names)
            
            st.pyplot(fig)
            plt.close()
        
        with col3:
            st.markdown("**Overlay Visualization**")
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(results['original_image'], cmap='gray')
            
            # Create colored overlays
            for class_id in range(1, len(self.class_names)):
                class_mask = (results['prediction'] == class_id)
                if np.any(class_mask):
                    colored_mask = np.zeros((*results['prediction'].shape, 4))
                    color = plt.cm.colors.to_rgba(self.class_colors[class_id])
                    colored_mask[class_mask] = (*color[:3], overlay_alpha)
                    ax.imshow(colored_mask)
            
            ax.set_title(f'Overlay (α={overlay_alpha})')
            ax.axis('off')
            st.pyplot(fig)
            plt.close()
        
        # Probability maps (if requested)
        if show_probabilities:
            st.markdown("### 📊 Class Probability Maps")
            
            cols = st.columns(len(self.class_names))
            for i, (col, class_name) in enumerate(zip(cols, self.class_names)):
                with col:
                    fig, ax = plt.subplots(figsize=(4, 4))
                    prob_map = results['probabilities'][i]
                    im = ax.imshow(prob_map, cmap='viridis', vmin=0, vmax=1)
                    ax.set_title(f'{class_name}\nProbability')
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, shrink=0.8)
                    st.pyplot(fig)
                    plt.close()
        
        # Detailed metrics table
        st.markdown("### 📈 Detailed Metrics")
        
        metrics_data = []
        for key, value in results['metrics'].items():
            if 'percentage' in key:
                metrics_data.append({
                    'Metric': key.replace('_', ' ').title(),
                    'Value': f"{value:.2f}%"
                })
            elif 'pixels' in key:
                metrics_data.append({
                    'Metric': key.replace('_', ' ').title(),
                    'Value': f"{value:,} pixels"
                })
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            st.table(metrics_df)
        
        # Download section
        st.markdown("### 💾 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download prediction as numpy array
            prediction_bytes = io.BytesIO()
            np.save(prediction_bytes, results['prediction'])
            st.download_button(
                label="📥 Download Segmentation",
                data=prediction_bytes.getvalue(),
                file_name="segmentation_mask.npy",
                mime="application/octet-stream"
            )
        
        with col2:
            # Download metrics as JSON
            metrics_json = json.dumps(results['metrics'], indent=2)
            st.download_button(
                label="📊 Download Metrics",
                data=metrics_json,
                file_name="segmentation_metrics.json",
                mime="application/json"
            )
    
    def render_about_section(self):
        """Render about section"""
        with st.expander("ℹ️ About This Demo", expanded=False):
            st.markdown("""
            ### 🔬 Technical Details
            
            **Model Architecture:**
            - U-Net with skip connections
            - 4-level encoder-decoder
            - ReLU activation, BatchNorm, Dropout
            - Mixed precision training support
            
            **Training Configuration:**
            - Combined loss: 0.3 × CrossEntropy + 0.7 × Dice
            - Adam optimizer with learning rate scheduling
            - Data augmentation: horizontal flip, rotation
            - Early stopping based on validation Dice
            
            **Performance Targets:**
            - Dice Coefficient: 75-80% (target for first demo)
            - Processing Time: 3-5 seconds per image
            - Memory Usage: <6GB GPU memory
            
            ### 📚 Dataset
            This demo is designed for the ACDC (Automated Cardiac Diagnosis Challenge) dataset:
            - 4-class segmentation: Background, RV, Myocardium, LV
            - Cardiac MRI images (ED/ES timepoints)
            - Resolution automatically adjusted to 256×256
            
            ### 🚀 Future Enhancements
            - Attention mechanisms for explainability
            - Multi-timepoint analysis
            - 3D volume processing
            - Clinical validation metrics
            """)
    
    def run(self):
        """Main application entry point"""
        # Render header
        self.render_header()
        
        # Render sidebar and get settings
        overlay_alpha, show_probabilities = self.render_sidebar()
        
        # Auto-load model on startup
        if not st.session_state.model_loaded:
            self.load_model()
        
        # Main content
        uploaded_file = self.render_file_upload()
        
        # Processing section
        self.render_processing_section(uploaded_file, overlay_alpha, show_probabilities)
        
        # About section
        self.render_about_section()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; margin-top: 2rem;">
        🫀 Cardiac MRI Segmentation Demo | Built with Streamlit & PyTorch<br>
        <em>Demo implementation for educational and research purposes</em>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    # Initialize and run the application
    app = CardiacSegmentationApp()
    app.run()