"""
ZSXT Web UI Application
Zero-Shot X-ray Style Translation - Interactive Web Interface
Built with Streamlit for easy deployment and user-friendly interaction
"""

import streamlit as st
import torch
import cv2
import numpy as np
from pathlib import Path
import tempfile
import os
from PIL import Image
import yaml
import time

# Import ZSXT modules
from inference import ZSXTInference
from src.model import GeneratorUNet

# ==================== Page Configuration ====================
st.set_page_config(
    page_title="ZSXT: Zero-Shot X-ray Style Translation",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== Custom Styling ====================
st.markdown("""
<style>
    .main-header {
        font-size: 3em;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5em;
    }
    .sub-header {
        font-size: 1.5em;
        color: #666;
        text-align: center;
        margin-bottom: 1em;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1.5em;
        border-radius: 0.5em;
        margin: 1em 0;
        border-left: 4px solid #1f77b4;
    }
    .metric-box {
        background-color: #e8f4f8;
        padding: 1em;
        border-radius: 0.5em;
        text-align: center;
        margin: 0.5em 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1em;
        border-radius: 0.5em;
        border-left: 4px solid #28a745;
        margin: 1em 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1em;
        border-radius: 0.5em;
        border-left: 4px solid #ffc107;
        margin: 1em 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== Session State Management ====================
if 'inference' not in st.session_state:
    st.session_state.inference = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'device' not in st.session_state:
    st.session_state.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==================== Sidebar Configuration ====================
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Device selection
    device_option = st.radio(
        "Select Device:",
        ["CUDA (GPU)", "CPU"],
        help="Choose device for inference"
    )
    st.session_state.device = 'cuda' if device_option == "CUDA (GPU)" else 'cpu'
    
    # Model checkpoint selection
    st.markdown("### 📦 Model Selection")
    checkpoint_path = st.text_input(
        "Checkpoint Path:",
        value="checkpoints/gen_best.pth",
        help="Path to model weights (e.g., checkpoints/gen_best.pth)"
    )
    
    config_path = st.text_input(
        "Config Path:",
        value="config.yaml",
        help="Path to configuration file"
    )
    
    # Load model button
    if st.button("🔄 Load Model", key="load_model", use_container_width=True):
        with st.spinner("Loading model..."):
            try:
                st.session_state.inference = ZSXTInference(
                    checkpoint_path=checkpoint_path,
                    config_path=config_path,
                    device=st.session_state.device
                )
                st.session_state.model_loaded = True
                st.success(f"✅ Model loaded successfully on {st.session_state.device.upper()}")
            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")
                st.session_state.model_loaded = False
    
    # Display system info
    st.markdown("### 🖥️ System Info")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Device", st.session_state.device.upper())
    with col2:
        st.metric("CUDA", "Available" if torch.cuda.is_available() else "N/A")
    
    if torch.cuda.is_available():
        col1, col2 = st.columns(2)
        with col1:
            st.metric("GPU", torch.cuda.get_device_name(0)[:20] + "...")
        with col2:
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            st.metric("VRAM", f"{vram:.1f} GB")

# ==================== Main Content ====================
st.markdown('<div class="main-header">🎨 ZSXT</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Zero-Shot X-ray Style Translation</div>', unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["🚀 Quick Start", "🖼️ Single Image", "📁 Batch Processing", "📊 Information"]
)

# ==================== Tab 1: Quick Start ====================
with tab1:
    st.markdown("""
    ## Welcome to ZSXT!
    
    This application provides an interactive interface for **Zero-Shot X-ray Style Translation**.
    
    ### 📋 Quick Steps:
    
    1. **Load Model** (in sidebar)
       - Set checkpoint path and config file
       - Click "Load Model" button
    
    2. **Process Images**
       - Choose between single image or batch processing
       - Upload your X-ray images (PNG/JPG)
    
    3. **Get Results**
       - View translations instantly
       - Download processed images
    """)
    
    # Feature highlight
    st.markdown('<div class="feature-box">', unsafe_allow_html=True)
    st.markdown("""
    ### ✨ Key Features:
    - ⚡ **Zero-Shot Learning**: No target domain data needed
    - 🎯 **Lightweight**: Only 37.7M parameters
    - 🚀 **Fast**: <50ms inference per image
    - 📈 **Accurate**: +74.5% mAP improvement
    - 🔄 **Batch Ready**: Process multiple images
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Model status
    if st.session_state.model_loaded:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("✅ **Model Status**: Ready for inference")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("⚠️ **Model Status**: Please load a model from the sidebar")
        st.markdown('</div>', unsafe_allow_html=True)

# ==================== Tab 2: Single Image Processing ====================
with tab2:
    st.markdown("## 🖼️ Single Image Processing")
    
    if not st.session_state.model_loaded:
        st.warning("⚠️ Please load a model first (use sidebar configuration)")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📤 Upload Image")
            uploaded_file = st.file_uploader(
                "Choose an X-ray image",
                type=['png', 'jpg', 'jpeg'],
                help="Supported formats: PNG, JPG, JPEG"
            )
        
        with col2:
            st.markdown("### ⚙️ Settings")
            enable_sr = st.checkbox("Enable Super-Resolution", value=True)
            show_comparison = st.checkbox("Show Before/After", value=True)
        
        if uploaded_file is not None:
            # Display progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Original image
                original_img = Image.open(tmp_path)
                original_np = np.array(original_img)
                
                progress_bar.progress(20)
                status_text.text("📥 Image loaded...")
                
                # Process image
                status_text.text("⚙️ Preprocessing...")
                progress_bar.progress(40)
                
                start_time = time.time()
                
                gray_tensor, original_size = st.session_state.inference.preprocess_image(tmp_path)
                
                progress_bar.progress(60)
                status_text.text("🎨 Generating translation...")
                
                rgb_img = st.session_state.inference.infer_single(gray_tensor)
                
                progress_bar.progress(80)
                status_text.text("✅ Processing complete...")
                
                # Convert to RGB for display
                rgb_display = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
                
                inference_time = time.time() - start_time
                progress_bar.progress(100)
                
                # Display results
                st.markdown("### 🎯 Results")
                
                if show_comparison:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(original_np, caption="Original Image", use_column_width=True)
                    with col2:
                        st.image(rgb_display, caption="ZSXT Translation", use_column_width=True)
                else:
                    st.image(rgb_display, caption="ZSXT Translation", use_column_width=True)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Inference Time", f"{inference_time*1000:.1f} ms")
                with col2:
                    st.metric("Output Size", f"{rgb_img.shape[1]}x{rgb_img.shape[0]}")
                with col3:
                    st.metric("Color Channels", "3 (RGB)")
                
                # Download button
                success_col1, success_col2 = st.columns(2)
                with success_col1:
                    # Save as numpy to convert
                    from io import BytesIO
                    img_pil = Image.fromarray(rgb_display)
                    buf = BytesIO()
                    img_pil.save(buf, format="PNG")
                    buf.seek(0)
                    
                    st.download_button(
                        label="⬇️ Download Result (PNG)",
                        data=buf.getvalue(),
                        file_name=f"zsxt_{uploaded_file.name}",
                        mime="image/png",
                        use_container_width=True
                    )
                
                # Cleanup
                os.remove(tmp_path)
                status_text.empty()
                progress_bar.empty()
                
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
                if 'tmp_path' in locals():
                    os.remove(tmp_path)

# ==================== Tab 3: Batch Processing ====================
with tab3:
    st.markdown("## 📁 Batch Processing")
    
    if not st.session_state.model_loaded:
        st.warning("⚠️ Please load a model first (use sidebar configuration)")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📁 Folder Settings")
            input_folder = st.text_input(
                "Input Folder:",
                value="datasets/Target_domain/CLC_extract/images",
                help="Path to folder containing input images"
            )
        
        with col2:
            st.markdown("### 💾 Output Settings")
            output_folder = st.text_input(
                "Output Folder:",
                value="datasets/Target_domain/CLC_extract_ZSXT/images",
                help="Path to save processed images"
            )
        
        col1, col2 = st.columns(2)
        with col1:
            enable_sr_batch = st.checkbox("Enable Super-Resolution", value=True, key="sr_batch")
        with col2:
            copy_labels = st.checkbox("Copy Labels Folder", value=True, key="copy_labels")
        
        if st.button("🚀 Start Batch Processing", use_container_width=True, key="batch_process"):
            try:
                if not os.path.exists(input_folder):
                    st.error(f"❌ Input folder not found: {input_folder}")
                else:
                    # Create placeholder for progress
                    progress_placeholder = st.empty()
                    status_placeholder = st.empty()
                    
                    # Get image files
                    image_files = [f for f in os.listdir(input_folder)
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    
                    if not image_files:
                        st.error(f"❌ No images found in {input_folder}")
                    else:
                        os.makedirs(output_folder, exist_ok=True)
                        
                        # Progress tracking
                        progress_bar = st.progress(0)
                        results_list = []
                        
                        start_time = time.time()
                        
                        for idx, img_file in enumerate(image_files):
                            try:
                                input_path = os.path.join(input_folder, img_file)
                                output_path = os.path.join(output_folder, img_file)
                                
                                # Process
                                gray_tensor, _ = st.session_state.inference.preprocess_image(input_path)
                                rgb_img = st.session_state.inference.infer_single(gray_tensor)
                                cv2.imwrite(output_path, rgb_img)
                                
                                results_list.append({
                                    'file': img_file,
                                    'status': '✅'
                                })
                                
                                progress = (idx + 1) / len(image_files)
                                progress_bar.progress(progress)
                                status_placeholder.text(f"Processing: {idx + 1}/{len(image_files)} - {img_file}")
                                
                            except Exception as e:
                                results_list.append({
                                    'file': img_file,
                                    'status': f'❌ {str(e)[:30]}'
                                })
                        
                        total_time = time.time() - start_time
                        
                        # Display results
                        progress_placeholder.empty()
                        status_placeholder.empty()
                        
                        st.markdown("### ✅ Processing Complete")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Images", len(image_files))
                        with col2:
                            successful = sum(1 for r in results_list if r['status'] == '✅')
                            st.metric("Successful", successful)
                        with col3:
                            st.metric("Total Time", f"{total_time:.2f}s")
                        with col4:
                            avg_time = total_time / len(image_files)
                            st.metric("Avg Time/Image", f"{avg_time*1000:.1f}ms")
                        
                        st.markdown("### 📝 Results Summary")
                        for result in results_list[:10]:
                            st.write(f"{result['status']} {result['file']}")
                        
                        if len(results_list) > 10:
                            st.write(f"... and {len(results_list) - 10} more files")
                        
                        st.success(f"✅ Batch processing completed! Results saved to: {output_folder}")
                        
            except Exception as e:
                st.error(f"❌ Error in batch processing: {str(e)}")

# ==================== Tab 4: Information ====================
with tab4:
    st.markdown("## 📊 Project Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Architecture")
        st.markdown("""
        **Generator**: 8-layer UNet
        - Input: 1 channel (Grayscale)
        - Output: 3 channels (RGB)
        - Parameters: 34.9M
        
        **Discriminator**: PatchGAN
        - Input: 4 channels (Gray + RGB)
        - Receptive Field: 70×70
        - Parameters: 2.77M
        
        **Total Parameters**: 37.7M
        """)
    
    with col2:
        st.markdown("### 💡 Key Features")
        st.markdown("""
        ✅ **Zero-Shot Learning**: No target domain data
        
        ✅ **Lightweight**: 67% fewer params than CUT
        
        ✅ **Fast**: <50ms inference per image
        
        ✅ **Accurate**: +74.5% mAP vs baseline
        
        ✅ **Smart**: Multi-modal parameter coverage
        """)
    
    st.markdown("### 🔄 Processing Pipeline")
    st.markdown("""
    1. **Decolorization** → Remove vendor pseudocoloring (ITU-R BT.601)
    2. **Multi-Modal Augmentation** → Synthetic device variation
       - Poisson noise, brightness, ripple, artifacts, lens flare
    3. **Four-Component Loss** → Quality metrics
       - L1 (70%), GAN (10%), Perceptual (15%), Color (5%)
    4. **Dynamic Scheduling** → 3-stage training progression
    """)
    
    st.markdown("### 📈 Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("mAP Improvement", "+74.5%")
    with col2:
        st.metric("Parameter Efficiency", "67%")
    with col3:
        st.metric("Inference Speed", "<50ms")
    with col4:
        st.metric("Model Size", "37.7M")
    
    st.markdown("### 📚 Documentation")
    st.markdown("""
    - **README.md** - Complete reference guide
    - **QUICKSTART.md** - 5-minute quick start
    - **ENVIRONMENT_SETUP.md** - Installation & setup
    - **PROJECT_OVERVIEW.md** - Architecture details
    """)
    
    st.markdown("### 📄 Citation")
    st.code("""
@inproceedings{zhang2025zsxt,
  title={Zero-Shot Pseudocolor X-ray Domain Translation 
         for Cross-Device Industrial Collaboration},
  author={Zhang, Xiaohao and others},
  booktitle={IEEE Conference},
  year={2025}
}
    """, language="bibtex")

# ==================== Footer ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p>ZSXT: Zero-Shot X-ray Style Translation | Web Interface v1.0</p>
    <p>Built with Streamlit | PyTorch | Python</p>
    <p>For issues or questions, please refer to the project documentation</p>
</div>
""", unsafe_allow_html=True)
