# 🎨 ZSXT Web UI Guide

## Overview

ZSXT Web UI is an interactive web interface built with **Streamlit** that provides an easy-to-use platform for:
- 🖼️ Single image translation
- 📁 Batch processing
- ⚙️ Configuration management
- 📊 Real-time performance monitoring

## Quick Start

### Step 1: Install Streamlit

```bash
pip install streamlit>=1.28.0
# Or update requirements:
pip install -r requirements.txt
```

### Step 2: Prepare Model

Ensure you have a trained model checkpoint:
```
checkpoints/gen_best.pth
```

### Step 3: Start the UI

**Option A: Using run_ui.py (Recommended)**
```bash
python run_ui.py
```

**Option B: Direct Streamlit**
```bash
streamlit run app.py
```

### Step 4: Open in Browser

The web interface will automatically open at:
```
http://localhost:8501
```

If it doesn't open automatically, manually navigate to the URL above.

---

## Interface Overview

### 🎯 Sidebar Configuration

#### Model Settings
- **Device Selection**: Choose between CUDA (GPU) or CPU
- **Checkpoint Path**: Path to model weights (default: `checkpoints/gen_best.pth`)
- **Config Path**: Path to config file (default: `config.yaml`)
- **Load Model Button**: Click to load/reload model

#### System Information
- Device type (CUDA/CPU)
- GPU availability
- GPU name (if available)
- VRAM size (if available)

### 📑 Main Tabs

#### 1️⃣ Quick Start Tab
- Welcome introduction
- Feature highlights
- Model status indicator
- Quick navigation guide

#### 2️⃣ Single Image Tab
- **📤 Image Upload**: Upload single X-ray image (PNG/JPG/JPEG)
- **⚙️ Settings**:
  - Enable/disable super-resolution
  - Toggle before/after comparison
- **🎯 Results Display**:
  - Original image
  - Translated image
  - Inference metrics (time, size, channels)
- **⬇️ Download Button**: Save result as PNG

**Key Metrics Displayed**:
- Inference Time (ms)
- Output Image Size (WxH)
- Color Channels

#### 3️⃣ Batch Processing Tab
- **📁 Folder Configuration**:
  - Input folder path
  - Output folder path
- **⚙️ Options**:
  - Enable super-resolution
  - Copy labels folder
- **🚀 Processing**:
  - Progress bar
  - Real-time status updates
  - Results summary
- **📊 Performance Metrics**:
  - Total images processed
  - Success count
  - Total processing time
  - Average time per image

#### 4️⃣ Information Tab
- **🎯 Architecture Details**:
  - Generator specs
  - Discriminator specs
  - Parameter counts
- **💡 Key Features**
- **🔄 Processing Pipeline**
- **📈 Performance Metrics**
- **📚 Documentation Links**
- **📄 Citation Format**

---

## Usage Examples

### Example 1: Single Image Translation

1. **Load Model** (Sidebar)
   - Keep default settings
   - Click "Load Model" button
   - Wait for "✅ Model loaded successfully"

2. **Navigate to Single Image Tab**
   - Click "🖼️ Single Image" tab

3. **Upload Image**
   - Click file uploader
   - Select an X-ray image (PNG/JPG)

4. **Configure Settings**
   - Enable Super-Resolution: ✓
   - Show Before/After: ✓

5. **View Results**
   - See original and translated images side-by-side
   - Check inference time

6. **Download**
   - Click "⬇️ Download Result (PNG)" button

### Example 2: Batch Processing

1. **Prepare Folders**
   - Input folder: `datasets/Target_domain/CLC_extract/images`
   - Output folder: `datasets/Target_domain/CLC_extract_ZSXT/images`

2. **Navigate to Batch Processing Tab**
   - Click "📁 Batch Processing" tab

3. **Configure Paths**
   - Input Folder: Paste input path
   - Output Folder: Paste output path

4. **Set Options**
   - Enable Super-Resolution: ✓
   - Copy Labels Folder: ✓

5. **Start Processing**
   - Click "🚀 Start Batch Processing" button
   - Monitor progress bar
   - Wait for completion

6. **View Results Summary**
   - Total images processed
   - Success rate
   - Processing time stats

### Example 3: Multi-Device Processing

1. **Load Different Model**
   - Sidebar → Checkpoint Path
   - Change to: `checkpoints/gen_best_v2.pth`
   - Click "Load Model"

2. **Process Images**
   - Use tabs to process with new model

3. **Compare Results**
   - Switch models and process same image
   - Compare translations visually

---

## Advanced Features

### Custom Configuration

**In the sidebar, you can modify:**
1. Device selection (CUDA/CPU)
2. Model checkpoint path
3. Configuration file path

### Performance Monitoring

**Real-time metrics include:**
- Inference time per image
- Processing throughput (images/second)
- Output size
- Success/failure counts

### Error Handling

If errors occur:
1. Check model path is correct
2. Verify config.yaml exists
3. Ensure input images are valid
4. Check GPU/CUDA availability
5. Refer to console output for details

---

## Troubleshooting

### Problem: "Model not found"
**Solution:**
- Check checkpoint path in sidebar
- Verify file exists: `checkpoints/gen_best.pth`
- Use full absolute path if relative path fails

### Problem: "Config file not found"
**Solution:**
- Ensure `config.yaml` is in project root
- Update path in sidebar if in different location

### Problem: "CUDA out of memory"
**Solution:**
- Switch to CPU mode in sidebar
- Reduce batch size in config.yaml
- Process smaller images

### Problem: "Streamlit not installed"
**Solution:**
```bash
pip install streamlit>=1.28.0
# Or
pip install -r requirements.txt
```

### Problem: "Port 8501 already in use"
**Solution:**
```bash
streamlit run app.py --server.port 8502
# Use different port (8502, 8503, etc.)
```

---

## Features Comparison

### Command-Line vs Web UI

| Feature | CLI | Web UI |
|---------|-----|--------|
| Batch Processing | ✅ | ✅ |
| Single Image | ✅ | ✅ |
| Visual Preview | ❌ | ✅ |
| Configuration | YAML only | GUI + YAML |
| Model Selection | CLI args | GUI dropdown |
| Results Download | Manual | One-click |
| Performance Stats | Log output | Dashboard |
| Ease of Use | Medium | High |

---

## Performance Optimization

### For Faster Processing:
1. **Enable CUDA** (sidebar)
2. **Disable Super-Resolution** if not needed
3. **Use CPU for testing** only

### For Better Quality:
1. **Enable Super-Resolution**
2. **Use larger batch size** (config.yaml)
3. **Enable comparison mode**

### For Production:
1. **Use CLI** (`inference.py`) for batch
2. **Parallel processing** on multiple GPUs
3. **Load balancing** with multiple instances

---

## Integration with Other Tools

### Use with YOLO Detection:
1. Process images with ZSXT UI
2. Run YOLO detection on output
3. Copy labels folder automatically

### Use with Analysis Pipeline:
1. Batch process images
2. Export results
3. Run custom analysis scripts

---

## Deployment Options

### Local Deployment (Current)
```bash
streamlit run app.py
# Runs on: http://localhost:8501
```

### Docker Deployment
```dockerfile
FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.headless", "true"]
```

### Cloud Deployment
- **Streamlit Cloud**: Free hosting
- **AWS**: EC2 + Streamlit
- **Google Cloud**: Run + Streamlit
- **Azure**: Container Instances

---

## Tips & Tricks

### Keyboard Shortcuts (Streamlit):
- `R`: Rerun app
- `C`: Clear cache
- `K`: List keyboard shortcuts

### Performance Tips:
- Cache model in session state (already done)
- Process images in batches
- Use GPU for inference
- Disable comparison mode for speed

### Advanced Usage:
- Modify `app.py` for custom features
- Add new tabs for analysis
- Integrate with external APIs
- Export metrics to CSV

---

## Technical Details

### Framework: Streamlit
- **Version**: >=1.28.0
- **Features**: Multi-page, state management, widgets
- **Performance**: Optimized with caching

### Session State Management
```python
if 'inference' not in st.session_state:
    st.session_state.inference = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
```

### Image Processing Pipeline
1. Upload → 2. Preprocess → 3. Inference → 4. Display/Download

---

## Future Enhancements

Planned features:
- [ ] Real-time image comparison slider
- [ ] Batch image gallery viewer
- [ ] Model training interface
- [ ] Results analytics dashboard
- [ ] Multi-model ensemble
- [ ] Advanced filtering options
- [ ] API endpoint for batch jobs
- [ ] Metrics export (CSV/JSON)

---

## Support & Documentation

- **README.md**: Project overview
- **QUICKSTART.md**: Quick start guide
- **ENVIRONMENT_SETUP.md**: Installation steps
- **PROJECT_OVERVIEW.md**: Architecture details

---

## Citation

If you use ZSXT Web UI, please cite:

```bibtex
@inproceedings{zhang2025zsxt,
  title={Zero-Shot Pseudocolor X-ray Domain Translation 
         for Cross-Device Industrial Collaboration},
  author={Zhang, Xiaohao and others},
  booktitle={IEEE Conference},
  year={2025}
}
```

---

**Last Updated**: November 30, 2025  
**Version**: 1.0  
**Status**: ✅ Ready for Use
