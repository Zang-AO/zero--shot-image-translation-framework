# 🚀 ZSXT Quick Reference Guide

## Launch Options

| Method | Command | Platform |
|--------|---------|----------|
| **Python Launcher** | `python run_ui.py` | All (Recommended) |
| **Streamlit CLI** | `streamlit run app.py` | All |
| **Batch Script** | Double-click `start_ui.bat` | Windows |
| **Shell Script** | `bash start_ui.sh` | Linux/Mac |

## Web UI Tabs

### 🏠 Quick Start
- Project overview and features
- Real-time model status
- Download checkpoints info
- Direct links to documentation

### 📸 Single Image
1. **Upload** an image or URL
2. **Preview** the original
3. **Process** with one click
4. **View** results with metrics
5. **Download** the translated image

### 📁 Batch Processing
1. **Select folder** containing images
2. **Preview** file count and formats
3. **Start processing** with progress bar
4. **View batch metrics**:
   - Success rate
   - Total inference time
   - Average time per image
5. **Download all** results as ZIP

### ℹ️ Information
- **Architecture**: Model structure and parameters
- **Features**: Capabilities and highlights
- **Performance**: Benchmark metrics
- **Citations**: Relevant references

## Sidebar Controls

| Control | Purpose | Default |
|---------|---------|---------|
| Device Selection | CPU or GPU processing | Auto-detect |
| Model Path | Custom generator checkpoint | `checkpoints/gen_best.pth` |
| Config Path | Training configuration | `config.yaml` |
| System Info | Display GPU/CPU status | On |

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+C` | Stop server (terminal) |
| `Ctrl+Shift+K` | Clear browser cache |
| `F5` | Refresh page |

## Common Issues

### Issue: "Port 8501 already in use"
```bash
streamlit run app.py --server.port 8502
```

### Issue: "CUDA out of memory"
1. Use **CPU mode** from sidebar
2. Or reduce batch size in config

### Issue: "Model checkpoint not found"
1. Verify path in sidebar
2. Download from releases page
3. Place in `checkpoints/` folder

## Performance Tips

- **GPU Processing**: 10-50ms per image
- **CPU Processing**: 100-500ms per image
- **Batch Mode**: ~2ms overhead per image
- **Memory**: ~2GB for GPU, ~500MB for CPU

## File Structure for Batch Processing

```
my_images/
├── image1.jpg
├── image2.png
├── subfolder/
│   ├── image3.jpg
│   └── image4.tiff
└── image5.bmp
```

✅ Automatically detects and processes all supported formats:
- `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`, etc.

## Command Line Alternatives

### Single Image Inference
```bash
python inference.py --input image.jpg --output output.jpg --gpu
```

### Batch Inference
```bash
python inference.py --input ./images --output ./results --gpu
```

### Training
```bash
python train.py --config config.yaml --gpu
```

## Configuration Management

Edit `config.yaml` for:
- Batch size
- Image dimensions
- Learning rate (training)
- Data paths
- Model parameters

## Environment Check

```bash
python verify_env.py
```

Should show:
```
✅ CUDA device
✅ Model checkpoint
✅ Configuration file
✅ All libraries
```

## Docker Deployment

```bash
docker build -t zsxt-ui .
docker run -p 8501:8501 zsxt-ui
```

## Requirements

- Python 3.8+
- CUDA 11.0+ (for GPU, optional)
- 50GB free disk space
- 8GB+ GPU memory (or 2GB+ for CPU)

## Support & Documentation

- 📖 **Full Guide**: See [UI_GUIDE.md](UI_GUIDE.md)
- 🎓 **Project Overview**: See [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)
- ⚙️ **Setup Guide**: See [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)
- 🚀 **Quick Start**: See [QUICKSTART.md](QUICKSTART.md)
- 📋 **README**: See [README.md](README.md)

---

**Last Updated**: 2025-11-30  
**Version**: 1.0.0
