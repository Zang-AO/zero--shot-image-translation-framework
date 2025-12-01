🎉 ZSXT Web UI - START HERE
=============================

## ⚡ Quick Launch (30 seconds)

```bash
python run_ui.py
```

**Browser opens automatically to:** http://localhost:8501

---

## 🚀 Launch Options

### Option 1: Python Launcher (Recommended)
```bash
python run_ui.py
```
✅ Checks dependencies  
✅ Validates model paths  
✅ Shows helpful messages  

### Option 2: Direct Streamlit
```bash
streamlit run app.py
```
✅ Direct launch  
✅ Full control  

### Option 3: Windows Batch
```
Double-click: start_ui.bat
```
✅ One-click on Windows  

### Option 4: Unix/Linux/Mac
```bash
bash start_ui.sh
```
✅ One-click on Unix systems  

---

## 📖 Documentation Map

| Document | Purpose | Read Time |
|----------|---------|-----------|
| **README.md** | Project overview & features | 5 min |
| **UI_GUIDE.md** | Complete UI documentation | 15 min |
| **QUICK_REFERENCE.md** | Quick lookup guide | 2 min |
| **QUICKSTART.md** | Step-by-step usage | 10 min |
| **COMPLETION_REPORT.md** | Implementation details | 10 min |

---

## 🎨 Web UI Features

### 🏠 Quick Start Tab
- Project overview
- Model status
- Feature highlights

### 📸 Single Image Tab
- Upload image
- Real-time preview
- One-click translate
- Download result

### 📁 Batch Processing Tab
- Select folder
- Batch translate
- View metrics
- Download all results

### ℹ️ Information Tab
- Model architecture
- Parameter details
- Performance metrics
- Citations

---

## ✅ Verification Checklist

Before first launch, verify:

```bash
# 1. Check Python environment
python verify_env.py
# Should show: ✅ All checks passed!

# 2. Check dependencies
pip list | grep streamlit
# Should show: streamlit (>=1.28.0)

# 3. Check GPU (optional)
python -c "import torch; print('GPU:', torch.cuda.is_available())"
```

---

## 🎯 Common Tasks

### Process Single Image
1. Open Web UI
2. Click "📸 Single Image" tab
3. Click "📤 Upload Image"
4. Select image file
5. Click "Process" button
6. Click "📥 Download Result"

### Batch Process Folder
1. Open Web UI
2. Click "📁 Batch Processing" tab
3. Click "📁 Select Folder"
4. Choose folder with images
5. Click "🚀 Start Batch Processing"
6. Wait for progress bar
7. Click "📦 Download All Results"

### Use Custom Model
1. Click device/config menu (top-right)
2. Expand "⚙️ Advanced Settings"
3. Enter custom model path
4. Click "Load Model"
5. New model is ready to use

### Check GPU Status
1. Look at sidebar (right side)
2. See "💻 System Information"
3. Shows GPU name and memory usage

---

## ⚡ Performance Tips

| Setting | Speed | Memory | Best For |
|---------|-------|--------|----------|
| GPU Mode | 🚀 Fast | ~2GB | Production |
| CPU Mode | 🐢 Slow | ~500MB | Testing |
| Batch Size 1 | 📊 Balanced | ~1GB | Memory-limited |
| Batch Size 3 | ⚡ Fast | ~2GB | Powerful GPU |

---

## 🆘 Common Issues

### "Port 8501 already in use"
```bash
streamlit run app.py --server.port 8502
```

### "CUDA out of memory"
1. Switch to CPU mode in sidebar
2. Or close other GPU applications
3. Or reduce batch size

### "Model checkpoint not found"
1. Ensure `checkpoints/gen_best.pth` exists
2. Or specify custom path in sidebar
3. Or download from releases

### "Dependencies not installed"
```bash
pip install -r requirements.txt --upgrade
python verify_env.py
```

---

## 📱 System Requirements

- **CPU**: Minimum i5 (Recommended i7+)
- **GPU**: NVIDIA with CUDA 11.0+ (Optional)
- **RAM**: 8GB+ for GPU, 4GB+ for CPU
- **Disk**: 50GB free space
- **Browser**: Chrome, Firefox, Safari, Edge
- **Python**: 3.8 or higher

---

## 🌐 Access from Other Devices

To access UI from another computer on same network:

```bash
# Start UI with network access
streamlit run app.py --server.address 0.0.0.0

# From another device, visit:
http://<your-ip>:8501
```

---

## 📊 File Structure

```
_code_EN/
├── 🎨 UI Files
│   ├── app.py                 Main interface
│   ├── run_ui.py              Python launcher
│   ├── start_ui.bat           Windows launcher
│   └── start_ui.sh            Unix launcher
├── 📚 Docs
│   ├── README.md              Main docs
│   ├── UI_GUIDE.md            UI documentation
│   ├── QUICK_REFERENCE.md     Quick guide
│   └── QUICKSTART.md          Getting started
├── 🚀 Scripts
│   ├── train.py               Training
│   ├── inference.py           Inference
│   └── verify_env.py          Setup check
└── 💾 Data
    ├── config.yaml            Settings
    ├── checkpoints/           Models
    └── datasets/              Data
```

---

## 🔗 Quick Links

- **GitHub**: [ZSXT Repository](#)
- **Documentation**: See README.md
- **Issues**: Report problems here
- **Contributing**: Help us improve!

---

## 💡 Pro Tips

1. **Faster Processing**: Use GPU mode
2. **Better Quality**: Try different models in checkpoints/
3. **Batch Processing**: Process multiple images at once
4. **Custom Models**: Load your own trained models
5. **Multiple Runs**: Results saved automatically

---

## 🎓 First Time Users

1. ✅ Read **README.md** (5 min)
2. ✅ Run `python verify_env.py` (1 min)
3. ✅ Launch with `python run_ui.py` (1 min)
4. ✅ Try **Single Image** tab (2 min)
5. ✅ Try **Batch Processing** tab (5 min)
6. ✅ Check **Information** tab (2 min)
7. ✅ Read **UI_GUIDE.md** for advanced features (15 min)

**Total Time**: ~30 minutes to master the UI

---

## ❓ FAQ

**Q: Do I need GPU?**  
A: No, CPU works fine. GPU is just faster (10-50ms vs 100-500ms).

**Q: Can I use my own model?**  
A: Yes! Specify model path in sidebar, click "Load Model".

**Q: How do I save results?**  
A: UI provides one-click download after processing.

**Q: Can I process folders with thousands of images?**  
A: Yes! Batch processing handles any number.

**Q: Is there a command-line version?**  
A: Yes, use `python inference.py` for CLI mode.

**Q: How do I train my own model?**  
A: Use `python train.py` with config.yaml settings.

---

## 📞 Support

If you encounter issues:

1. Check **QUICK_REFERENCE.md** → Troubleshooting
2. Check **UI_GUIDE.md** → Troubleshooting
3. Run `python verify_env.py` for diagnostics
4. Check project GitHub issues

---

**Ready to start?**

```bash
python run_ui.py
```

Then visit: http://localhost:8501

Enjoy! 🚀

---

**Version**: 1.0.0  
**Updated**: 2025-11-30
