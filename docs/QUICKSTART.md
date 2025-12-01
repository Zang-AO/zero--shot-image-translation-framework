# ZSXT Quick Start Guide ⚡

Get started in **5 minutes** with ZSXT. Perfect for first-time users!

---

## 🚀 Three-Step Setup

### Step 1️⃣: Install Environment (2 min)

```bash
# Clone repository
git clone https://github.com/Zang-AO/zero--shot-image-translation-framework.git
cd zero--shot-image-translation-framework

# Create conda environment
conda create -n zsxt python=3.9 -y
conda activate zsxt

# Install all dependencies (includes PyTorch)
pip install torch torchvision opencv-python numpy pyyaml tqdm matplotlib scikit-image scipy pillow
```

**✅ Verify**: `python verify_env.py` should pass all checks

### Step 2️⃣: Prepare Data (1 min)

Create this folder structure and place **RGB X-ray images** inside:

```
datasets/Source_domain/KDXray/
└── train/
    └── images/          ← PUT YOUR IMAGES HERE
        ├── image001.png
        ├── image002.png
        └── ... (300+ images recommended)
```

**That's it!** Grayscale images are generated automatically during training.

### Step 3️⃣: Train (2 min setup, then let it run)

```bash
python train.py
```

**Live Output** (watch these numbers):
```
[Epoch 25/50] Loss_G: 0.8745 | Loss_D: 0.9123 | Gap: 0.0378
MAE↓: 0.0234 | FID↓: 18.45 | LPIPS↓: 0.0321 | Color-KL↓: 0.0156
✓ Best MAE model saved: checkpoints/gen_best_mae.pth
```

**Expected Time**: ~3 minutes per epoch (150 min total for 50 epochs)

---

## 🎯 Zero-Shot Inference (2 min)

Once training completes, translate target domain images **without retraining**:

```bash
python inference.py \
  --input datasets/Target_domain/CLC_extract/images \
  --output datasets/Target_domain/CLC_extract_ZSXT/images \
  --checkpoint checkpoints/gen_best.pth
```

**Output Example**:
```
=====================================================
🔄 ZSXT Zero-Shot Inference
=====================================================
Input:  datasets/Target_domain/CLC_extract/images
Output: datasets/Target_domain/CLC_extract_ZSXT/images
Count:  542 images

Progress: 100%|████████████████| 542/542 [00:23<00:00, 23.1 fps]

✅ Inference complete: 542 images
✅ Labels folder copied
=====================================================
```

**That's it!** Images are translated and ready for detection.

---

## 📊 Expected Results

After training (50 epochs), you should see:

### Training Metrics
- **Loss_G**: ~0.85 (stable)
- **Loss_D**: ~0.90 (balanced with Loss_G)
- **MAE**: <0.03 (pixel accuracy)
- **FID**: <20 (generation quality)

### Detection Performance (with YOLOv10n)
- **Source-Only**: 28.2% mAP
- **ZSXT**: 49.2% mAP (**+74.5% improvement**)

### Visual Quality
- Preserved microstructure details
- Accurate color reconstruction
- No blurring artifacts

---

## ⚙️ Common Configurations

### Low-End GPU (8GB VRAM)

```yaml
# config.yaml
batch_size: 2
img_width: 256
img_height: 256
num_augments: 2
```

### High-End GPU (24GB VRAM)

```yaml
# config.yaml
batch_size: 16
img_width: 640
img_height: 640
num_augments: 4
```

### Fast Prototyping (10 epochs)

```yaml
# config.yaml
num_epochs: 10
save_interval: 2
sample_interval: 2
```

---

## 🐛 Common Issues & Solutions

### Issue 1: Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Fix**:
```yaml
# config.yaml - Reduce batch size
batch_size: 1
img_width: 256
img_height: 256
```

### Issue 2: Training Not Starting

**Check**:
1. Is `images/` folder populated?
2. Are images in valid format (.png/.jpg)?
3. Run: `ls datasets/Source_domain/KDXray/train/images/ | wc -l`

### Issue 3: Poor Visual Quality

**Solution**: Adjust loss weights
```yaml
# config.yaml - Increase perceptual weight
loss_weights:
  l1: [0.5, 5.0, 50]
  gan: [1.0, 1.0, 0.5]
  perceptual: [5.0, 4.0, 2.0]  # Increased from [3.0, 2.0, 1.0]
  color: [20.0, 30.0, 30.0]
```

---

## 📈 Monitoring Training

### Real-Time Console Output

```
[Epoch 25/50] Loss_G: 0.8745 | Loss_D: 0.9123 | Gap: 0.0378
  Weights:
   L1=10.00 GAN=0.20 Perc=4.00 Color=0.30
  Ratios: L1=72.3% Perc=15.8% GAN=8.1% Color=3.8%

[评估中...] 计算MAE↓ FID↓ LPIPS↓ Color-KL↓
  MAE↓:       0.023400
  FID↓:       18.4523
  LPIPS↓:     0.032100
  Color-KL↓:  0.015600
  ✓ 最佳MAE模型已保存: checkpoints/gen_best_mae.pth
```

### Visualization Files

Check `generated_images/` folder:
- `epoch_N_samples.png`: Visual comparison (Input | Generated | Ground Truth)
- `training_curves.png`: Loss curves over epochs
- `evaluation_metrics_curves.png`: Metric trends

---

## 🎓 Next Steps

### 1. Fine-Tune Hyperparameters

Experiment with:
- `num_augments`: 1-5 (data augmentation multiplier)
- `learning_rate`: 0.0001-0.0003
- `loss_weights`: Adjust [early, mid, late] ratios

### 2. Evaluate on Detection Task

```bash
# Use translated images with YOLO detector
cd ultralytics
python detect.py \
  --weights yolov10n.pt \
  --source ../datasets/Target_domain/CLC_extract_ZSXT/images
```

### 3. Export Model for Production

```python
# export_onnx.py
import torch
from src.model import GeneratorUNet

model = GeneratorUNet().eval()
model.load_state_dict(torch.load('checkpoints/gen_best.pth'))

dummy_input = torch.randn(1, 1, 640, 640)
torch.onnx.export(model, dummy_input, 'zsxt_model.onnx')
```

---

## 📚 Resources

- **Full Documentation**: [README.md](README.md)
- **Paper**: [Coming Soon - IEEE Conference 2025]
- **Issues**: [GitHub Issues](https://github.com/Zang-AO/zero--shot-image-translation-framework/issues)
- **Dataset**: [PDSXray on Figshare](https://figshare.com/s/70c31a8d9c7d0f0f8fc5)

---

## ✅ Checklist

Before starting:
- [ ] Environment installed (`conda activate zsxt`)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset placed in `datasets/Source_domain/.../train/images/`
- [ ] Config file reviewed (`config.yaml`)
- [ ] GPU available (`nvidia-smi`)

After training:
- [ ] Training completed successfully (50 epochs)
- [ ] Checkpoints saved in `checkpoints/`
- [ ] Sample images generated in `generated_images/`
- [ ] MAE < 0.03 and FID < 20
- [ ] Inference script tested on target domain

---

**Happy Training! 🚀**

For detailed documentation, see [README.md](README.md).
