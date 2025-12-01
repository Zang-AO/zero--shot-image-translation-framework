# 📚 ZSXT Project Overview

Complete reference guide for project structure and organization.

---

## 📁 Project Structure

```
_code_EN/
├── 📋 Documentation
│   ├── README.md                    ← START: Main guide
│   ├── QUICKSTART.md               ← 5-minute setup
│   ├── ENVIRONMENT_SETUP.md        ← Detailed installation
│   └── PROJECT_OVERVIEW.md         ← This file
│
├── ⚙️ Configuration & Tools
│   ├── config.yaml                 ← Hyperparameters
│   ├── requirements.txt            ← Dependencies
│   └── verify_env.py               ← Verification script
│
├── 🚀 Scripts
│   ├── train.py                    ← Training (source domain)
│   └── inference.py                ← Zero-shot inference
│
├── 🔧 Core Modules (src/)
│   ├── model.py                    ← Networks (UNet + PatchGAN)
│   ├── losses.py                   ← Loss functions (dynamic)
│   ├── preprocess_pipeline.py      ← Data processing
│   ├── super_resolution.py         ← SR post-processing
│   └── img_process_train.py        ← Augmentation utils
│
├── 📊 Data (datasets/)
│   ├── Source_domain/KDXray/       ← Training images (RGB)
│   └── Target_domain/CLC/          ← Inference images
│
└── 💾 Outputs
    ├── checkpoints/                ← Model weights
    └── generated_images/           ← Visualizations
```

---

## 🚀 Quick Navigation

### 👤 For New Users

| Goal | Read | Time |
|------|------|------|
| **Get started now** | [QUICKSTART.md](QUICKSTART.md) | 5 min |
| **Setup environment** | [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) | 15 min |
| **Learn everything** | [README.md](README.md) | 30 min |
| **Understand architecture** | [PROJECT_OVERVIEW.md](#architecture-guide) | 10 min |

### 👨‍💻 For Developers

| Topic | File | Lines |
|-------|------|-------|
| Generator & Discriminator | `src/model.py` | 85-150 |
| Loss Function Design | `src/losses.py` | 25-120 |
| Data Processing | `src/preprocess_pipeline.py` | All |
| Augmentation Strategy | `src/img_process_train.py` | 1-80 |

---

## 🔧 Configuration Files

### config.yaml

**Purpose**: Single source of truth for all hyperparameters

**Key Sections**:
```yaml
# Training
batch_size: 3
num_epochs: 50
learning_rate: 0.0002

# Data paths

---

## ⚙️ Configuration Reference

### config.yaml Structure

```yaml
# 📊 Dataset Configuration
dataset:
  source: 'KDXray'                # Source domain
  data_root: "datasets/Source_domain/KDXray"
  images_folder: "train/images"
  batch_size: 32

# 🎯 Loss Weights (3-stage dynamic)
loss_weights:
  l1: [0.5, 5.0, 50]              # Early → Late epochs
  gan: [1.0, 1.0, 0.5]            # Adversarial balance
  perceptual: [3.0, 2.0, 1.0]     # Feature similarity
  color: [20.0, 30.0, 30.0]       # Color preservation

# 📈 Training Hyperparameters
train:
  epochs: 50
  learning_rate: 0.0002
  batch_size: 32
  num_workers: 4

# 🖼️ Image Settings
image:
  img_width: 512
  img_height: 512
```

### Key Settings by Use Case

| Scenario | Setting | Value | Notes |
|----------|---------|-------|-------|
| **Fast training** | batch_size | 64 | Requires 40GB+ VRAM |
| **Limited VRAM** | batch_size | 2 | ~6GB VRAM required |
| **High resolution** | img_width/height | 640 | Slower but better quality |
| **Quick test** | epochs | 5 | For testing only |

---

## 🚀 Scripts Guide

### 🎓 train.py - Training Script

**Function**: Train ZSXT on source domain

**Key Features**:
- ✅ Automatic grayscale generation
- ✅ Dynamic 3-stage loss scheduling  
- ✅ Real-time metric evaluation
- ✅ Best model checkpoint selection

**Quick Start**:
```bash
python train.py                    # Uses config.yaml
```

**Example: Resume Training**:
```yaml
# In config.yaml
pretrained_gen: "checkpoints/gen_epoch_20.pth"
```

**Outputs**:
```
checkpoints/
├── gen_best.pth           ← Best overall model
├── gen_best_mae.pth       ← Best pixel accuracy
└── gen_epoch_N.pth        ← Periodic checkpoints

generated_images/
├── epoch_10_samples.png   ← Visual progression
├── training_curves.png    ← Loss curves
└── evaluation_metrics.png ← Metric plots
```

---

### 🎯 inference.py - Zero-Shot Inference

**Function**: Translate target domain images (no retraining)

**Command**:
```bash
python inference.py \
  --input path/to/images \
  --output path/to/output \
  --checkpoint checkpoints/gen_best.pth
```

| Parameter | Purpose | Example |
|-----------|---------|---------|
| `--input` | Source images | `datasets/Target/images` |
| `--output` | Output folder | `datasets/Output/images` |
| `--checkpoint` | Model path | `checkpoints/gen_best.pth` |

**Output Example**:
```
✅ Processing 542 images
Progress: 100%|████| 542/542 [00:23]
✅ Completed! Results saved.
```

---

### 🔍 verify_env.py - Environment Check

**Function**: Validate all dependencies and GPU setup

**Command**:
```bash
python verify_env.py
```

**Expected Output**:
```
✓ Python: 3.9.18
✓ PyTorch: 2.1.0
✓ CUDA: Available
✓ GPU: NVIDIA RTX 3090 (24GB)
✓ All dependencies: OK
✅ Ready for training!
```

---

## 🧩 Core Modules Overview

### src/model.py - Neural Networks

**Components**:

```
┌─────────────────────────────────────┐
│   GeneratorUNet                     │
│   (8-layer, 34.9M parameters)      │
│   Gray (1-ch) → RGB (3-ch)         │
└─────────────────────────────────────┘
              ↓
        Real Domain RGB
              ↓
┌─────────────────────────────────────┐
│   PatchGANDiscriminator             │
│   (5-layer, 2.77M parameters)       │
│   70×70 receptive field             │
└─────────────────────────────────────┘
```

**Architecture Table**:
| Component | Type | Params | Input | Output |
|-----------|------|--------|-------|--------|
| Generator | UNet | 34.9M | [B,1,H,W] | [B,3,H,W] |
| Discriminator | PatchGAN | 2.77M | [B,4,H,W] | [B,1,70,70] |
| **Total** | - | **37.7M** | - | - |

**Key Usage**:
```python
generator = GeneratorUNet(in_ch=1, out_ch=3)
discriminator = PatchGANDiscriminator(in_ch=4)

fake_rgb = generator(gray_image)           # [B,1,H,W]→[B,3,H,W]
disc_pred = discriminator(cat([gray, rgb])) # Judge authenticity
```

---

### src/losses.py - Loss Functions

**Four-Component Design**:

| Component | Weight | Purpose | Formula |
|-----------|--------|---------|---------|
| **L1 Loss** | 70% | Pixel accuracy | ∑\|fake - real\| |
| **GAN Loss** | 10% | Adversarial training | BCE |
| **Perceptual** | 15% | Feature matching | VGG19 distance |
| **Color Loss** | 5% | Histogram alignment | KL divergence |

**Dynamic Scheduling** (3 Stages):

```
Epoch Progress: 0% ─────────┬──────────┬──────────→ 100%
                    Stage 1 │ Stage 2  │ Stage 3

L1 Weight:        0.5 ─────→ 5.0 ────→ 50 (focuses on detail)
GAN Weight:       1.0 ─────→ 1.0 ────→ 0.5 (reduces collapse)
Perceptual:       3.0 ─────→ 2.0 ────→ 1.0
Color Loss:      20.0 ────→ 30.0 ────→ 30.0 (constant preservation)
```

**Usage Example**:
```python
loss_fn = CombinedLoss(total_epochs=50, weights=config.loss_weights)
loss_fn.set_epoch(15)  # Sets weights based on epoch

loss_g, detail_dict = loss_fn.forward_generator(
    fake=gen_output,
    real=real_image,
    disc_output=disc_pred
)
```

---

### src/preprocess_pipeline.py - Data Processing

**Pipeline Stages**:

```
RGB Image (640×640)
      ↓ [1] Super-Resolution (2×)
RGB Image (1280×1280)
      ↓ [2] Resize to target
RGB Image (512×512)
      ↓ [3] Decolorization (ITU-R BT.601)
Gray Image (512×512)
      ↓ [4] Multi-modal Augmentation (×3)
Augmented Gray Images (3 variants each)
```

**Augmentation Strategy**:

| Augmentation | Parameters | Purpose |
|--------------|-----------|---------|
| Poisson Noise | σ=0.05 | Quantum noise simulation |
| Brightness | 0.8-1.2× | Illumination variation |
| Ripple | 5px amplitude | Motion artifacts |
| Metal Artifacts | 0.5 intensity | Equipment artifacts |
| Lens Flare | 0.5 intensity | Optical artifacts |

**Code Example**:
```python
pipeline = PreprocessPipeline('config.yaml')
pipeline.check_and_generate(
    images_folder='datasets/train/images',
    gray_folder='datasets/train/images_gray'
)
# Generates 3 augmented versions per image
```

---

### src/super_resolution.py

**Methods**:
- **Bicubic interpolation** (GPU-accelerated, default)
- **Deep SR model** (optional, ESPCN-like)

**Usage**:
```python
sr = SuperResolution(scale_factor=2, device='cuda')

# NumPy interface
upsampled = sr.upsample_numpy(image_np)  # [H, W, 3] → [2H, 2W, 3]

# Tensor interface
upsampled_tensor = sr.upsample(image_tensor)  # [B, 3, H, W] → [B, 3, 2H, 2W]
```

---

## 📊 Expected Outputs

### Training Outputs

**Checkpoints** (`checkpoints/`):
```
gen_best.pth           # Best overall loss (recommended for general use)
gen_best_mae.pth       # Best MAE (recommended for pixel accuracy)
gen_final.pth          # Final epoch
gen_epoch_10.pth       # Periodic checkpoint (every save_interval)
```

**Visualizations** (`generated_images/`):
```
epoch_50_samples.png              # Input | Generated | Ground Truth
training_curves.png               # Loss curves (G, D, L1, GAN, Perc, Color)
evaluation_metrics_curves.png     # Metric curves (MAE, FID, LPIPS, Color-KL)
```

### Inference Outputs

**Translated Images**:
```
datasets/Target_domain/CLC_extract_ZSXT/
├── images/                  # Translated images (same resolution as config)
│   ├── image001.png
│   ├── image002.png
│   └── ...
└── labels/                  # Copied from input (if --no-copy-labels not set)
    ├── image001.txt
    └── ...
```

---

## 📈 Performance Benchmarks

### Training Speed (RTX 3090, batch_size=3, 640×640)

| Operation | Time | Throughput |
|-----------|------|------------|
| Single epoch | ~3 min | ~30 img/s |
| Full training (50 epochs) | ~2.5 hours | - |
| Preprocessing | ~10 min (9k images) | ~15 img/s |

### Inference Speed

| GPU | Resolution | Time/Image | Throughput |
|-----|------------|------------|------------|
| RTX 3090 | 640×640 | 42ms | 24 fps |
| RTX 3090 | 960×960 | 89ms | 11 fps |
| RTX 3060 | 640×640 | 68ms | 15 fps |
| CPU (i7-9700K) | 640×640 | 2.3s | 0.4 fps |

### GPU Memory Usage

| Batch Size | Resolution | Memory (Training) | Memory (Inference) |
|------------|------------|-------------------|---------------------|
| 3 | 640×640 | ~8GB | ~2GB |
| 8 | 640×640 | ~18GB | - |
| 16 | 640×640 | ~32GB (A100) | - |
| 1 | 960×960 | ~4GB | ~3GB |

---

## 🔍 Troubleshooting Quick Reference

### Training Issues

| Issue | Solution |
|-------|----------|
| OOM error | Reduce `batch_size` in config.yaml |
| Discriminator collapse | Increase `loss_weights.gan` |
| Slow training | Enable `torch.backends.cudnn.benchmark = True` |
| Poor visual quality | Increase `loss_weights.perceptual` |

### Inference Issues

| Issue | Solution |
|-------|----------|
| Output too dark/bright | Check input images are RGB (not BGR) |
| Blurry results | Ensure using `gen_best_mae.pth` checkpoint |
| Slow inference | Reduce `inference.img_width/height` |
| Labels not copied | Remove `--no-copy-labels` flag |

### Environment Issues

| Issue | Solution |
|-------|----------|
| CUDA not found | Reinstall PyTorch with CUDA |
| Import cv2 fails | `pip install opencv-python==4.8.0` |
| VGG19 download fails | Pre-download: `python -c "import torchvision; torchvision.models.vgg19(weights='VGG19_Weights.IMAGENET1K_V1')"` |

---

## 📚 Additional Resources

### Paper & Dataset
- **Paper**: [Coming Soon - IEEE Conference 2025]
- **Dataset**: [PDSXray on Figshare](https://figshare.com/s/70c31a8d9c7d0f0f8fc5)

### Code References
- **PyTorch**: https://pytorch.org/
- **OpenCV**: https://opencv.org/
- **Baseline Methods**: CycleGAN, CUT, UVCGAN, EnCo

### Community
- **GitHub Issues**: [Report bugs](https://github.com/Zang-AO/zero--shot-image-translation-framework/issues)
- **Email**: syx2821@cau.edu.cn (Corresponding Author)

---

## ✅ Checklist for New Users

### Before Training
- [ ] Environment verified (`python verify_env.py`)
- [ ] Dataset placed in `datasets/Source_domain/.../train/images/`
- [ ] Config reviewed (`config.yaml`)
- [ ] GPU available (`nvidia-smi`)

### During Training
- [ ] Monitor console logs (Loss_G, Loss_D, Gap)
- [ ] Check sample images (`generated_images/epoch_N_samples.png`)
- [ ] Track metrics (MAE↓, FID↓, LPIPS↓, Color-KL↓)

### After Training
- [ ] Best checkpoint saved (`checkpoints/gen_best_mae.pth`)
- [ ] Final metrics: MAE<0.03, FID<20
- [ ] Inference tested on target domain
- [ ] Detection accuracy evaluated (optional)

---

**Version**: 1.0.0  
**Last Updated**: 2025-01-XX  
**Maintainer**: Xiaohao Zhang (Corresponding: Yinxue Shi)
