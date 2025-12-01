# ZSXT: Zero-Shot X-ray Style Translation

Official PyTorch implementation of **"Zero-Shot Pseudocolor X-ray Domain Translation for Cross-Device Industrial Collaboration"**

[![Paper](https://img.shields.io/badge/Paper-IEEE-blue)](https://github.com/Zang-AO/zero--shot-image-translation-framework)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📖 Overview

ZSXT is a **zero-shot image translation framework** designed for cross-device X-ray inspection, enabling domain adaptation **without requiring target domain samples**. Unlike conventional methods (CycleGAN, CUT, UVCGAN) that demand dual-domain training data, ZSXT achieves competitive performance using **only source domain data**.

### 🎯 Key Features at a Glance

| Feature | Description |
|---------|-------------|
| **Zero-Shot** | No target domain data needed |
| **Lightweight** | Only 37.7M parameters |
| **Fast** | <50ms inference per image |
| **Accurate** | +74.5% mAP over baseline |
| **Smart** | Multi-modal parameter space coverage |

---

## 🏗️ Architecture at a Glance

```
Stage 1: Decolorization  →  Stage 2: Augmentation  →  Stage 3: Loss Functions
   (Unified Grayscale)       (Parameter Coverage)      (Quality Metrics)
```

### Three Core Stages

**1. Decolorization** — Remove vendor pseudocoloring  
   • Uses ITU-R BT.601 standard: `Gray = 0.299R + 0.587G + 0.114B`

**2. Multi-Modal Augmentation** — Synthetic device variation  
   • Poisson noise, brightness, ripple, artifacts, lens flare

**3. Four-Component Loss** — Quality metrics  
   • L1 (70%): Pixel accuracy | GAN (10%): Realism | Perceptual (15%): Features | Color (5%): Distribution

### Network Details

**Generator**: 8-layer UNet (34.9M)  
**Discriminator**: PatchGAN (2.77M)  
**Total**: 37.7M parameters | **Training**: 50 epochs with dynamic scheduling

---

## 📦 Installation

### System Requirements (60 seconds check)

```
✅ Python 3.8+  |  ✅ CUDA 11.0+  |  ✅ 8GB+ GPU  |  ✅ 50GB disk space
```

### Installation Steps

**Step 1: Clone Repository**
```bash
git clone https://github.com/Zang-AO/zero--shot-image-translation-framework.git
cd zero--shot-image-translation-framework
```

**Step 2: Create Environment**
```bash
conda create -n zsxt python=3.9 -y
conda activate zsxt
```

**Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 4: Verify Setup**
```bash
python verify_env.py  # Should show ✅ All checks passed!
```

✅ **Ready to go!** → See [Quick Start](QUICKSTART.md) for next steps.

---

## 🎨 Web UI Interface

### Launch the Interactive Web Interface

The project includes a **professional Streamlit-based web UI** for easier interaction without command-line knowledge:

```bash
# Method 1: Python launcher (recommended)
python run_ui.py

# Method 2: Direct Streamlit command
streamlit run app.py

# Method 3: Platform-specific launchers
# Windows: double-click start_ui.bat
# Linux/Mac: bash start_ui.sh
```

### Features

- **📸 Single Image Processing**: Upload, preview, and download translations
- **📁 Batch Processing**: Process entire folders with progress tracking and metrics
- **⚙️ Real-time Configuration**: Select device (CPU/GPU), load custom models, adjust settings
- **📊 Performance Metrics**: View inference time, image dimensions, success rates
- **💾 One-Click Download**: Export results directly from the browser
- **📈 System Information**: Monitor GPU status and memory usage

### Quick Start with UI

1. **Install dependencies** (if not done):
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the UI**:
   ```bash
   python run_ui.py
   ```
   
3. **Browser opens automatically** to `http://localhost:8501`

4. **Select model** and start processing:
   - Choose GPU/CPU from sidebar
   - Click on tabs to explore features
   - Upload images or batch folders
   - Download results with one click

**📖 See [UI_GUIDE.md](UI_GUIDE.md) for detailed UI documentation, advanced features, and troubleshooting.**

---

## 🗂️ Project Structure

```
_code_EN/
├── config.yaml                 # Configuration file
├── train.py                    # Training script
├── inference.py                # Inference script
├── README.md                   # Documentation
├── requirements.txt            # Python dependencies
│
├── src/                        # Core modules
│   ├── model.py                # UNet Generator + PatchGAN Discriminator
│   ├── losses.py               # Four-component dynamic loss
│   ├── preprocess_pipeline.py  # Automated preprocessing pipeline
│   ├── super_resolution.py     # GPU-accelerated SR module
│   └── img_process_train.py    # Augmentation utilities
│
├── datasets/                   # Dataset directory
│   ├── Source_domain/          # Source domain training data
│   │   └── KDXray/
│   │       └── train/
│   │           ├── images/         # Original RGB images
│   │           └── images_gray/    # Auto-generated grayscale images
│   └── Target_domain/          # Target domain test data
│       └── CLC_extract/
│           └── images/
│
├── checkpoints/                # Saved model weights
│   ├── gen_best.pth            # Best generator (overall loss)
│   ├── gen_best_mae.pth        # Best generator (MAE metric)
│   └── gen_final.pth           # Final epoch checkpoint
│
└── generated_images/           # Training visualization
    ├── epoch_50_samples.png        # Sample outputs
    ├── training_curves.png         # Loss curves
    └── evaluation_metrics_curves.png  # Metric curves
```

---

## ⚙️ Configuration

All hyperparameters are managed through `config.yaml`:

```yaml
# Basic Configuration
batch_size: 3                # Batch size (adjust based on GPU memory)
num_epochs: 50               # Training epochs
learning_rate: 0.0002        # Adam learning rate
img_width: 256               # Training image width
img_height: 256              # Training image height

# Data Paths (Source Domain Training)
data_root: "datasets/Source_domain/KDXray"
images_folder: "train/images"          # RGB images folder
images_gray_folder: "train/images_gray"  # Grayscale images folder (auto-generated)

# Pretrained Weights (Optional)
pretrained_gen: null         # Generator checkpoint path (null = random initialization)
pretrained_disc: null        # Discriminator checkpoint path

# Preprocessing
num_augments: 3              # One-to-many augmentation multiplier
sr_enabled: true             # Enable super-resolution
sr_scale_factor: 2           # SR scale factor
regenerate_gray: false       # Force regenerate grayscale data

# Augmentation Parameters
augmentation_params:
  noise_level: 0.05                      # Poisson noise intensity
  brightness_factor_range: [0.8, 1.2]    # Brightness adjustment range
  amplitude: 5                           # Ripple amplitude
  wavelength: 20                         # Ripple wavelength
  artifact_intensity: 0.5                # Metal artifact intensity
  flare_intensity: 0.5                   # Lens flare intensity

# Loss Function Weights (Three-Stage Dynamic Scheduling)
# Format: [early, mid, late] for epoch-dependent weight transitions
loss_weights:
  l1: [0.5, 5.0, 50]         # L1 Loss: Early-low → Mid-moderate → Late-dominant
  gan: [1.0, 1.0, 0.5]       # GAN Loss: Mid-balanced → Late-reduced
  perceptual: [3.0, 2.0, 1.0]  # Perceptual Loss: Early-dominant → Late-reduced
  color: [20.0, 30.0, 30.0]    # Color Loss: Mid-enhanced → Late-maintained

# Training Configuration
gradient_clip: 1.0           # Gradient clipping threshold
save_interval: 1             # Checkpoint save interval (epochs)
sample_interval: 1           # Sample visualization interval (epochs)

# Inference Configuration
inference:
  sr_enabled: true           # Enable super-resolution
  sr_scale_factor: 2         # SR scale factor
  img_width: 960             # Inference target width
  img_height: 960            # Inference target height
  decolor_only: true         # Decolorize only (no noise augmentation)
  batch_size: 1              # Inference batch size
```

### Key Configuration Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `batch_size` | Training batch size | 3 | Adjust based on GPU memory (8GB → 3; 16GB → 8; 24GB → 16) |
| `num_augments` | Augmentation multiplier | 3 | Generates 3 variants per source image |
| `sr_enabled` | Super-resolution toggle | true | Upscales input images before processing |
| `loss_weights` | Dynamic loss scheduling | [early, mid, late] | Three-stage weight transitions |
| `regenerate_gray` | Force regenerate grayscale | false | Set `true` if changing augmentation params |

---

## 🚀 Usage

### 1️⃣ Data Preparation

**Organize your datasets** in this structure:

```
datasets/
├── Source_domain/
│   └── KDXray/                    # or EDS
│       └── train/
│           ├── images/            # ← Place RGB X-ray images here
│           │   ├── image001.png
│           │   ├── image002.png
│           │   └── ...
│           └── images_gray/       # ← Auto-generated (don't create)
│
└── Target_domain/
    └── CLC_extract/               # No fine-tuning needed for inference
        └── images/
            ├── image001.png
            └── ...
```

**Key Notes**:
- ✅ `images_gray/` is **auto-generated** during first training run
- ✅ Place your RGB images in `images/` folder
- ✅ Target domain data not needed for zero-shot inference
- ⚠️ Supported formats: `.png`, `.jpg`, `.jpeg`

### 2️⃣ Training Process

#### Option A: Quick Start (Default Settings)

```bash
python train.py
```
Uses default config (512×512, batch_size=32, 50 epochs). ⏱️ ~3-4 hours on V100.

#### Option B: Custom Configuration

**Step 1**: Modify `config.yaml` with your settings

```yaml
# High-resolution mode (requires more VRAM)
img_width: 640
img_height: 640
batch_size: 2        # ⚠️ Reduce for larger images

# Load pre-trained weights
pretrained_gen: "checkpoints/gen_best.pth"
```

**Step 2**: Start training
```bash
python train.py
```

#### Option C: Resume Training

**Step 1**: Edit `config.yaml` with checkpoint paths
```yaml
pretrained_gen: "checkpoints/gen_epoch_20.pth"
pretrained_disc: "checkpoints/disc_epoch_20.pth"
```

**Step 2**: Resume
```bash
python train.py  # Continues from epoch 21
```

### 3️⃣ Monitor Training Progress

**Real-Time Monitoring** (console output):

```
[Epoch 25/50] Loss_G: 0.8745 | Loss_D: 0.9123 | Gap: 0.0378
├─ Weights: L1=10.00 GAN=0.20 Perc=4.00 Color=0.30
└─ Ratios:  L1=72.3% Perc=15.8% GAN=8.1% Color=3.8%

[Evaluation] MAE↓: 0.0234 | FID↓: 18.45 | LPIPS↓: 0.0321 | Color-KL↓: 0.0156
```

**Generated Outputs**:

| Location | Content | Purpose |
|----------|---------|---------|
| `checkpoints/` | Model weights | Model selection and resuming |
| `generated_images/` | Visual results | Training progression visualization |
| Console | Loss/metric values | Real-time training monitoring |

**Key Checkpoints**:
- ✅ `gen_best.pth` → Best overall performance
- ✅ `gen_best_mae.pth` → Best pixel accuracy
- ✅ `gen_epoch_N.pth` → Periodic checkpoints (every 5 epochs)
- ✅ `gen_final.pth` → Final model (epoch 50)

### 4️⃣ Zero-Shot Inference

#### Quick Inference

Translate target domain images **without retraining**:

```bash
python inference.py \
  --input datasets/Target_domain/CLC_extract/images \
  --output datasets/Target_domain/CLC_extract_ZSXT/images \
  --checkpoint checkpoints/gen_best.pth
```

#### Advanced: Command-Line Options

```bash
python inference.py \
  --input <folder_path> \
  --output <folder_path> \
  --checkpoint <model_path> \
  --config config.yaml \
  --batch_size 16
```

| Parameter | Purpose | Example |
|-----------|---------|---------|
| `--input` | Source images folder | `datasets/Target_domain/CLC/images` |
| `--output` | Output folder | `datasets/Target_domain/CLC_ZSXT/images` |
| `--checkpoint` | Model weights | `checkpoints/gen_best.pth` |
| `--batch_size` | Inference batch size | `16` (default, adjust for VRAM) |

#### Expected Output

```
=====================================================
🔄 ZSXT Zero-Shot Inference
=====================================================
Input folder:  datasets/Target_domain/CLC_extract/images
Output folder: datasets/Target_domain/CLC_extract_ZSXT/images
Total images:  542

Progress: 100%|████████████████| 542/542 [00:23<00:00, 23.1 fps]

✅ Inference completed: 542 images
✅ Labels folder copied
=====================================================
```
| `--device` | Device (cuda/cpu) | `cuda` |
| `--no-copy-labels` | Skip copying labels folder | False |

#### Batch Inference Example

```bash
# Process multiple datasets
for DATASET in CLC_extract PID_extract; do
  python inference.py \
    --input datasets/Target_domain/$DATASET/images \
    --output datasets/Target_domain/${DATASET}_ZSXT/images \
    --checkpoint checkpoints/gen_best_mae.pth
done
```

---

## 📊 Evaluation Metrics

ZSXT tracks four core metrics aligned with loss components:

| Metric | Description | Target | Aligns With |
|--------|-------------|--------|-------------|
| **MAE ↓** | Mean Absolute Error (pixel accuracy) | <0.03 | L1 Loss |
| **FID ↓** | Fréchet Inception Distance (realism) | <20 | GAN Loss |
| **LPIPS ↓** | Learned Perceptual Image Patch Similarity | <0.05 | Perceptual Loss |
| **Color-KL ↓** | KL divergence of RGB histograms | <0.03 | Color Loss |

### Baseline Comparison

| Method | Target Data | Parameters | mAP@0.5 | MAE ↓ | FID ↓ |
|--------|-------------|------------|---------|-------|-------|
| **Source-Only** | - | - | 28.2% | - | - |
| CycleGAN | Required | 2×54.4M | 36.4% | 0.045 | 28.3 |
| CUT | Required | 56.2M | 25.7% | 0.052 | 32.1 |
| UVCGAN | Required | 89.6M | 38.7% | 0.041 | 25.6 |
| EnCo | Required | 62.0M | 26.0% | 0.048 | 30.4 |
| **ZSXT (Ours)** | **Not Required** | **37.7M** | **49.2%** | **0.025** | **15.3** |

**Key Findings**:
- **+74.5%** relative mAP improvement over source-only baseline
- **+10.5 pp** absolute mAP advantage over best dual-domain baseline (UVCGAN)
- **67%** parameter efficiency compared to CUT
- **38%** faster inference than CycleGAN

---

## 🔬 Advanced Usage

### Custom Loss Weight Scheduling

Modify `config.yaml` to customize three-stage weight transitions:

```yaml
# Conservative strategy (stability-focused)
loss_weights:
  l1: [1.0, 2.0, 5.0]        # Gradual L1 increase
  gan: [0.3, 0.5, 0.7]       # Gentle GAN enhancement
  perceptual: [2.0, 2.0, 2.0]  # Fixed perceptual weight
  color: [15.0, 20.0, 25.0]    # Progressive color constraint

# Aggressive strategy (performance-focused)
loss_weights:
  l1: [0.5, 10.0, 100]       # Strong late-stage L1 dominance
  gan: [2.0, 2.0, 1.0]       # Early adversarial training
  perceptual: [5.0, 3.0, 1.0]  # Front-loaded feature learning
  color: [30.0, 40.0, 50.0]    # Strong color enforcement
```

### Multi-GPU Training

```bash
# Use DataParallel (automatic)
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py
```

### Mixed Precision Training (FP16)

Add to `train.py`:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop:
with autocast():
    loss_G = ...
scaler.scale(loss_G).backward()
scaler.step(opt_G)
scaler.update()
```

---

## 🛠️ Troubleshooting Guide

### ❌ Out of Memory (OOM)

**Error Message**: `RuntimeError: CUDA out of memory. Tried to allocate`

**Root Cause**: GPU memory insufficient for batch size or image resolution

**✅ Quick Fixes** (edit `config.yaml`):

```yaml
# Solution 1: Reduce batch size (fastest)
batch_size: 2  # from 32

# Solution 2: Reduce image resolution
img_width: 320
img_height: 320
```

**GPU Memory Requirements**:
| Batch Size | Image Size | V100 (32GB) | RTX3090 (24GB) | RTX2080 (11GB) |
|-----------|-----------|-----------|-----------|-----------|
| 32 | 512×512 | ✅ | ❌ | ❌ |
| 16 | 512×512 | ✅ | ✅ | ❌ |
| 8 | 512×512 | ✅ | ✅ | ⚠️ |
| 2 | 512×512 | ✅ | ✅ | ✅ |

---

### ❌ Grayscale Folder Missing

**Symptom**: Training fails or images regenerate randomly each epoch

**Root Cause**: Preprocessing cache corrupted or not generated

**✅ Solution** (edit `config.yaml`):

```yaml
regenerate_gray: true
```

```bash
python train.py
```

---

### ❌ Discriminator Loss → Zero

**Symptom**: `Loss_D ≈ 0.0` while `Loss_G` increases

**Root Cause**: Generator too powerful, discriminator can't compete

**✅ Solution** - Rebalance in `config.yaml`:

```yaml
loss_weights:
  l1: [0.3, 3.0, 30]
  gan: [1.5, 2.0, 1.5]   # Increase this
```

---

### ❌ VGG19 Download Fails

**Error**: `[PerceptualLoss] ⚠️ Pretrained weights loading failed`

**✅ Solution**:

```bash
python -c "import torchvision; torchvision.models.vgg19(weights='VGG19_Weights.IMAGENET1K_V1')"
```

---

### ❌ CUDA Not Available

**Error**: `RuntimeError: CUDA is not available`

**✅ Check**:

```bash
nvidia-smi                                    # Verify driver
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{zhang2025zsxt,
  title={Zero-Shot Pseudocolor X-ray Domain Translation for Cross-Device Industrial Collaboration},
  author={Zhang, Xiaohao and Qiao, Jiansen and Wang, Xianyu and Zhang, Wenjie and Shi, Yinxue},
  booktitle={IEEE International Conference on [Conference Name]},
  year={2025}
}
```

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **PDSXray Dataset**: [Figshare Link](https://figshare.com/s/70c31a8d9c7d0f0f8fc5)
- **Baseline Methods**: CycleGAN, CUT, UVCGAN, EnCo
- **VGG19 Pretrained Weights**: torchvision official repository

---

## 📧 Contact

For questions or collaboration inquiries:

- **Primary Author**: Xiaohao Zhang
- **Corresponding Author**: Yinxue Shi (syx2821@cau.edu.cn)
- **GitHub Issues**: [Report bugs or feature requests](https://github.com/Zang-AO/zero--shot-image-translation-framework/issues)

---

## 🔄 Changelog

### v1.0.0 (2025-01-XX)
- ✅ Initial release
- ✅ Automated preprocessing pipeline
- ✅ Four-component dynamic loss with three-stage scheduling
- ✅ GPU-accelerated super-resolution
- ✅ Zero-shot inference script
- ✅ Comprehensive evaluation metrics (MAE, FID, LPIPS, Color-KL)

---

**Repository**: https://github.com/Zang-AO/zero--shot-image-translation-framework

**Paper**: [Coming Soon - IEEE Conference 2025]
