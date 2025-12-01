# 🔧 ZSXT Environment Setup Guide

Complete step-by-step instructions for setting up your development environment.

---

## 📋 System Requirements

### 🖥️ Hardware Specifications

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| **GPU** | GTX 1660 (6GB) | RTX 3090 (24GB) | NVIDIA only |
| **CPU** | 4 cores | 8+ cores | Intel/AMD |
| **RAM** | 16GB | 32GB+ | System memory |
| **Storage** | 50GB free | 100GB+ SSD | For models + datasets |
| **OS** | Ubuntu 18.04+ | Ubuntu 20.04+ | Or Windows 10+ |

### 📦 Software Stack

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.8 - 3.11 | Runtime |
| CUDA | 11.0+ | GPU support |
| cuDNN | 8.0+ | Deep learning ops |
| Git | 2.0+ | Version control |

---

## ⚡ Quick Start (5 minutes)

**Choose your installation method**:

## ⚡ Quick Start (5 minutes)

**Choose your installation method**:

```
┌─────────────────────────────────────────────────────┐
│  Installation Method Selector                       │
├─────────────────────────────────────────────────────┤
│                                                     │
│  1. 🐍 Conda (Recommended)                         │
│     • Best for beginners                            │
│     • Automatic CUDA setup                          │
│     • Easy environment management                   │
│                                                     │
│  2. 📦 Pip (Fast)                                  │
│     • Lightweight                                   │
│     • Requires manual CUDA setup                    │
│                                                     │
│  3. 🐳 Docker (Isolated)                           │
│     • Pre-configured environment                    │
│     • No local setup needed                         │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 🐍 Method 1: Conda Installation (Recommended)

#### Step 1: Install Miniconda

**Linux/macOS**:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

**Windows**:
1. Download from: https://docs.conda.io/en/latest/miniconda.html
2. Run installer: `Miniconda3-latest-Windows-x86_64.exe`
3. Add to PATH when prompted

#### Step 2: Create Environment

```bash
# Create new environment
conda create -n zsxt python=3.9 -y

# Activate environment
conda activate zsxt

# Verify Python version
python --version  # Should show: Python 3.9.x
```

#### Step 3: Install PyTorch with CUDA

**CUDA 11.8 (Recommended)**:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

**CUDA 12.1**:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

**CPU Only (No GPU)**:
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

#### Step 4: Verify PyTorch Installation

```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"
```

Expected output:
```
PyTorch: 2.1.0
CUDA Available: True
CUDA Version: 11.8
```

#### Step 5: Install Other Dependencies

```bash
# Install from requirements.txt
pip install opencv-python numpy pyyaml tqdm matplotlib scikit-image scipy pillow

# Or install individually
pip install opencv-python>=4.6.0
pip install numpy>=1.21.0
pip install pyyaml>=6.0
pip install tqdm>=4.64.0
pip install matplotlib>=3.5.0
pip install scikit-image>=0.19.0
pip install scipy>=1.7.0
pip install pillow>=9.0.0
```

---

## 📦 Method 2: Pip + Virtual Environment

**Best for**: Lightweight setups, existing PyTorch installations

#### Step 1️⃣: Create Virtual Environment

```bash
# Create Python 3.9 environment
python3.9 -m venv zsxt_env

# Activate
# 🐧 Linux/macOS:
source zsxt_env/bin/activate

# 🪟 Windows:
zsxt_env\Scripts\activate
```

#### Step 2️⃣: Upgrade Pip

```bash
pip install --upgrade pip setuptools wheel
```

#### Step 3️⃣: Install PyTorch

Go to **https://pytorch.org/get-started/locally/** and copy the pip command for your setup:

```bash
# Example for CUDA 11.8 on Linux
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Step 4️⃣: Install ZSXT Dependencies

```bash
pip install -r requirements.txt
```

#### Verify Installation

```bash
python verify_env.py
```

---

## 🐳 Method 3: Docker (Containerized)

**Best for**: Reproducible environments, CI/CD pipelines

#### Create Dockerfile

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Project code
COPY . .

CMD ["python", "train.py"]
```

#### Build and Run

```bash
# Build image
docker build -t zsxt:latest .

# Run training
docker run --gpus all -v $(pwd)/datasets:/workspace/datasets zsxt:latest

# Interactive mode
docker run --gpus all -it -v $(pwd):/workspace zsxt:latest /bin/bash
```

---

## ✅ Environment Verification

### Automated Verification (Recommended)

**Run verification script**:

```bash
python verify_env.py
```

**Expected Output**:
```
============================================================
✅ ZSXT Environment Verification
============================================================

✓ Python: 3.9.0
✓ PyTorch: 2.1.0+cu118
✓ CUDA Available: True
✓ CUDA Version: 11.8
✓ GPU Count: 1
  - GPU 0: NVIDIA A100-PCIE-40GB
✓ GPU Memory: 40.0 GB
✓ cv2: installed
✓ numpy: installed
✓ yaml: installed
✓ tqdm: installed
✓ matplotlib: installed
✓ scipy: installed
✓ PIL: installed

============================================================
✅ All checks passed! Environment is ready.
============================================================
```

### Manual Verification Commands

**Check Python**:
```bash
python --version  # Should show 3.8+ 
```

**Check PyTorch**:
```bash
python -c "import torch; print(torch.__version__); print(f'CUDA: {torch.cuda.is_available()}')"
```

**Check GPU**:
```bash
nvidia-smi  # Lists all NVIDIA GPUs
```

**Check Dependencies**:
```bash
python -c "import cv2, yaml, numpy; print('✓ All imports OK')"
```

if __name__ == '__main__':
    main()
```

Run verification:
```bash
python verify_env.py
```

Expected output:
```
============================================================
ZSXT Environment Verification
============================================================

✓ Python: 3.9.18
✓ PyTorch: 2.1.0
✓ CUDA Available: True
✓ CUDA Version: 11.8
✓ GPU Count: 1
  - GPU 0: NVIDIA GeForce RTX 3090
✓ cv2: installed
✓ numpy: installed
✓ yaml: installed
✓ tqdm: installed
✓ matplotlib: installed
✓ scipy: installed
✓ PIL: installed
✓ GPU Memory: 24.0 GB

============================================================
✅ All checks passed! Environment is ready.
============================================================
```

---

## 🐛 Troubleshooting

### Issue 1: CUDA Not Detected

**Symptom**:
```python
torch.cuda.is_available()  # Returns False
```

**Solutions**:

1. **Check NVIDIA Driver**:
```bash
nvidia-smi
```
If command not found, install driver:
```bash
# Ubuntu
sudo apt install nvidia-driver-535

# Check after reboot
nvidia-smi
```

2. **Reinstall PyTorch with Correct CUDA**:
```bash
# Uninstall existing PyTorch
pip uninstall torch torchvision torchaudio

# Reinstall with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. **Check CUDA Toolkit Version**:
```bash
nvcc --version  # Should match PyTorch CUDA version
```

---

### Issue 2: Import Errors

**Error**: `ModuleNotFoundError: No module named 'cv2'`

**Solution**:
```bash
# Uninstall conflicting packages
pip uninstall opencv-python opencv-python-headless opencv-contrib-python

# Reinstall
pip install opencv-python==4.8.0
```

---

### Issue 3: Out of Memory During Installation

**Error**: `Killed` during `pip install`

**Solution**:
```bash
# Increase swap space (Linux)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Install with --no-cache-dir
pip install --no-cache-dir -r requirements.txt
```

---

### Issue 4: Conda Environment Activation Fails

**Error**: `conda: command not found`

**Solution**:
```bash
# Add conda to PATH
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Or use full path
~/miniconda3/bin/conda activate zsxt
```

---

## 📦 Package Version Compatibility Matrix

| PyTorch | CUDA | cuDNN | Python | torchvision |
|---------|------|-------|--------|-------------|
| 2.1.0 | 11.8 | 8.7 | 3.8-3.11 | 0.16.0 |
| 2.0.1 | 11.8 | 8.7 | 3.8-3.11 | 0.15.2 |
| 1.13.1 | 11.7 | 8.5 | 3.7-3.10 | 0.14.1 |
| 1.12.1 | 11.6 | 8.3 | 3.7-3.10 | 0.13.1 |

**Recommendation**: Use **PyTorch 2.1.0 + CUDA 11.8** for best stability.

---

## 🔄 Updating Environment

### Update All Packages

```bash
# Activate environment
conda activate zsxt

# Update conda packages
conda update --all -y

# Update pip packages
pip list --outdated
pip install --upgrade <package_name>
```

### Update PyTorch

```bash
# Check current version
python -c "import torch; print(torch.__version__)"

# Update to latest
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia --force-reinstall
```

---

## 🚀 Performance Optimization

### Enable TensorFloat-32 (TF32)

Add to `train.py`:
```python
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

**Speedup**: ~1.5x faster on Ampere GPUs (RTX 30XX, A100)

### Enable cuDNN Benchmark

```python
torch.backends.cudnn.benchmark = True
```

**Speedup**: ~10-20% for fixed input sizes

### Use Mixed Precision (FP16)

See [Advanced Usage](README.md#advanced-usage) in main README.

---

## 📚 Additional Resources

- **PyTorch Installation**: https://pytorch.org/get-started/locally/
- **CUDA Toolkit**: https://developer.nvidia.com/cuda-toolkit
- **cuDNN**: https://developer.nvidia.com/cudnn
- **Conda Docs**: https://docs.conda.io/

---

## ✉️ Support

If you encounter issues not covered here:

1. Check [Troubleshooting](#troubleshooting) section
2. Run `python verify_env.py` and share output
3. Open an issue: https://github.com/Zang-AO/zero--shot-image-translation-framework/issues

---

**Last Updated**: 2025-01-XX
