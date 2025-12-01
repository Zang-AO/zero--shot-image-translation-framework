#!/usr/bin/env python3
"""
ZSXT Environment Verification Script
Checks all dependencies and system requirements
"""

import sys
import subprocess

def print_section(title):
    print(f"\n{'='*60}")
    print(f"{title}")
    print('='*60)

def check_python():
    """Check Python version"""
    version = sys.version_info
    print(f"✓ Python: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3:
        print("✗ Python 3.x required")
        return False
    if version.minor < 8:
        print("✗ Python 3.8+ required")
        return False
    
    return True

def check_pytorch():
    """Check PyTorch and CUDA availability"""
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        
        # Check CUDA
        cuda_available = torch.cuda.is_available()
        print(f"{'✓' if cuda_available else '⚠'} CUDA Available: {cuda_available}")
        
        if cuda_available:
            print(f"✓ CUDA Version: {torch.version.cuda}")
            print(f"✓ cuDNN Version: {torch.backends.cudnn.version()}")
            
            # GPU info
            gpu_count = torch.cuda.device_count()
            print(f"✓ GPU Count: {gpu_count}")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  - GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")
                
                if gpu_mem < 6:
                    print(f"    ⚠ Warning: {gpu_mem:.1f}GB VRAM may cause OOM errors (recommend 8GB+)")
        else:
            print("⚠ CUDA not available - training will be very slow on CPU")
            print("  Recommendation: Install CUDA-enabled PyTorch")
        
        return True
    except ImportError:
        print("✗ PyTorch not installed")
        print("  Install: pip install torch torchvision")
        return False
    except Exception as e:
        print(f"✗ Error checking PyTorch: {e}")
        return False

def check_dependencies():
    """Check all required packages"""
    packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'yaml': 'pyyaml',
        'tqdm': 'tqdm',
        'matplotlib': 'matplotlib',
        'scipy': 'scipy',
        'PIL': 'pillow',
        'skimage': 'scikit-image'
    }
    
    all_installed = True
    missing = []
    
    for import_name, package_name in packages.items():
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {package_name}: {version}")
        except ImportError:
            print(f"✗ {package_name}: NOT INSTALLED")
            missing.append(package_name)
            all_installed = False
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
    
    return all_installed

def check_gpu_capabilities():
    """Check GPU compute capabilities"""
    try:
        import torch
        if not torch.cuda.is_available():
            return True
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            compute_cap = f"{props.major}.{props.minor}"
            print(f"✓ GPU {i} Compute Capability: {compute_cap}")
            
            # Check for tensor cores (compute capability >= 7.0)
            if props.major >= 7:
                print(f"  ✓ Tensor Cores available (FP16 training supported)")
            else:
                print(f"  ⚠ No Tensor Cores (FP16 training may not work)")
        
        return True
    except Exception as e:
        print(f"⚠ Could not check GPU capabilities: {e}")
        return True

def test_basic_operations():
    """Test basic PyTorch operations"""
    try:
        import torch
        
        # CPU tensor
        x = torch.randn(10, 10)
        y = torch.matmul(x, x.t())
        print("✓ CPU tensor operations: OK")
        
        # GPU tensor (if available)
        if torch.cuda.is_available():
            x_gpu = torch.randn(10, 10, device='cuda')
            y_gpu = torch.matmul(x_gpu, x_gpu.t())
            print("✓ GPU tensor operations: OK")
            
            # Check memory allocation
            torch.cuda.synchronize()
            mem_allocated = torch.cuda.memory_allocated(0) / 1024**2
            print(f"✓ GPU memory allocated: {mem_allocated:.2f} MB")
        
        return True
    except Exception as e:
        print(f"✗ Tensor operations failed: {e}")
        return False

def check_disk_space():
    """Check available disk space"""
    try:
        import shutil
        import os
        
        # Check current directory
        cwd = os.getcwd()
        stat = shutil.disk_usage(cwd)
        
        free_gb = stat.free / 1024**3
        print(f"✓ Free disk space: {free_gb:.1f} GB")
        
        if free_gb < 20:
            print("  ⚠ Warning: <20GB free space (recommend 50GB+ for datasets)")
            return False
        
        return True
    except Exception as e:
        print(f"⚠ Could not check disk space: {e}")
        return True

def check_file_permissions():
    """Check write permissions"""
    try:
        import os
        import tempfile
        
        # Try to create a test file
        with tempfile.NamedTemporaryFile(mode='w', delete=True) as f:
            f.write('test')
        
        print("✓ Write permissions: OK")
        return True
    except Exception as e:
        print(f"✗ Write permissions failed: {e}")
        return False

def main():
    print_section("ZSXT Environment Verification")
    
    results = {
        'Python': check_python(),
        'PyTorch': check_pytorch(),
        'Dependencies': check_dependencies(),
        'GPU Capabilities': check_gpu_capabilities(),
        'Basic Operations': test_basic_operations(),
        'Disk Space': check_disk_space(),
        'File Permissions': check_file_permissions()
    }
    
    print_section("Summary")
    
    all_passed = True
    for component, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {component}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ All checks passed! Environment is ready for ZSXT.")
        print("\nNext steps:")
        print("1. Prepare dataset in datasets/Source_domain/.../train/images/")
        print("2. Review config.yaml")
        print("3. Run: python train.py")
    else:
        print("⚠ Some checks failed. Please fix the issues above.")
        print("\nCommon solutions:")
        print("- Missing packages: pip install -r requirements.txt")
        print("- No CUDA: Install CUDA toolkit and PyTorch with CUDA")
        print("- Low disk space: Free up space or use external drive")
    print("="*60)
    
    sys.exit(0 if all_passed else 1)

if __name__ == '__main__':
    main()
