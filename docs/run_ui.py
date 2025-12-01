"""
ZSXT Web UI - Quick Start & Setup Guide
Run this script to start the interactive web interface
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = {
        'streamlit': 'streamlit',
        'torch': 'torch',
        'PIL': 'Pillow',
        'cv2': 'opencv-python',
        'yaml': 'pyyaml',
    }
    
    missing_packages = []
    for module, package_name in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print(f"Install them with: pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def main():
    print("\n" + "="*70)
    print("🎨 ZSXT Web UI - Interactive Interface")
    print("="*70)
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    if not check_dependencies():
        print("\n❌ Please install missing packages first")
        sys.exit(1)
    
    # Check model checkpoint
    print("\n🔍 Checking for model checkpoint...")
    if not os.path.exists("checkpoints/gen_best.pth"):
        print("\n⚠️  Warning: Could not find checkpoints/gen_best.pth")
        print("   You can:")
        print("   1. Place model in checkpoints/gen_best.pth")
        print("   2. Configure path in the web UI sidebar")
    else:
        print("✅ Model checkpoint found!")
    
    # Check config
    print("\n🔍 Checking configuration...")
    if not os.path.exists("config.yaml"):
        print("⚠️  Warning: Could not find config.yaml")
    else:
        print("✅ Configuration file found!")
    
    # Start Streamlit app
    print("\n" + "="*70)
    print("🚀 Starting ZSXT Web UI...")
    print("="*70)
    print("\n📱 The web interface will open in your browser")
    print("   If not, navigate to: http://localhost:8501")
    print("\n💡 Tips:")
    print("   - Use the sidebar to configure model and settings")
    print("   - Upload images in the 'Single Image' tab")
    print("   - Process folders using 'Batch Processing' tab")
    print("   - Check 'Information' tab for project details")
    print("\n✋ Press Ctrl+C to stop the server\n")
    
    # Run Streamlit app
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--logger.level=info",
        "--client.showErrorDetails=true"
    ])

if __name__ == "__main__":
    main()
