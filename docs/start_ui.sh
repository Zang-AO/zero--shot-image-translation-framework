#!/bin/bash

# ====================================================
# ZSXT Web UI - Quick Start Script for Unix/Linux/Mac
# ====================================================

echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║  🎨 ZSXT Web UI - Starting Interface           ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python not found! Please install Python 3.8+"
        exit 1
    fi
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

echo "✅ Python found"

# Check if streamlit is installed
$PYTHON_CMD -c "import streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Streamlit not installed"
    echo "Installing Streamlit..."
    $PYTHON_CMD -m pip install streamlit>=1.28.0
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install Streamlit"
        exit 1
    fi
fi

echo "✅ Streamlit available"

# Check model checkpoint
if [ ! -f "checkpoints/gen_best.pth" ]; then
    echo "⚠️  Warning: checkpoints/gen_best.pth not found"
    echo "You can configure model path in the UI sidebar"
fi

echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║  🚀 Starting Web Interface                     ║"
echo "║  Browser: http://localhost:8501               ║"
echo "╚════════════════════════════════════════════════╝"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start Streamlit
$PYTHON_CMD -m streamlit run app.py
