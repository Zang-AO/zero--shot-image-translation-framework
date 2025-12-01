@echo off
REM ====================================================
REM ZSXT Web UI - Quick Start Script for Windows
REM ====================================================

echo.
echo ╔════════════════════════════════════════════════╗
echo ║  🎨 ZSXT Web UI - Starting Interface           ║
echo ╚════════════════════════════════════════════════╝
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.8+
    pause
    exit /b 1
)

echo ✅ Python found

REM Check if streamlit is installed
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Streamlit not installed
    echo Installing Streamlit...
    pip install streamlit>=1.28.0
    if errorlevel 1 (
        echo ❌ Failed to install Streamlit
        pause
        exit /b 1
    )
)

echo ✅ Streamlit available

REM Check model checkpoint
if not exist "checkpoints\gen_best.pth" (
    echo ⚠️  Warning: checkpoints\gen_best.pth not found
    echo You can configure model path in the UI sidebar
)

echo.
echo ╔════════════════════════════════════════════════╗
echo ║  🚀 Starting Web Interface                     ║
echo ║  Browser: http://localhost:8501               ║
echo ╚════════════════════════════════════════════════╝
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start Streamlit
python -m streamlit run app.py

pause
