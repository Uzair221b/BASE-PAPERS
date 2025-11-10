# Installation Script for Glaucoma Detection Training Environment
# Run this in PowerShell: .\install_dependencies.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Glaucoma Detection - Setup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "[1/5] Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
Write-Host "  Found: $pythonVersion" -ForegroundColor Green
Write-Host ""

# Navigate to preprocessing directory
Write-Host "[2/5] Navigating to preprocessing directory..." -ForegroundColor Yellow
cd preprocessing
Write-Host "  Current directory: $PWD" -ForegroundColor Green
Write-Host ""

# Upgrade pip
Write-Host "[3/5] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip
Write-Host ""

# Install requirements
Write-Host "[4/5] Installing required packages (this may take 5-10 minutes)..." -ForegroundColor Yellow
Write-Host "  Installing: TensorFlow, OpenCV, NumPy, Pandas, and more..." -ForegroundColor Cyan
pip install -r requirements.txt
Write-Host ""

# Verify installation
Write-Host "[5/5] Verifying installation..." -ForegroundColor Yellow
Write-Host ""

Write-Host "  Checking TensorFlow..." -ForegroundColor Cyan
python -c "import tensorflow as tf; print('    ✓ TensorFlow', tf.__version__, 'installed successfully')" 2>&1

Write-Host "  Checking GPU availability..." -ForegroundColor Cyan
python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print('    ✓ GPU Available:', len(gpus) > 0); print('    ✓ GPU Devices:', gpus if gpus else 'None (will use CPU)')" 2>&1

Write-Host "  Checking OpenCV..." -ForegroundColor Cyan
python -c "import cv2; print('    ✓ OpenCV', cv2.__version__, 'installed successfully')" 2>&1

Write-Host "  Checking NumPy..." -ForegroundColor Cyan
python -c "import numpy as np; print('    ✓ NumPy', np.__version__, 'installed successfully')" 2>&1

Write-Host "  Checking Pandas..." -ForegroundColor Cyan
python -c "import pandas as pd; print('    ✓ Pandas', pd.__version__, 'installed successfully')" 2>&1

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Installation Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. If GPU shows 'None', install CUDA Toolkit 12.2" -ForegroundColor White
Write-Host "  2. Read SETUP_TRAINING_ENVIRONMENT.md for detailed guide" -ForegroundColor White
Write-Host "  3. Start with: python preprocess_and_save.py --help" -ForegroundColor White
Write-Host ""
Write-Host "Need help? Check docs/guides/START_HERE.md" -ForegroundColor Cyan
Write-Host ""

# Return to base directory
cd ..

