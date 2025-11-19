@echo off
REM Auto-start training with monitoring
REM Just double-click this file to start!

echo ================================================================================
echo AUTO TRAINING WITH MONITORING
echo ================================================================================
echo.
echo This will:
echo   1. Start training automatically
echo   2. Monitor it every 10 minutes
echo   3. Auto-restart if it crashes
echo   4. Run in background
echo.
echo You can close this window - training will continue!
echo.
echo ================================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python first.
    pause
    exit /b 1
)

REM Check if training script exists
if not exist "simple_resume_training.py" (
    echo ERROR: Training script not found!
    echo Please make sure simple_resume_training.py exists.
    pause
    exit /b 1
)

REM Start the monitor (which will start training)
echo Starting auto-training monitor...
echo.
python auto_train_monitor.py

pause

