@echo off
REM Start training with real-time progress display
REM This will show progress bars and updates in real-time

echo ================================================================================
echo STARTING TRAINING WITH REAL-TIME PROGRESS
echo ================================================================================
echo.
echo Training will show:
echo   - Progress bars for each epoch
echo   - Real-time accuracy and loss
echo   - Estimated time remaining
echo.
echo Press Ctrl+C to stop (but training will continue in background)
echo.
echo ================================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    pause
    exit /b 1
)

REM Start training directly (not through monitor, so we can see progress)
echo Starting training...
echo.
python simple_resume_training.py

pause


