@echo off
REM Simple training starter - just double-click this!
cd /d "%~dp0"

echo.
echo ================================================================================
echo STARTING TRAINING
echo ================================================================================
echo.
echo IMPORTANT: Keep this window open while training runs!
echo.
echo You will see progress bars and updates in real-time.
echo.
echo ================================================================================
echo.

python simple_resume_training.py

echo.
echo ================================================================================
echo Training finished or stopped.
echo ================================================================================
pause


