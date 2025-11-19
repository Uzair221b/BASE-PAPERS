@echo off
cd /d "%~dp0"
title Glaucoma Training - DO NOT CLOSE THIS WINDOW
color 0A

echo ================================================================================
echo STARTING TRAINING
echo ================================================================================
echo.
echo Training will run in THIS window.
echo DO NOT CLOSE THIS WINDOW while training is running!
echo.
echo You will see progress bars and updates in real-time.
echo.
echo ================================================================================
echo.

python simple_resume_training.py

echo.
echo ================================================================================
echo TRAINING COMPLETED OR STOPPED
echo ================================================================================
pause
