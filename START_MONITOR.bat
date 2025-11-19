@echo off
cd /d "%~dp0"
title Monitoring Preprocessed Training

echo ================================================================================
echo CONTINUOUS MONITOR FOR PREPROCESSED TRAINING
echo ================================================================================
echo.
echo This will monitor and auto-restart preprocessed training if it stops.
echo Won't interrupt original training.
echo.
echo Press Ctrl+C to stop monitoring.
echo.
echo ================================================================================
echo.

python monitor_preprocessed_training.py

pause

