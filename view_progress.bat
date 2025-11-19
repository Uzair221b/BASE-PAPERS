@echo off
cd /d "%~dp0"
title View Training Progress

echo ================================================================================
echo VIEWING TRAINING PROGRESS
echo ================================================================================
echo.

python view_progress.py

echo.
echo ================================================================================
echo.
echo To watch in real-time, run: python view_progress.py --watch
echo.
pause
