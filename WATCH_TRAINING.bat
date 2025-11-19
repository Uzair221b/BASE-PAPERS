@echo off
cd /d "%~dp0"
title Watching Training Progress - Real-Time

echo ================================================================================
echo WATCHING TRAINING PROGRESS - REAL-TIME
echo ================================================================================
echo.
echo This will show training progress as it happens.
echo Press Ctrl+C to stop watching (training will continue).
echo.
echo ================================================================================
echo.

python watch_training_live.py

pause

