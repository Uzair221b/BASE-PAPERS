@echo off
REM Open HTML file in browser to convert to PDF
cd /d "%~dp0"

echo ================================================================================
echo OPENING HTML FILE - CONVERT TO PDF
echo ================================================================================
echo.
echo The HTML file will open in your browser.
echo.
echo To convert to PDF:
echo   1. Press Ctrl+P (or File -^> Print)
echo   2. Select "Save as PDF" as the printer
echo   3. Click "Save"
echo.
echo ================================================================================
echo.

start UNDERSTANDING_TRAINING_PROGRESS.html

pause

