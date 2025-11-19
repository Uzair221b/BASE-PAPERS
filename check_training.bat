@echo off
cd /d "%~dp0"
echo ================================================================================
echo CHECKING IF TRAINING IS RUNNING
echo ================================================================================
echo.

tasklist | findstr /I python.exe
if %errorlevel% == 0 (
    echo.
    echo [OK] Training IS running!
    echo.
) else (
    echo.
    echo [INFO] Training is NOT running.
    echo.
)

echo ================================================================================
echo LATEST TRAINING LOG
echo ================================================================================
echo.

if exist "training_logs\training_*.log" (
    for /f "delims=" %%i in ('dir /b /o-d training_logs\training_*.log') do (
        echo Latest log: %%i
        echo.
        powershell -Command "Get-Content 'training_logs\%%i' -Tail 20"
        goto :done
    )
) else (
    echo No training logs found.
)

:done
echo.
echo ================================================================================
pause
