@echo off
setlocal
cd /d "%~dp0"
title SLM Forge - Setup and Run
echo ============================================
echo   SLM Forge - environment setup and launch
echo   Working folder: %cd%
echo ============================================

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found on PATH.
    echo Install Python 3.10+ from https://www.python.org/downloads/
    echo and check "Add python.exe to PATH" during install. Then run this file again.
    pause
    exit /b 1
)

if not exist venv (
    echo [1/4] Creating virtual environment in %cd%\venv ...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create the virtual environment.
        echo Do NOT run this file as Administrator - just double-click it normally.
        pause
        exit /b 1
    )
)

call venv\Scripts\activate.bat

echo [2/4] Checking torch...
python -m pip show torch >nul 2>nul
if errorlevel 1 (
    echo torch not found - installing CPU build now ^(this can take a few minutes^).
    echo If you have an NVIDIA GPU with CUDA drivers, you can later replace it
    echo with the CUDA build from https://pytorch.org/get-started/locally for speed.
    python -m pip install --default-timeout=180 --retries 5 torch --index-url https://download.pytorch.org/whl/cpu
    if errorlevel 1 (
        echo [ERROR] torch install failed - probably a slow/unstable internet connection.
        echo Try running this file again ^(pip resumes/retries downloads^).
        pause
        exit /b 1
    )
)

echo [3/4] Installing remaining dependencies from requirements.txt...
python -m pip install --default-timeout=180 --retries 5 -r requirements.txt
if errorlevel 1 (
    echo [ERROR] pip install failed - see the error above.
    echo This is usually a slow/unstable internet connection or a firewall/VPN
    echo blocking pypi.org and files.pythonhosted.org. Try running this file again.
    pause
    exit /b 1
)

echo [4/4] Starting SLM Forge...
python -m app.main

echo.
echo SLM Forge has stopped.
pause
