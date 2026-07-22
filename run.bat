@echo off
title ResumeBot AI Launcher
echo ===================================================
echo               ResumeBot AI Launcher
echo ===================================================
echo.

cd /d "%~dp0"
set PYTHONUTF8=1

:: Check if virtual environment exists
if not exist .venv (
    echo [ERROR] Virtual environment .venv not found.
    echo Please run setup first, or create the venv.
    pause
    exit /b
)

:: Activate virtual environment
call .venv\Scripts\activate.bat

:: Train if model doesn't exist
if not exist resume_chatbot.keras (
    echo [INFO] Chatbot model not found. Training model first...
    python train_chatbot.py
)

echo [INFO] Starting Streamlit Application...
streamlit run app.py

pause
