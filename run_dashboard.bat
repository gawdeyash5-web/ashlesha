@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Please run run_model.bat once first to install dependencies.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m streamlit run app.py
pause
