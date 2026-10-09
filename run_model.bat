@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Creating Python virtual environment...
  py -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python main.py
pause
