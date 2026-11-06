@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main_3d.py
) else (
    python main_3d.py
)
pause
