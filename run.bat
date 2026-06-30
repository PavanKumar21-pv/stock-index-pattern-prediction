@echo off
cd /d "%~dp0"

if exist "venv37\Scripts\python.exe" (
    "venv37\Scripts\python.exe" Main.py
) else (
    python Main.py
)

pause
