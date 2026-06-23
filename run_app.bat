@echo off
cd /d "%~dp0"
echo Checking for desktop shortcut...
powershell.exe -ExecutionPolicy Bypass -File "create_desktop_shortcut.ps1"
echo.
echo Starting ALPR System...
echo Press Ctrl+C in this window to stop.
streamlit run indian_alpr_demo.py --server.port 8502
pause
