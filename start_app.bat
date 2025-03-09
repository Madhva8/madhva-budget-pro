@echo off
echo ====================================
echo    Madhva Budget Pro - Quick Start
echo ====================================
echo.

echo Changing to src directory...
cd /d "%~dp0src"

echo Starting application...
python main_pyside6.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error: The application encountered a problem.
    echo.
    echo Diagnostic information:
    echo ----------------------
    python -c "import sys; print('Python version:', sys.version)"
    
    echo.
    echo Checking required modules:
    python -c "import PySide6; print('PySide6: Installed')" 2>NUL || echo PySide6: Not found
    python -c "import matplotlib; print('Matplotlib: Installed')" 2>NUL || echo Matplotlib: Not found
    python -c "import pandas; print('Pandas: Installed')" 2>NUL || echo Pandas: Not found
    python -c "import pdfplumber; print('PDFPlumber: Installed')" 2>NUL || echo PDFPlumber: Not found
    
    echo.
    echo To install all dependencies, run: pip install -r requirements.txt
    echo.
    pause
)

exit /b %ERRORLEVEL%
