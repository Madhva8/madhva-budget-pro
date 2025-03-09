@echo off
setlocal EnableDelayedExpansion

REM Set colors for Windows console
set "GREEN=[92m"
set "BLUE=[94m"
set "RED=[91m"
set "NC=[0m"

echo %BLUE%=====================================%NC%
echo %BLUE%   Madhva Budget Pro - Windows Start %NC%
echo %BLUE%=====================================%NC%
echo.

REM Get the directory where this batch file is located
set "DIR=%~dp0"
set "DIR=%DIR:~0,-1%"

echo Working directory: %GREEN%%DIR%%NC%

REM Check if Python is installed and in PATH
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo %RED%Error: Python not found in PATH%NC%
    echo.
    echo Please install Python from python.org and ensure it's added to your PATH
    echo during installation (check the "Add Python to PATH" option).
    echo.
    echo After installing Python, try running this script again.
    echo.
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=*" %%a in ('python --version 2^>^&1') do set "PYTHON_VERSION=%%a"
echo %GREEN%Found %PYTHON_VERSION%%NC%

REM Check if we're in the src directory or root
if exist "%DIR%\src\main_pyside6.py" (
    echo Changing to src directory...
    cd /d "%DIR%\src"
) else if exist "%DIR%\main_pyside6.py" (
    echo Main script found in current directory
) else (
    echo %RED%Error: Could not find main_pyside6.py in either src directory or current directory%NC%
    echo Searched in:
    echo - %DIR%\src\main_pyside6.py
    echo - %DIR%\main_pyside6.py
    echo.
    pause
    exit /b 1
)

REM Check if database exists, if not create it from sample
if not exist "%DIR%\financial_planner.db" (
    if exist "%DIR%\financial_planner.db.sample" (
        echo %BLUE%Database not found. Setting up from sample...%NC%
        copy "%DIR%\financial_planner.db.sample" "%DIR%\financial_planner.db" >nul
        
        REM Add sample data for better visualization if script exists
        if exist "%DIR%\fix_pie_chart.py" (
            echo %BLUE%Adding sample expense data for better visualization...%NC%
            pushd "%DIR%"
            python fix_pie_chart.py
            popd
        )
    )
)

REM Check if required packages are installed
echo %BLUE%Checking required packages...%NC%
python -c "import sys; packages = ['PySide6', 'pandas', 'matplotlib', 'pdfplumber', 'numpy']; missing = [p for p in packages if p not in sys.modules and not __import__(p, fromlist=['']) in globals()]; sys.exit(1 if missing else 0)" 2>nul

if %ERRORLEVEL% NEQ 0 (
    echo %RED%Some required packages are missing. Installing now...%NC%
    echo.
    
    REM Try to install required packages
    if exist "%DIR%\requirements.txt" (
        python -m pip install --user -r "%DIR%\requirements.txt"
    ) else (
        python -m pip install --user PySide6 pandas matplotlib pdfplumber numpy
    )
    
    if %ERRORLEVEL% NEQ 0 (
        echo %RED%Failed to install required packages%NC%
        echo.
        echo Please try running this command manually:
        echo python -m pip install --user PySide6 pandas matplotlib pdfplumber numpy
        echo.
        pause
        exit /b 1
    )
)

REM Create a shortcut if it doesn't exist yet
if not exist "%USERPROFILE%\Desktop\Madhva Budget Pro.lnk" (
    echo %BLUE%Creating desktop shortcut...%NC%
    
    REM Create a temporary VBScript to make the shortcut
    echo Set oWS = WScript.CreateObject("WScript.Shell") > "%TEMP%\CreateShortcut.vbs"
    echo sLinkFile = "%USERPROFILE%\Desktop\Madhva Budget Pro.lnk" >> "%TEMP%\CreateShortcut.vbs"
    echo Set oLink = oWS.CreateShortcut(sLinkFile) >> "%TEMP%\CreateShortcut.vbs"
    echo oLink.TargetPath = "%DIR%\start_app.bat" >> "%TEMP%\CreateShortcut.vbs"
    echo oLink.WorkingDirectory = "%DIR%" >> "%TEMP%\CreateShortcut.vbs"
    echo oLink.Description = "Launch Madhva Budget Pro" >> "%TEMP%\CreateShortcut.vbs"
    echo If FSO.FileExists("%DIR%\logo.png") Then >> "%TEMP%\CreateShortcut.vbs"
    echo     oLink.IconLocation = "%DIR%\logo.png" >> "%TEMP%\CreateShortcut.vbs"
    echo End If >> "%TEMP%\CreateShortcut.vbs"
    echo oLink.Save >> "%TEMP%\CreateShortcut.vbs"
    
    REM Run the VBScript
    cscript //nologo "%TEMP%\CreateShortcut.vbs"
    del "%TEMP%\CreateShortcut.vbs"
)

echo %BLUE%Starting Madhva Budget Pro...%NC%
echo.

REM Run the application
python main_pyside6.py

REM Check if application crashed
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo %RED%The application encountered an error (Exit code: %ERRORLEVEL%)%NC%
    echo.
    echo %BLUE%Diagnostic information:%NC%
    echo %BLUE%----------------------%NC%
    
    echo Python version:
    python --version
    echo.
    
    echo Checking required modules:
    call :check_module PySide6
    call :check_module matplotlib
    call :check_module pandas
    call :check_module pdfplumber
    call :check_module numpy
    
    echo.
    echo If modules are missing, try running:
    echo python -m pip install --user -r requirements.txt
    echo.
    pause
    exit /b %ERRORLEVEL%
)

exit /b 0

:check_module
python -c "import %~1; print('%~1: Installed')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo %RED%%~1: Not found%NC%
) else (
    echo %GREEN%%~1: Installed%NC%
)
goto :eof
