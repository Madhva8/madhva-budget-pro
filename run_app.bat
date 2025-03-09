@echo off
REM ====================================================
REM Madhva Budget Pro - Financial Management Application
REM Windows Launcher
REM ====================================================

echo ===============================================
echo           Madhva Budget Pro
echo      Financial Management Application
echo ===============================================
echo.

REM Get the directory where this script is located
set "APP_DIR=%~dp0"
cd /d "%APP_DIR%"

echo Using application directory: %APP_DIR%

REM Check for Python installation
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed or not in your PATH.
    echo Please install Python 3.9 or higher and try again.
    pause
    exit /b 1
)

echo Python found: 
python --version

REM Check for required packages
echo Checking dependencies...
set MISSING_DEPS=

REM Define dependencies to check
set DEPS=PySide6 keyring matplotlib pandas pdfplumber

for %%D in (%DEPS%) do (
    python -c "import %%D" >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo Installing missing dependency: %%D
        python -m pip install %%D
        if %ERRORLEVEL% NEQ 0 (
            set MISSING_DEPS=!MISSING_DEPS! %%D
        )
    ) else (
        echo Found dependency: %%D
    )
)

if not "%MISSING_DEPS%"=="" (
    echo ERROR: Failed to install required dependencies: %MISSING_DEPS%
    echo Please run: pip install %MISSING_DEPS%
    pause
    exit /b 1
)

REM Ensure necessary directories exist
if not exist "%APP_DIR%\backups" mkdir "%APP_DIR%\backups"
if not exist "%APP_DIR%\logs" mkdir "%APP_DIR%\logs"

REM Ensure database exists
if not exist "%APP_DIR%\financial_planner.db" (
    echo Creating database...
    if exist "%APP_DIR%\financial_planner.db.sample" (
        copy "%APP_DIR%\financial_planner.db.sample" "%APP_DIR%\financial_planner.db"
        echo Database created from sample
    ) else (
        type nul > "%APP_DIR%\financial_planner.db"
        echo Empty database created
    )
)

REM Create the credential setup script
echo Setting up user credentials...
echo import sqlite3 > "%APP_DIR%\fix_credentials.py"
echo import hashlib >> "%APP_DIR%\fix_credentials.py"
echo import secrets >> "%APP_DIR%\fix_credentials.py"
echo import os >> "%APP_DIR%\fix_credentials.py"
echo import sys >> "%APP_DIR%\fix_credentials.py"
echo from datetime import datetime >> "%APP_DIR%\fix_credentials.py"
echo. >> "%APP_DIR%\fix_credentials.py"
echo def fix_credentials(): >> "%APP_DIR%\fix_credentials.py"
echo     """Fix login credentials to ensure users can always log in.""" >> "%APP_DIR%\fix_credentials.py"
echo     try: >> "%APP_DIR%\fix_credentials.py"
echo         # Connect to the database (or create if it doesn't exist) >> "%APP_DIR%\fix_credentials.py"
echo         db_path = "financial_planner.db" >> "%APP_DIR%\fix_credentials.py"
echo         conn = sqlite3.connect(db_path) >> "%APP_DIR%\fix_credentials.py"
echo         cursor = conn.cursor() >> "%APP_DIR%\fix_credentials.py"
echo. >> "%APP_DIR%\fix_credentials.py"
echo         # Make sure users table exists with proper structure >> "%APP_DIR%\fix_credentials.py"
echo         cursor.execute('''CREATE TABLE IF NOT EXISTS users ( >> "%APP_DIR%\fix_credentials.py"
echo             id INTEGER PRIMARY KEY AUTOINCREMENT, >> "%APP_DIR%\fix_credentials.py"
echo             username TEXT UNIQUE NOT NULL, >> "%APP_DIR%\fix_credentials.py"
echo             salt TEXT NOT NULL, >> "%APP_DIR%\fix_credentials.py"
echo             password_hash TEXT NOT NULL, >> "%APP_DIR%\fix_credentials.py"
echo             email TEXT, >> "%APP_DIR%\fix_credentials.py"
echo             full_name TEXT, >> "%APP_DIR%\fix_credentials.py"
echo             is_admin BOOLEAN DEFAULT 0, >> "%APP_DIR%\fix_credentials.py"
echo             is_active BOOLEAN DEFAULT 1, >> "%APP_DIR%\fix_credentials.py"
echo             last_login TIMESTAMP, >> "%APP_DIR%\fix_credentials.py"
echo             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, >> "%APP_DIR%\fix_credentials.py"
echo             updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP >> "%APP_DIR%\fix_credentials.py"
echo         )''') >> "%APP_DIR%\fix_credentials.py"
echo. >> "%APP_DIR%\fix_credentials.py"
echo         # Default users to create or update >> "%APP_DIR%\fix_credentials.py"
echo         users = [ >> "%APP_DIR%\fix_credentials.py"
echo             ("mohna", "Mohna@30", "mohna@example.com", "Mohna User", False), >> "%APP_DIR%\fix_credentials.py"
echo             ("Madhva", "Mohna@30", "madhva@example.com", "Madhva User", False), >> "%APP_DIR%\fix_credentials.py"
echo             ("admin", "admin", "admin@example.com", "Administrator", True), >> "%APP_DIR%\fix_credentials.py"
echo             ("demo", "demo", "demo@example.com", "Demo User", False) >> "%APP_DIR%\fix_credentials.py"
echo         ] >> "%APP_DIR%\fix_credentials.py"
echo. >> "%APP_DIR%\fix_credentials.py"
echo         for username, password, email, full_name, is_admin in users: >> "%APP_DIR%\fix_credentials.py"
echo             # Generate salt for password hashing >> "%APP_DIR%\fix_credentials.py"
echo             salt = secrets.token_hex(16) >> "%APP_DIR%\fix_credentials.py"
echo. >> "%APP_DIR%\fix_credentials.py"
echo             # Hash the password using the salt >> "%APP_DIR%\fix_credentials.py"
echo             password_bytes = password.encode('utf-8') >> "%APP_DIR%\fix_credentials.py"
echo             salt_bytes = bytes.fromhex(salt) >> "%APP_DIR%\fix_credentials.py"
echo. >> "%APP_DIR%\fix_credentials.py"
echo             password_hash = hashlib.pbkdf2_hmac('sha256', password_bytes, salt_bytes, 100000).hex() >> "%APP_DIR%\fix_credentials.py"
echo. >> "%APP_DIR%\fix_credentials.py"
echo             # Check if user already exists >> "%APP_DIR%\fix_credentials.py"
echo             cursor.execute("SELECT id FROM users WHERE username = ?", (username,)) >> "%APP_DIR%\fix_credentials.py"
echo             result = cursor.fetchone() >> "%APP_DIR%\fix_credentials.py"
echo. >> "%APP_DIR%\fix_credentials.py"
echo             now = datetime.now().strftime("%%Y-%%m-%%d %%H:%%M:%%S") >> "%APP_DIR%\fix_credentials.py"
echo. >> "%APP_DIR%\fix_credentials.py"
echo             if result: >> "%APP_DIR%\fix_credentials.py"
echo                 # Update existing user >> "%APP_DIR%\fix_credentials.py"
echo                 cursor.execute( >> "%APP_DIR%\fix_credentials.py"
echo                     "UPDATE users SET salt = ?, password_hash = ?, email = ?, full_name = ?, is_admin = ?, is_active = 1, updated_at = ? WHERE username = ?", >> "%APP_DIR%\fix_credentials.py"
echo                     (salt, password_hash, email, full_name, 1 if is_admin else 0, now, username) >> "%APP_DIR%\fix_credentials.py"
echo                 ) >> "%APP_DIR%\fix_credentials.py"
echo                 print(f"Updated user: {username}") >> "%APP_DIR%\fix_credentials.py"
echo             else: >> "%APP_DIR%\fix_credentials.py"
echo                 # Create new user >> "%APP_DIR%\fix_credentials.py"
echo                 cursor.execute( >> "%APP_DIR%\fix_credentials.py"
echo                     "INSERT INTO users (username, salt, password_hash, email, full_name, is_admin, is_active, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)", >> "%APP_DIR%\fix_credentials.py"
echo                     (username, salt, password_hash, email, full_name, 1 if is_admin else 0, now, now) >> "%APP_DIR%\fix_credentials.py"
echo                 ) >> "%APP_DIR%\fix_credentials.py"
echo                 print(f"Created user: {username}") >> "%APP_DIR%\fix_credentials.py"
echo. >> "%APP_DIR%\fix_credentials.py"
echo         # Also make sure there's at least some transaction data for visuals >> "%APP_DIR%\fix_credentials.py"
echo         cursor.execute('''CREATE TABLE IF NOT EXISTS transactions ( >> "%APP_DIR%\fix_credentials.py"
echo             id INTEGER PRIMARY KEY AUTOINCREMENT, >> "%APP_DIR%\fix_credentials.py"
echo             date TEXT NOT NULL, >> "%APP_DIR%\fix_credentials.py"
echo             description TEXT NOT NULL, >> "%APP_DIR%\fix_credentials.py"
echo             amount REAL NOT NULL, >> "%APP_DIR%\fix_credentials.py"
echo             category TEXT, >> "%APP_DIR%\fix_credentials.py"
echo             account_id INTEGER, >> "%APP_DIR%\fix_credentials.py"
echo             is_expense BOOLEAN DEFAULT 1, >> "%APP_DIR%\fix_credentials.py"
echo             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP >> "%APP_DIR%\fix_credentials.py"
echo         )''') >> "%APP_DIR%\fix_credentials.py"
echo. >> "%APP_DIR%\fix_credentials.py"
echo         # Commit changes and close >> "%APP_DIR%\fix_credentials.py"
echo         conn.commit() >> "%APP_DIR%\fix_credentials.py"
echo         conn.close() >> "%APP_DIR%\fix_credentials.py"
echo         print("Credentials and sample data set up successfully!") >> "%APP_DIR%\fix_credentials.py"
echo         return True >> "%APP_DIR%\fix_credentials.py"
echo     except Exception as e: >> "%APP_DIR%\fix_credentials.py"
echo         print(f"Error setting up credentials: {e}") >> "%APP_DIR%\fix_credentials.py"
echo         return False >> "%APP_DIR%\fix_credentials.py"
echo. >> "%APP_DIR%\fix_credentials.py"
echo if __name__ == "__main__": >> "%APP_DIR%\fix_credentials.py"
echo     success = fix_credentials() >> "%APP_DIR%\fix_credentials.py"
echo     sys.exit(0 if success else 1) >> "%APP_DIR%\fix_credentials.py"

REM Run the credential fix script
python "%APP_DIR%\fix_credentials.py"
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Credentials setup failed. Authentication may not work correctly.
) else (
    echo User credentials set up successfully.
)

REM Set environment variables for the application
set "PYTHONPATH=%APP_DIR%"
set "OS_MODULE_FIX=1"
set "MADHVA_APP_ENV=production"

REM Determine the main script to run
set "MAIN_SCRIPT="
if exist "%APP_DIR%\main_pyside6.py" (
    set "MAIN_SCRIPT=%APP_DIR%\main_pyside6.py"
) else if exist "%APP_DIR%\src\main_pyside6.py" (
    set "MAIN_SCRIPT=%APP_DIR%\src\main_pyside6.py"
) else (
    echo ERROR: Could not find main application script.
    pause
    exit /b 1
)

echo.
echo ==== Starting Madhva Budget Pro ====
echo.

REM Launch the application
python "%MAIN_SCRIPT%"

if %ERRORLEVEL% NEQ 0 (
    echo Application exited with error code: %ERRORLEVEL%
    echo Please check the logs for details.
    pause
    exit /b %ERRORLEVEL%
)

echo Application closed normally
exit /b 0