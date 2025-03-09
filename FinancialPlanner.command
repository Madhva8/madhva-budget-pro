#!/bin/bash

# ====================================================
# Financial Planner Pro - Financial Management Application
# Launcher with Touch ID Support
# ====================================================

# Display app startup message with fancy formatting
echo "╭────────────────────────────────────────────────╮"
echo "│         🧮 Financial Planner Pro              │"
echo "│       Financial Management Application         │"
echo "╰────────────────────────────────────────────────╯"
echo "🔐 Touch ID authentication will be automatically enabled"

# Determine app location
APP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$APP_DIR"

echo "📂 Using application directory: $APP_DIR"

# Check Python installation
if ! command -v python3 &> /dev/null; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        osascript -e 'display dialog "Python 3 is not installed. Please install Python 3 to run this application." buttons {"OK"} default button "OK" with icon stop with title "Python Not Found"'
    else
        echo "❌ ERROR: Python 3 is not installed. Please install Python 3 to run this application."
    fi
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check for required packages and install if needed
echo "🔍 Checking dependencies..."
DEPENDENCIES=("PySide6" "keyring" "matplotlib" "pandas" "pdfplumber" "numpy" "pillow" "PyQt5" "SQLAlchemy")
MISSING_DEPS=()

for dep in "${DEPENDENCIES[@]}"; do
    if ! python3 -c "import $dep" &> /dev/null; then
        echo "📦 Installing missing dependency: $dep"
        python3 -m pip install $dep
        if [ $? -ne 0 ]; then
            MISSING_DEPS+=("$dep")
        fi
    else
        echo "✅ Found dependency: $dep"
    fi
done

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    MISSING_LIST=$(IFS=", "; echo "${MISSING_DEPS[*]}")
    echo "❌ ERROR: Failed to install required dependencies: $MISSING_LIST"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        osascript -e "display dialog \"Could not install required dependencies: $MISSING_LIST. Please run: pip install ${MISSING_LIST// /, }\" buttons {\"OK\"} default button \"OK\" with icon stop with title \"Dependency Error\""
    fi
    exit 1
fi

# Ensure necessary directories exist
DIRS_TO_CREATE=("backups" "logs")
for dir in "${DIRS_TO_CREATE[@]}"; do
    if [ ! -d "$APP_DIR/$dir" ]; then
        echo "📁 Creating $dir directory..."
        mkdir -p "$APP_DIR/$dir"
    fi
done

# Ensure database exists
DB_PATH="$APP_DIR/financial_planner.db"
if [ ! -f "$DB_PATH" ]; then
    echo "🗄️ Creating database..."
    if [ -f "$APP_DIR/financial_planner.db.sample" ]; then
        cp "$APP_DIR/financial_planner.db.sample" "$DB_PATH"
        echo "✅ Database created from sample"
    else
        touch "$DB_PATH"
        echo "✅ Empty database created"
    fi
fi

# Create and run login fix script
echo "🔑 Setting up user credentials..."
cat > "$APP_DIR/fix_credentials.py" << 'EOF'
#!/usr/bin/env python3
import sqlite3
import hashlib
import secrets
import os
import sys
from datetime import datetime

def fix_credentials():
    """Fix login credentials to ensure users can always log in."""
    try:
        # Connect to the database (or create if it doesn't exist)
        db_path = "financial_planner.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Make sure users table exists with proper structure
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            salt TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT,
            full_name TEXT,
            is_admin BOOLEAN DEFAULT 0,
            is_active BOOLEAN DEFAULT 1,
            last_login TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Default users to create or update with generic credentials
        users = [
            ("user1", "password1", "user1@example.com", "Regular User", False),
            ("user2", "password2", "user2@example.com", "Regular User", False),
            ("admin", "admin", "admin@example.com", "Administrator", True),
            ("demo", "demo", "demo@example.com", "Demo User", False)
        ]
        
        for username, password, email, full_name, is_admin in users:
            # Generate salt for password hashing
            salt = secrets.token_hex(16)
            
            # Hash the password using the salt
            password_bytes = password.encode('utf-8')
            salt_bytes = bytes.fromhex(salt)
            
            password_hash = hashlib.pbkdf2_hmac(
                'sha256',
                password_bytes,
                salt_bytes,
                100000  # Number of iterations
            ).hex()
            
            # Check if user already exists
            cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
            result = cursor.fetchone()
            
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if result:
                # Update existing user
                cursor.execute(
                    "UPDATE users SET salt = ?, password_hash = ?, email = ?, full_name = ?, is_admin = ?, is_active = 1, updated_at = ? WHERE username = ?",
                    (salt, password_hash, email, full_name, 1 if is_admin else 0, now, username)
                )
                print(f"Updated user: {username}")
            else:
                # Create new user
                cursor.execute(
                    "INSERT INTO users (username, salt, password_hash, email, full_name, is_admin, is_active, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)",
                    (username, salt, password_hash, email, full_name, 1 if is_admin else 0, now, now)
                )
                print(f"Created user: {username}")
        
        # Also make sure there's at least some transaction data for visuals
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            description TEXT NOT NULL,
            amount REAL NOT NULL,
            category TEXT,
            account_id INTEGER,
            is_expense BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Add some sample transactions if table is empty
        cursor.execute("SELECT COUNT(*) FROM transactions")
        count = cursor.fetchone()[0]
        
        if count == 0:
            print("Adding sample transactions...")
            sample_transactions = [
                ("2025-03-01", "Grocery shopping", -75.50, "Food", 1, 1),
                ("2025-03-02", "Salary", 3000.00, "Income", 1, 0),
                ("2025-03-03", "Netflix subscription", -12.99, "Entertainment", 1, 1),
                ("2025-03-04", "Gas station", -45.00, "Transportation", 1, 1),
                ("2025-03-05", "Rent payment", -1200.00, "Housing", 1, 1),
                ("2025-03-06", "Dinner at restaurant", -65.75, "Food", 1, 1),
                ("2025-03-07", "Movie tickets", -25.00, "Entertainment", 1, 1),
                ("2025-03-08", "Utility bill", -120.00, "Utilities", 1, 1),
                ("2025-03-09", "Clothing purchase", -85.50, "Shopping", 1, 1),
                ("2025-03-10", "Pharmacy", -35.25, "Health", 1, 1)
            ]
            
            for date, desc, amount, category, account_id, is_expense in sample_transactions:
                cursor.execute(
                    "INSERT INTO transactions (date, description, amount, category, account_id, is_expense) VALUES (?, ?, ?, ?, ?, ?)",
                    (date, desc, amount, category, account_id, is_expense)
                )
                
        # Commit changes and close
        conn.commit()
        conn.close()
        print("Credentials and sample data set up successfully!")
        return True
    except Exception as e:
        print(f"Error setting up credentials: {e}")
        return False

if __name__ == "__main__":
    success = fix_credentials()
    sys.exit(0 if success else 1)
EOF

# Make script executable
chmod +x "$APP_DIR/fix_credentials.py"

# Run the credential fix script
python3 "$APP_DIR/fix_credentials.py"
if [ $? -ne 0 ]; then
    echo "⚠️ Credentials setup failed. Authentication may not work correctly."
else
    echo "✅ User credentials set up successfully."
fi

# Set up Touch ID
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🔐 Setting up Touch ID access..."
    # Request permission with reduced prompts (only if not already granted)
    if [ ! -f "$APP_DIR/.touchid_setup_complete" ]; then
        # Create a helper script that will run with admin privileges once
        HELPER_SCRIPT=$(mktemp)
        cat > "$HELPER_SCRIPT" <<EOL
#!/bin/bash
echo "Touch ID permissions granted" > "$APP_DIR/.touchid_setup_complete"
chmod 600 "$APP_DIR/.touchid_setup_complete"
EOL
        chmod +x "$HELPER_SCRIPT"
        
        # Run the helper script with admin privileges (this will prompt once)
        echo "🔑 You may be asked for your password once to set up Touch ID permissions"
        if command -v osascript &> /dev/null; then
            osascript -e "do shell script \"$HELPER_SCRIPT\" with administrator privileges"
        else
            sudo "$HELPER_SCRIPT"
        fi
        rm "$HELPER_SCRIPT"
    fi
    
    # Set up environment variable for Touch ID
    export TOUCHID_PREAUTH="1"
    echo "✅ Touch ID enabled for this session"
fi

# Run the application with all necessary settings
echo "🚀 Launching Financial Planner Pro..."

# Set environment variables - CRITICAL for UI component paths
export PYTHONPATH="$APP_DIR:$APP_DIR/src"
export OS_MODULE_FIX="1"
export FINANCIAL_PLANNER_APP_ENV="production"
export TOUCHID_ENABLED="1"
export PYTHONIOENCODING="utf-8"
export MODERN_UI_ENABLED="1"
export FINANCIAL_PLANNER_DEBUG="1"
export TEMPLATE_PATH="$APP_DIR/src/ui"

# Fix path to help matplotlib find its data files
export MATPLOTLIBDATA="$APP_DIR/venv/lib/python3.8/site-packages/matplotlib/mpl-data"

# This ensures PyQt/PySide can find the right display
export QT_QPA_PLATFORM_PLUGIN_PATH="$APP_DIR/venv/lib/python3.8/site-packages/PySide6/plugins/platforms"

# Determine the main script to run
MAIN_SCRIPT=""
if [ -f "$APP_DIR/src/main_pyside6.py" ]; then
    MAIN_SCRIPT="$APP_DIR/src/main_pyside6.py"
elif [ -f "$APP_DIR/main_pyside6.py" ]; then
    MAIN_SCRIPT="$APP_DIR/main_pyside6.py"  
elif [ -f "$APP_DIR/src/main.py" ]; then
    MAIN_SCRIPT="$APP_DIR/src/main.py"
else
    echo "❌ ERROR: Could not find main application script."
    exit 1
fi

echo "▶️ Running: $MAIN_SCRIPT"

# Run the application
if [[ "$OSTYPE" == "darwin"* ]]; then
    # For macOS: Try different methods to run the app with proper GUI support
    
    # Check if we have a virtual environment and prefer it
    if [ -d "$APP_DIR/venv/bin" ]; then
        PYTHON_CMD="$APP_DIR/venv/bin/python3"
        echo "Using virtual environment Python: $PYTHON_CMD"
    else
        PYTHON_CMD="python3"
        echo "Using system Python: $PYTHON_CMD"
    fi
    
    # Create a robust launcher script with all necessary environment variables
    TEMP_LAUNCHER=$(mktemp)
    cat > "$TEMP_LAUNCHER" << EOF
#!/bin/bash
# Change to the application directory
cd "$APP_DIR"

# Set up all required environment variables
export PYTHONPATH="$APP_DIR:$APP_DIR/src"
export OS_MODULE_FIX="1"
export FINANCIAL_PLANNER_APP_ENV="production"
export TOUCHID_ENABLED="1"
export TOUCHID_PREAUTH="1"
export PYTHONIOENCODING="utf-8"
export MODERN_UI_ENABLED="1"
export FINANCIAL_PLANNER_DEBUG="1"
export TEMPLATE_PATH="$APP_DIR/src/ui"

# Fix paths for matplotlib and PySide6
export MATPLOTLIBDATA="$APP_DIR/venv/lib/python3.8/site-packages/matplotlib/mpl-data"
export QT_QPA_PLATFORM_PLUGIN_PATH="$APP_DIR/venv/lib/python3.8/site-packages/PySide6/plugins/platforms"

# Check for Python interpreter in virtual environment
if [ -f "$APP_DIR/venv/bin/python3" ]; then
    PYTHON="$APP_DIR/venv/bin/python3"
else
    PYTHON="python3"
fi

echo "Launching Financial Planner Pro with \$PYTHON"
echo "Main script: $MAIN_SCRIPT"

# Run with Python's -u flag for unbuffered output
\$PYTHON -u "$MAIN_SCRIPT"

# Keep terminal open if there was an error
if [ \$? -ne 0 ]; then
    echo "Application exited with an error. Press Enter to close this window."
    read -p ""
fi
EOF
    chmod +x "$TEMP_LAUNCHER"
    
    # Run the temporary launcher script
    echo "▶️ Launching application with all required paths and environment variables..."
    "$TEMP_LAUNCHER"
    
    # If direct execution fails, try opening in a new terminal window
    if [ $? -ne 0 ]; then
        echo "⚠️ Direct execution failed, opening in a new terminal window..."
        open -a Terminal.app "$TEMP_LAUNCHER"
    fi
    
else
    # Standard execution for other platforms
    if [ -d "$APP_DIR/venv/bin" ]; then
        "$APP_DIR/venv/bin/python3" "$MAIN_SCRIPT"
    else
        python3 "$MAIN_SCRIPT"
    fi
fi

# Check exit status
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ Application exited with code: $EXIT_CODE"
    
    # Check for common error causes and provide specific guidance
    if grep -q "ImportError: No module named" "$APP_DIR/logs/financial_planner.log" 2>/dev/null; then
        echo "💡 Missing dependency detected. Try installing required packages:"
        echo "   $PYTHON_CMD -m pip install -r requirements.txt"
        
        if [[ "$OSTYPE" == "darwin"* ]] && command -v osascript &> /dev/null; then
            osascript -e 'display dialog "Missing Python dependency detected. Please run:\npip install -r requirements.txt" buttons {"OK"} default button "OK" with icon caution with title "Missing Dependency"'
        fi
    elif grep -q "ModuleNotFoundError: No module named" "$APP_DIR/logs/financial_planner.log" 2>/dev/null; then
        echo "💡 Python module not found. Try installing required packages:"
        echo "   $PYTHON_CMD -m pip install -r requirements.txt"
        
        if [[ "$OSTYPE" == "darwin"* ]] && command -v osascript &> /dev/null; then
            osascript -e 'display dialog "Python module not found. Please run:\npip install -r requirements.txt" buttons {"OK"} default button "OK" with icon caution with title "Module Not Found"'
        fi
    elif grep -q "no such table:" "$APP_DIR/logs/financial_planner.log" 2>/dev/null; then
        echo "💡 Database issue detected. Try resetting the database:"
        echo "   cp $APP_DIR/financial_planner.db.sample $APP_DIR/financial_planner.db"
        
        if [[ "$OSTYPE" == "darwin"* ]] && command -v osascript &> /dev/null; then
            osascript -e 'display dialog "Database issue detected. The database may be corrupted.\n\nTry restoring from the sample database." buttons {"OK"} default button "OK" with icon caution with title "Database Error"'
        fi
    else
        # Generic error message
        if [[ "$OSTYPE" == "darwin"* ]] && command -v osascript &> /dev/null; then
            osascript -e 'display dialog "The application exited unexpectedly. Please check the logs for details." buttons {"OK"} default button "OK" with icon caution with title "Application Error"'
        else
            echo "The application exited unexpectedly. Please check the logs for details."
        fi
    fi
    
    # Create logs directory if it doesn't exist
    mkdir -p "$APP_DIR/logs"
    
    # Capture environment info for debugging
    echo "--- Environment Information ---" > "$APP_DIR/logs/environment_info.log"
    echo "Date: $(date)" >> "$APP_DIR/logs/environment_info.log"
    echo "OS: $(uname -a)" >> "$APP_DIR/logs/environment_info.log"
    echo "Python: $($PYTHON_CMD --version 2>&1)" >> "$APP_DIR/logs/environment_info.log"
    echo "PYTHONPATH: $PYTHONPATH" >> "$APP_DIR/logs/environment_info.log"
    echo "Working directory: $(pwd)" >> "$APP_DIR/logs/environment_info.log"
    echo "Main script: $MAIN_SCRIPT" >> "$APP_DIR/logs/environment_info.log"
    echo "--------------------------" >> "$APP_DIR/logs/environment_info.log"
    
    echo "Environment information saved to logs/environment_info.log"
    
    exit $EXIT_CODE
fi

echo "✨ Application closed normally"
exit 0