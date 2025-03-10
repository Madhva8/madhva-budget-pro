#!/bin/bash

# ====================================================
# Madhva Budget Pro - Financial Management Application
# Linux Launcher
# ====================================================

echo "======================================================"
echo "           Madhva Budget Pro                         "
echo "      Financial Management Application               "
echo "======================================================"

# Determine app location
APP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$APP_DIR"

echo "📂 Using application directory: $APP_DIR"

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "❌ ERROR: Python 3 is not installed. Please install Python 3 to run this application."
    echo "Try: sudo apt install python3 python3-pip  # For Debian/Ubuntu"
    echo "Or:  sudo dnf install python3 python3-pip  # For Fedora/RHEL"
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check for required packages and install if needed
echo "🔍 Checking dependencies..."
DEPENDENCIES=("PySide6" "keyring" "matplotlib" "pandas" "pdfplumber")
MISSING_DEPS=()

for dep in "${DEPENDENCIES[@]}"; do
    if ! python3 -c "import $dep" &> /dev/null; then
        echo "📦 Installing missing dependency: $dep"
        python3 -m pip install --user $dep
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
    echo "Please run: pip3 install --user $MISSING_LIST"
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
        
        # Default users to create or update
        users = [
            ("mohna", "Mohna@30", "mohna@example.com", "Mohna User", False),
            ("Madhva", "Mohna@30", "madhva@example.com", "Madhva User", False),
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

# Run the application with all necessary settings
echo "🚀 Launching Madhva Budget Pro..."

# Set environment variables
export PYTHONPATH="$APP_DIR"
export OS_MODULE_FIX="1"
export MADHVA_APP_ENV="production"

# Determine the main script to run
MAIN_SCRIPT=""
if [ -f "$APP_DIR/main_pyside6.py" ]; then
    MAIN_SCRIPT="$APP_DIR/main_pyside6.py"
elif [ -f "$APP_DIR/src/main_pyside6.py" ]; then
    MAIN_SCRIPT="$APP_DIR/src/main_pyside6.py"
else
    echo "❌ ERROR: Could not find main application script."
    exit 1
fi

echo "▶️ Running: $MAIN_SCRIPT"

# Check for X11 or Wayland
if [ -z "$DISPLAY" ] && [ -z "$WAYLAND_DISPLAY" ]; then
    echo "❌ ERROR: No display server detected. This application requires a graphical environment."
    exit 1
fi

# Run the application
python3 "$MAIN_SCRIPT"

# Check exit status
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ Application exited with code: $EXIT_CODE"
    echo "Please check the logs for details."
    exit $EXIT_CODE
fi

echo "✨ Application closed normally"
exit 0