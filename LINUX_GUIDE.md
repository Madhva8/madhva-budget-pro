# Financial Planner Pro - Linux Installation Guide

This guide provides step-by-step instructions for installing and running Financial Planner Pro on Linux systems.

## Supported Distributions

- Ubuntu 20.04 LTS or newer
- Debian 11 or newer
- Fedora 34 or newer
- Other distributions with Python 3.9+ support

## Quick Start Guide

### Step 1: Install Python & Dependencies

Most Linux distributions come with Python pre-installed. You'll need Python 3.9 or newer.

#### For Ubuntu/Debian:
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv python3-tk git
```

#### For Fedora/RHEL:
```bash
sudo dnf install python3 python3-pip python3-tkinter git
```

### Step 2: Download Financial Planner Pro

#### Option 1: Using Git (Recommended)
```bash
git clone https://github.com/username/financial-planner-pro.git
cd financial-planner-pro
```

#### Option 2: Direct Download
1. Go to the GitHub repository
2. Click the green "Code" button
3. Select "Download ZIP"
4. Extract the ZIP file to your home directory or preferred location

### Step 3: Run the Application

1. Open a terminal in the financial-planner-pro directory
2. Make the launcher executable:
   ```bash
   chmod +x run_app.sh
   ```
3. Run the application:
   ```bash
   ./run_app.sh
   ```

That's it! The application will:
- Check for Python on your system
- Install any missing dependencies
- Set up the database with sample data
- Launch the application

## Troubleshooting

### Missing Dependencies

If you encounter errors about missing modules:
```bash
pip3 install --user -r requirements.txt
```

### Display Issues

If you see errors about missing display:
1. Make sure you're running a desktop environment
2. Check that you have X11 or Wayland configured properly
3. Install Python's Tk package:
   ```bash
   # Ubuntu/Debian
   sudo apt install python3-tk
   
   # Fedora/RHEL
   sudo dnf install python3-tkinter
   ```

### Database Issues

If you encounter database errors:
```bash
cp financial_planner.db.sample financial_planner.db
python3 create_sample_db.py
```

## Desktop Integration

To create a desktop shortcut:

1. Create a .desktop file:
```bash
cat > ~/Desktop/FinancialPlannerPro.desktop << EOL
[Desktop Entry]
Type=Application
Name=Financial Planner Pro
Comment=Financial Management Application
Exec=$(pwd)/run_app.sh
Icon=$(pwd)/logo.png
Terminal=false
Categories=Office;Finance;
EOL
```

2. Make it executable:
```bash
chmod +x ~/Desktop/FinancialPlannerPro.desktop
```

## Back Up Your Data

Regularly back up your financial data:
```bash
cp financial_planner.db ~/Documents/financial_planner_$(date +%Y%m%d).db
```

## Getting Help

If you encounter issues:
1. Check the main [README.md](README.md) file
2. Visit the GitHub repository for updates
3. Report issues on the [GitHub issues page](https://github.com/username/financial-planner-pro/issues)