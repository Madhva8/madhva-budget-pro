# Financial Planner Pro - Windows Installation Guide

This guide provides step-by-step instructions for installing and running Financial Planner Pro on Windows systems.

## Installation Requirements

- Windows 10 or 11 (Windows 7 and 8 may work but are not officially supported)
- Internet connection (for downloading dependencies)
- 500MB free disk space
- Python 3.9 or newer

## Quick Start Guide

### Step 1: Install Python (if not already installed)

1. Download Python from the official website: [python.org/downloads](https://www.python.org/downloads/)
2. Run the installer
3. **IMPORTANT**: Check the box "Add Python to PATH" during installation
4. Click "Install Now"

### Step 2: Download Financial Planner Pro

#### Option 1: Using Git (Recommended)

1. Download and install Git from [git-scm.com](https://git-scm.com/download/win)
2. Open Command Prompt (search for "cmd" in the Start menu)
3. Run the following commands:

```
cd C:\Users\YourUsername\Documents
git clone https://github.com/username/financial-planner-pro.git
cd financial-planner-pro
```

#### Option 2: Direct Download

1. Go to the GitHub repository
2. Click the green "Code" button
3. Select "Download ZIP"
4. Extract the ZIP file to a location you can easily find (e.g., Documents folder)

### Step 3: Run the Application

1. Navigate to the Financial Planner Pro folder in File Explorer
2. **Double-click** the file named `start_app.bat` or `run_app.bat`

That's it! The application will:
- Automatically check for Python
- Install required dependencies if needed
- Set up the database with sample data
- Launch the application

## Troubleshooting Common Issues

### "Python is not recognized as an internal or external command"

This means Python was not added to your PATH. Solutions:
1. Re-run the Python installer and check the "Add Python to PATH" option
2. Or manually add Python to your PATH:
   - Search for "Environment Variables" in the Start menu
   - Edit the PATH variable to include your Python installation folder

### Missing Modules/Dependencies

If the script reports missing modules:
1. Open Command Prompt as Administrator
2. Navigate to your application folder:
   ```
   cd C:\path\to\financial-planner-pro
   ```
3. Install dependencies manually:
   ```
   pip install -r requirements.txt
   ```

### Database Issues

If you see database errors:
1. Delete the existing database file if it's corrupted
2. Create a fresh database by running:
   ```
   copy financial_planner.db.sample financial_planner.db
   python fix_pie_chart.py
   ```

## Creating a Backup

It's a good practice to back up your financial data regularly:
1. Close the application
2. Copy `financial_planner.db` to a safe location
3. Date your backups (e.g., `financial_planner_20250310.db`)

## Uninstalling

To remove Financial Planner Pro:
1. Delete the application folder
2. Delete the desktop shortcut

No registry entries or system files are modified by the application.

## Getting Help

If you encounter any issues, please:
1. Check the main [README.md](README.md) file
2. Visit the GitHub repository for updates
3. Report issues on the GitHub repository
