# Madhva Budget Pro - macOS Installation Guide

This guide provides step-by-step instructions for downloading, installing, and running Madhva Budget Pro on macOS with Touch ID support.

## Quick Start Guide

### Step 1: Download the Application

1. Visit [github.com/Madhva8/madhva-budget-pro](https://github.com/Madhva8/madhva-budget-pro)
2. Click the green "Code" button
3. Select "Download ZIP"
4. Once downloaded, open Finder and go to your Downloads folder
5. Double-click the ZIP file to extract it

### Step 2: Run the Application

1. Open the extracted folder (named "madhva-budget-pro-master")
2. **Right-click** on `FinancialPlanner.command` or `StartApp.command` and select "Open"
   * On the first run, you may see a security warning - click "Open" to proceed
3. The application will start automatically!

That's it! The launcher will:
- Detect Python on your system
- Install any missing dependencies
- Set up the database
- Configure Touch ID support (if available)
- Start the application

## Touch ID Support

This application supports Touch ID authentication on compatible Mac devices:

1. When you first run the app, you may be asked for your password to set up Touch ID
2. After setup, you can use Touch ID for future logins
3. Touch ID will automatically fill your password after successful fingerprint verification

## Detailed Installation Steps

### Prerequisites

- macOS 10.13 or newer
- Python 3.9+ (we recommend using Anaconda)

### Installing Python

#### Option 1: Install Anaconda (Recommended)

1. Download Anaconda from [anaconda.com/download](https://www.anaconda.com/download)
2. Run the installer and follow the prompts
3. Make sure to add Anaconda to your PATH when prompted

#### Option 2: Install Python directly

1. Download Python from [python.org/downloads](https://www.python.org/downloads/)
2. Run the installer
3. Follow the installation prompts

### Resolving First-Launch Security Issues

macOS protects you from running applications from unidentified developers. To run our application:

1. Right-click (or Control+click) on `SimpleStart.command`
2. Select "Open" from the context menu
3. When the warning appears, click "Open"

This only needs to be done once - future launches can be done with a normal double-click.

### Making the Application Executable

If you encounter "Permission denied" errors:

1. Open Terminal
2. Navigate to your application directory:
   ```
   cd /path/to/madhva-budget-pro-master
   ```
3. Make the launcher executable:
   ```
   chmod +x SimpleStart.command
   ```

## Creating a Desktop Shortcut (Alias)

For easier access:

1. In Finder, navigate to the madhva-budget-pro folder
2. Right-click on `SimpleStart.command`
3. Hold down the Option key and select "Make Alias"
4. Drag the alias to your Desktop or Applications folder

## Troubleshooting

### "Command not found" Error

This usually means the script isn't executable:

1. Open Terminal
2. Navigate to the application folder:
   ```
   cd /path/to/madhva-budget-pro-master
   ```
3. Make all scripts executable:
   ```
   chmod +x *.command *.sh
   ```

### Missing Dependencies

If you see module import errors:

1. Open Terminal
2. Navigate to the application folder
3. Run:
   ```
   pip install -r requirements.txt
   ```

### Database Issues

If you encounter database errors:

1. Navigate to the application folder in Terminal
2. Reset the database:
   ```
   cp financial_planner.db.sample financial_planner.db
   python fix_pie_chart.py
   ```

## Creating Backups

To back up your financial data:

1. Close the application
2. In Finder, navigate to the application folder
3. Copy `financial_planner.db` to a secure location
4. Use a date in the filename (e.g., `financial_planner_20250310.db`)

## System Requirements

- macOS 10.13 or newer
- 4GB RAM minimum
- 200MB free disk space
- Python 3.9 or newer

## Getting Help

If you encounter issues:

1. Check the main [README.md](README.md) file
2. Visit the GitHub repository for updates
3. Report issues on the [GitHub issues page](https://github.com/Madhva8/madhva-budget-pro/issues)

## Updates

To update to the latest version:

1. Download the latest version from GitHub
2. Copy your `financial_planner.db` file to the new folder to preserve your data
