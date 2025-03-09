# Financial Planner Pro - Launch Guide

This guide provides detailed instructions on how to run Financial Planner Pro on different operating systems without using an IDE.

## Quick Start

For the simplest launch experience, use the provided launcher scripts:

### macOS

1. **Open Finder** and navigate to your Financial Planner Pro folder
2. **Double-click** `FinancialPlanner.command`
3. If you see a security warning, right-click the file and select "Open" instead

### Windows

1. **Open File Explorer** and navigate to your Financial Planner Pro folder
2. **Double-click** `start_app.bat`
3. The script will automatically check for Python, install dependencies if needed, and even create a desktop shortcut

### Linux

1. **Open Terminal** in your Financial Planner Pro folder
2. Run `./start_app.sh`

## Troubleshooting

If the application doesn't launch, try these steps:

### Dependencies Issues

1. Run the dependency installer:
   ```
   ./install_dependencies.sh
   ```

2. Or manually install required packages:
   ```
   pip install PySide6 pandas pdfplumber numpy matplotlib keyring
   ```

### macOS Script Permissions

If you see "Permission denied" errors:

1. Open Terminal and navigate to your application directory
2. Run: `chmod +x *.command install_dependencies.sh`

### Database Issues

If you see database errors:

1. Ensure you have a valid database file:
   ```
   cp financial_planner.db.sample financial_planner.db
   ```

2. Add sample data for better visualization:
   ```
   python fix_pie_chart.py
   ```

## Creating Desktop Shortcuts

### macOS

1. Open Finder and navigate to your application folder
2. Right-click on `FinancialPlanner.command` and select "Make Alias"
3. Move the alias to your Desktop or Applications folder

### Windows

### Automatic Method (Recommended)
The `start_app.bat` script will automatically create a desktop shortcut for you the first time you run it.

### Manual Method
1. Right-click on your Desktop and select New > Shortcut
2. In the location field, enter:
   ```
   C:\path\to\Financial_planner\start_app.bat
   ```
3. Name the shortcut "Financial Planner Pro" and click Finish

### Linux

1. Create a .desktop file:
   ```
   [Desktop Entry]
   Type=Application
   Name=Financial Planner Pro
   Exec=bash -c "cd /path/to/Financial_planner && ./run_app.sh"
   Icon=/path/to/Financial_planner/logo.png
   Terminal=false
   ```
2. Save it to `~/.local/share/applications/` or your Desktop

## Running From Source

If you prefer to run the application directly:

1. Navigate to the `src` directory:
   ```
   cd src
   ```

2. Run the main script:
   ```
   python main_pyside6.py
   ```

## Advanced: Creating Standalone Executables

For a more traditional application experience, you can create standalone executables:

### Using PyInstaller (All Platforms)

1. Install PyInstaller:
   ```
   pip install pyinstaller
   ```

2. Create the executable:
   ```
   # macOS
   ./mac_build.sh
   
   # Windows
   windows_build.bat
   
   # Linux
   ./linux_build.sh
   ```

3. Look for the executable in the `dist` folder

## Docker Support

For containerized deployment:

1. Build and start the Docker container:
   ```
   docker compose up -d
   ```

2. Stop the container when done:
   ```
   docker compose down
   ```

## Further Assistance

If you continue to experience issues, check the project README.md for more detailed information or report the issue on the project's GitHub page.
