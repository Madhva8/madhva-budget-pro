# Financial Planner Pro

A comprehensive personal finance management application specifically designed for Sparkasse bank customers in Germany. Features automatic parsing of German Sparkasse bank statements with English translation, AI-powered transaction categorization, budgeting tools, and rich financial analytics, all within a beautiful Adobe-inspired interface optimized for readability and professional use.

<img src="logo.png" alt="Financial Planner Pro Logo" width="120"/>

## Quick Start Guide

| Platform | Launch Method | Detailed Guide |
|----------|--------------|----------------|
| **macOS** | Double-click `FinancialPlanner.command` or `StartApp.command` | [macOS Guide](MACOS_GUIDE.md) |
| **Windows** | Double-click `start_app.bat` | [Windows Guide](WINDOWS_GUIDE.md) |
| **Linux** | Run `./run_app.sh` or `./start_app.sh` | [Linux Guide](LINUX_GUIDE.md) |

## New in March 2025 Update
- **Enhanced Touch ID Support** - Seamless authentication on macOS devices 
- **Password Auto-Fill** - Touch ID now automatically fills your password
- **Cross-Platform Launchers** - Easy-to-use startup scripts for all platforms
- **Improved UI/UX** - Polished interface with Adobe-inspired design

## Dashboard Showcase

The dashboard provides a comprehensive overview of your financial situation with interactive visualizations:

- Spending by Category (pie chart)
- Income vs Expenses comparison
- Monthly Trends with future forecasting
- Recent Transactions list

> **Specialized for Sparkasse Bank Customers**: Upload your Sparkasse PDF statements in German and view your finances in English with automatic translation and smart categorization

## New Adobe-Inspired Design System

Financial Planner Pro has been completely redesigned with an Adobe-inspired design system that enhances readability, visual organization, and professional appearance. The new UI features:

- **Clean Visual Hierarchy**: Improved typography and spacing for better information scanning
- **Card-Based Layout**: Content organization with clear visual boundaries
- **Modern Component Library**: Consistently styled buttons, cards, and form elements
- **Professional Color Scheme**: Based on Adobe's design language
- **Enhanced Visualizations**: Improved charts and data representations
- **Responsive Design**: Improved layout adaptation to different screen sizes

## Key Features

### 🏦 Financial Management
- **Transaction Tracking**: Efficiently manage income and expenses with a user-friendly transaction interface
- **Category Management**: Organize transactions into customizable categories
- **Comprehensive Dashboard**: Visual overview of your financial health with charts and analytics
- **Search & Filter**: Quickly find transactions with powerful search and filtering capabilities

### 📊 Analytics & Insights
- **Financial Trends**: Visualize spending patterns over time
- **Category Analysis**: Understand where your money goes with detailed category breakdowns
- **Income vs. Expenses**: Track your cash flow with detailed comparisons
- **Custom Reports**: Generate financial reports based on various parameters

### 🤖 Automation & AI
- **Sparkasse PDF Import**: Automatically parse German Sparkasse bank statements
- **German-to-English Translation**: View your German bank statements in English
- **AI-Powered Categorization**: Smart categorization of transactions using AI algorithms
- **Subscription Detection**: AI identifies and tracks recurring payments and subscriptions
- **Batch Operations**: Efficiently manage multiple transactions at once

### 🔐 Security
- **User Authentication**: Secure login system with username/password
- **Data Encryption**: Secure storage of sensitive financial information
- **Local-First**: All data stays on your computer, not in the cloud
- **Touch ID Support**: macOS biometric authentication integration

### 🎨 Modern UI/UX
- **Adobe-Inspired Design**: Professional and clean interface based on Adobe's design language
- **Light/Dark Mode**: Complete support for both light and dark themes
- **Responsive Design**: Adapts to different window sizes with improved layout adjustments
- **Intuitive Navigation**: Enhanced tab-based interface with improved visual feedback
- **Accessibility**: Better text contrast and visual organization for improved readability

### 🚀 Advanced Features
- **Budget Planning**: Set and track budgets by category
- **Goal Tracking**: Define and monitor financial goals
- **Financial Calendar**: View upcoming bills and income
- **Export Options**: Save reports and data in various formats
- **Docker Support**: Run in containerized environments

## Installation Guide

### Prerequisites

Before installing Financial Planner Pro, ensure you have:

- **Python 3.9 or newer** installed on your system
- **pip** (Python package manager)
- **Git** (for cloning the repository)

#### Installing Python

- **macOS**: Install using Homebrew: `brew install python` or download [Anaconda](https://www.anaconda.com/download)
- **Windows**: Download installer from [python.org](https://www.python.org/downloads/)
- **Linux**: Use your distribution's package manager (e.g., `apt install python3 python3-pip`)

### Installation Guide

#### Option 1: Direct Download (Recommended)

1. **Download the latest version**:
   - Visit the GitHub Releases page
   - Download the version for your operating system

2. **Launch the application**:
   - **macOS**: Double-click `FinancialPlanner.command`
   - **Windows**: Double-click `start_app.bat`
   - **Linux**: Run `./run_app.sh`

#### Option 2: Clone from GitHub

1. **Clone the repository**:
   ```bash
   git clone https://github.com/username/financial-planner-pro.git
   cd financial-planner-pro
   ```

2. **Run the application launcher for your platform**:
   - **macOS**: `./FinancialPlanner.command`
   - **Windows**: `start_app.bat`
   - **Linux**: `./run_app.sh`

The launchers automatically:
- Check for and install required dependencies
- Set up the database with sample data
- Configure Touch ID on macOS (if available)
- Run the application with optimal settings

### Platform-Specific Notes

#### macOS

The macOS launcher provides the following features:
- **Touch ID Support**: Secure authentication using your fingerprint
- **Automatic Dependencies**: Installs required packages if needed
- **Database Setup**: Creates and populates the database automatically

If Touch ID is available, you'll be prompted to authorize its use the first time you run the application.

#### Windows 

The Windows launcher:
- Automatically checks for Python and installs dependencies
- Creates the required database structure
- Configures proper environment variables

#### Linux

The Linux launcher:
- Works across most major distributions (Ubuntu, Fedora, etc.)
- Checks for required GUI components
- Sets up the proper Python environment

For all platforms, the application will remember your login credentials for quicker future access.

#### Docker Support

For containerized deployment:

```bash
# Run the application in Docker (first time may take a while to build)
docker compose up -d

# Access the application through your browser at:
http://localhost:8080

# Stop the container when done
docker compose down
```

#### Standalone Application (Advanced)

You can build a standalone executable that doesn't require Python to be installed:

```bash
# macOS
./mac_build.sh

# Windows
windows_build.bat

# Linux
./linux_build.sh
```

The standalone executables will be created in the `dist/` directory.

### Troubleshooting

- **Missing dependencies**: Ensure all dependencies are installed with `pip install -r requirements.txt`
- **Python version issues**: Confirm you're using Python 3.9+ with `python --version`
- **Database errors**: Make sure you've created the database using the instructions above
- **Permission issues**: Ensure script files are executable (`chmod +x filename` on macOS/Linux)

### Authentication

The application features a secure login system with biometric support to protect your financial data.

#### Login System

- **Secure Authentication**: Password hashing with salt for maximum security
- **Multiple Users**: Support for different user accounts with varying permissions
- **Remember Me**: Option to securely store credentials for quicker logins
- **Account Management**: Create and manage your own user accounts

#### Biometric Authentication

- **Touch ID Support**: On macOS devices with Touch ID, use fingerprint for fast, secure login
- **Password Auto-Fill**: Biometric authentication automatically fills credentials
- **Enhanced Security**: All biometric data is handled securely by your operating system
- **Seamless Experience**: Quick authentication without compromising security

## Technical Details

### Architecture
- **Design System**: Custom Adobe-inspired component library with consistent styling
- **Frontend**: PySide6/PyQt5 for the UI
- **Backend**: Pure Python with SQLite database
- **Analytics**: Matplotlib for visualization, Pandas for data processing
- **PDF Processing**: PDFPlumber for Sparkasse statement parsing
- **Natural Language Processing**: AI modules for transaction categorization
- **Translation**: German to English translation for bank statements

### System Requirements
- **Operating Systems**: macOS 10.14+, Windows 10+, Linux
- **Python**: 3.9 or newer
- **Memory**: 4GB RAM recommended
- **Storage**: 200MB + space for your financial data
- **Language Support**: English UI with German bank statement processing

## Recent Updates

### March 2025 Update
- Fixed issues with pie chart rendering in the Dashboard tab
- Added additional data visualization options
- Improved compatibility with latest PySide6
- Enhanced dark mode support
- Fixed bugs related to database connection

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Adobe for design inspiration
- Icons from various free icon libraries
- PySide6 and Qt for the UI framework
- Python community for the excellent libraries

## Support & Contributions

Contributions, issues, and feature requests are welcome! Feel free to check the issues page or submit pull requests.

---

Developed with ❤️ by Madhva (madhva.fakare@uni-muenster.de)

## Usage

### Importing Transactions

1. Click the "Import" button in the toolbar
2. Select a PDF bank statement
3. Review the extracted transactions
4. Confirm import to add to the database

### Managing Transactions

- View all transactions in the Transactions tab
- Use batch selection mode to select multiple transactions
- Apply bulk operations like delete or change category
- Filter transactions by date, category, or type

### Financial Planning

- Set budget goals in the Budget tab
- Track your financial goals in the Goals tab
- Generate reports in the Reports tab
- View financial summary in the Dashboard tab

## Project Structure

- `src/` - Main source code directory
  - `ai/` - AI components for transaction processing
  - `database/` - Database management
  - `models/` - Data models
  - `ui/` - User interface components
    - `adobe_*.py` - Adobe-styled UI components
    - `design_system.py` - The Adobe-inspired design system
  - `main_pyside6.py` - Application entry point

## License

This project is licensed under the MIT License - see the LICENSE file for details.