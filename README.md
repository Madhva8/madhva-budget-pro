# Madhva Budget Pro

A comprehensive personal finance management application specifically designed for Sparkasse bank customers in Germany. Features automatic parsing of German Sparkasse bank statements with English translation, AI-powered transaction categorization, budgeting tools, and rich financial analytics, all within a beautiful Adobe-inspired interface optimized for readability and professional use.

<img src="logo.png" alt="Madhva Budget Pro Logo" width="120"/>

## Dashboard Showcase

The dashboard provides a comprehensive overview of your financial situation with interactive visualizations:

- Spending by Category (pie chart)
- Income vs Expenses comparison
- Monthly Trends with future forecasting
- Recent Transactions list

> **Specialized for Sparkasse Bank Customers**: Upload your Sparkasse PDF statements in German and view your finances in English with automatic translation and smart categorization

## New Adobe-Inspired Design System

Madhva Budget Pro has been completely redesigned with an Adobe-inspired design system that enhances readability, visual organization, and professional appearance. The new UI features:

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

Before installing Madhva Budget Pro, ensure you have:

- **Python 3.9 or newer** installed on your system
- **pip** (Python package manager)
- **Git** (for cloning the repository)

#### Installing Python

- **macOS**: Install using Homebrew: `brew install python`
- **Windows**: Download installer from [python.org](https://www.python.org/downloads/)
- **Linux**: Use your distribution's package manager (e.g., `apt install python3 python3-pip`)

### Step-by-Step Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourname/Financial_planner.git
   cd Financial_planner
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   
   # Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   # Use the provided script for all dependencies
   ./install_dependencies.sh
   
   # Or install manually
   pip install -r requirements.txt
   ```

4. **Setup the database**:
   ```bash
   # Use the sample database
   cp financial_planner.db.sample financial_planner.db
   
   # Or generate fresh sample data
   python create_sample_db.py
   
   # For pie chart testing
   python fix_pie_chart.py
   ```

### Running the Application

#### macOS Quick Launch

Use the provided command file for quick launch on macOS:

```bash
# Make the launcher executable (first time only)
chmod +x FinancialPlanner.command

# Run the application
./FinancialPlanner.command
```

#### Standard Launch

You can also run the application directly:

**macOS/Linux:**
```bash
python main_pyside6.py
```

**Windows:**
```bash
python main_pyside6.py
```

#### Docker Support

For containerized deployment:

```bash
# Run the application in Docker (first time may take a while to build)
./run-in-docker.sh
```

#### Standalone Build

Build a standalone executable (experimental):

```bash
# Create a standalone executable for your platform
./build-standalone.sh
```

For containerized deployment:

```bash
# Build and start the container
docker compose up -d

# Stop the container when done
docker compose down
```

### Troubleshooting

- **Missing dependencies**: Ensure all dependencies are installed with `pip install -r requirements.txt`
- **Python version issues**: Confirm you're using Python 3.9+ with `python --version`
- **Database errors**: Make sure you've created the database using the instructions above
- **Permission issues**: Ensure script files are executable (`chmod +x filename` on macOS/Linux)

### Authentication

The application features a secure login system to protect your financial data.
- For testing purposes, the sample database includes two accounts
- Security measures prevent bypassing the authentication system
- Contact the administrator for access credentials if needed

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

Developed with ❤️ by Madhva

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