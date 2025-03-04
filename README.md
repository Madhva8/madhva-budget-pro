# Madhva Budget Pro

A comprehensive personal finance management application specifically designed for Sparkasse bank customers in Germany. Features automatic parsing of German Sparkasse bank statements with English translation, AI-powered transaction categorization, budgeting tools, and rich financial analytics, all within a beautiful macOS-optimized interface.

<!-- Add screenshot here once available -->
<!-- ![Madhva Budget Pro](docs/screenshot.png) -->

> **Specialized for Sparkasse Bank Customers**: Upload your Sparkasse PDF statements in German and view your finances in English with automatic translation and smart categorization

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
- **macOS Native Look & Feel**: Beautiful interface optimized for macOS
- **Light/Dark Mode**: Support for system appearance preferences
- **Responsive Design**: Adapts to different window sizes
- **Intuitive Navigation**: User-friendly tab-based interface

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
   git clone https://github.com/yourusername/madhva-budget-pro.git
   cd madhva-budget-pro
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
   pip install -r requirements.txt
   ```

4. **Setup the database**:
   ```bash
   # Rename the sample database
   cp financial_planner.db.sample financial_planner.db
   
   # Or generate fresh sample data
   python create_sample_db.py
   mv sample_financial_planner.db financial_planner.db
   ```

### Running the Application

#### macOS

The simplest way to run on macOS:

```bash
# Make the launcher executable (first time only)
chmod +x direct_run.command

# Run the application
./direct_run.command
```

This launches the application with full security features enabled.

#### Windows

```bash
python src\main_pyside6.py
```

#### Linux

```bash
python3 src/main_pyside6.py
```

#### Docker Support

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
- **Frontend**: PySide6/PyQt5 for the UI
- **Backend**: Pure Python with SQLite database
- **Analytics**: Matplotlib for visualization, Pandas for data processing
- **PDF Processing**: PDFPlumber for Sparkasse statement parsing
- **Natural Language Processing**: AI modules for transaction categorization
- **Translation**: German to English translation for bank statements

### System Requirements
- **Operating Systems**: macOS 10.14+ (optimized), Windows 10+, Linux
- **Python**: 3.9 or newer
- **Memory**: 4GB RAM recommended
- **Storage**: 200MB + space for your financial data
- **Language Support**: English UI with German bank statement processing

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

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
  - `main_pyside6.py` - Application entry point

## License

This project is licensed under the MIT License - see the LICENSE file for details.