#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Financial Planner - Main Application Module

This is the main entry point for the Financial Planner application.
It integrates various components and initializes the main UI.
"""

import sys
import os
import logging
import traceback
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import Qt, QSettings

# Add current directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import UI components
try:
    from ui.main_window import MainWindow
except ImportError as e:
    print(f"Error importing UI components: {e}")
    print("Make sure PyQt5 is installed: pip install PyQt5")
    sys.exit(1)

# Import database module
try:
    from database.database_manager import DatabaseManager
except ImportError as e:
    print(f"Error importing database components: {e}")
    sys.exit(1)

# Import AI modules (conditional imports to handle missing dependencies)
try:
    from ai.sparkasse_parser import SparkasseParser
    SPARKASSE_PARSER_AVAILABLE = True
    print("SparkasseParser module successfully imported")
except ImportError as e:
    SPARKASSE_PARSER_AVAILABLE = False
    print(f"Error importing SparkasseParser: {e}")
    logging.warning("Sparkasse parser not available - bank statement import will be limited")

try:
    from ai.statement_translator import StatementTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError as e:
    TRANSLATOR_AVAILABLE = False
    print(f"Error importing StatementTranslator: {e}")
    logging.warning("Statement translator not available - German translations will be limited")

try:
    from ai.transaction_categories import TransactionCategorizer
    CATEGORIZER_AVAILABLE = True
except ImportError as e:
    CATEGORIZER_AVAILABLE = False
    print(f"Error importing TransactionCategorizer: {e}")
    logging.warning("Transaction categorizer not available - automatic categorization will be limited")

try:
    from ai.financial_ai import FinancialAI
    FINANCIAL_AI_AVAILABLE = True
except ImportError as e:
    FINANCIAL_AI_AVAILABLE = False
    print(f"Error importing FinancialAI: {e}")
    logging.warning("Financial AI not available - financial insights will be limited")

try:
    from ai.budget_optimizer import BudgetOptimizer
    BUDGET_OPTIMIZER_AVAILABLE = True
except ImportError as e:
    BUDGET_OPTIMIZER_AVAILABLE = False
    print(f"Error importing BudgetOptimizer: {e}")
    logging.warning("Budget optimizer not available - budget optimization will be limited")

try:
    from ai.investment_advisor import InvestmentAdvisor
    INVESTMENT_ADVISOR_AVAILABLE = True
except ImportError as e:
    INVESTMENT_ADVISOR_AVAILABLE = False
    print(f"Error importing InvestmentAdvisor: {e}")
    logging.warning("Investment advisor not available - investment advice will be limited")

# Check for optional dependencies
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
    print("pdfplumber module successfully imported")
except ImportError as e:
    PDFPLUMBER_AVAILABLE = False
    print(f"Error importing pdfplumber: {e}")
    logging.warning("pdfplumber not available - PDF statement import will be limited")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
    print("pandas module successfully imported")
except ImportError as e:
    PANDAS_AVAILABLE = False
    print(f"Error importing pandas: {e}")
    logging.warning("pandas not available - advanced analysis will be limited")

try:
    from PyQt5.QtChart import QChart
    QTCHART_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    QTCHART_AVAILABLE = False
    print(f"Error importing QChart: {e}")
    logging.warning("PyQtChart not available - charts will not be displayed")


def setup_logging():
    """Set up logging for the application."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create log directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "financial_planner.log")

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def initialize_ai_components(db_manager):
    """
    Initialize AI components of the application.

    Args:
        db_manager: Database manager instance

    Returns:
        Dictionary with initialized AI components
    """
    ai_components = {}

    # Initialize Sparkasse parser if available and pdfplumber is available
    if SPARKASSE_PARSER_AVAILABLE and PDFPLUMBER_AVAILABLE:
        try:
            ai_components['sparkasse_parser'] = SparkasseParser()
            print("Successfully initialized SparkasseParser")
            logging.info("SparkasseParser initialized successfully")
        except Exception as e:
            print(f"Error initializing SparkasseParser: {e}")
            logging.error(f"Error initializing SparkasseParser: {e}")
            traceback.print_exc()
    else:
        if SPARKASSE_PARSER_AVAILABLE and not PDFPLUMBER_AVAILABLE:
            print("SparkasseParser available but pdfplumber missing - PDF import will not work")
            logging.warning("SparkasseParser available but pdfplumber missing - PDF import will not work")

    # Initialize statement translator if available
    if TRANSLATOR_AVAILABLE:
        try:
            ai_components['translator'] = StatementTranslator()
            print("Successfully initialized StatementTranslator")
            logging.info("StatementTranslator initialized successfully")
        except Exception as e:
            print(f"Error initializing StatementTranslator: {e}")
            logging.error(f"Error initializing StatementTranslator: {e}")
            traceback.print_exc()

    # Initialize transaction categorizer if available
    if CATEGORIZER_AVAILABLE:
        try:
            ai_components['categorizer'] = TransactionCategorizer()
            print("Successfully initialized TransactionCategorizer")
            logging.info("TransactionCategorizer initialized successfully")
        except Exception as e:
            print(f"Error initializing TransactionCategorizer: {e}")
            logging.error(f"Error initializing TransactionCategorizer: {e}")
            traceback.print_exc()

    # Initialize financial AI if available
    if FINANCIAL_AI_AVAILABLE:
        try:
            ai_components['financial_ai'] = FinancialAI(db_manager)
            print("Successfully initialized FinancialAI")
            logging.info("FinancialAI initialized successfully")
        except Exception as e:
            print(f"Error initializing FinancialAI: {e}")
            logging.error(f"Error initializing FinancialAI: {e}")
            traceback.print_exc()

    # Initialize budget optimizer if available
    if BUDGET_OPTIMIZER_AVAILABLE:
        try:
            ai_components['budget_optimizer'] = BudgetOptimizer()
            print("Successfully initialized BudgetOptimizer")
            logging.info("BudgetOptimizer initialized successfully")
        except Exception as e:
            print(f"Error initializing BudgetOptimizer: {e}")
            logging.error(f"Error initializing BudgetOptimizer: {e}")
            traceback.print_exc()

    # Initialize investment advisor if available
    if INVESTMENT_ADVISOR_AVAILABLE:
        try:
            ai_components['investment_advisor'] = InvestmentAdvisor()
            print("Successfully initialized InvestmentAdvisor")
            logging.info("InvestmentAdvisor initialized successfully")
        except Exception as e:
            print(f"Error initializing InvestmentAdvisor: {e}")
            logging.error(f"Error initializing InvestmentAdvisor: {e}")
            traceback.print_exc()

    return ai_components


def main():
    """Main application entry point."""
    # Set up logging
    setup_logging()

    # Log application start
    logging.info("Starting Financial Planner application")

    # Create QApplication instance
    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("Financial Planner")
    app.setOrganizationName("FinPlanner")
    app.setApplicationVersion("1.0.0")

    try:
        # Initialize database manager
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "financial_planner.db")
        logging.info(f"Database path: {db_path}")
        db_manager = DatabaseManager(db_path)
        logging.info("Database manager initialized successfully")

        # Initialize AI components
        logging.info("Initializing AI components...")
        ai_components = initialize_ai_components(db_manager)

        # Check available AI capabilities
        if ai_components:
            logging.info(f"AI capabilities: {', '.join(ai_components.keys())}")
        else:
            logging.warning("No AI components available")

        # Create and show main window
        logging.info("Creating main window...")
        main_window = MainWindow(db_manager, ai_components)
        main_window.show()

        # Display first-run message if database was just created
        if not os.path.exists(db_path) or os.path.getsize(db_path) < 10000:
            QMessageBox.information(
                main_window,
                "Welcome to Financial Planner",
                "Welcome to Financial Planner!\n\n"
                "This application helps you manage your finances with a focus on "
                "international students in Germany.\n\n"
                "Start by adding your transactions or importing bank statements."
            )
        else:
            # Show a message about the new duplicate detection feature
            QMessageBox.information(
                main_window,
                "New Features Available",
                "New features are now available:\n\n"
                "• Duplicate Transaction Detection: Find and remove duplicate transactions\n"
                "• Delete Button: Easily delete transactions from the list\n"
                "• Enhanced Categorization: Better auto-categorization of bank transactions\n\n"
                "Right-click on any transaction to access these features."
            )

        # Start event loop
        logging.info("Starting event loop...")
        sys.exit(app.exec_())

    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        QMessageBox.critical(
            None,
            "Error Starting Application",
            f"An error occurred while starting the application:\n\n{str(e)}\n\n"
            "Check the logs for more information."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()