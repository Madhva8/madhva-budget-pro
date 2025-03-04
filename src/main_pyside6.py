#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Financial Planner - PySide6 Version

A desktop application for managing personal finances with a modern macOS interface.
Uses PySide6 for better macOS integration.
"""

import os
import sys
import logging
import datetime
from typing import Dict, Any, Optional, List, Set

from PySide6.QtWidgets import QApplication, QSplashScreen
from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QPixmap, QFont

# Import from the project modules
from database.database_manager import DatabaseManager
# Import the Modern UI components directly (not through ui.__init__)
from ui.modern_main_window import ModernMainWindow
# Will add more imports as needed

# Configure logging
def setup_logging():
    """Configure the logging system."""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_file = os.path.join(log_dir, "financial_planner.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create a root logger
    logger = logging.getLogger()
    return logger

def initialize_ai_components(db_manager=None):
    """
    Initialize AI components for the application with enhanced model training.
    
    Args:
        db_manager: Database manager instance for AI model training
        
    Returns:
        Dictionary of initialized AI components
    """
    ai_components = {}
    
    # Try to import and initialize SparkasseParser
    try:
        from ai.sparkasse_parser import SparkasseParser
        print("SparkasseParser module successfully imported")
        
        # Check for pdfplumber and pandas dependencies
        try:
            import pdfplumber
            print("pdfplumber module successfully imported")
        except ImportError:
            print("Failed to import pdfplumber - PDF parsing will not be available")
            
        try:
            import pandas
            print("pandas module successfully imported")
        except ImportError:
            print("Failed to import pandas - data analysis will be limited")
            
        # Try to initialize the SparkasseParser
        try:
            sparkasse_parser = SparkasseParser()
            ai_components['sparkasse_parser'] = sparkasse_parser
            print("Successfully initialized SparkasseParser")
        except Exception as e:
            logging.error(f"Failed to initialize SparkasseParser: {e}")
    except ImportError:
        logging.warning("SparkasseParser module not available")
        
    # Import and initialize more AI components as needed
    try:
        from ai.statement_translator import StatementTranslator
        translator = StatementTranslator()
        ai_components['statement_translator'] = translator
        print("Successfully initialized StatementTranslator")
    except ImportError:
        logging.warning("StatementTranslator module not available")
    
    # Initialize transaction categorizer with potential training data
    try:
        from ai.transaction_categories import TransactionCategorizer
        categorizer = TransactionCategorizer()
        ai_components['transaction_categorizer'] = categorizer
        print("Successfully initialized TransactionCategorizer")
        
        # Attempt to load previous training data
        if hasattr(categorizer, '_load_training_data'):
            try:
                categorizer._load_training_data()
                print("Loaded transaction categorizer training data")
            except Exception as e:
                logging.warning(f"Failed to load transaction categorizer training data: {e}")
    except ImportError:
        logging.warning("TransactionCategorizer module not available")
    
    # Initialize Financial AI with database manager for training/history
    try:
        from ai.financial_ai import FinancialAI
        financial_ai = FinancialAI(db_manager)
        ai_components['financial_ai'] = financial_ai
        print("Successfully initialized FinancialAI")
        
        # Attempt auto-training if manager available
        if db_manager:
            try:
                from models.transaction import Transaction
                training_transactions = Transaction.get_training_transactions()
                
                if training_transactions:
                    # Convert to format expected by train_from_transactions
                    tx_dicts = [tx.to_dict() for tx in training_transactions]
                    financial_ai.train_from_transactions(tx_dicts)
                    print(f"Auto-trained FinancialAI with {len(tx_dicts)} historical transactions")
                else:
                    print("No training transactions found - AI will use default models")
            except Exception as e:
                logging.warning(f"Auto-training failed: {e}")
    except ImportError:
        logging.warning("FinancialAI module not available")
        
    # Budget optimizer with enhanced AI and ML capabilities
    try:
        from ai.budget_optimizer import BudgetOptimizer
        budget_optimizer = BudgetOptimizer(db_manager)  # Pass DB manager for model persistence
        ai_components['budget_optimizer'] = budget_optimizer
        print("Successfully initialized BudgetOptimizer with ML capabilities")
    except ImportError:
        logging.warning("BudgetOptimizer module not available")
        
    # Investment advisor with enhanced ML capabilities  
    try:
        from ai.investment_advisor import InvestmentAdvisor
        investment_advisor = InvestmentAdvisor()
        ai_components['investment_advisor'] = investment_advisor
        print("Successfully initialized InvestmentAdvisor")
    except ImportError:
        logging.warning("InvestmentAdvisor module not available")
        
    # Initialize NLP libraries if available for multiple components
    try:
        import nltk
        # Check if required NLTK data is downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading required NLTK data for text processing...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            print("NLTK data downloaded successfully")
            
        print("Successfully initialized NLP libraries for advanced text processing")
    except ImportError:
        logging.warning("NLTK not available - some advanced text processing features will be limited")
        
    return ai_components

def main():
    """Run the main application."""
    # Set up logging
    logger = setup_logging()
    
    try:
        # Create the Qt Application
        app = QApplication(sys.argv)
        
        # Set application name and organization for settings
        app.setApplicationName("Madhva Budget Pro")
        app.setOrganizationName("Madhva Finance")
        
        # Set default font - cross-platform
        default_font = QFont()
        
        # Use system-appropriate font based on platform
        if sys.platform == "darwin":  # macOS
            default_font.setFamily("SF Pro Text")
        elif sys.platform == "win32":  # Windows
            default_font.setFamily("Segoe UI")
        else:  # Linux and others
            default_font.setFamily("Noto Sans")
            
        default_font.setPointSize(13)
        app.setFont(default_font)
        
        # Initialize the database
        db_manager = DatabaseManager('financial_planner.db')
        
        # Check if login is required
        login_required = db_manager.get_setting("login_required", "1") == "1"
        current_user = None
        
        # Skip login for debugging if environment variable is set
        if os.environ.get("SKIP_LOGIN") == "1":
            current_user = "admin"
            logger.info("Login skipped due to SKIP_LOGIN environment variable. Using admin user.")
        elif login_required:
            # Import the login dialog
            from ui.login_dialog import ModernLoginDialog
            
            # Show login dialog
            try:
                login_dialog = ModernLoginDialog(db_manager)
                login_result = login_dialog.exec()
                
                if login_result != QDialog.DialogCode.Accepted:
                    logger.info("Login cancelled or failed. Exiting application.")
                    sys.exit(0)
                    
                # Get the logged-in username
                current_user = login_dialog.username_edit.text()
                logger.info(f"User '{current_user}' logged in successfully")
            except Exception as e:
                logger.error(f"Error during login: {e}", exc_info=True)
                # Use admin as fallback
                current_user = "admin"
                logger.info("Using admin user as fallback due to login error.")
        
        # Initialize AI components with database manager for training
        ai_components = initialize_ai_components(db_manager)
        
        # Add subscription analyzer to AI components
        try:
            from ai.subscription_analyzer import SubscriptionAnalyzer
            subscription_analyzer = SubscriptionAnalyzer()
            ai_components['subscription_analyzer'] = subscription_analyzer
            print(f"Successfully initialized SubscriptionAnalyzer and added to AI components: {list(ai_components.keys())}")
            logging.info(f"AI components after adding SubscriptionAnalyzer: {list(ai_components.keys())}")
            
            # Connect budget optimizer with subscription analyzer if both are available
            if 'budget_optimizer' in ai_components and 'subscription_analyzer' in ai_components:
                # Enable integration between components
                ai_components['budget_optimizer'].user_preferences['integrations'] = {
                    'subscription_analyzer': subscription_analyzer
                }
                print("Connected BudgetOptimizer with SubscriptionAnalyzer for enhanced functionality")
        except ImportError:
            logging.warning("SubscriptionAnalyzer module not available - subscription analysis will be limited")
        except Exception as e:
            logging.error(f"Error initializing SubscriptionAnalyzer: {e}", exc_info=True)
            
        # Log startup status
        ai_status = {
            "total_components": len(ai_components),
            "nlp_available": any(hasattr(comp, 'nlp_enabled') and comp.nlp_enabled for comp in ai_components.values()),
            "ml_available": any(hasattr(comp, 'ml_enabled') and comp.ml_enabled for comp in ai_components.values()),
            "trained_models": any(hasattr(comp, 'is_trained') and comp.is_trained for comp in ai_components.values()),
        }
        print(f"AI System Status: {ai_status}")
        logging.info(f"AI System Status: {ai_status}")
        
        # Create and show the main window with current user
        main_window = ModernMainWindow(db_manager, ai_components, current_user)
        main_window.show()
        
        # Start the application
        sys.exit(app.exec())
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"An error occured while starting an application {e} Check the logs for more information")
        sys.exit(1)

if __name__ == "__main__":
    main()