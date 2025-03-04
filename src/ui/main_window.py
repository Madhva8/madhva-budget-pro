#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main Window Module

This module defines the main application window for the Financial Planner.
"""

import os
import sys
import logging
import shutil
import datetime
from typing import Dict, Any, Optional
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTabWidget, QStackedWidget, QSplitter, QFrame, QToolBar, QAction,
    QMenu, QStatusBar, QFileDialog, QMessageBox, QSizePolicy, QGraphicsDropShadowEffect,
    QComboBox, QCheckBox
)
from PyQt5.QtCore import Qt, QSize, QSettings
from PyQt5.QtGui import QIcon, QFont, QPixmap, QColor

# Import UI components
from ui.transactions_tab import TransactionsTab
from ui.dashboard_tab import DashboardTab
from ui.add_transaction_dialog import AddTransactionDialog
from ui.styled_button import StyledButton

# Import AI components (if available)
try:
    from ai.sparkasse_parser import SparkasseParser

    SPARKASSE_PARSER_AVAILABLE = True
except ImportError:
    SPARKASSE_PARSER_AVAILABLE = False
    logging.warning("Failed to import SparkasseParser in main_window.py")


class MainWindow(QMainWindow):
    """Main window for the Financial Planner application."""

    def __init__(self, db_manager, ai_components=None):
        """
        Initialize the main window.

        Args:
            db_manager: Database manager instance
            ai_components: Dictionary with AI component instances
        """
        super().__init__()

        # Store references to managers and components
        self.db_manager = db_manager
        self.ai_components = ai_components or {}
        self.logger = logging.getLogger(__name__)

        # Debug: Log available AI components
        self.logger.info(f"Initializing MainWindow with AI components: {list(self.ai_components.keys())}")

        # Initialize UI
        self.init_ui()

        # Load settings
        self.load_settings()

    def init_ui(self):
        """Initialize the user interface."""
        # Set window properties
        self.setWindowTitle("Financial Planner")
        self.setMinimumSize(1000, 700)
        
        # Set application font - use system default font for better compatibility
        font = QFont()
        font.setPointSize(13)
        self.setFont(font)

        # Create central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Main layout with margins for floating appearance
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(15, 15, 15, 15)
        self.main_layout.setSpacing(10)

        # Create toolbar
        self.create_toolbar()

        # Create tab widget with modern appearance
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)  # More modern look
        self.tab_widget.setTabPosition(QTabWidget.North)
        self.tab_widget.setMovable(True)  # Allow reordering tabs
        
        # No shadow effect for better compatibility
        self.main_layout.addWidget(self.tab_widget)

        # Create tabs
        self.create_dashboard_tab()
        self.create_transactions_tab()
        self.create_budget_tab()
        self.create_reports_tab()
        self.create_goals_tab()
        self.create_settings_tab()

        # Create status bar with modern styling
        status_bar = QStatusBar()
        status_bar.setSizeGripEnabled(False)  # Remove size grip for cleaner look
        self.setStatusBar(status_bar)
        self.statusBar().showMessage("Ready")

        # Set up event handlers
        self.setup_event_handlers()

    def create_toolbar(self):
        """Create the application toolbar with macOS-inspired styling."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(20, 20))
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        
        # Add spacing at the beginning for macOS-like appearance
        spacer = QWidget()
        spacer.setFixedWidth(8)
        toolbar.addWidget(spacer)
        
        self.addToolBar(toolbar)

        # Create macOS-style toolbar buttons
        
        # Add Transaction
        add_action = QAction(QIcon.fromTheme("list-add"), "Add", self)
        add_action.triggered.connect(self.show_add_transaction_dialog)
        add_action.setToolTip("Add a new transaction")
        toolbar.addAction(add_action)

        # Add small space
        spacer1 = QWidget()
        spacer1.setFixedWidth(12)
        toolbar.addWidget(spacer1)

        # Import
        import_action = QAction(QIcon.fromTheme("document-open"), "Import", self)
        import_action.triggered.connect(self.show_import_dialog)
        import_action.setToolTip("Import transactions from file")
        toolbar.addAction(import_action)

        # Export
        export_action = QAction(QIcon.fromTheme("document-save"), "Export", self)
        export_action.triggered.connect(self.show_export_dialog)
        export_action.setToolTip("Export transactions to file")
        toolbar.addAction(export_action)

        # Add flexible spacer to push the theme toggle to the right
        flexible_spacer = QWidget()
        flexible_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(flexible_spacer)

        # Theme Toggle - on the right side like macOS controls
        self.theme_action = QAction(QIcon.fromTheme("weather-clear-night"), "Theme", self)
        self.theme_action.triggered.connect(self.toggle_theme)
        self.theme_action.setToolTip("Toggle between light and dark mode")
        toolbar.addAction(self.theme_action)
        
        # Add spacing at the end for macOS-like appearance
        end_spacer = QWidget()
        end_spacer.setFixedWidth(8)
        toolbar.addWidget(end_spacer)

    def create_dashboard_tab(self):
        """Create the dashboard tab."""
        self.dashboard_tab = DashboardTab(self.db_manager)
        self.tab_widget.addTab(self.dashboard_tab, "Dashboard")

    def create_transactions_tab(self):
        """Create the transactions tab."""
        self.transactions_tab = TransactionsTab(self.db_manager)
        self.tab_widget.addTab(self.transactions_tab, "Transactions")

    def create_budget_tab(self):
        """Create the budget tab."""
        budget_widget = QWidget()
        budget_layout = QVBoxLayout(budget_widget)

        # Placeholder for budget content
        budget_layout.addWidget(QLabel("Budget - Coming Soon"))

        self.tab_widget.addTab(budget_widget, "Budget")

    def create_reports_tab(self):
        """Create the reports tab."""
        reports_widget = QWidget()
        reports_layout = QVBoxLayout(reports_widget)

        # Placeholder for reports content
        reports_layout.addWidget(QLabel("Reports - Coming Soon"))

        self.tab_widget.addTab(reports_widget, "Reports")

    def create_goals_tab(self):
        """Create the goals tab."""
        goals_widget = QWidget()
        goals_layout = QVBoxLayout(goals_widget)

        # Placeholder for goals content
        goals_layout.addWidget(QLabel("Financial Goals - Coming Soon"))

        self.tab_widget.addTab(goals_widget, "Goals")

    def create_settings_tab(self):
        """Create the settings tab with enhanced UI."""
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        settings_layout.setSpacing(15)
        settings_layout.setContentsMargins(20, 20, 20, 20)
        
        # App Settings heading
        app_settings_label = QLabel("Application Settings")
        app_settings_label.setFont(QFont("", 16, QFont.Bold))
        settings_layout.addWidget(app_settings_label)
        
        # Theme setting
        theme_layout = QHBoxLayout()
        theme_label = QLabel("Theme:")
        theme_label.setFixedWidth(150)
        self.theme_toggle = QPushButton("Toggle Theme")
        self.theme_toggle.clicked.connect(self.toggle_theme)
        theme_layout.addWidget(theme_label)
        theme_layout.addWidget(self.theme_toggle)
        theme_layout.addStretch()
        settings_layout.addLayout(theme_layout)
        
        # Currency setting
        currency_layout = QHBoxLayout()
        currency_label = QLabel("Currency:")
        currency_label.setFixedWidth(150)
        self.currency_combo = QComboBox()
        self.currency_combo.addItems(["EUR", "USD", "GBP", "CHF", "JPY"])
        self.currency_combo.setCurrentText(self.db_manager.get_setting("currency", "EUR"))
        self.currency_combo.currentTextChanged.connect(lambda text: self.db_manager.update_setting("currency", text))
        currency_layout.addWidget(currency_label)
        currency_layout.addWidget(self.currency_combo)
        currency_layout.addStretch()
        settings_layout.addLayout(currency_layout)
        
        # Date format setting
        date_layout = QHBoxLayout()
        date_label = QLabel("Date Format:")
        date_label.setFixedWidth(150)
        self.date_format_combo = QComboBox()
        self.date_format_combo.addItems(["dd/MM/yyyy", "MM/dd/yyyy", "yyyy-MM-dd"])
        self.date_format_combo.setCurrentText(self.db_manager.get_setting("date_format", "dd/MM/yyyy"))
        self.date_format_combo.currentTextChanged.connect(lambda text: self.db_manager.update_setting("date_format", text))
        date_layout.addWidget(date_label)
        date_layout.addWidget(self.date_format_combo)
        date_layout.addStretch()
        settings_layout.addLayout(date_layout)
        
        # Add a separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        settings_layout.addWidget(line)
        
        # Data Management heading
        data_label = QLabel("Data Management")
        data_label.setFont(QFont("", 16, QFont.Bold))
        settings_layout.addWidget(data_label)
        
        # Backup database
        backup_layout = QHBoxLayout()
        backup_label = QLabel("Database Backup:")
        backup_label.setFixedWidth(150)
        self.backup_button = StyledButton("Create Backup")
        self.backup_button.clicked.connect(self.create_database_backup)
        backup_layout.addWidget(backup_label)
        backup_layout.addWidget(self.backup_button)
        backup_layout.addStretch()
        settings_layout.addLayout(backup_layout)
        
        # Backup frequency
        backup_freq_layout = QHBoxLayout()
        backup_freq_label = QLabel("Backup Frequency:")
        backup_freq_label.setFixedWidth(150)
        self.backup_freq_combo = QComboBox()
        self.backup_freq_combo.addItems(["Manual", "Daily", "Weekly", "Monthly"])
        current_freq = self.db_manager.get_setting("backup_frequency", "weekly").capitalize()
        if current_freq == "Weekly":
            self.backup_freq_combo.setCurrentText("Weekly")
        else:
            self.backup_freq_combo.setCurrentIndex(0)
        self.backup_freq_combo.currentTextChanged.connect(
            lambda text: self.db_manager.update_setting("backup_frequency", text.lower())
        )
        backup_freq_layout.addWidget(backup_freq_label)
        backup_freq_layout.addWidget(self.backup_freq_combo)
        backup_freq_layout.addStretch()
        settings_layout.addLayout(backup_freq_layout)
        
        # Add a separator
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setFrameShadow(QFrame.Sunken)
        settings_layout.addWidget(line2)
        
        # Features heading
        features_label = QLabel("Features")
        features_label.setFont(QFont("", 16, QFont.Bold))
        settings_layout.addWidget(features_label)
        
        # AI Assistant toggle
        ai_layout = QHBoxLayout()
        ai_label = QLabel("AI Assistant:")
        ai_label.setFixedWidth(150)
        self.ai_checkbox = QCheckBox("Enable AI features")
        self.ai_checkbox.setChecked(self.db_manager.get_setting("ai_assistant_enabled", "1") == "1")
        self.ai_checkbox.stateChanged.connect(
            lambda state: self.db_manager.update_setting("ai_assistant_enabled", "1" if state else "0")
        )
        ai_layout.addWidget(ai_label)
        ai_layout.addWidget(self.ai_checkbox)
        ai_layout.addStretch()
        settings_layout.addLayout(ai_layout)
        
        # Add a separator
        line3 = QFrame()
        line3.setFrameShape(QFrame.HLine)
        line3.setFrameShadow(QFrame.Sunken)
        settings_layout.addWidget(line3)
        
        # About section
        about_label = QLabel("About")
        about_label.setFont(QFont("", 16, QFont.Bold))
        settings_layout.addWidget(about_label)
        
        about_text = QLabel("Financial Planner v1.0.0\n© 2024 All rights reserved")
        settings_layout.addWidget(about_text)
        
        # Add stretch to push everything to the top
        settings_layout.addStretch()
        
        self.tab_widget.addTab(settings_widget, "Settings")
        
    def create_database_backup(self):
        """Create a backup of the database file."""
        try:
            # Get the database path
            db_path = self.db_manager.db_path
            
            # Check if the database file exists
            if not os.path.exists(db_path):
                QMessageBox.warning(
                    self, "Backup Error",
                    "Database file not found."
                )
                return
                
            # Create backup directory if it doesn't exist
            backup_dir = os.path.join(os.path.dirname(db_path), "backups")
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
                
            # Create backup filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            db_name = os.path.basename(db_path)
            backup_path = os.path.join(backup_dir, f"{os.path.splitext(db_name)[0]}_{timestamp}.db")
            
            # Copy the database file
            shutil.copy2(db_path, backup_path)
            
            # Show success message
            QMessageBox.information(
                self, "Backup Created",
                f"Database backup created successfully at:\n{backup_path}"
            )
            
            # Log backup creation
            self.logger.info(f"Database backup created at: {backup_path}")
            
            # Update status bar
            self.statusBar().showMessage(f"Database backup created at: {backup_path}", 5000)
            
        except Exception as e:
            # Show error message
            QMessageBox.critical(
                self, "Backup Error",
                f"Failed to create database backup: {str(e)}"
            )
            
            # Log error
            self.logger.error(f"Failed to create database backup: {e}", exc_info=True)

    def setup_event_handlers(self):
        """Set up event handlers."""
        # Tab change handler
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

    def on_tab_changed(self, index):
        """
        Handle tab change event.

        Args:
            index: New tab index
        """
        # Refresh data when switching to certain tabs
        if index == 0:  # Dashboard tab
            self.dashboard_tab.refresh_dashboard()
        elif index == 1:  # Transactions tab
            self.transactions_tab.load_transactions()

    def show_add_transaction_dialog(self):
        """Show the add transaction dialog."""
        # Debug: Get transaction count before
        count_before = len(self.db_manager.get_transactions())
        self.logger.info(f"Transaction count before adding: {count_before}")

        dialog = AddTransactionDialog(self.db_manager)
        if dialog.exec_():
            # Refresh tabs based on current tab
            current_tab = self.tab_widget.currentIndex()
            if current_tab == 0:
                self.dashboard_tab.refresh_dashboard()
            elif current_tab == 1:
                self.transactions_tab.load_transactions()

            # Debug: Get transaction count after
            count_after = len(self.db_manager.get_transactions())
            self.logger.info(f"Transaction count after adding: {count_after}")

            self.statusBar().showMessage("Transaction added successfully", 3000)

    def show_import_dialog(self):
        """Show the import dialog."""
        # Debug: Check available AI components
        self.logger.info(f"Available AI components: {list(self.ai_components.keys())}")
        
        # Check if PDF import is available first
        pdf_support = 'sparkasse_parser' in self.ai_components
        
        # Create file filter based on available importers
        if pdf_support:
            file_filter = "CSV Files (*.csv);;PDF Files (*.pdf);;All Files (*)"
        else:
            file_filter = "CSV Files (*.csv);;All Files (*)"
            
        print(f"PDF support available: {pdf_support}")
        print(f"Using file filter: {file_filter}")

        options = QFileDialog.Options()
        file_path, selected_filter = QFileDialog.getOpenFileName(
            self, "Import Transactions", "",
            file_filter,
            options=options
        )
        
        print(f"Selected file: {file_path}")
        print(f"Selected filter: {selected_filter}")

        if file_path:
            # First check what the file actually is (not just the extension)
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(20)  # Read first 20 bytes to check file type
                    
                    # Check if it's a PDF (starts with %PDF)
                    if header.startswith(b'%PDF'):
                        file_type = 'pdf'
                    elif b',' in header and (b'\n' in header or b'\r' in header) and not header.startswith(b'%'):
                        # Simple CSV detection - has commas and line breaks
                        file_type = 'csv'  
                    else:
                        # Fall back to extension
                        file_ext = os.path.splitext(file_path)[1].lower()
                        file_type = file_ext.lstrip('.')
            except Exception as e:
                self.logger.error(f"Error detecting file type: {e}")
                # Fall back to extension
                file_ext = os.path.splitext(file_path)[1].lower()
                file_type = file_ext.lstrip('.')
            
            print(f"Detected file type: {file_type}")
            
            if file_type == 'csv':
                self.import_csv(file_path)
            elif file_type == 'pdf' and pdf_support:
                self.import_pdf(file_path)
            else:
                QMessageBox.warning(
                    self, "Unsupported File",
                    f"The file type '{file_type}' is not supported for import or the required parser is not available."
                )

    def import_csv(self, file_path):
        """
        Import transactions from a CSV file.

        Args:
            file_path: Path to the CSV file
        """
        try:
            import csv
            import pandas as pd
            
            # First check if this is our own exported CSV format
            try:
                # Try to read with pandas to better handle different CSV formats
                df = pd.read_csv(file_path)
                
                # Check if this is our exported format (has expected columns)
                our_format = all(col in df.columns for col in 
                                ['date', 'description', 'amount', 'is_income', 'category'])
                
                self.logger.info(f"Detected CSV format: {'our format' if our_format else 'generic format'}")
                print(f"CSV has columns: {list(df.columns)}")
                
                if our_format:
                    # This is our exported format, use the enhanced import
                    return self._import_csv_enhanced(df)
            except Exception as e:
                self.logger.warning(f"Could not analyze CSV with pandas: {e}")
                our_format = False
            
            # Fall back to basic CSV import if not our format
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)  # Skip header

                count = 0
                for row in reader:
                    if len(row) >= 3:
                        date = row[0]
                        description = row[1]
                        amount = float(row[2])

                        # Add to database (simplified)
                        transaction_id = self.db_manager.add_transaction(
                            date=date,
                            amount=abs(amount),
                            description=description,
                            category_id=1,  # Default category
                            is_income=amount > 0
                        )
                        self.logger.info(f"Added transaction with ID: {transaction_id}")
                        count += 1

            # Refresh tabs based on current tab
            current_tab = self.tab_widget.currentIndex() 
            if current_tab == 0:
                self.dashboard_tab.refresh_dashboard()
            elif current_tab == 1:
                self.transactions_tab.load_transactions()

            self.statusBar().showMessage(f"Imported {count} transactions successfully", 3000)
            QMessageBox.information(
                self, "Import Successful",
                f"Successfully imported {count} transactions."
            )

        except Exception as e:
            self.logger.error(f"Error importing CSV: {e}")
            QMessageBox.critical(
                self, "Import Error",
                f"An error occurred while importing: {str(e)}"
            )
            
    def _import_csv_enhanced(self, df):
        """
        Import transactions from a DataFrame in our CSV format.
        
        Args:
            df: Pandas DataFrame with transaction data
        """
        try:
            self.logger.info(f"Importing {len(df)} transactions from enhanced CSV")
            
            # Filter out balance entries and any rows without essential data
            df = df[~df['description'].str.contains('Kontostand', na=False)]
            df = df.dropna(subset=['date', 'amount', 'description'])
            
            self.logger.info(f"After filtering: {len(df)} transactions to import")
            
            # Process each transaction
            count = 0
            for _, row in df.iterrows():
                # Determine category
                category_id = 1  # Default category
                
                # Try to match category name if it exists
                if 'category' in row and row['category'] and row['category'] != 'Uncategorized':
                    # Get existing categories from database
                    categories = self.db_manager.get_categories()
                    for cat in categories:
                        if cat['name'].lower() == row['category'].lower():
                            category_id = cat['id']
                            break
                            
                    # If category doesn't exist, create it
                    if category_id == 1 and row['category'] != 'Uncategorized':
                        category_id = self.db_manager.add_category(str(row['category']))
                
                # Handle booleans properly
                is_income = False
                if 'is_income' in row:
                    if isinstance(row['is_income'], bool):
                        is_income = row['is_income']
                    elif isinstance(row['is_income'], str):
                        is_income = row['is_income'].lower() in ['true', '1', 'yes']
                
                # Determine other parameters
                merchant = row.get('merchant', '') if 'merchant' in row else ''
                notes = row.get('notes', '') if 'notes' in row else ''
                        
                # Add to database
                transaction_id = self.db_manager.add_transaction(
                    date=row['date'],
                    amount=abs(float(row['amount'])),
                    description=row['description'],
                    category_id=category_id,
                    is_income=is_income,
                    merchant=merchant,
                    notes=notes
                )
                
                if transaction_id:
                    self.logger.info(f"Added transaction with ID: {transaction_id}")
                    count += 1
                else:
                    self.logger.warning(f"Failed to add transaction: {row['description']}")
            
            # Refresh tabs based on current tab
            current_tab = self.tab_widget.currentIndex() 
            if current_tab == 0:
                self.dashboard_tab.refresh_dashboard()
            elif current_tab == 1:
                self.transactions_tab.load_transactions()

            self.statusBar().showMessage(f"Imported {count} transactions successfully", 3000)
            QMessageBox.information(
                self, "Import Successful",
                f"Successfully imported {count} transactions."
            )
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error importing enhanced CSV: {e}")
            QMessageBox.critical(
                self, "Import Error",
                f"An error occurred while importing: {str(e)}"
            )
            return False

    def import_pdf(self, file_path):
        """
        Import transactions from a PDF file.

        Args:
            file_path: Path to the PDF file
        """
        # Debug: More detailed logging for PDF import
        self.logger.info(f"Attempting to import PDF: {file_path}")
        self.logger.info(f"AI components available: {list(self.ai_components.keys())}")
        
        # Print components to console for debugging
        print(f"PDF import requested for: {file_path}")
        print(f"AI components available: {list(self.ai_components.keys())}")
        
        # Check if the file exists and can be read
        if not os.path.exists(file_path):
            self.logger.error(f"PDF file does not exist: {file_path}")
            QMessageBox.critical(
                self, "File Error",
                "The selected file does not exist or cannot be accessed."
            )
            return
            
        # Make sure it's actually a PDF file by checking content, not just extension
        try:
            with open(file_path, 'rb') as f:
                header = f.read(4)
                if header != b'%PDF':
                    self.logger.error(f"File is not a valid PDF: {file_path}")
                    QMessageBox.warning(
                        self, "Invalid PDF",
                        "The selected file is not a valid PDF file."
                    )
                    return
        except Exception as e:
            self.logger.error(f"Error checking PDF file: {e}")
            QMessageBox.critical(
                self, "File Error",
                f"Could not read the selected file: {str(e)}"
            )
            return

        # Check if Sparkasse parser is available
        if 'sparkasse_parser' in self.ai_components:
            self.statusBar().showMessage("Parsing PDF statement...", 3000)
            self.logger.info("Sparkasse parser found, proceeding with import")

            try:
                # Get the parser
                parser = self.ai_components['sparkasse_parser']

                # Parse the PDF
                transactions = parser.parse_pdf(file_path)
                self.logger.info(f"Parsed {len(transactions)} transactions from PDF")
                print(f"Parsed transactions: {len(transactions)}")
                
                # Debug first few transactions
                for i, tx in enumerate(transactions[:5]):
                    print(f"Transaction {i}: Date={tx.get('date', 'N/A')}, Description={tx.get('description', 'N/A')[:30]}..., Amount={tx.get('amount', 0)}")

                if not transactions:
                    QMessageBox.warning(
                        self, "No Transactions Found",
                        "No transactions could be extracted from the PDF. This might not be a supported bank statement format."
                    )
                    return
                
                # Filter out any initial balance entries
                transactions = [tx for tx in transactions if tx.get('category', '') != 'Initial Balance' and 'Kontostand' not in tx.get('description', '')]
                self.logger.info(f"After filtering initial balances: {len(transactions)} transactions")
                print(f"After filtering: {len(transactions)} transactions")

                # Import the transactions
                count = 0
                for tx in transactions:
                    # Skip entries that look like balance statements rather than transactions
                    if 'Kontostand' in tx.get('description', '') or tx.get('category', '') == 'Initial Balance':
                        continue
                        
                    # Determine category
                    category_id = 1  # Default category
                    
                    # Try to derive category ID from transaction category string
                    tx_category = tx.get('category', '')
                    if tx_category:
                        # Get existing categories from database
                        categories = self.db_manager.get_categories()
                        for cat in categories:
                            if cat['name'].lower() == tx_category.lower():
                                category_id = cat['id']
                                break
                                
                        # If category doesn't exist, create it
                        if category_id == 1 and tx_category != 'Uncategorized':
                            category_id = self.db_manager.add_category(tx_category)
                    
                    try:
                        # Get transaction details
                        date = tx.get('date', '')
                        description = tx.get('description', '')
                        amount = tx.get('amount', 0)
                        is_income = tx.get('is_income', False)
                        merchant = tx.get('merchant', '')
                        
                        # Skip if any required field is missing
                        if not date or not description or amount == 0:
                            self.logger.warning(f"Skipping transaction with missing data: {tx}")
                            continue
                            
                        # Format truncated description if needed
                        if len(description) > 100:
                            description = description[:100] + "..."
                        
                        # Add to database
                        transaction_id = self.db_manager.add_transaction(
                            date=date,
                            amount=amount,
                            description=description,
                            category_id=category_id,
                            is_income=is_income,
                            merchant=merchant
                        )
                        
                        if transaction_id:
                            self.logger.info(f"Added transaction with ID: {transaction_id} from PDF")
                            count += 1
                        else:
                            self.logger.warning(f"Failed to add transaction: {tx}")
                    except Exception as e:
                        self.logger.error(f"Error adding transaction: {e}", exc_info=True)
                        continue

                # Refresh tabs based on current tab
                current_tab = self.tab_widget.currentIndex()
                if current_tab == 0:
                    self.dashboard_tab.refresh_dashboard()
                elif current_tab == 1:
                    self.transactions_tab.load_transactions()

                # Show summary
                if count > 0:
                    self.statusBar().showMessage(f"Imported {count} transactions from PDF", 5000)
                    QMessageBox.information(
                        self, "Import Successful",
                        f"Successfully imported {count} transactions from the PDF statement."
                    )
                else:
                    self.statusBar().showMessage("No transactions were imported", 5000)
                    QMessageBox.warning(
                        self, "Import Issue",
                        "No transactions could be imported from the PDF."
                    )

            except Exception as e:
                self.logger.error(f"Error importing PDF: {e}", exc_info=True)
                QMessageBox.critical(
                    self, "Import Error",
                    f"An error occurred while importing: {str(e)}"
                )
        else:
            self.logger.warning("PDF import failed: Sparkasse parser not available")
            QMessageBox.warning(
                self, "Feature Not Available",
                "PDF statement parsing is not available.\n\nPlease install the required module with:\npip install pdfplumber"
            )

    def show_export_dialog(self):
        """Show the export dialog."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Transactions", "",
            "CSV Files (*.csv);;All Files (*)",
            options=options
        )

        if file_path:
            # Add .csv extension if not present
            if not file_path.endswith('.csv'):
                file_path += '.csv'

            # Get all transactions
            transactions = self.db_manager.get_transactions()

            # Export to CSV
            try:
                import csv

                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)

                    # Write header
                    writer.writerow([
                        'Date', 'Description', 'Amount', 'Category',
                        'Type', 'Notes', 'Merchant'
                    ])

                    # Write data
                    for tx in transactions:
                        writer.writerow([
                            tx.get('date', ''),
                            tx.get('description', ''),
                            tx.get('amount', 0),
                            tx.get('category_name', ''),
                            'Income' if tx.get('is_income') else 'Expense',
                            tx.get('notes', ''),
                            tx.get('merchant', '')
                        ])

                self.statusBar().showMessage(f"Exported {len(transactions)} transactions to {file_path}", 3000)

            except Exception as e:
                self.logger.error(f"Error exporting to CSV: {e}")
                QMessageBox.critical(
                    self, "Export Error",
                    f"An error occurred while exporting: {str(e)}"
                )

    def toggle_theme(self):
        """Toggle between light and dark themes."""
        current_theme = self.db_manager.get_setting("theme", "light")

        # Toggle theme
        new_theme = "dark" if current_theme == "light" else "light"
        self.db_manager.update_setting("theme", new_theme)

        # Apply the new theme
        self.apply_theme(new_theme)

        self.statusBar().showMessage(f"Switched to {new_theme} theme", 3000)

    def apply_theme(self, theme):
        """
        Apply a theme to the application.

        Args:
            theme: Theme name ('light' or 'dark')
        """
        # Simplified dark theme implementation
        if theme == "dark":
            # Dark theme with simpler styling
            self.setStyleSheet("""
                QMainWindow, QWidget {
                    background-color: #2B2B2B;
                    color: #E4E4E4;
                }
                
                QLabel {
                    color: #E4E4E4;
                }
                
                QFrame {
                    background-color: #343434;
                    border: 1px solid #444444;
                }
                
                QToolBar {
                    background-color: #323232;
                    border-bottom: 1px solid #4A4A4A;
                    spacing: 8px;
                    padding: 4px;
                }
                
                QStatusBar {
                    background-color: #333333;
                    color: #BBBBBB;
                    border-top: 1px solid #4A4A4A;
                }
                
                QPushButton {
                    background-color: #444444;
                    color: #E4E4E4;
                    border: 1px solid #555555;
                    padding: 6px 12px;
                }
                
                QPushButton:hover {
                    background-color: #505050;
                }
                
                QPushButton:pressed {
                    background-color: #383838;
                }
                
                QTabWidget::pane {
                    border: 1px solid #444444;
                    background-color: #2B2B2B;
                }
                
                QTabBar::tab {
                    background-color: #323232;
                    color: #BBBBBB;
                    border: 1px solid #444444;
                    padding: 8px 16px;
                    margin-right: 2px;
                }
                
                QTabBar::tab:selected {
                    background-color: #444444;
                    color: #E4E4E4;
                }
                
                QTableWidget {
                    background-color: #343434;
                    alternate-background-color: #3A3A3A;
                    color: #E4E4E4;
                    border: 1px solid #444444;
                    gridline-color: #444444;
                }
                
                QHeaderView::section {
                    background-color: #404040;
                    color: #BBBBBB;
                    border: 1px solid #444444;
                    padding: 4px;
                }
                
                QLineEdit, QDateEdit, QTextEdit, QComboBox {
                    background-color: #404040;
                    color: #E4E4E4;
                    border: 1px solid #555555;
                    padding: 4px;
                    selection-background-color: #0057D9;
                }
                
                QCheckBox::indicator {
                    width: 16px;
                    height: 16px;
                    border: 1px solid #666666;
                    background-color: #404040;
                }
                
                QCheckBox::indicator:checked {
                    background-color: #0057D9;
                    border: 1px solid #0057D9;
                }
            """)

            # Update theme icon
            self.theme_action.setIcon(QIcon.fromTheme("weather-clear"))
        else:
            # Simplified light theme 
            self.setStyleSheet("""
                QMainWindow, QWidget {
                    background-color: #F0F0F0;
                    color: #333333;
                }
                
                QLabel {
                    color: #333333;
                }
                
                QFrame {
                    background-color: white;
                    border: 1px solid #E0E0E0;
                }
                
                QToolBar {
                    background-color: #F5F5F5;
                    border-bottom: 1px solid #E0E0E0;
                    spacing: 8px;
                    padding: 4px;
                }
                
                QStatusBar {
                    background-color: #F5F5F5;
                    color: #666666;
                    border-top: 1px solid #E0E0E0;
                }
                
                QPushButton {
                    background-color: #F5F5F5;
                    color: #333333;
                    border: 1px solid #D0D0D0;
                    padding: 6px 12px;
                }
                
                QPushButton:hover {
                    background-color: #EBEBEB;
                }
                
                QPushButton:pressed {
                    background-color: #E1E1E1;
                }
                
                QTabWidget::pane {
                    border: 1px solid #E0E0E0;
                    background-color: #F0F0F0;
                }
                
                QTabBar::tab {
                    background-color: #E8E8E8;
                    color: #666666;
                    border: 1px solid #E0E0E0;
                    padding: 8px 16px;
                    margin-right: 2px;
                }
                
                QTabBar::tab:selected {
                    background-color: #F8F8F8;
                    color: #333333;
                }
                
                QTableWidget {
                    background-color: white;
                    alternate-background-color: #F9F9F9;
                    color: #333333;
                    border: 1px solid #E0E0E0;
                    gridline-color: #E8E8E8;
                }
                
                QHeaderView::section {
                    background-color: #F0F0F0;
                    color: #666666;
                    border: 1px solid #E0E0E0;
                    padding: 4px;
                }
                
                QLineEdit, QDateEdit, QTextEdit, QComboBox {
                    background-color: white;
                    color: #333333;
                    border: 1px solid #D5D5D5;
                    padding: 4px;
                    selection-background-color: #0078FF;
                }
                
                QCheckBox::indicator {
                    width: 16px;
                    height: 16px;
                    border: 1px solid #BBBBBB;
                    background-color: white;
                }
                
                QCheckBox::indicator:checked {
                    background-color: #0078FF;
                    border: 1px solid #0078FF;
                }
            """)

            # Update theme icon
            self.theme_action.setIcon(QIcon.fromTheme("weather-clear-night"))

    def load_settings(self):
        """Load application settings."""
        # Apply theme
        theme = self.db_manager.get_setting("theme", "light")
        self.apply_theme(theme)

    def closeEvent(self, event):
        """
        Handle application close event.

        Args:
            event: Close event
        """
        # Save settings, etc.
        # Close database connection
        self.db_manager.close()
        event.accept()