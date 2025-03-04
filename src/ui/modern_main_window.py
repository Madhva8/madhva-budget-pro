#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modern Main Window Module

This module defines the main application window for the Financial Planner
using PySide6 for a modern macOS look and feel.
"""

import os
import sys
import logging
import datetime
import shutil
import matplotlib
matplotlib.use('QtAgg')  # Use generic QtAgg backend which works with both PySide6 and PyQt5
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from typing import Dict, Any, Optional, List

# Import PySide6 components
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTabWidget, QSplitter, QFrame, QToolBar, QMenu, QStatusBar, 
    QFileDialog, QMessageBox, QSizePolicy, QApplication, QLineEdit,
    QDateEdit, QComboBox, QDialogButtonBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QScrollArea, QGroupBox, QMenuBar, QDialog
)
from PySide6.QtCore import Qt, QSize, QSettings, Slot, Signal, Property, QDate, QKeyCombination, QEvent
from PySide6.QtGui import QIcon, QFont, QPixmap, QColor, QAction, QPalette, QBrush, QKeySequence, QShortcut

# Import modern UI components
from ui.modern_styled_button import ModernStyledButton, ModernDangerButton, ModernSuccessButton
from ui.modern_transactions_tab import ModernTransactionsTab
from ui.modern_dashboard_tab import ModernDashboardTab
# Will add more components later

# Import AI components (if available)
SPARKASSE_PARSER_AVAILABLE = False
try:
    import ai.sparkasse_parser
    SPARKASSE_PARSER_AVAILABLE = True
except ImportError:
    logging.warning("Failed to import SparkasseParser in modern_main_window.py")


class ModernMainWindow(QMainWindow):
    """Modern main window for the Financial Planner application with macOS-style UI."""

    def __init__(self, db_manager, ai_components=None, current_user=None):
        """
        Initialize the main window.

        Args:
            db_manager: Database manager instance
            ai_components: Dictionary with AI component instances
            current_user: Currently logged in username
        """
        super().__init__()

        # Store references to managers and components
        self.db_manager = db_manager
        self.ai_components = ai_components or {}
        self.current_user = current_user if current_user else None
        self.logger = logging.getLogger(__name__)
        
        # Debug log for initialization
        self.logger.info(f"Initializing window with user: {self.current_user}")

        # Debug: Log available AI components
        self.logger.info(f"Initializing ModernMainWindow with AI components: {list(self.ai_components.keys())}")

        # Initialize UI
        self.init_ui()

        # Connect resize event to handle UI scaling
        self.resizeEvent = self.on_resize

        # Load settings
        self.load_settings()

    def create_menu_bar(self):
        """Create standard macOS menu bar with all expected menus."""
        menu_bar = self.menuBar()
        
        # File Menu
        file_menu = menu_bar.addMenu("&File")
        
        # New Transaction
        new_action = QAction("New &Transaction...", self)
        new_action.setShortcut(QKeySequence.New)
        new_action.triggered.connect(self.show_add_transaction_dialog)
        file_menu.addAction(new_action)
        
        # Import
        import_action = QAction("&Import Transactions...", self)
        import_action.setShortcut(QKeySequence("Ctrl+I"))
        import_action.triggered.connect(self.show_import_dialog)
        file_menu.addAction(import_action)
        
        # Export
        export_action = QAction("&Export Transactions...", self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(self.show_export_dialog)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        # Backup
        backup_action = QAction("Create &Backup...", self)
        backup_action.triggered.connect(self.create_database_backup)
        file_menu.addAction(backup_action)
        
        file_menu.addSeparator()
        
        # Close - standard macOS behavior
        close_action = QAction("&Close Window", self)
        close_action.setShortcut(QKeySequence.Close)
        close_action.triggered.connect(self.close)
        file_menu.addAction(close_action)
        
        # Edit Menu
        edit_menu = menu_bar.addMenu("&Edit")
        
        # Undo/Redo (placeholders for future implementation)
        undo_action = QAction("&Undo", self)
        undo_action.setShortcut(QKeySequence.Undo)
        edit_menu.addAction(undo_action)
        
        redo_action = QAction("&Redo", self)
        redo_action.setShortcut(QKeySequence.Redo)
        edit_menu.addAction(redo_action)
        
        edit_menu.addSeparator()
        
        # Cut/Copy/Paste
        cut_action = QAction("Cu&t", self)
        cut_action.setShortcut(QKeySequence.Cut)
        edit_menu.addAction(cut_action)
        
        copy_action = QAction("&Copy", self)
        copy_action.setShortcut(QKeySequence.Copy)
        edit_menu.addAction(copy_action)
        
        paste_action = QAction("&Paste", self)
        paste_action.setShortcut(QKeySequence.Paste)
        edit_menu.addAction(paste_action)
        
        select_all_action = QAction("Select &All", self)
        select_all_action.setShortcut(QKeySequence.SelectAll)
        edit_menu.addAction(select_all_action)
        
        edit_menu.addSeparator()
        
        # Find
        find_action = QAction("&Find...", self)
        find_action.setShortcut(QKeySequence.Find)
        edit_menu.addAction(find_action)
        
        # View Menu
        view_menu = menu_bar.addMenu("&View")
        
        # Theme toggle
        theme_action = QAction("Toggle &Theme", self)
        theme_action.setShortcut(QKeySequence("Ctrl+T"))
        theme_action.triggered.connect(self.toggle_theme)
        view_menu.addAction(theme_action)
        
        view_menu.addSeparator()
        
        # Refresh
        refresh_action = QAction("&Refresh", self)
        refresh_action.setShortcut(QKeySequence.Refresh)
        refresh_action.triggered.connect(self.refresh_current_tab)
        view_menu.addAction(refresh_action)
        
        # Window Menu (standard macOS menu)
        window_menu = menu_bar.addMenu("&Window")
        
        # Minimize
        minimize_action = QAction("&Minimize", self)
        minimize_action.setShortcut(QKeySequence("Ctrl+M"))
        minimize_action.triggered.connect(self.showMinimized)
        window_menu.addAction(minimize_action)
        
        # Zoom (maximize)
        zoom_action = QAction("&Zoom", self)
        zoom_action.triggered.connect(lambda: self.setWindowState(self.windowState() ^ Qt.WindowMaximized))
        window_menu.addAction(zoom_action)
        
        window_menu.addSeparator()
        
        # Bring All to Front
        bring_to_front_action = QAction("Bring All to &Front", self)
        bring_to_front_action.triggered.connect(lambda: self.setWindowState(Qt.WindowActive))
        window_menu.addAction(bring_to_front_action)
        
        # Help Menu
        help_menu = menu_bar.addMenu("&Help")
        
        # Budget Pro Help
        help_action = QAction("Madhva Budget Pro &Help", self)
        help_action.setShortcut(QKeySequence("F1"))
        help_action.triggered.connect(self.show_help)
        help_menu.addAction(help_action)
        
        # Keyboard Shortcuts
        shortcuts_action = QAction("&Keyboard Shortcuts", self)
        shortcuts_action.triggered.connect(self.show_shortcuts)
        help_menu.addAction(shortcuts_action)
        
        help_menu.addSeparator()
        
        # Check for Updates
        updates_action = QAction("Check for &Updates...", self)
        updates_action.triggered.connect(self.check_updates)
        help_menu.addAction(updates_action)
        
        help_menu.addSeparator()
        
        # About
        about_action = QAction("&About Madhva Budget Pro", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def init_ui(self):
        """Initialize the user interface with macOS style."""
        # Set window properties
        self.setWindowTitle("Madhva Budget Pro")
        self.setMinimumSize(800, 600)  # Smaller minimum size for better adaptability
        
        # Set the application icon
        try:
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../logo.png")
            self.logger.info(f"Setting application icon from: {icon_path}")
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
                # Also set as application icon
                QApplication.setWindowIcon(QIcon(icon_path))
        except Exception as e:
            self.logger.error(f"Error setting application icon: {e}")
        
        # Try to set application style to match macOS if available
        try:
            QApplication.setStyle("macintosh")
        except Exception:
            # Use whatever style is available
            pass
        
        # Set application font
        font = QFont()
        font.setPointSize(13)
        QApplication.setFont(font)

        # Create standard macOS menu bar
        self.create_menu_bar()

        # Create central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Main layout with margins for macOS appearance
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(12)
        
        # Add a header with the logo and app name
        self.header_layout = QHBoxLayout()
        
        # Logo container
        self.logo_label = QLabel()
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../logo.png")
        if os.path.exists(icon_path):
            logo_pixmap = QPixmap(icon_path)
            scaled_logo = logo_pixmap.scaled(48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.logo_label.setPixmap(scaled_logo)
            self.logo_label.setFixedSize(48, 48)
        
        # App name label with large, bold font
        self.app_name_label = QLabel("Madhva Budget Pro")
        app_name_font = QFont()
        app_name_font.setPointSize(24)
        app_name_font.setBold(True)
        self.app_name_label.setFont(app_name_font)
        self.app_name_label.setStyleSheet("color: var(--text-color, black);")
        
        # Add to header layout
        self.header_layout.addWidget(self.logo_label)
        self.header_layout.addWidget(self.app_name_label)
        self.header_layout.addStretch(1)  # Push everything to the left
        
        # Add user info and logout button if a user is logged in
        if self.current_user and self.current_user not in (None, ''):
            # User info with icon
            user_layout = QHBoxLayout()
            user_layout.setSpacing(8)
            
            # User icon
            user_icon_label = QLabel()
            try:
                user_icon = QIcon.fromTheme("user")
                if user_icon.isNull():
                    # Fallback icon using text
                    user_icon_label.setText("👤")
                    user_icon_label.setStyleSheet("font-size: 18px;")
                else:
                    user_icon_label.setPixmap(user_icon.pixmap(24, 24))
            except Exception:
                # Fallback to text emoji
                user_icon_label.setText("👤")
                user_icon_label.setStyleSheet("font-size: 18px;")
            user_icon_label.setFixedSize(24, 24)
            
            # Username label
            user_label = QLabel(f"Hello, {self.current_user}")
            user_font = QFont()
            user_font.setPointSize(14)
            user_label.setFont(user_font)
            user_label.setStyleSheet("color: var(--text-color, #333); font-weight: 500;")
            
            # Add to user layout
            user_layout.addWidget(user_icon_label)
            user_layout.addWidget(user_label)
            
            # Logout button
            self.logout_button = ModernDangerButton("Logout")
            self.logout_button.setFixedSize(100, 32)
            self.logout_button.clicked.connect(self.logout)
            
            # Add to header layout
            self.header_layout.addLayout(user_layout)
            self.header_layout.addWidget(self.logout_button)
        
        # Add header to main layout
        self.main_layout.addLayout(self.header_layout)
        
        # Add a separator line
        self.separator = QFrame()
        self.separator.setFrameShape(QFrame.HLine)
        self.separator.setFrameShadow(QFrame.Sunken)
        self.separator.setStyleSheet("background-color: var(--border-color, #E5E5E5);")
        self.main_layout.addWidget(self.separator)

        # Create toolbar - more macOS-like
        self.create_toolbar()

        # Create tab widget with macOS appearance
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)  # More macOS-like appearance
        self.tab_widget.setTabPosition(QTabWidget.North)
        self.tab_widget.setMovable(True)  # Allow reordering tabs
        self.tab_widget.setUsesScrollButtons(True)  # Enable scroll buttons for tabs when space is limited
        self.tab_widget.setElideMode(Qt.ElideRight)  # Elide text with "..." when needed
        
        self.main_layout.addWidget(self.tab_widget)

        # Create placeholder tabs (will implement these in separate files)
        self.create_dashboard_tab()
        self.create_transactions_tab()  # Now uses a placeholder
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
        
        # Apply responsive layout
        self.apply_responsive_layout(self.width())
        
        # Apply theme based on macOS appearance
        self.apply_theme(self.db_manager.get_setting("theme", "light"))

    def create_toolbar(self):
        """Create the application toolbar with macOS-style appearance."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(20, 20))
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        
        # Add spacing at the beginning for macOS-like appearance
        spacer = QWidget()
        spacer.setFixedWidth(8)
        toolbar.addWidget(spacer)
        
        self.addToolBar(toolbar)

        # Create macOS-style toolbar buttons that adapt to available space
        
        # Add Transaction - use SF Symbols style icon
        add_action = QAction(QIcon.fromTheme("list-add"), "Add", self)
        add_action.triggered.connect(self.show_add_transaction_dialog)
        add_action.setToolTip("Add a new transaction")
        toolbar.addAction(add_action)

        # Add small space
        spacer1 = QWidget()
        spacer1.setFixedWidth(12)
        toolbar.addWidget(spacer1)

        # Create a toolbar button container for optional buttons
        self.toolbar_button_container = QWidget()
        toolbar_buttons_layout = QHBoxLayout(self.toolbar_button_container)
        toolbar_buttons_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_buttons_layout.setSpacing(8)
        
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
        
        # Make buttons adaptable to window size
        self.toolbar_actions = [add_action, import_action, export_action]

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
        """Create the dashboard tab with full modern implementation."""
        self.logger.info(f"Creating dashboard tab with AI components: {list(self.ai_components.keys()) if self.ai_components else 'None'}")
        self.dashboard_tab = ModernDashboardTab(self.db_manager, self.ai_components)
        self.tab_widget.addTab(self.dashboard_tab, "Dashboard")

    def create_transactions_tab(self):
        """Create the transactions tab with full functionality."""
        # Use the full implementation that supports batch operations
        self.transactions_tab = ModernTransactionsTab(self.db_manager)
        self.tab_widget.addTab(self.transactions_tab, "Transactions")

    def create_budget_tab(self):
        """Create the budget tab with budget planning tools."""
        budget_widget = QWidget()
        budget_layout = QVBoxLayout(budget_widget)
        
        # Set macOS-appropriate margins
        budget_layout.setContentsMargins(20, 20, 20, 20)
        budget_layout.setSpacing(16)

        # Budget title
        title_label = QLabel("Budget Planning")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: var(--text-color, black);")
        budget_layout.addWidget(title_label)
        
        # Monthly budget overview
        overview_frame = QFrame()
        overview_frame.setFrameShape(QFrame.StyledPanel)
        overview_frame.setStyleSheet("""
            QFrame {
                background-color: var(--card-bg-color, white);
                border: 1px solid var(--border-color, #E5E5E5);
                border-radius: 10px;
            }
        """)
        
        overview_layout = QVBoxLayout(overview_frame)
        overview_layout.setContentsMargins(20, 20, 20, 20)
        
        overview_title = QLabel("Monthly Budget Overview")
        overview_title.setStyleSheet("font-size: 18px; font-weight: bold; color: var(--text-color, black);")
        overview_layout.addWidget(overview_title)
        
        budget_layout.addWidget(overview_frame)
        
        # Category budgets section
        categories_frame = QFrame()
        categories_frame.setFrameShape(QFrame.StyledPanel)
        categories_frame.setStyleSheet("""
            QFrame {
                background-color: var(--card-bg-color, white);
                border: 1px solid var(--border-color, #E5E5E5);
                border-radius: 10px;
            }
        """)
        
        categories_layout = QVBoxLayout(categories_frame)
        categories_layout.setContentsMargins(20, 20, 20, 20)
        
        categories_title = QLabel("Category Budgets")
        categories_title.setStyleSheet("font-size: 18px; font-weight: bold; color: var(--text-color, black);")
        categories_layout.addWidget(categories_title)
        
        # Sample category rows
        category_names = ["Housing", "Food", "Transportation", "Entertainment", "Utilities"]
        
        for category in category_names:
            category_row = QWidget()
            category_row.setStyleSheet("background-color: transparent;")
            row_layout = QHBoxLayout(category_row)
            row_layout.setContentsMargins(0, 8, 0, 8)
            
            # Category name
            cat_label = QLabel(category)
            cat_label.setStyleSheet("font-weight: 500; color: var(--text-color, black);")
            row_layout.addWidget(cat_label, 2)
            
            # Budget amount input
            amount_input = QLineEdit("0.00")
            amount_input.setStyleSheet("""
                QLineEdit {
                    border: 1px solid var(--border-color, #E0E0E0);
                    border-radius: 4px;
                    padding: 5px;
                    background-color: var(--input-bg-color, #F9F9F9);
                    color: var(--text-color, black);
                }
            """)
            amount_input.setFixedWidth(120)
            amount_input.setAlignment(Qt.AlignRight)
            row_layout.addWidget(amount_input, 1)
            
            # Euro sign
            euro_label = QLabel("€")
            euro_label.setStyleSheet("color: var(--text-color, black);")
            row_layout.addWidget(euro_label, 0)
            
            categories_layout.addWidget(category_row)
        
        # Add Save Budget button
        save_button = ModernStyledButton("Save Budget")
        save_button.setFixedWidth(150)
        categories_layout.addWidget(save_button, 0, Qt.AlignRight)
        
        budget_layout.addWidget(categories_frame)
        
        # Coming soon section
        coming_frame = QFrame()
        coming_frame.setFrameShape(QFrame.StyledPanel)
        coming_frame.setObjectName("comingSoonFrame")
        coming_frame.setStyleSheet("""
            #comingSoonFrame {
                background-color: var(--card-highlight-color, rgba(0, 122, 255, 0.1));
                border: 1px solid var(--border-highlight-color, rgba(0, 122, 255, 0.3));
                border-radius: 10px;
            }
        """)
        coming_layout = QVBoxLayout(coming_frame)
        coming_layout.setContentsMargins(16, 16, 16, 16)
        
        coming_title = QLabel("✨ Budget Features Coming Soon")
        coming_title.setStyleSheet("font-weight: bold; color: var(--button-primary-bg, #007AFF); font-size: 16px;")
        coming_layout.addWidget(coming_title)
        
        coming_features = QLabel(
            "• Budget vs. actual spending comparison\n"
            "• Budget alerts and notifications\n"
            "• Budget history and trends\n"
            "• Custom budget categories\n"
            "• Budget templates"
        )
        coming_features.setStyleSheet("line-height: 1.5; color: var(--text-color, black);")
        coming_layout.addWidget(coming_features)
        
        budget_layout.addWidget(coming_frame)
        budget_layout.addStretch(1)  # Push content to the top

        self.tab_widget.addTab(budget_widget, "Budget")

    def create_reports_tab(self):
        """Create the reports tab with financial reporting tools."""
        reports_widget = QWidget()
        reports_layout = QVBoxLayout(reports_widget)
        
        # Set macOS-appropriate margins
        reports_layout.setContentsMargins(20, 20, 20, 20)
        reports_layout.setSpacing(16)

        # Reports title
        title_label = QLabel("Financial Reports")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: var(--text-color, black);")
        reports_layout.addWidget(title_label)
        
        # Report type selection
        report_options = QFrame()
        report_options.setFrameShape(QFrame.StyledPanel)
        report_options.setStyleSheet("""
            QFrame {
                background-color: var(--card-bg-color, white);
                border: 1px solid var(--border-color, #E5E5E5);
                border-radius: 10px;
            }
        """)
        
        options_layout = QVBoxLayout(report_options)
        options_layout.setContentsMargins(20, 20, 20, 20)
        options_layout.setSpacing(16)
        
        options_title = QLabel("Choose Report Type")
        options_title.setStyleSheet("font-size: 18px; font-weight: bold; color: var(--text-color, black);")
        options_layout.addWidget(options_title)
        
        # Report selection buttons in a horizontal layout
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(12)
        
        # Spending by Category button
        cat_report_btn = ModernStyledButton("Spending by Category")
        cat_report_btn.setFixedHeight(40)
        buttons_layout.addWidget(cat_report_btn)
        
        # Monthly Summary button
        monthly_report_btn = ModernStyledButton("Monthly Summary")
        monthly_report_btn.setFixedHeight(40)
        buttons_layout.addWidget(monthly_report_btn)
        
        # Income vs Expenses button
        income_exp_btn = ModernStyledButton("Income vs Expenses")
        income_exp_btn.setFixedHeight(40)
        buttons_layout.addWidget(income_exp_btn)
        
        options_layout.addLayout(buttons_layout)
        
        # Date range selection
        date_range = QFrame()
        date_range.setStyleSheet("""
            QFrame {
                background-color: var(--alt-bg-color, #F9F9F9);
                border-radius: 6px;
            }
        """)
        
        date_layout = QHBoxLayout(date_range)
        date_layout.setContentsMargins(12, 12, 12, 12)
        
        date_range_label = QLabel("Date Range:")
        date_range_label.setStyleSheet("color: var(--text-color, black);")
        date_layout.addWidget(date_range_label)
        
        from_date_label = QLabel("From:")
        from_date_label.setStyleSheet("font-weight: 500; color: var(--text-color, black);")
        date_layout.addWidget(from_date_label)
        
        from_date = QDateEdit()
        from_date.setCalendarPopup(True)
        from_date.setDate(QDate.currentDate().addMonths(-6))
        from_date.setStyleSheet("""
            QDateEdit {
                border: 1px solid var(--border-color, #E0E0E0);
                border-radius: 4px;
                padding: 5px;
                background-color: var(--input-bg-color, #F9F9F9);
                color: var(--text-color, black);
            }
        """)
        date_layout.addWidget(from_date)
        
        to_date_label = QLabel("To:")
        to_date_label.setStyleSheet("font-weight: 500; color: var(--text-color, black);")
        date_layout.addWidget(to_date_label)
        
        to_date = QDateEdit()
        to_date.setCalendarPopup(True)
        to_date.setDate(QDate.currentDate())
        to_date.setStyleSheet("""
            QDateEdit {
                border: 1px solid var(--border-color, #E0E0E0);
                border-radius: 4px;
                padding: 5px;
                background-color: var(--input-bg-color, #F9F9F9);
                color: var(--text-color, black);
            }
        """)
        date_layout.addWidget(to_date)
        
        generate_btn = ModernStyledButton("Generate Report")
        date_layout.addWidget(generate_btn)
        
        options_layout.addWidget(date_range)
        
        reports_layout.addWidget(report_options)
        
        # Report preview area
        preview_frame = QFrame()
        preview_frame.setFrameShape(QFrame.StyledPanel)
        preview_frame.setStyleSheet("""
            QFrame {
                background-color: var(--card-bg-color, white);
                border: 1px solid var(--border-color, #E5E5E5);
                border-radius: 10px;
            }
        """)
        
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(20, 20, 20, 20)
        
        preview_title = QLabel("Report Preview")
        preview_title.setStyleSheet("font-size: 18px; font-weight: bold; color: var(--text-color, black);")
        preview_layout.addWidget(preview_title)
        
        preview_placeholder = QLabel("Select a report type and generate it to see the preview here.")
        preview_placeholder.setAlignment(Qt.AlignCenter)
        preview_placeholder.setStyleSheet("color: var(--header-text-color, #8E8E93); padding: 40px;")
        preview_layout.addWidget(preview_placeholder)
        
        reports_layout.addWidget(preview_frame, 1)  # 1 = stretch factor
        
        # Coming soon section
        coming_frame = QFrame()
        coming_frame.setFrameShape(QFrame.StyledPanel)
        coming_frame.setObjectName("comingSoonFrame")
        coming_frame.setStyleSheet("""
            #comingSoonFrame {
                background-color: var(--card-highlight-color, rgba(0, 122, 255, 0.1));
                border: 1px solid var(--border-highlight-color, rgba(0, 122, 255, 0.3));
                border-radius: 10px;
            }
        """)
        coming_layout = QVBoxLayout(coming_frame)
        coming_layout.setContentsMargins(16, 16, 16, 16)
        
        coming_title = QLabel("✨ Report Features Coming Soon")
        coming_title.setStyleSheet("font-weight: bold; color: var(--button-primary-bg, #007AFF); font-size: 16px;")
        coming_layout.addWidget(coming_title)
        
        coming_features = QLabel(
            "• PDF report export\n"
            "• Interactive charts and visualizations\n"
            "• Custom report builder\n"
            "• Year-over-year comparisons\n"
            "• Expense trend analysis"
        )
        coming_features.setStyleSheet("line-height: 1.5; color: var(--text-color, black);")
        coming_layout.addWidget(coming_features)
        
        reports_layout.addWidget(coming_frame)

        self.tab_widget.addTab(reports_widget, "Reports")

    def create_goals_tab(self):
        """Create the goals tab with financial goal tracking."""
        goals_widget = QWidget()
        goals_layout = QVBoxLayout(goals_widget)
        
        # Set macOS-appropriate margins
        goals_layout.setContentsMargins(20, 20, 20, 20)
        goals_layout.setSpacing(16)

        # Goals title
        title_label = QLabel("Financial Goals")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: var(--text-color, black);")
        goals_layout.addWidget(title_label)
        
        # Summary row with current goals
        summary_layout = QHBoxLayout()
        summary_layout.setSpacing(16)
        
        # Goal progress card
        progress_card = QFrame()
        progress_card.setFrameShape(QFrame.StyledPanel)
        progress_card.setObjectName("progressCard")
        progress_card.setStyleSheet("""
            #progressCard {
                background-color: var(--card-highlight-color, #F0F8FF);
                border: 1px solid var(--border-highlight-color, #D1DFE7);
                border-radius: 10px;
            }
        """)
        progress_layout = QVBoxLayout(progress_card)
        progress_layout.setContentsMargins(16, 16, 16, 16)
        
        progress_title = QLabel("Goals Progress")
        progress_title.setStyleSheet("font-weight: bold; color: var(--button-primary-bg, #007AFF); font-size: 16px;")
        progress_layout.addWidget(progress_title)
        
        progress_label = QLabel("2 of 5 goals completed")
        progress_label.setStyleSheet("font-size: 18px; font-weight: bold; color: var(--text-color, black);")
        progress_layout.addWidget(progress_label)
        
        summary_layout.addWidget(progress_card)
        
        # Next goal card
        next_card = QFrame()
        next_card.setFrameShape(QFrame.StyledPanel)
        next_card.setObjectName("nextCard")
        next_card.setStyleSheet("""
            #nextCard {
                background-color: var(--card-highlight-color, #FFF9E6);
                border: 1px solid var(--border-highlight-color, #E7E0D1);
                border-radius: 10px;
            }
        """)
        next_layout = QVBoxLayout(next_card)
        next_layout.setContentsMargins(16, 16, 16, 16)
        
        next_title = QLabel("Next Goal")
        next_title.setStyleSheet("font-weight: bold; color: var(--button-success-bg, #FF9500); font-size: 16px;")
        next_layout.addWidget(next_title)
        
        next_label = QLabel("Save €1,000 for Emergency Fund")
        next_label.setStyleSheet("font-size: 18px; font-weight: bold; color: var(--text-color, black);")
        next_layout.addWidget(next_label)
        
        summary_layout.addWidget(next_card)
        
        goals_layout.addLayout(summary_layout)
        
        # Goals list
        goals_list_frame = QFrame()
        goals_list_frame.setFrameShape(QFrame.StyledPanel)
        goals_list_frame.setStyleSheet("""
            QFrame {
                background-color: var(--card-bg-color, white);
                border: 1px solid var(--border-color, #E5E5E5);
                border-radius: 10px;
            }
        """)
        
        goals_list_layout = QVBoxLayout(goals_list_frame)
        goals_list_layout.setContentsMargins(20, 20, 20, 20)
        goals_list_layout.setSpacing(16)
        
        goals_list_title = QLabel("Your Financial Goals")
        goals_list_title.setStyleSheet("font-size: 18px; font-weight: bold; color: var(--text-color, black);")
        goals_list_layout.addWidget(goals_list_title)
        
        # Add new goal button
        add_goal_btn = ModernStyledButton("+ Add New Goal")
        add_goal_btn.setFixedWidth(150)
        goals_list_layout.addWidget(add_goal_btn, 0, Qt.AlignRight)
        
        # Sample goals
        goal_items = [
            {"name": "Emergency Fund", "target": 3000, "current": 1000, "deadline": "2025-09-01"},
            {"name": "New Car", "target": 15000, "current": 2500, "deadline": "2026-06-01"},
            {"name": "Vacation", "target": 2000, "current": 1800, "deadline": "2025-07-15"},
            {"name": "Home Down Payment", "target": 50000, "current": 10000, "deadline": "2028-01-01"},
            {"name": "Retirement", "target": 500000, "current": 50000, "deadline": "2045-01-01"}
        ]
        
        for goal in goal_items:
            goal_frame = QFrame()
            goal_frame.setFrameShape(QFrame.StyledPanel)
            goal_frame.setStyleSheet("""
                QFrame {
                    background-color: var(--alt-bg-color, #F9F9F9);
                    border-radius: 8px;
                    padding: 4px;
                }
            """)
            
            goal_frame_layout = QVBoxLayout(goal_frame)
            goal_frame_layout.setContentsMargins(12, 12, 12, 12)
            
            goal_header = QHBoxLayout()
            
            goal_name = QLabel(goal["name"])
            goal_name.setStyleSheet("font-weight: bold; font-size: 16px; color: var(--text-color, black);")
            goal_header.addWidget(goal_name)
            
            goal_header.addStretch()
            
            goal_amount = QLabel(f"€{goal['current']:,} / €{goal['target']:,}")
            goal_amount.setStyleSheet("font-weight: bold; color: var(--button-primary-bg, #007AFF);")
            goal_header.addWidget(goal_amount)
            
            goal_frame_layout.addLayout(goal_header)
            
            # Progress bar (simplified for demonstration)
            progress_percent = min(100, int(goal["current"] / goal["target"] * 100))
            progress_text = QLabel(f"Progress: {progress_percent}%")
            progress_text.setStyleSheet("color: var(--text-color, black);")
            goal_frame_layout.addWidget(progress_text)
            
            # Deadline
            deadline = QLabel(f"Target Date: {goal['deadline']}")
            deadline.setStyleSheet("color: var(--header-text-color, #8E8E93); font-size: 12px;")
            goal_frame_layout.addWidget(deadline)
            
            goals_list_layout.addWidget(goal_frame)
        
        goals_layout.addWidget(goals_list_frame, 1)  # 1 = stretch factor
        
        # Coming soon section
        coming_frame = QFrame()
        coming_frame.setFrameShape(QFrame.StyledPanel)
        coming_frame.setObjectName("comingSoonFrame")
        coming_frame.setStyleSheet("""
            #comingSoonFrame {
                background-color: var(--card-highlight-color, rgba(0, 122, 255, 0.1));
                border: 1px solid var(--border-highlight-color, rgba(0, 122, 255, 0.3));
                border-radius: 10px;
            }
        """)
        coming_layout = QVBoxLayout(coming_frame)
        coming_layout.setContentsMargins(16, 16, 16, 16)
        
        coming_title = QLabel("✨ Goal Features Coming Soon")
        coming_title.setStyleSheet("font-weight: bold; color: var(--button-primary-bg, #007AFF); font-size: 16px;")
        coming_layout.addWidget(coming_title)
        
        coming_features = QLabel(
            "• Goal progress notifications\n"
            "• Smart goal suggestions\n"
            "• Goal categories and templates\n"
            "• Goal sharing with family members\n"
            "• Milestone tracking and celebrations"
        )
        coming_features.setStyleSheet("line-height: 1.5; color: var(--text-color, black);")
        coming_layout.addWidget(coming_features)
        
        goals_layout.addWidget(coming_frame)

        self.tab_widget.addTab(goals_widget, "Goals")

    def create_settings_tab(self):
        """Create the settings tab using QGroupBox for clearer section grouping."""
        # Create main settings widget
        settings_widget = QWidget()
        
        # Create a scroll area to handle many settings gracefully
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)  # Hide the frame
        
        # Container for scrollable content
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(30, 30, 30, 30)
        scroll_layout.setSpacing(30)
        
        # Settings title
        title = QLabel("Settings")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)
        scroll_layout.addWidget(title)
        
        # ===== APPEARANCE SECTION =====
        appearance_group = QGroupBox("Appearance")
        appearance_group.setStyleSheet("""
            QGroupBox {
                font-size: 18px;
                font-weight: bold;
                border: 1px solid #CCCCCC;
                border-radius: 6px;
                margin-top: 18px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 5px 10px;
                background-color: white;
            }
        """)
        
        appearance_layout = QVBoxLayout(appearance_group)
        appearance_layout.setContentsMargins(20, 30, 20, 20)  # Extra top margin
        
        # Theme toggle
        theme_layout = QHBoxLayout()
        theme_label = QLabel("Theme:")
        default_font = QFont()
        default_font.setPointSize(14)
        theme_label.setFont(default_font)
        theme_label.setStyleSheet("border: none;")
        theme_layout.addWidget(theme_label)
        
        theme_layout.addStretch()
        
        current_theme = self.db_manager.get_setting("theme", "light")
        theme_text = "Dark Mode" if current_theme == "light" else "Light Mode"
        
        theme_button = QPushButton(theme_text)
        theme_button.setFont(default_font)
        theme_button.setMinimumSize(180, 40)
        theme_button.clicked.connect(self.toggle_theme)
        theme_layout.addWidget(theme_button)
        
        appearance_layout.addLayout(theme_layout)
        scroll_layout.addWidget(appearance_group)
        
        # ===== DATA MANAGEMENT SECTION =====
        data_group = QGroupBox("Data Management")
        data_group.setStyleSheet("""
            QGroupBox {
                font-size: 18px;
                font-weight: bold;
                border: 1px solid #CCCCCC;
                border-radius: 6px;
                margin-top: 18px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 5px 10px;
                background-color: white;
            }
        """)
        
        data_layout = QVBoxLayout(data_group)
        data_layout.setContentsMargins(20, 30, 20, 20)  # Extra top margin
        
        # Backup option
        backup_layout = QHBoxLayout()
        backup_label = QLabel("Database Backup:")
        backup_label.setFont(default_font)
        backup_label.setStyleSheet("border: none;")
        backup_layout.addWidget(backup_label)
        
        backup_layout.addStretch()
        
        self.backup_button = QPushButton("Create Backup")
        self.backup_button.setFont(default_font)
        self.backup_button.setMinimumSize(180, 40)
        self.backup_button.clicked.connect(self.create_database_backup)
        backup_layout.addWidget(self.backup_button)
        
        data_layout.addLayout(backup_layout)
        
        # Add spacer
        data_layout.addSpacing(15)
        
        # Import option
        import_layout = QHBoxLayout()
        import_label = QLabel("Import Data:")
        import_label.setFont(default_font)
        import_label.setStyleSheet("border: none;")
        import_layout.addWidget(import_label)
        
        import_layout.addStretch()
        
        import_button = QPushButton("Import")
        import_button.setFont(default_font)
        import_button.setMinimumSize(180, 40)
        import_button.clicked.connect(self.show_import_dialog)
        import_layout.addWidget(import_button)
        
        data_layout.addLayout(import_layout)
        
        # Add spacer
        data_layout.addSpacing(15)
        
        # Export option
        export_layout = QHBoxLayout()
        export_label = QLabel("Export Data:")
        export_label.setFont(default_font)
        export_label.setStyleSheet("border: none;")
        export_layout.addWidget(export_label)
        
        export_layout.addStretch()
        
        export_button = QPushButton("Export")
        export_button.setFont(default_font)
        export_button.setMinimumSize(180, 40)
        export_button.clicked.connect(self.show_export_dialog)
        export_layout.addWidget(export_button)
        
        data_layout.addLayout(export_layout)
        scroll_layout.addWidget(data_group)
        
        # ===== CATEGORY MANAGEMENT SECTION =====
        category_group = QGroupBox("Category Management")
        category_group.setStyleSheet("""
            QGroupBox {
                font-size: 18px;
                font-weight: bold;
                border: 1px solid #CCCCCC;
                border-radius: 6px;
                margin-top: 18px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 5px 10px;
                background-color: white;
            }
        """)
        
        category_layout = QVBoxLayout(category_group)
        category_layout.setContentsMargins(20, 30, 20, 20)  # Extra top margin
        
        # Description
        category_desc = QLabel("Customize transaction categories to better organize your finances.")
        category_desc.setFont(default_font)
        category_desc.setWordWrap(True)
        category_desc.setStyleSheet("border: none;")
        category_layout.addWidget(category_desc)
        
        # Add spacer
        category_layout.addSpacing(15)
        
        # Button container
        cat_button_layout = QHBoxLayout()
        cat_button_layout.addStretch()
        
        edit_categories_button = QPushButton("Edit Categories")
        edit_categories_button.setFont(default_font)
        edit_categories_button.setMinimumSize(180, 40)
        cat_button_layout.addWidget(edit_categories_button)
        
        category_layout.addLayout(cat_button_layout)
        scroll_layout.addWidget(category_group)
        
        # ===== ABOUT SECTION =====
        about_group = QGroupBox("About")
        about_group.setStyleSheet("""
            QGroupBox {
                font-size: 18px;
                font-weight: bold;
                border: 1px solid #CCCCCC;
                border-radius: 6px;
                margin-top: 18px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 5px 10px;
                background-color: white;
            }
        """)
        
        about_layout = QVBoxLayout(about_group)
        about_layout.setContentsMargins(20, 30, 20, 20)  # Extra top margin
        
        # App name
        app_name = QLabel("Madhva Budget Pro")
        app_name_font = QFont()
        app_name_font.setPointSize(16)
        app_name_font.setBold(True)
        app_name.setFont(app_name_font)
        app_name.setStyleSheet("border: none;")
        about_layout.addWidget(app_name)
        
        # Version
        version = QLabel("Version 1.1.0")
        version.setFont(default_font)
        version.setStyleSheet("border: none;")
        about_layout.addWidget(version)
        
        # Add spacer
        about_layout.addSpacing(15)
        
        # Horizontal line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        about_layout.addWidget(line)
        
        # Add spacer
        about_layout.addSpacing(15)
        
        # Copyright
        copyright_info = QLabel("© 2025 Madhva Finance")
        copyright_info.setFont(default_font)
        copyright_info.setStyleSheet("border: none;")
        about_layout.addWidget(copyright_info)
        
        # Developer info with heart emoji
        developer_info = QLabel("Developed with ❤️ by Madhva")
        developer_info.setFont(default_font)
        developer_info.setStyleSheet("border: none;")
        about_layout.addWidget(developer_info)
        
        scroll_layout.addWidget(about_group)
        
        # Add bottom spacing
        scroll_layout.addStretch(1)
        
        # Set up scroll area
        scroll_area.setWidget(scroll_content)
        
        # Add scroll area to main layout
        main_layout = QVBoxLayout(settings_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll_area)
        
        # Apply global style to ensure texts are visible
        settings_widget.setStyleSheet("""
            QLabel, QPushButton, QGroupBox {
                color: black;
            }
            QPushButton {
                background-color: #F0F0F0;
                border: 1px solid #CCCCCC;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #E0E0E0;
            }
        """)
        
        self.tab_widget.addTab(settings_widget, "Settings")

    def setup_event_handlers(self):
        """Set up event handlers."""
        # Tab change handler
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
    @Slot(int)
    def on_tab_changed(self, index):
        """
        Handle tab change event.

        Args:
            index: New tab index
        """
        # Update the appropriate tab when it's selected
        tab_text = self.tab_widget.tabText(index)
        
        if tab_text == "Transactions":
            # Refresh transactions when tab is selected
            self.transactions_tab.load_transactions()
            
            # Apply responsive layout to transactions tab
            if hasattr(self.transactions_tab, 'parent_resized'):
                try:
                    self.transactions_tab.parent_resized(self.width())
                except Exception as e:
                    self.logger.error(f"Failed to apply responsive layout: {e}")
            
            # Update dashboard data when transactions change to keep dashboard in sync
            if hasattr(self, 'dashboard_tab'):
                try:
                    self.dashboard_tab.refresh_dashboard()
                except Exception as e:
                    self.logger.error(f"Failed to update dashboard after transaction update: {e}")
                    
        elif tab_text == "Dashboard":
            # Update dashboard data when tab is selected
            if hasattr(self, 'dashboard_tab'):
                try:
                    self.dashboard_tab.refresh_dashboard()
                except Exception as e:
                    self.logger.error(f"Failed to update dashboard: {e}")
                
        # Force a resize of the window to ensure charts are properly scaled
        # This helps the pie chart maintain proper size relationship with its container
        current_size = self.size()
        QApplication.processEvents()  # Process any pending events
        self.resize(current_size.width(), current_size.height())
                
        # Apply responsive layout to the active tab
        current_tab = self.tab_widget.currentWidget()
        if current_tab and hasattr(current_tab, 'parent_resized'):
            try:
                current_tab.parent_resized(self.width())
            except Exception as e:
                # Ignore errors during layout adjustment
                pass
        
    def show_add_transaction_dialog(self):
        """Show the add transaction dialog."""
        # Import the dialog here to avoid circular imports
        from ui.modern_transaction_dialog import ModernTransactionDialog
        
        try:
            # Create and show the dialog
            dialog = ModernTransactionDialog(self.db_manager, None, self)
            if dialog.exec_():
                # Refresh all data regardless of current tab
                if hasattr(self, 'transactions_tab'):
                    self.transactions_tab.load_transactions()
                
                # Always refresh dashboard after a transaction changes
                if hasattr(self, 'dashboard_tab'):
                    self.logger.info("Refreshing dashboard after transaction change")
                    self.dashboard_tab.refresh_dashboard()
                
                # If on dashboard tab, force a UI update
                if self.tab_widget.currentIndex() == 0:  # Dashboard tab
                    # Force a resize to update the charts
                    current_size = self.size()
                    QApplication.processEvents()
                    self.resize(current_size.width(), current_size.height())
                    
                self.statusBar().showMessage("Transaction added successfully", 3000)
            else:
                self.statusBar().showMessage("Transaction addition cancelled", 3000)
        except Exception as e:
            self.logger.error(f"Error adding transaction: {e}")
            QMessageBox.critical(
                self, "Error",
                f"An error occurred while adding the transaction: {str(e)}"
            )

    def show_import_dialog(self):
        """Show the import dialog with native file picker and detailed feedback."""
        # Use native file dialog for better macOS integration
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Import Transactions")
        file_dialog.setNameFilter("All Supported Files (*.csv *.pdf);;CSV Files (*.csv);;PDF Files (*.pdf)")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setAcceptMode(QFileDialog.AcceptOpen)
        
        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                file_path = selected_files[0]
                file_type = os.path.splitext(file_path)[1].lower()
                
                # Show detailed feedback with a message box
                if file_type == '.csv':
                    self._show_import_details(file_path, "CSV")
                elif file_type == '.pdf':
                    self._show_import_details(file_path, "PDF")
                else:
                    QMessageBox.warning(
                        self,
                        "Unsupported File Type",
                        f"The file type '{file_type}' is not supported for import."
                    )
            else:
                self.statusBar().showMessage("No file selected", 3000)
        else:
            self.statusBar().showMessage("Import cancelled", 3000)
            
    def _show_import_details(self, file_path, file_type):
        """Process import and show details about the results."""
        if file_type == "CSV":
            # For CSV we'll show just a placeholder for now
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle(f"Import {file_type} File")
            msg_box.setText(
                f"CSV file selected: {os.path.basename(file_path)}\n\n"
                "In the new interface, this file would be processed to:\n"
                "• Extract transaction data\n"
                "• Categorize transactions automatically\n"
                "• Add all transactions to your database\n\n"
                "This feature is being updated to match modern macOS design standards."
            )
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setStandardButtons(QMessageBox.Ok)
            
            # Style the message box to match macOS
            msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: white;
                    font-size: 14px;
                }
                QPushButton {
                    background-color: #007AFF;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 8px 16px;
                    min-width: 100px;
                }
                QPushButton:hover {
                    background-color: #0062CC;
                }
            """)
            
            msg_box.exec_()
            self.statusBar().showMessage(f"{file_type} file selected: {os.path.basename(file_path)}", 3000)
            
        else:  # PDF
            # For PDF, let's actually try to process it using the existing parser
            pdf_file = file_path
            filename = os.path.basename(pdf_file)
            
            # Check if the parser is available in AI components
            if 'sparkasse_parser' in self.ai_components:
                try:
                    self.statusBar().showMessage(f"Parsing PDF statement {filename}...", 3000)
                    
                    # Get the parser instance
                    parser = self.ai_components['sparkasse_parser']
                    
                    # Parse the PDF to get transactions
                    transactions = parser.parse_pdf(pdf_file)
                    
                    if transactions:
                        # Create a results dialog
                        msg_box = QMessageBox(self)
                        msg_box.setWindowTitle("PDF Import Results")
                        
                        # Display summary of found transactions
                        msg_box.setText(
                            f"Successfully processed: {filename}\n\n"
                            f"Found {len(transactions)} transactions.\n\n"
                            "Would you like to import these transactions?"
                        )
                        msg_box.setIcon(QMessageBox.Question)
                        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                        
                        # Style the message box to match macOS
                        msg_box.setStyleSheet("""
                            QMessageBox {
                                background-color: white;
                                font-size: 14px;
                            }
                            QPushButton {
                                background-color: #007AFF;
                                color: white;
                                border: none;
                                border-radius: 4px;
                                padding: 8px 16px;
                                min-width: 100px;
                            }
                            QPushButton:hover {
                                background-color: #0062CC;
                            }
                        """)
                        
                        # If user confirms, add the transactions
                        if msg_box.exec_() == QMessageBox.Yes:
                            imported_count = 0
                            
                            # Import transactions to database
                            for tx in transactions:
                                # Find or create category (use default if needed)
                                category_id = 1  # Default category
                                
                                # Add to database
                                tx_id = self.db_manager.add_transaction(
                                    date=tx.get('date', ''),
                                    amount=tx.get('amount', 0),
                                    description=tx.get('description', ''),
                                    category_id=category_id,
                                    is_income=tx.get('is_income', False),
                                    merchant=tx.get('merchant', '')
                                )
                                
                                if tx_id:
                                    imported_count += 1
                            
                            # Show confirmation
                            confirm_msg = QMessageBox(self)
                            confirm_msg.setWindowTitle("Import Complete")
                            confirm_msg.setText(f"Successfully imported {imported_count} transactions.")
                            confirm_msg.setIcon(QMessageBox.Information)
                            confirm_msg.setStandardButtons(QMessageBox.Ok)
                            confirm_msg.setStyleSheet(msg_box.styleSheet())
                            confirm_msg.exec_()
                            
                            # Refresh dashboard if transactions were imported
                            if imported_count > 0 and hasattr(self, 'dashboard_tab'):
                                self.dashboard_tab.refresh_dashboard()
                            
                            self.statusBar().showMessage(f"Imported {imported_count} transactions from {filename}", 3000)
                        else:
                            self.statusBar().showMessage("Import cancelled", 3000)
                    else:
                        QMessageBox.warning(
                            self, "No Transactions Found",
                            "No transactions could be extracted from the PDF. This might not be a supported bank statement format."
                        )
                except Exception as e:
                    self.logger.error(f"Error importing PDF: {e}", exc_info=True)
                    QMessageBox.critical(
                        self, "Import Error",
                        f"An error occurred while importing: {str(e)}"
                    )
            else:
                # Fallback if parser not available
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("Parser Not Available")
                msg_box.setText(
                    f"PDF statement selected: {filename}\n\n"
                    "The Sparkasse PDF parser module is not available in this build.\n"
                    "Please install the required dependencies:\n\n"
                    "• pdfplumber\n"
                    "• pandas\n\n"
                    "Then restart the application."
                )
                msg_box.setIcon(QMessageBox.Warning)
                msg_box.setStandardButtons(QMessageBox.Ok)
                
                # Style the message box to match macOS
                msg_box.setStyleSheet("""
                    QMessageBox {
                        background-color: white;
                        font-size: 14px;
                    }
                    QPushButton {
                        background-color: #007AFF;
                        color: white;
                        border: none;
                        border-radius: 4px;
                        padding: 8px 16px;
                        min-width: 100px;
                    }
                    QPushButton:hover {
                        background-color: #0062CC;
                    }
                """)
                
                msg_box.exec_()

    def show_export_dialog(self):
        """Show the export dialog with native file dialog and detailed feedback."""
        # Use native save dialog for better macOS integration
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Export Transactions")
        file_dialog.setNameFilter("CSV Files (*.csv)")
        file_dialog.setFileMode(QFileDialog.AnyFile)
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setDefaultSuffix("csv")
        
        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                file_path = selected_files[0]
                # Ensure file has .csv extension
                if not file_path.lower().endswith('.csv'):
                    file_path += '.csv'
                    
                # Show confirmation dialog
                self._show_export_details(file_path)
            else:
                self.statusBar().showMessage("No file selected", 3000)
        else:
            self.statusBar().showMessage("Export cancelled", 3000)
            
    def _show_export_details(self, file_path):
        """Show details about a successful export with confirmation."""
        # Create a styled message dialog
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Export Transactions")
        msg_box.setText(
            f"File selected for export: {os.path.basename(file_path)}\n\n"
            "In the new interface, this would export:\n"
            "• All transactions matching your current filters\n"
            "• Formatted data ready for spreadsheet applications\n"
            "• Complete transaction details including categories\n\n"
            "This feature is being updated to match modern macOS design standards."
        )
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setStandardButtons(QMessageBox.Ok)
        
        # Style the message box to match macOS
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: white;
                font-size: 14px;
            }
            QPushButton {
                background-color: #007AFF;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #0062CC;
            }
        """)
        
        msg_box.exec_()
        self.statusBar().showMessage(f"Export prepared for: {os.path.basename(file_path)}", 3000)

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

    def toggle_theme(self):
        """Toggle between light and dark themes."""
        current_theme = self.db_manager.get_setting("theme", "light")

        # Toggle theme
        new_theme = "dark" if current_theme == "light" else "light"
        self.db_manager.update_setting("theme", new_theme)

        # Apply the new theme
        self.apply_theme(new_theme)
        
        # Update settings button text when theme changes
        for widget in self.findChildren(ModernStyledButton):
            if hasattr(widget, 'text') and widget.text() in ["Dark Mode", "Light Mode"]:
                widget.setText("Dark Mode" if new_theme == "light" else "Light Mode")

        self.statusBar().showMessage(f"Switched to {new_theme} theme", 3000)

    def apply_theme(self, theme):
        """
        Apply a theme to the application.

        Args:
            theme: Theme name ('light' or 'dark')
        """
        if theme == "dark":
            # Use Qt's palette system for better macOS integration
            dark_palette = QPalette()
            
            # Set dark color scheme with deeper blue accents
            dark_palette.setColor(QPalette.Window, QColor(30, 30, 35))  # Slightly bluish dark background
            dark_palette.setColor(QPalette.WindowText, QColor(230, 230, 240))
            dark_palette.setColor(QPalette.Base, QColor(22, 22, 27))  # Slightly bluish dark
            dark_palette.setColor(QPalette.AlternateBase, QColor(35, 35, 40))
            dark_palette.setColor(QPalette.ToolTipBase, QColor(30, 30, 35))
            dark_palette.setColor(QPalette.ToolTipText, QColor(230, 230, 240))
            dark_palette.setColor(QPalette.Text, QColor(230, 230, 240))
            dark_palette.setColor(QPalette.Button, QColor(45, 45, 55))  # Slightly bluish buttons
            dark_palette.setColor(QPalette.ButtonText, QColor(230, 230, 240))
            dark_palette.setColor(QPalette.BrightText, Qt.red)
            dark_palette.setColor(QPalette.Link, QColor(64, 156, 255))  # Brighter blue for better visibility
            dark_palette.setColor(QPalette.Highlight, QColor(0, 102, 204))  # Deeper blue highlight
            dark_palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
            
            # Apply palette to application
            QApplication.setPalette(dark_palette)
            
            # Define theme CSS variables for components
            theme_vars = """
                * {
                    --card-bg-color: rgba(40, 40, 50, 0.7);
                    --border-color: #3D3D48;
                    --bg-color: #282832;
                    --alt-bg-color: #32323E; 
                    --text-color: #E6E6F0;
                    --separator-color: #3D3D48;
                    --header-bg-color: #323240;
                    --header-text-color: #A0A0B0;
                    --selection-color: rgba(64, 156, 255, 0.25);
                    --card-highlight-color: rgba(64, 156, 255, 0.2);
                    --border-highlight-color: rgba(64, 156, 255, 0.4);
                    
                    /* Button styling variables */
                    --button-primary-bg: #4080D0;
                    --button-primary-text: white;
                    --button-primary-hover-bg: #5090E0;
                    --button-primary-pressed-bg: #3070C0;
                    
                    --button-secondary-bg: rgba(40, 40, 50, 0.7);
                    --button-secondary-text: #80A0E0;
                    --button-secondary-border: #4D4D58;
                    --button-secondary-hover-bg: rgba(64, 128, 208, 0.2);
                    --button-secondary-hover-border: #5D5D68;
                    --button-secondary-pressed-bg: rgba(64, 128, 208, 0.3);
                    
                    --button-danger-bg: #D04040;
                    --button-danger-text: white;
                    --button-danger-hover-bg: #E05050;
                    --button-danger-pressed-bg: #C03030;
                    --button-danger-disabled-bg: #804040;
                    
                    --button-success-bg: #40A060;
                    --button-success-text: white;
                    --button-success-hover-bg: #50B070; 
                    --button-success-pressed-bg: #308050;
                    --button-success-disabled-bg: #407050;
                    
                    --button-disabled-text: #6D6D78;
                    --button-disabled-bg: rgba(50, 50, 60, 0.5);
                    --button-disabled-border: #4D4D58;
                    --button-disabled-accent-bg: #405080;
                }
            """
            # Apply theme variables to application
            self.setStyleSheet(theme_vars)
            
            # Update theme icon
            self.theme_action.setIcon(QIcon.fromTheme("weather-clear"))
            
            # Fix toolbar text in dark mode
            try:
                for toolbar in self.findChildren(QToolBar):
                    toolbar.setStyleSheet("""
                        QToolBar {
                            background-color: rgb(45, 45, 55);
                            border: none;
                        }
                        QToolButton {
                            color: #E6E6F0;
                            background-color: transparent;
                            border: none;
                            border-radius: 4px;
                            padding: 4px;
                        }
                        QToolButton:hover {
                            background-color: rgba(64, 156, 255, 0.2);
                        }
                        QToolButton:pressed {
                            background-color: rgba(64, 156, 255, 0.3);
                        }
                    """)
            except Exception as e:
                self.logger.error(f"Error applying toolbar dark style: {e}")
        else:
            # Light theme - use default palette
            QApplication.setPalette(QApplication.style().standardPalette())
            
            # Update theme icon
            self.theme_action.setIcon(QIcon.fromTheme("weather-clear-night"))
            
            # Reset theme variables for light mode
            theme_vars = """
                * {
                    --card-bg-color: rgba(255, 255, 255, 0.7);
                    --border-color: #E5E5E5;
                    --bg-color: white;
                    --alt-bg-color: #F9F9F9; 
                    --text-color: black;
                    --separator-color: #F0F0F0;
                    --header-bg-color: #F5F5F5;
                    --header-text-color: #666666;
                    --selection-color: rgba(0, 122, 255, 0.1);
                    --card-highlight-color: rgba(0, 122, 255, 0.1);
                    --border-highlight-color: rgba(0, 122, 255, 0.3);
                    
                    /* Button styling variables */
                    --button-primary-bg: #007AFF;
                    --button-primary-text: white;
                    --button-primary-hover-bg: #0062CC;
                    --button-primary-pressed-bg: #0051A8;
                    
                    --button-secondary-bg: #F5F5F7;
                    --button-secondary-text: #007AFF;
                    --button-secondary-border: #DDDDDD;
                    --button-secondary-hover-bg: rgba(0, 122, 255, 0.05);
                    --button-secondary-hover-border: #BBBBBB;
                    --button-secondary-pressed-bg: rgba(0, 122, 255, 0.1);
                    
                    --button-danger-bg: #FF3B30;
                    --button-danger-text: white;
                    --button-danger-hover-bg: #E0342B;
                    --button-danger-pressed-bg: #C12E26;
                    --button-danger-disabled-bg: #FFBFBC;
                    
                    --button-success-bg: #34C759;
                    --button-success-text: white;
                    --button-success-hover-bg: #30B350;
                    --button-success-pressed-bg: #2A9F47;
                    --button-success-disabled-bg: #B8ECC5;
                    
                    --button-disabled-text: #AAAAAA;
                    --button-disabled-bg: rgba(240, 240, 240, 0.8);
                    --button-disabled-border: #E5E5E5;
                    --button-disabled-accent-bg: #B0D0FF;
                }
            """
            # Apply light theme variables
            self.setStyleSheet(theme_vars)
            
            # Reset toolbar style
            try:
                for toolbar in self.findChildren(QToolBar):
                    toolbar.setStyleSheet("")
            except Exception as e:
                self.logger.error(f"Error resetting toolbar style: {e}")

            
    def load_settings(self):
        """Load application settings."""
        # Apply theme
        theme = self.db_manager.get_setting("theme", "light")
        self.apply_theme(theme)

    def on_resize(self, event):
        """
        Handle window resize event to adapt UI accordingly.
        
        Args:
            event: Resize event
        """
        # Apply responsive layout based on new width
        self.apply_responsive_layout(event.size().width())
        
        # Handle chart resizing in dashboard
        current_tab = self.tab_widget.currentWidget()
        if self.tab_widget.tabText(self.tab_widget.currentIndex()) == "Dashboard":
            # If we're on the dashboard, we need to update the chart
            try:
                if hasattr(self, 'chart_canvas'):
                    # Trigger chart refresh by calling update_expense_chart
                    # with the current transactions data
                    # This will recreate the chart with the correct size
                    QApplication.processEvents()  # Process any pending events first
                    # No need to regenerate data, just trigger a canvas update
                    self.chart_canvas.draw_idle()
            except Exception as e:
                self.logger.error(f"Error resizing chart: {e}")
        
        # Call the parent class implementation
        super().resizeEvent(event)
    
    def apply_responsive_layout(self, width):
        """
        Adjust UI elements based on window width.
        
        Args:
            width: Current window width
        """
        
    def refresh_current_tab(self):
        """Refresh data in the current tab."""
        current_tab_index = self.tab_widget.currentIndex()
        tab_widget = self.tab_widget.widget(current_tab_index)
        
        # Call appropriate refresh method based on the current tab
        tab_text = self.tab_widget.tabText(current_tab_index)
        
        if tab_text == "Transactions" and hasattr(self, 'transactions_tab'):
            self.transactions_tab.load_transactions()
            self.statusBar().showMessage("Transactions refreshed", 3000)
            
        elif tab_text == "Dashboard" and hasattr(self, 'dashboard_tab'):
            self.dashboard_tab.refresh_dashboard()
            self.statusBar().showMessage("Dashboard refreshed", 3000)
            
        else:
            self.statusBar().showMessage(f"Refreshed {tab_text} tab", 3000)
        
        # Force redraw of the window
        self.update()
    
    def show_help(self):
        """Show help documentation."""
        QMessageBox.information(
            self,
            "Madhva Budget Pro Help",
            "Welcome to Madhva Budget Pro!\n\n"
            "This application helps you manage your personal finances in a simple and intuitive way.\n\n"
            "• Dashboard: Overview of your finances\n"
            "• Transactions: Manage your income and expenses\n"
            "• Budget: Create and monitor your budget\n"
            "• Reports: Generate financial reports\n"
            "• Goals: Set and track financial goals\n\n"
            "For more detailed help, please visit our website or contact support."
        )
    
    def show_shortcuts(self):
        """Show keyboard shortcuts dialog."""
        shortcuts_text = """
<h3>Keyboard Shortcuts</h3>
<table>
<tr><td><b>General</b></td><td></td></tr>
<tr><td>Ctrl+N</td><td>New Transaction</td></tr>
<tr><td>Ctrl+I</td><td>Import Transactions</td></tr>
<tr><td>Ctrl+E</td><td>Export Transactions</td></tr>
<tr><td>Ctrl+T</td><td>Toggle Theme</td></tr>
<tr><td>F5</td><td>Refresh Current Tab</td></tr>
<tr><td>Ctrl+W</td><td>Close Window</td></tr>
<tr><td>F1</td><td>Help</td></tr>
<tr><td><b>Navigation</b></td><td></td></tr>
<tr><td>Ctrl+Tab</td><td>Next Tab</td></tr>
<tr><td>Ctrl+Shift+Tab</td><td>Previous Tab</td></tr>
<tr><td>Ctrl+1-5</td><td>Switch to Tab 1-5</td></tr>
</table>
"""
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Keyboard Shortcuts")
        msg_box.setText(shortcuts_text)
        msg_box.setTextFormat(Qt.RichText)
        msg_box.setStandardButtons(QMessageBox.Ok)
        
        # Style the message box to match macOS
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: white;
                font-size: 14px;
            }
            QLabel {
                min-width: 400px;
            }
            QPushButton {
                background-color: #007AFF;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #0062CC;
            }
        """)
        
        msg_box.exec_()
    
    def check_updates(self):
        """Check for application updates."""
        QMessageBox.information(
            self,
            "Check for Updates",
            "Madhva Budget Pro v1.1.0\n\n"
            "You are running the latest version!"
        )
    
    def show_about(self):
        """Show about dialog with app information."""
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../logo.png")
        
        about_text = f"""
<div style='text-align: center;'>
<h2>Madhva Budget Pro</h2>
<p>Version 1.1.0</p>
<p>A personal finance manager for macOS</p>
<p>&copy; 2025 Madhva Finance</p>
<p>Developed by Madhva</p>
<p>Made with ❤️ in Silicon Valley</p>
</div>
"""
        
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("About Madhva Budget Pro")
        
        if os.path.exists(icon_path):
            logo_pixmap = QPixmap(icon_path)
            scaled_logo = logo_pixmap.scaled(128, 128, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            msg_box.setIconPixmap(scaled_logo)
            
        msg_box.setText(about_text)
        msg_box.setTextFormat(Qt.RichText)
        msg_box.setStandardButtons(QMessageBox.Ok)
        
        # Style the message box to match macOS
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: white;
                font-size: 14px;
            }
            QLabel {
                min-width: 400px;
            }
            QPushButton {
                background-color: #007AFF;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #0062CC;
            }
        """)
        
        msg_box.exec_()
    
    def logout(self):
        """Log out the current user and restart the application."""
        reply = QMessageBox.question(
            self, 
            "Confirm Logout", 
            "Are you sure you want to log out?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.logger.info(f"User '{self.current_user}' logging out")
            
            # Close the current window
            self.close()
            
            # Restart the application to show the login dialog
            program = sys.executable
            args = sys.argv
            self.logger.info(f"Restarting application: {program} {args}")
            
            # Exit with a special code that can be caught by a wrapper script
            QApplication.exit(42)  # Special exit code for restart
        # Compact mode thresholds
        compact_threshold = 900
        very_compact_threshold = 700
        
        # Adjust toolbar based on width
        if width < compact_threshold:
            # In compact mode, hide text on toolbar buttons
            try:
                # Apply to the toolbar instead of individual actions
                for toolbar in self.findChildren(QToolBar):
                    toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)
            except Exception as e:
                self.logger.error(f"Error adjusting toolbar style: {e}")
        else:
            # In normal mode, show text under icons
            try:
                # Apply to the toolbar instead of individual actions
                for toolbar in self.findChildren(QToolBar):
                    toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            except Exception as e:
                self.logger.error(f"Error adjusting toolbar style: {e}")
                
        # Handle very compact mode
        if width < very_compact_threshold:
            # For extremely small windows, adjust layouts further
            self.main_layout.setContentsMargins(10, 10, 10, 10)
            self.main_layout.setSpacing(6)
        else:
            # Restore normal padding
            self.main_layout.setContentsMargins(20, 20, 20, 20)
            self.main_layout.setSpacing(12)
            
        # Forward resize event to active tab for responsive layout
        current_tab = self.tab_widget.currentWidget()
        if current_tab and hasattr(current_tab, 'parent_resized'):
            try:
                current_tab.parent_resized(width)
            except Exception as e:
                # Ignore errors during layout adjustment
                pass
    
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


# Example usage - this would be called from main.py
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Would need a database manager and AI components
    # window = ModernMainWindow(db_manager, ai_components)
    # window.show()
    
    sys.exit(app.exec())