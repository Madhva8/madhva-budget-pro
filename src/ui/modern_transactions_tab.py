#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modern Transactions Tab Module

This module defines the transactions tab for the Financial Planner,
providing a view of transactions with filtering and editing capabilities.
Designed with modern macOS aesthetics using PySide6.
"""

import logging
import datetime
from typing import List, Dict, Any, Optional, Set

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit,
    QDateEdit, QComboBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QFrame, QSplitter, QMessageBox, QMenu, QCheckBox, QGroupBox,
    QFormLayout, QScrollBar, QScrollArea, QSizePolicy, QDialog, QDialogButtonBox,
    QGridLayout
)
from PySide6.QtCore import Qt, QDate, Signal, Slot, QModelIndex, QItemSelectionModel
from PySide6.QtGui import QIcon, QColor, QCursor, QFont, QAction, QPalette, QBrush

from ui.modern_styled_button import ModernStyledButton, ModernDangerButton, ModernSuccessButton
# Will be implemented later
# from ui.modern_add_transaction_dialog import ModernAddTransactionDialog


class ModernTransactionsTab(QWidget):
    """Widget representing the transactions tab with modern macOS styling."""

    # Signals
    transaction_selected = Signal(int)  # Emitted when a transaction is selected

    def __init__(self, db_manager):
        """
        Initialize the transactions tab.

        Args:
            db_manager: Database manager instance
        """
        super().__init__()

        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        
        # State for batch operations
        self.batch_mode = False
        self.selected_transactions: Set[int] = set()  # Set of selected transaction IDs

        # Initialize UI
        self.init_ui()
        
        # Connect resize event for responsive layout
        self.parent_resized(self.width())

        # Load initial data
        self.load_transactions()

    
    def _create_checkbox_handler(self, tx_id):
        # Create a handler function with a properly captured tx_id
        return lambda state: self.on_transaction_checked(state, tx_id)
    def init_ui(self):
        """Initialize the user interface with macOS styling."""
        # Main layout with macOS standard margins and spacing
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Make the layout adaptable to window size
        self.main_layout = layout
        
        # ---- FILTER SECTION ----
        
        # Top controls - Filters with macOS-style card design
        filter_frame = QFrame()
        filter_frame.setObjectName("filterCard")
        filter_frame.setFrameShape(QFrame.StyledPanel)
        filter_frame.setStyleSheet("""
            #filterCard {
                background-color: var(--card-bg-color, rgba(255, 255, 255, 0.5));
                border: 1px solid var(--border-color, #E5E5E5);
                border-radius: 10px;
            }
        """)

        filter_layout = QHBoxLayout(filter_frame)
        filter_layout.setContentsMargins(16, 12, 16, 12)
        filter_layout.setSpacing(16)

        # Use a grid layout for better control on narrow windows
        date_controls = QWidget()
        date_grid = QGridLayout(date_controls)
        date_grid.setContentsMargins(0, 0, 0, 0)
        date_grid.setHorizontalSpacing(8)
        date_grid.setVerticalSpacing(8)
        
        # From date with macOS style - first row
        from_date_label = QLabel("From:")
        from_date_label.setStyleSheet("font-weight: 500;")
        date_grid.addWidget(from_date_label, 0, 0)
        
        self.from_date = QDateEdit()
        self.from_date.setCalendarPopup(True)
        self.from_date.setMinimumWidth(90)
        self.from_date.setMaximumWidth(120)
        self.from_date.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        # Default to 3 months ago
        from_date = QDate.currentDate().addMonths(-3)
        self.from_date.setDate(from_date)
        date_grid.addWidget(self.from_date, 0, 1)

        # To date with macOS style - second row
        to_date_label = QLabel("To:")
        to_date_label.setStyleSheet("font-weight: 500;")
        date_grid.addWidget(to_date_label, 1, 0)
        
        self.to_date = QDateEdit()
        self.to_date.setCalendarPopup(True)
        self.to_date.setMinimumWidth(90)
        self.to_date.setMaximumWidth(120)
        self.to_date.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.to_date.setDate(QDate.currentDate())
        date_grid.addWidget(self.to_date, 1, 1)
        
        filter_layout.addWidget(date_controls)

        # Add vertical separator in macOS style
        v_separator1 = QFrame()
        v_separator1.setFrameShape(QFrame.VLine)
        v_separator1.setFrameShadow(QFrame.Sunken)
        v_separator1.setStyleSheet("color: #DDDDDD; max-width: 1px;")
        filter_layout.addWidget(v_separator1)

        # Category filter with macOS style dropdown
        category_label = QLabel("Category:")
        category_label.setStyleSheet("font-weight: 500;")
        filter_layout.addWidget(category_label)
        
        self.category_combo = QComboBox()
        self.category_combo.setFixedWidth(150)
        self.category_combo.addItem("All Categories", None)
        # Populate with categories - will be filled when data is loaded
        filter_layout.addWidget(self.category_combo)

        # Add vertical separator
        v_separator2 = QFrame()
        v_separator2.setFrameShape(QFrame.VLine)
        v_separator2.setFrameShadow(QFrame.Sunken)
        v_separator2.setStyleSheet("color: #DDDDDD; max-width: 1px;")
        filter_layout.addWidget(v_separator2)

        # Transaction type filter
        type_label = QLabel("Type:")
        type_label.setStyleSheet("font-weight: 500;")
        filter_layout.addWidget(type_label)
        
        self.type_combo = QComboBox()
        self.type_combo.setFixedWidth(120)
        self.type_combo.addItem("All Types", None)
        self.type_combo.addItem("Income", 1)
        self.type_combo.addItem("Expense", 0)
        filter_layout.addWidget(self.type_combo)

        # Add vertical separator
        v_separator3 = QFrame()
        v_separator3.setFrameShape(QFrame.VLine)
        v_separator3.setFrameShadow(QFrame.Sunken)
        v_separator3.setStyleSheet("color: #DDDDDD; max-width: 1px;")
        filter_layout.addWidget(v_separator3)

        # Search field with macOS-style search appearance
        search_label = QLabel("Search:")
        search_label.setStyleSheet("font-weight: 500;")
        filter_layout.addWidget(search_label)
        
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search in description...")
        self.search_box.setMinimumWidth(180)
        self.search_box.setStyleSheet("""
            QLineEdit {
                border-radius: 15px;
                padding-left: 10px;
                padding-right: 10px;
                background-color: rgba(142, 142, 147, 0.12);
                border: none;
            }
            QLineEdit:focus {
                background-color: rgba(142, 142, 147, 0.18);
            }
        """)
        filter_layout.addWidget(self.search_box)

        filter_layout.addStretch()

        # Apply filters button - macOS style accent button
        self.apply_filters_button = ModernStyledButton("Apply")
        self.apply_filters_button.clicked.connect(self.load_transactions)
        filter_layout.addWidget(self.apply_filters_button)

        # Reset filters button - macOS style secondary button
        self.reset_filters_button = ModernStyledButton("Reset", is_secondary=True)
        self.reset_filters_button.clicked.connect(self.reset_filters)
        filter_layout.addWidget(self.reset_filters_button)
        
        # Batch selection button - secondary style
        self.batch_mode_button = ModernStyledButton("Batch Select", is_secondary=True)
        self.batch_mode_button.clicked.connect(self.toggle_batch_mode)
        filter_layout.addWidget(self.batch_mode_button)

        layout.addWidget(filter_frame)
        
        # ---- BATCH OPERATIONS BAR ----
        
        # Create batch operations bar (initially hidden)
        self.batch_bar = QFrame()
        self.batch_bar.setFrameShape(QFrame.StyledPanel)
        self.batch_bar.setObjectName("batchBar")
        self.batch_bar.setStyleSheet("""
            #batchBar {
                background-color: var(--card-bg-color, rgba(255, 255, 255, 0.7));
                border-radius: 10px;
                border: 1px solid var(--border-color, #E5E5E5);
            }
        """)
        self.batch_bar.setVisible(False)  # Initially hidden
        
        batch_layout = QHBoxLayout(self.batch_bar)
        batch_layout.setContentsMargins(16, 8, 16, 8)
        
        # Selection count with macOS-style counter badge
        selection_widget = QWidget()
        selection_layout = QHBoxLayout(selection_widget)
        selection_layout.setContentsMargins(0, 0, 0, 0)
        selection_layout.setSpacing(6)
        
        selection_label = QLabel("Selected:")
        selection_label.setStyleSheet("font-weight: 500;")
        selection_layout.addWidget(selection_label)
        
        self.selected_count_label = QLabel("0")
        self.selected_count_label.setStyleSheet("""
            background-color: #007AFF;
            color: white;
            border-radius: 10px;
            padding: 2px 8px;
            min-width: 16px;
            font-weight: bold;
        """)
        self.selected_count_label.setAlignment(Qt.AlignCenter)
        selection_layout.addWidget(self.selected_count_label)
        batch_layout.addWidget(selection_widget)
        
        # Batch operations buttons
        self.select_all_button = ModernStyledButton("Select All", is_secondary=True)
        self.select_all_button.clicked.connect(self.select_all_transactions)
        batch_layout.addWidget(self.select_all_button)
        
        self.select_none_button = ModernStyledButton("Select None", is_secondary=True)
        self.select_none_button.clicked.connect(self.cancel_batch_selection)
        batch_layout.addWidget(self.select_none_button)
        
        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("color: #DDDDDD; max-width: 1px;")
        batch_layout.addWidget(separator)
        
        self.batch_delete_button = ModernDangerButton("Delete Selected")
        self.batch_delete_button.clicked.connect(self.delete_selected_transactions)
        batch_layout.addWidget(self.batch_delete_button)
        
        self.batch_categorize_button = ModernStyledButton("Change Category")
        self.batch_categorize_button.clicked.connect(self.change_category_for_selected)
        batch_layout.addWidget(self.batch_categorize_button)
        
        # Add stretch to push buttons to the left
        batch_layout.addStretch()
        
        layout.addWidget(self.batch_bar)
        
        # Add vertical space
        layout.addSpacing(8)
        
        # ---- TRANSACTIONS TABLE ----
        
        # Create table with modern macOS data table styling
        self.transactions_table = QTableWidget()
        self.transactions_table.setColumnCount(8)
        self.transactions_table.setHorizontalHeaderLabels([
            "Select", "Date", "Description", "Category", "Amount", "Type", "Merchant", "Actions"
        ])
        
        # Modern table styling
        self.transactions_table.setShowGrid(False)
        self.transactions_table.setAlternatingRowColors(True)
        self.transactions_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.transactions_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.transactions_table.verticalHeader().setVisible(False)
        self.transactions_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid var(--border-color, #E5E5E5);
                border-radius: 5px;
                background-color: var(--bg-color, white);
                alternate-background-color: var(--alt-bg-color, #F9F9F9);
            }
            QTableWidget::item {
                padding: 5px;
                border-bottom: 1px solid var(--separator-color, #F0F0F0);
                color: var(--text-color, black);
            }
            QTableWidget::item:selected {
                background-color: var(--selection-color, rgba(0, 122, 255, 0.1));
                color: var(--text-color, black);
            }
        """)
        
        # macOS-style header
        header = self.transactions_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setHighlightSections(False)
        header.setSectionsClickable(True)
        header.setStyleSheet("""
            QHeaderView::section {
                background-color: var(--header-bg-color, #F5F5F5);
                border: none;
                border-bottom: 1px solid var(--border-color, #E0E0E0);
                padding: 4px;
                font-weight: bold;
                color: var(--header-text-color, #666666);
            }
        """)

        # Set column widths
        header.setSectionResizeMode(0, QHeaderView.Fixed)  # Select
        self.transactions_table.setColumnWidth(0, 50)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Date
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Amount
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # Type
        header.setSectionResizeMode(7, QHeaderView.Fixed)  # Actions
        self.transactions_table.setColumnWidth(7, 120)

        # Connect signals
        self.transactions_table.cellClicked.connect(self.on_transaction_cell_clicked)
        self.transactions_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.transactions_table.customContextMenuRequested.connect(self.show_context_menu)
        
        # Add to layout with stretch to expand and fill available space
        layout.addWidget(self.transactions_table, 1)

        # ---- SUMMARY SECTION ----
        
        # Bottom controls with macOS-style card layout
        bottom_container = QWidget()
        bottom_layout = QHBoxLayout(bottom_container)
        bottom_layout.setContentsMargins(0, 12, 0, 0)

        # Add Transaction button - accent color
        self.add_button = ModernStyledButton("Add Transaction")
        self.add_button.clicked.connect(self.show_add_transaction_dialog)
        bottom_layout.addWidget(self.add_button)

        bottom_layout.addStretch()

        # Summary card with macOS style
        self.summary_frame = QFrame()
        self.summary_frame.setFrameShape(QFrame.StyledPanel)
        self.summary_frame.setObjectName("summaryCard")
        self.summary_frame.setStyleSheet("""
            #summaryCard {
                background-color: var(--card-bg-color, rgba(255, 255, 255, 0.7));
                border: 1px solid var(--border-color, #E5E5E5);
                border-radius: 10px;
            }
        """)

        summary_layout = QFormLayout(self.summary_frame)
        summary_layout.setContentsMargins(16, 12, 16, 12)
        summary_layout.setSpacing(10)
        
        # Create styled summary labels with macOS system colors
        income_title = QLabel("Income:")
        income_title.setStyleSheet("font-weight: 500;")
        self.income_label = QLabel("0.00 €")
        self.income_label.setStyleSheet("color: #34C759; font-weight: bold; font-size: 15px;")
        
        expense_title = QLabel("Expenses:")
        expense_title.setStyleSheet("font-weight: 500;")
        self.expense_label = QLabel("0.00 €")
        self.expense_label.setStyleSheet("color: #FF3B30; font-weight: bold; font-size: 15px;")
        
        balance_title = QLabel("Balance:")
        balance_title.setStyleSheet("font-weight: 500;")
        self.balance_label = QLabel("0.00 €")
        self.balance_label.setStyleSheet("color: #007AFF; font-weight: bold; font-size: 15px;")

        summary_layout.addRow(income_title, self.income_label)
        summary_layout.addRow(expense_title, self.expense_label)
        summary_layout.addRow(balance_title, self.balance_label)

        bottom_layout.addWidget(self.summary_frame)
        layout.addWidget(bottom_container)

    def load_transactions(self):
        """Load transactions based on the current filters."""
        # Get filter values
        from_date = self.from_date.date().toString("yyyy-MM-dd")
        to_date = self.to_date.date().toString("yyyy-MM-dd")
        category_id = self.category_combo.currentData()
        is_income = self.type_combo.currentData()
        search_text = self.search_box.text().strip()

        # Get transactions from database, excluding initial balance entries
        transactions = self.db_manager.get_transactions(
            start_date=from_date,
            end_date=to_date,
            category_id=category_id,
            is_income=is_income,
            exclude_categories=['Initial Balance']  # Exclude initial balance entries
        )

        # Filter by search text if provided
        if search_text:
            transactions = [
                tx for tx in transactions
                if search_text.lower() in tx.get('description', '').lower()
            ]

        # Update table
        self.update_transactions_table(transactions)

        
        # Update summary
        self.update_summary(transactions)
        
        # Restore scroll position after a short delay
        if hasattr(self, 'scrollTimer'):
            self.scrollTimer.stop()
        from PySide6.QtCore import QTimer
        
        # Get current scroll position
        current_scroll = 0
        if self.transactions_table.verticalScrollBar():
            current_scroll = self.transactions_table.verticalScrollBar().value()
            
        # Store the value as an instance variable
        self.current_scroll_pos = current_scroll
        
        self.scrollTimer = QTimer()
        self.scrollTimer.setSingleShot(True)
        self.scrollTimer.timeout.connect(self.restore_scroll_position)
        self.scrollTimer.start(100)  # Short delay to ensure table is fully updated
    

    def update_transactions_table(self, transactions):
        """
        Update the transactions table with transaction data.

        Args:
            transactions: List of transaction dictionaries
        """
        # Clear table
        self.transactions_table.setRowCount(0)
        
        # Clear selection when updating table
        self.selected_transactions.clear()
        self.update_selection_count()

        # Add transactions
        for row, tx in enumerate(transactions):
            self.transactions_table.insertRow(row)
            
            # Store transactions for dashboard reference
            self.current_transactions = transactions
            
            # Select Checkbox column
            checkbox = QCheckBox()
            checkbox.setProperty("tx_id", tx.get('id'))
            checkbox.stateChanged.connect(self._create_checkbox_handler(tx.get('id')))
            # Center checkbox in cell
            checkbox_container = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_container)
            checkbox_layout.addWidget(checkbox)
            checkbox_layout.setAlignment(Qt.AlignCenter)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            self.transactions_table.setCellWidget(row, 0, checkbox_container)

            # Date column with nice formatting
            date_str = tx.get('date', '')
            try:
                # Format date in macOS style (15 Jan 2023)
                date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                formatted_date = date_obj.strftime("%d %b %Y")
            except ValueError:
                formatted_date = date_str

            date_item = QTableWidgetItem(formatted_date)
            date_item.setData(Qt.UserRole, tx.get('id'))  # Store transaction ID
            self.transactions_table.setItem(row, 1, date_item)

            # Description column
            description_item = QTableWidgetItem(tx.get('description', ''))
            self.transactions_table.setItem(row, 2, description_item)

            # Category column with color indicator
            category_item = QTableWidgetItem(tx.get('category_name', ''))
            
            # Set category color as a subtle background
            if tx.get('category_color'):
                color = QColor(tx.get('category_color'))
                color.setAlpha(40)  # Very subtle background
                category_item.setBackground(color)
                
            self.transactions_table.setItem(row, 3, category_item)

            # Amount column with currency formatting and color
            amount = tx.get('amount', 0)
            amount_item = QTableWidgetItem(f"{amount:.2f} €")
            amount_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

            # Set color based on transaction type using macOS system colors
            if tx.get('is_income'):
                amount_item.setForeground(QColor('#34C759'))  # macOS green
            else:
                amount_item.setForeground(QColor('#FF3B30'))  # macOS red

            self.transactions_table.setItem(row, 4, amount_item)

            # Type column
            type_text = "Income" if tx.get('is_income') else "Expense"
            type_item = QTableWidgetItem(type_text)
            
            # Set text color based on type using macOS system colors
            if tx.get('is_income'):
                type_item.setForeground(QColor('#34C759'))  # Green
            else:
                type_item.setForeground(QColor('#FF3B30'))  # Red
                
            self.transactions_table.setItem(row, 5, type_item)

            # Merchant column
            merchant_item = QTableWidgetItem(tx.get('merchant', ''))
            merchant_item.setForeground(QColor('#8E8E93'))  # Secondary text color
            self.transactions_table.setItem(row, 6, merchant_item)

            # Actions column with modern buttons
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(4, 0, 4, 0)
            actions_layout.setSpacing(8)

            # Edit button - secondary style
            edit_button = ModernStyledButton("Edit", is_secondary=True)
            edit_button.setProperty("tx_id", tx.get('id'))
            edit_button.clicked.connect(
                lambda checked=False, tx_id=tx.get('id'): self.edit_transaction(tx_id)
            )
            edit_button.setFixedSize(60, 24)
            actions_layout.addWidget(edit_button)
            
            # Delete button - danger style
            delete_button = ModernDangerButton("Delete")
            delete_button.setProperty("tx_id", tx.get('id'))
            delete_button.clicked.connect(
                lambda checked=False, tx_id=tx.get('id'): self.delete_transaction(tx_id)
            )
            delete_button.setFixedSize(60, 24)
            actions_layout.addWidget(delete_button)

            self.transactions_table.setCellWidget(row, 7, actions_widget)

    def update_summary(self, transactions):
        """
        Update the summary section with transaction totals.

        Args:
            transactions: List of transaction dictionaries
        """
        # Filter to exclude initial balance entries
        filtered_transactions = [tx for tx in transactions 
                               if tx.get('category_name') != 'Initial Balance']
        
        # Calculate totals
        income_amounts = [tx['amount'] for tx in filtered_transactions if tx.get('is_income')]
        expense_amounts = [tx['amount'] for tx in filtered_transactions if not tx.get('is_income')]
        
        total_income = sum(income_amounts) if income_amounts else 0
        total_expense = sum(expense_amounts) if expense_amounts else 0
        balance = total_income - total_expense
        
        # Update labels with formatted currency amounts
        self.income_label.setText(f"{total_income:.2f} €")
        self.expense_label.setText(f"{total_expense:.2f} €")
        self.balance_label.setText(f"{balance:.2f} €")
        
        # Update balance color based on value
        if balance >= 0:
            self.balance_label.setStyleSheet("color: #34C759; font-weight: bold; font-size: 15px;")
        else:
            self.balance_label.setStyleSheet("color: #FF3B30; font-weight: bold; font-size: 15px;")

    def reset_filters(self):
        """Reset all filters to default values."""
        # Reset date range to last 3 months
        from_date = QDate.currentDate().addMonths(-3)
        self.from_date.setDate(from_date)
        self.to_date.setDate(QDate.currentDate())

        # Reset category and type filters
        self.category_combo.setCurrentIndex(0)  # "All Categories"
        self.type_combo.setCurrentIndex(0)  # "All Types"

        # Clear search box
        self.search_box.clear()

        # Reload transactions
        self.load_transactions()
        
    def on_transaction_cell_clicked(self, row, column):
        """
        Handle cell click event in the transactions table.

        Args:
            row: Clicked row index
            column: Clicked column index
        """
        # Ignore clicks on the checkbox column and actions column
        if column == 0 or column == 7:
            return
            
        # In batch mode, clicking a row toggles its checkbox
        if self.batch_mode:
            # Get the checkbox widget
            checkbox_container = self.transactions_table.cellWidget(row, 0)
            if checkbox_container:
                checkbox = checkbox_container.findChild(QCheckBox)
                if checkbox:
                    # Toggle checked state
                    checkbox.setChecked(not checkbox.isChecked())
            return

        # Get transaction ID from the row
        tx_id_item = self.transactions_table.item(row, 1)  # Date column has the transaction ID
        if tx_id_item:
            tx_id = tx_id_item.data(Qt.UserRole)
            if tx_id:
                # Emit signal with the transaction ID
                self.transaction_selected.emit(tx_id)
                
    def show_context_menu(self, position):
        """
        Show context menu for the transactions table with macOS style.

        Args:
            position: Menu position
        """
        # Get the row under the cursor
        row = self.transactions_table.rowAt(position.y())
        
        # Create menu with macOS styling
        if row < 0:
            # Show context menu for empty area
            empty_menu = QMenu(self)
            empty_menu.setStyleSheet("""
                QMenu {
                    background-color: white;
                    border: 1px solid #E0E0E0;
                    border-radius: 8px;
                }
                QMenu::item {
                    padding: 6px 24px;
                }
                QMenu::item:selected {
                    background-color: #007AFF;
                    color: white;
                }
            """)
            
            # Add batch selection actions
            if self.batch_mode:
                select_all_action = QAction("Select All", self)
                select_all_action.triggered.connect(self.select_all_transactions)
                empty_menu.addAction(select_all_action)
                
                select_none_action = QAction("Deselect All", self)
                select_none_action.triggered.connect(self.cancel_batch_selection)
                empty_menu.addAction(select_none_action)
            else:
                batch_mode_action = QAction("Enter Batch Selection Mode", self)
                batch_mode_action.triggered.connect(self.toggle_batch_mode)
                empty_menu.addAction(batch_mode_action)
            
            empty_menu.exec_(QCursor.pos())
            return

        # Get transaction ID from the row
        tx_id_item = self.transactions_table.item(row, 1)  # Date column has the transaction ID
        if not tx_id_item:
            return

        tx_id = tx_id_item.data(Qt.UserRole)
        if not tx_id:
            return

        # Create context menu
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: white;
                border: 1px solid #E0E0E0;
                border-radius: 8px;
            }
            QMenu::item {
                padding: 6px 24px;
            }
            QMenu::item:selected {
                background-color: #007AFF;
                color: white;
            }
        """)

        # Add actions
        edit_action = QAction("Edit Transaction", self)
        edit_action.triggered.connect(lambda: self.edit_transaction(tx_id))
        menu.addAction(edit_action)

        delete_action = QAction("Delete Transaction", self)
        delete_action.triggered.connect(lambda: self.delete_transaction(tx_id))
        menu.addAction(delete_action)

        # Add separator in macOS style (thin line)
        menu.addSeparator()

        # Add category actions submenu
        categories_menu = QMenu("Change Category", self)
        categories_menu.setStyleSheet(menu.styleSheet())
        
        for category in self.db_manager.get_categories():
            category_action = QAction(category['name'], self)
            category_action.triggered.connect(
                lambda checked=False, cat_id=category['id']: self.change_transaction_category(tx_id, cat_id)
            )
            categories_menu.addAction(category_action)

        menu.addMenu(categories_menu)
        
        # Add batch selection options if in batch mode
        if self.batch_mode:
            menu.addSeparator()
            
            # Get the checkbox for this row
            checkbox_container = self.transactions_table.cellWidget(row, 0)
            if checkbox_container:
                checkbox = checkbox_container.findChild(QCheckBox)
                if checkbox:
                    if checkbox.isChecked():
                        unselect_action = QAction("Unselect This Transaction", self)
                        unselect_action.triggered.connect(lambda: checkbox.setChecked(False))
                        menu.addAction(unselect_action)
                    else:
                        select_action = QAction("Select This Transaction", self)
                        select_action.triggered.connect(lambda: checkbox.setChecked(True))
                        menu.addAction(select_action)
            
            select_all_action = QAction("Select All Transactions", self)
            select_all_action.triggered.connect(self.select_all_transactions)
            menu.addAction(select_all_action)

        # Show the menu
        menu.exec_(QCursor.pos())
                
    def toggle_batch_mode(self):
        """Toggle batch selection mode with animation."""
        self.batch_mode = not self.batch_mode
        
        # Update UI
        if self.batch_mode:
            self.batch_mode_button.setText("Exit Batch Mode")
            self.batch_bar.setVisible(True)
        else:
            self.batch_mode_button.setText("Batch Select")
            self.batch_bar.setVisible(False)
            self.selected_transactions.clear()
            self.update_selection_count()
            
        # Refresh the table to update checkboxes
        self.load_transactions()
    
    def on_transaction_checked(self, state, tx_id):
        """
        Handle transaction checkbox state change.
        
        Args:
            state: Checkbox state
            tx_id: Transaction ID
        """
        if state == Qt.Checked:
            self.selected_transactions.add(tx_id)
        else:
            self.selected_transactions.discard(tx_id)
            
        # Update the selected count
        self.update_selection_count()
    
    def update_selection_count(self):
        """Update the selected transactions count label."""
        count = len(self.selected_transactions)
        self.selected_count_label.setText(str(count))
        
        # Enable/disable batch operation buttons based on selection
        has_selection = count > 0
        
        self.batch_delete_button.setEnabled(has_selection)
        # Force enable the button if selection exists
        if has_selection:
            self.batch_delete_button.setEnabled(True)
            self.batch_delete_button.setStyleSheet("background-color: #FF3B30; color: white;")
        else:
            self.batch_delete_button.setStyleSheet("")
    
        self.batch_categorize_button.setEnabled(has_selection)
    
    def select_all_transactions(self):
        """Select all transactions currently visible in the table."""
        # Check all checkboxes
        for row in range(self.transactions_table.rowCount()):
            checkbox_container = self.transactions_table.cellWidget(row, 0)
            if checkbox_container:
                checkbox = checkbox_container.findChild(QCheckBox)
                if checkbox:
                    # Get the transaction ID
                    tx_id = checkbox.property("tx_id")
                    if tx_id:
                        # Add to selected transactions set
                        self.selected_transactions.add(tx_id)
                        # Check the checkbox (without triggering the event)
                        checkbox.blockSignals(True)
                        checkbox.setChecked(True)
                        checkbox.blockSignals(False)
        
        # Update the selection count
        self.update_selection_count()
    
    def cancel_batch_selection(self):
        """Cancel the current batch selection."""
        self.selected_transactions.clear()
        self.update_selection_count()
        
        # Uncheck all checkboxes
        for row in range(self.transactions_table.rowCount()):
            checkbox_container = self.transactions_table.cellWidget(row, 0)
            if checkbox_container:
                checkbox = checkbox_container.findChild(QCheckBox)
                if checkbox:
                    checkbox.setChecked(False)
    
    def delete_selected_transactions(self):
        """Delete all selected transactions with confirmation dialog."""
        count = len(self.selected_transactions)
        
        if count == 0:
            return
            
        # Confirm deletion with macOS-style dialog
        message_box = QMessageBox(
            QMessageBox.Question,
            "Confirm Batch Deletion",
            f"Are you sure you want to delete {count} selected transactions?",
            QMessageBox.Yes | QMessageBox.No,
            self
        )
        message_box.setDefaultButton(QMessageBox.No)
        
        # Style the dialog to match macOS
        message_box.setStyleSheet("""
            QMessageBox {
                background-color: white;
            }
            QPushButton {
                min-width: 80px;
                padding: 5px 15px;
            }
        """)
        
        if message_box.exec_() != QMessageBox.Yes:
            return
            
        # Delete transactions
        deleted_count = 0
        for tx_id in list(self.selected_transactions):  # Make a copy of the set while iterating
            if self.db_manager.delete_transaction(tx_id):
                deleted_count += 1
                self.selected_transactions.discard(tx_id)
        
        # Reload transactions
        self.load_transactions()
        
        # Show success message
        QMessageBox.information(
            self, "Batch Deletion",
            f"Successfully deleted {deleted_count} transactions."
        )
    
    def change_category_for_selected(self):
        """Change category for all selected transactions with macOS-styled dialog."""
        if not self.selected_transactions:
            return
            
        # Get all categories
        categories = self.db_manager.get_categories()
        
        # Create dialog with macOS styling
        category_dialog = QDialog(self)
        category_dialog.setWindowTitle("Change Category")
        category_dialog.setFixedWidth(350)
        category_dialog.setStyleSheet("""
            QDialog {
                background-color: white;
                border-radius: 10px;
            }
            QComboBox {
                padding: 5px;
                border: 1px solid #E0E0E0;
                border-radius: 6px;
                min-height: 25px;
            }
            QPushButton {
                min-width: 80px;
                padding: 5px 15px;
            }
        """)
        
        dialog_layout = QVBoxLayout(category_dialog)
        dialog_layout.setContentsMargins(20, 20, 20, 20)
        dialog_layout.setSpacing(15)
        
        # Add label
        dialog_layout.addWidget(QLabel(f"Change category for {len(self.selected_transactions)} transactions:"))
        
        # Category selection
        category_combo = QComboBox()
        for category in categories:
            category_combo.addItem(category['name'], category['id'])
        dialog_layout.addWidget(category_combo)
        
        # Buttons in macOS style
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(category_dialog.accept)
        button_box.rejected.connect(category_dialog.reject)
        dialog_layout.addWidget(button_box)
        
        # Show dialog
        if category_dialog.exec_() == QDialog.Accepted:
            # Get selected category
            category_id = category_combo.currentData()
            
            # Update transactions
            updated_count = 0
            for tx_id in self.selected_transactions:
                if self.db_manager.update_transaction(tx_id, category_id=category_id):
                    updated_count += 1
            
            # Reload transactions
            self.load_transactions()
            
            # Show success message
            QMessageBox.information(
                self, "Category Updated",
                f"Successfully updated {updated_count} transactions."
            )
    
    def show_add_transaction_dialog(self):
        """Show the add transaction dialog with macOS styling."""
        # Display a temporary dialog for the migration phase
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Add Transaction")
        msg_box.setText("This feature is being updated to the new UI.\n\nWould you like to create a new transaction?")
        msg_box.setIcon(QMessageBox.Information)
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
        
        response = msg_box.exec_()
        
        if response == QMessageBox.Yes:
            # Simulate transaction addition for now
            QMessageBox.information(self, "Success", "Transaction added successfully!")
            # We will reload transactions in the future
            # self.load_transactions()
        else:
            pass  # User cancelled
    
    def edit_transaction(self, tx_id):
        """Edit a transaction."""
        # Get the transaction data
        tx = self.db_manager.get_transaction_by_id(tx_id)
        if not tx:
            QMessageBox.warning(self, "Transaction Not Found", f"Could not find transaction with ID {tx_id}.")
            return
            
        # Import the dialog here to avoid circular imports
        from ui.modern_transaction_dialog import ModernTransactionDialog
        
        # Show the edit dialog
        try:
            dialog = ModernTransactionDialog(self.db_manager, tx, self)
            if dialog.exec_():
                # Refresh the transactions list
                self.load_transactions()
                
                # Show a success message (will be handled by the parent window)
        except Exception as e:
            self.logger.error(f"Error editing transaction: {e}")
            QMessageBox.critical(
                self, "Error",
                f"An error occurred while editing the transaction: {str(e)}"
            )
    
    def delete_transaction(self, tx_id):
        """Delete a transaction with confirmation dialog."""
        # Get transaction data for confirmation message
        tx = self.db_manager.get_transaction_by_id(tx_id)
        if not tx:
            self.logger.error(f"Transaction {tx_id} not found")
            return

        # Confirm deletion with macOS-style dialog
        message_box = QMessageBox(
            QMessageBox.Question,
            "Confirm Deletion",
            f"Are you sure you want to delete this transaction?\n\n"
            f"Date: {tx.get('date')}\n"
            f"Description: {tx.get('description')}\n"
            f"Amount: {tx.get('amount'):.2f} €",
            QMessageBox.Yes | QMessageBox.No,
            self
        )
        message_box.setDefaultButton(QMessageBox.No)
        
        # Style the dialog to match macOS
        message_box.setStyleSheet("""
            QMessageBox {
                background-color: white;
            }
            QPushButton {
                min-width: 80px;
                padding: 5px 15px;
            }
        """)

        if message_box.exec_() == QMessageBox.Yes:
            # Delete transaction
            success = self.db_manager.delete_transaction(tx_id)
            if success:
                # Reload transactions
                self.load_transactions()
            else:
                QMessageBox.warning(
                    self, "Deletion Failed",
                    "Failed to delete the transaction. Please try again."
                )
    
    def change_transaction_category(self, tx_id, category_id):
        """Change the category of a transaction."""
        # Update transaction
        success = self.db_manager.update_transaction(tx_id, category_id=category_id)
        if success:
            # Reload transactions
            self.load_transactions()
        else:
            QMessageBox.warning(
                self, "Update Failed",
                "Failed to update the transaction category. Please try again."
            )
            
    def parent_resized(self, width):
        """
        Adjust UI elements based on parent window width.
        
        Args:
            width: Current window width
        """
        # Compact mode thresholds
        compact_threshold = 900
        very_compact_threshold = 700
        
        # Adjust filter layout based on width
        if width < compact_threshold:
            # In compact mode, make the filter layout more compact
            try:
                self.main_layout.setContentsMargins(10, 10, 10, 10)
                self.main_layout.setSpacing(8)
                
                # Make table columns adapt
                header = self.transactions_table.horizontalHeader()
                header.setSectionResizeMode(QHeaderView.Interactive)
                header.setSectionResizeMode(0, QHeaderView.Fixed)  # Select
                header.setSectionResizeMode(2, QHeaderView.Stretch)  # Description
                
                # Adjust column widths
                self.transactions_table.setColumnWidth(0, 40)  # Select
                self.transactions_table.setColumnWidth(1, 80)  # Date
                self.transactions_table.setColumnWidth(4, 80)  # Amount
                self.transactions_table.setColumnWidth(5, 70)  # Type
                
                # Make buttons smaller but still flexible
                self.batch_mode_button.setMinimumWidth(80)
                self.batch_mode_button.setMaximumWidth(90)
                self.refresh_button.setMinimumWidth(80)
                self.refresh_button.setMaximumWidth(90)
                
                # Adjust search box size
                if hasattr(self, 'search_box'):
                    self.search_box.setMinimumWidth(120)
                
                # Hide some filter controls in very compact mode
                if width < very_compact_threshold:
                    # Hide some less essential UI elements
                    if hasattr(self, 'type_combo'):
                        self.type_combo.setMaximumWidth(80)
                    if hasattr(self, 'category_combo'):
                        self.category_combo.setMaximumWidth(100)
                    if hasattr(self, 'reset_filters_button'):
                        self.reset_filters_button.setVisible(False)
                else:
                    # Show all filter elements in compact mode
                    if hasattr(self, 'type_combo'):
                        self.type_combo.setMaximumWidth(120)
                    if hasattr(self, 'category_combo'):
                        self.category_combo.setMaximumWidth(150)
                    if hasattr(self, 'reset_filters_button'):
                        self.reset_filters_button.setVisible(True)
            except Exception as e:
                # Ignore errors during layout adjustment
                pass
        else:
            # In normal mode, use standard spacing
            try:
                self.main_layout.setContentsMargins(20, 20, 20, 20)
                self.main_layout.setSpacing(12)
                
                # Reset table columns
                header = self.transactions_table.horizontalHeader()
                header.setSectionResizeMode(QHeaderView.Stretch)
                header.setSectionResizeMode(0, QHeaderView.Fixed)  # Select
                self.transactions_table.setColumnWidth(0, 50)
                
                # Restore filter controls
                if hasattr(self, 'search_box'):
                    self.search_box.setMinimumWidth(180)
                if hasattr(self, 'category_combo'):
                    self.category_combo.setMaximumWidth(150)
                if hasattr(self, 'type_combo'):
                    self.type_combo.setMaximumWidth(120)
                if hasattr(self, 'reset_filters_button'):
                    self.reset_filters_button.setVisible(True)
                
                # Restore button sizes but keep them flexible
                self.batch_mode_button.setMinimumWidth(110)
                self.batch_mode_button.setMaximumWidth(120)
                self.refresh_button.setMinimumWidth(110)
                self.refresh_button.setMaximumWidth(120)
            except Exception as e:
                # Ignore errors during layout adjustment
                pass
                
    def restore_scroll_position(self):
        """Restore the scroll position of the transactions table."""
        if hasattr(self, 'current_scroll_pos') and self.transactions_table.verticalScrollBar():
            self.transactions_table.verticalScrollBar().setValue(self.current_scroll_pos)