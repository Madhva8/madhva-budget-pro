#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transactions Tab Module

This module defines the transactions tab for the Financial Planner,
providing a view of transactions with filtering and editing capabilities.
"""

import logging
import datetime
from typing import List, Dict, Any, Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit,
    QDateEdit, QComboBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QFrame, QSplitter, QMessageBox, QMenu, QAction, QCheckBox, QGroupBox,
    QFormLayout, QGraphicsDropShadowEffect
)
from PyQt5.QtCore import Qt, QDate, pyqtSignal
from PyQt5.QtGui import QIcon, QColor, QCursor, QFont

from ui.add_transaction_dialog import AddTransactionDialog
from ui.styled_button import StyledButton, DangerButton, SuccessButton


class TransactionsTab(QWidget):
    """Widget representing the transactions tab in the application."""

    # Signals
    transaction_selected = pyqtSignal(int)  # Emitted when a transaction is selected

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
        self.selected_transactions = set()  # Set of selected transaction IDs

        # Initialize UI
        self.init_ui()

        # Load initial data
        self.load_transactions()

    def init_ui(self):
        """Initialize the user interface with macOS-inspired styling."""
        # Main layout with margins for modern look
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Batch operations bar with simpler styling
        self.batch_bar = QFrame()
        self.batch_bar.setFrameShape(QFrame.StyledPanel)
        self.batch_bar.setFixedHeight(50)
        self.batch_bar.setVisible(False)  # Initially hidden
        
        batch_layout = QHBoxLayout(self.batch_bar)
        batch_layout.setContentsMargins(16, 8, 16, 8)
        
        # Selection label with macOS-style counter badge
        selection_widget = QWidget()
        selection_layout = QHBoxLayout(selection_widget)
        selection_layout.setContentsMargins(0, 0, 0, 0)
        selection_layout.setSpacing(6)
        
        selection_label = QLabel("Selected:")
        selection_label.setStyleSheet("font-weight: medium;")
        selection_layout.addWidget(selection_label)
        
        self.selected_count_label = QLabel("0")
        self.selected_count_label.setStyleSheet("""
            background-color: #007AFF;
            color: white;
            border-radius: 10px;
            padding: 2px 8px;
            min-width: 20px;
            text-align: center;
            font-weight: bold;
        """)
        selection_layout.addWidget(self.selected_count_label)
        batch_layout.addWidget(selection_widget)
        
        # Batch operation buttons styled as macOS toolbar buttons
        self.select_all_button = StyledButton("Select All", is_secondary=True)
        self.select_all_button.clicked.connect(self.select_all_transactions)
        batch_layout.addWidget(self.select_all_button)
        
        self.select_none_button = StyledButton("Select None", is_secondary=True)
        self.select_none_button.clicked.connect(self.cancel_batch_selection)
        batch_layout.addWidget(self.select_none_button)
        
        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("color: #CCCCCC;")
        separator.setFixedWidth(1)
        batch_layout.addWidget(separator)
        
        self.batch_delete_button = DangerButton("Delete Selected")
        self.batch_delete_button.clicked.connect(self.delete_selected_transactions)
        batch_layout.addWidget(self.batch_delete_button)
        
        self.batch_categorize_button = StyledButton("Change Category")
        self.batch_categorize_button.clicked.connect(self.change_category_for_selected)
        batch_layout.addWidget(self.batch_categorize_button)
        
        # Add stretch to push buttons to the left
        batch_layout.addStretch()
        
        layout.addWidget(self.batch_bar)

        # Top controls - Filters with simpler styling
        filter_frame = QFrame()
        filter_frame.setObjectName("filterCard")
        filter_frame.setFrameShape(QFrame.StyledPanel)
        filter_frame.setMinimumHeight(70)

        filter_layout = QHBoxLayout(filter_frame)
        filter_layout.setContentsMargins(20, 16, 20, 16)
        filter_layout.setSpacing(16)

        # Date range filter with macOS style
        date_layout = QHBoxLayout()
        date_layout.setSpacing(8)

        # From date
        from_date_label = QLabel("From:")
        from_date_label.setStyleSheet("font-weight: medium;")
        date_layout.addWidget(from_date_label)
        
        self.from_date = QDateEdit()
        self.from_date.setCalendarPopup(True)
        self.from_date.setFixedWidth(120)
        # Default to 3 months ago
        from_date = QDate.currentDate().addMonths(-3)
        self.from_date.setDate(from_date)
        date_layout.addWidget(self.from_date)

        # To date
        to_date_label = QLabel("To:")
        to_date_label.setStyleSheet("font-weight: medium;")
        date_layout.addWidget(to_date_label)
        
        self.to_date = QDateEdit()
        self.to_date.setCalendarPopup(True)
        self.to_date.setFixedWidth(120)
        self.to_date.setDate(QDate.currentDate())
        date_layout.addWidget(self.to_date)

        filter_layout.addLayout(date_layout)

        # Add vertical separator
        v_separator1 = QFrame()
        v_separator1.setFrameShape(QFrame.VLine)
        v_separator1.setFrameShadow(QFrame.Sunken)
        v_separator1.setStyleSheet("color: #DDDDDD;")
        filter_layout.addWidget(v_separator1)

        # Category filter
        category_label = QLabel("Category:")
        category_label.setStyleSheet("font-weight: medium;")
        filter_layout.addWidget(category_label)
        
        self.category_combo = QComboBox()
        self.category_combo.setFixedWidth(150)
        self.category_combo.addItem("All Categories", None)
        # Populate with categories
        categories = self.db_manager.get_categories()
        for category in categories:
            self.category_combo.addItem(category['name'], category['id'])
        filter_layout.addWidget(self.category_combo)

        # Add vertical separator
        v_separator2 = QFrame()
        v_separator2.setFrameShape(QFrame.VLine)
        v_separator2.setFrameShadow(QFrame.Sunken)
        v_separator2.setStyleSheet("color: #DDDDDD;")
        filter_layout.addWidget(v_separator2)

        # Transaction type filter
        type_label = QLabel("Type:")
        type_label.setStyleSheet("font-weight: medium;")
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
        v_separator3.setStyleSheet("color: #DDDDDD;")
        filter_layout.addWidget(v_separator3)

        # Search box with macOS-style search field
        search_label = QLabel("Search:")
        search_label.setStyleSheet("font-weight: medium;")
        filter_layout.addWidget(search_label)
        
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search in description...")
        self.search_box.setMinimumWidth(180)
        filter_layout.addWidget(self.search_box)

        filter_layout.addStretch()

        # Apply filters button
        self.apply_filters_button = StyledButton("Apply Filters")
        self.apply_filters_button.clicked.connect(self.load_transactions)
        filter_layout.addWidget(self.apply_filters_button)

        # Reset filters button
        self.reset_filters_button = StyledButton("Reset", is_secondary=True)
        self.reset_filters_button.clicked.connect(self.reset_filters)
        filter_layout.addWidget(self.reset_filters_button)
        
        # Batch selection button
        self.batch_mode_button = StyledButton("Batch Select", is_secondary=True)
        self.batch_mode_button.clicked.connect(self.toggle_batch_mode)
        filter_layout.addWidget(self.batch_mode_button)

        layout.addWidget(filter_frame)
        
        # Add spacing between filter card and table
        layout.addSpacing(16)

        # Transactions table with cleaner styling
        table_container = QWidget()
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)
        
        self.transactions_table = QTableWidget()
        self.transactions_table.setColumnCount(8)  # Added a select column
        self.transactions_table.setHorizontalHeaderLabels([
            "Select", "Date", "Description", "Category", "Amount", "Type", "Merchant", "Actions"
        ])
        
        # Modern table styling - simpler with better compatibility
        self.transactions_table.setAlternatingRowColors(True)
        self.transactions_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.transactions_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.transactions_table.verticalHeader().setVisible(False)  # Hide row numbers
        
        # Customize header
        header = self.transactions_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setHighlightSections(False)
        header.setSectionsClickable(True)

        # Set column widths
        header.setSectionResizeMode(0, QHeaderView.Fixed)  # Select
        self.transactions_table.setColumnWidth(0, 50)  # Select width
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Date
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Amount
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # Type
        header.setSectionResizeMode(7, QHeaderView.Fixed)  # Actions
        self.transactions_table.setColumnWidth(7, 120)  # Actions width

        # Connect signals
        self.transactions_table.cellClicked.connect(self.on_transaction_cell_clicked)
        self.transactions_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.transactions_table.customContextMenuRequested.connect(self.show_context_menu)
        
        table_layout.addWidget(self.transactions_table)
        layout.addWidget(table_container, 1)  # 1 = stretch factor

        # Bottom controls with macOS-style card design
        bottom_container = QWidget()
        bottom_layout = QHBoxLayout(bottom_container)
        bottom_layout.setContentsMargins(0, 16, 0, 0)

        # Add Transaction button
        self.add_button = StyledButton("Add Transaction")
        self.add_button.clicked.connect(self.show_add_transaction_dialog)
        bottom_layout.addWidget(self.add_button)

        bottom_layout.addStretch()

        # Summary section with simpler styling
        self.summary_frame = QFrame()
        self.summary_frame.setFrameShape(QFrame.StyledPanel)
        self.summary_frame.setObjectName("summaryCard")

        summary_layout = QFormLayout(self.summary_frame)
        summary_layout.setContentsMargins(20, 16, 20, 16)
        summary_layout.setSpacing(12)
        
        # Create styled summary labels with system colors
        income_title = QLabel("Income:")
        income_title.setFont(QFont("", -1, QFont.Bold))
        self.income_label = QLabel("0.00 €")
        self.income_label.setStyleSheet("color: green; font-weight: bold;")
        
        expense_title = QLabel("Expenses:")
        expense_title.setFont(QFont("", -1, QFont.Bold))
        self.expense_label = QLabel("0.00 €")
        self.expense_label.setStyleSheet("color: red; font-weight: bold;")
        
        balance_title = QLabel("Balance:")
        balance_title.setFont(QFont("", -1, QFont.Bold))
        self.balance_label = QLabel("0.00 €")
        self.balance_label.setStyleSheet("color: blue; font-weight: bold;")

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
            
            # Select Checkbox - styled for macOS
            checkbox_widget = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_widget)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            checkbox_layout.setAlignment(Qt.AlignCenter)
            
            checkbox = QCheckBox()
            checkbox.setProperty("tx_id", tx.get('id'))
            checkbox.stateChanged.connect(
                lambda state, tx_id=tx.get('id'): self.on_transaction_checked(state, tx_id)
            )
            checkbox_layout.addWidget(checkbox)
            
            self.transactions_table.setCellWidget(row, 0, checkbox_widget)

            # Date with macOS formatting
            date_str = tx.get('date', '')
            try:
                # Format date for display (if valid)
                date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                formatted_date = date_obj.strftime("%d.%m.%Y")
            except ValueError:
                formatted_date = date_str

            date_item = QTableWidgetItem(formatted_date)
            date_item.setData(Qt.UserRole, tx.get('id'))  # Store transaction ID
            date_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            date_item.setFont(QFont("SF Pro Text", -1, QFont.Medium))
            self.transactions_table.setItem(row, 1, date_item)

            # Description with better formatting
            description_item = QTableWidgetItem(tx.get('description', ''))
            description_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.transactions_table.setItem(row, 2, description_item)

            # Category with simpler styling
            category_item = QTableWidgetItem(tx.get('category_name', ''))
            
            # Set category color as background
            if tx.get('category_color'):
                color = QColor(tx.get('category_color'))
                # Make it more transparent for better readability
                color.setAlpha(50)
                category_item.setBackground(color)
                
            self.transactions_table.setItem(row, 3, category_item)

            # Amount with macOS-style formatting
            amount = tx.get('amount', 0)
            amount_item = QTableWidgetItem(f"{amount:.2f} €")
            amount_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            amount_item.setFont(QFont("SF Pro Text", -1, QFont.Bold))

            # Set color based on transaction type with macOS system colors
            if tx.get('is_income'):
                amount_item.setForeground(QColor('#34C759'))  # macOS green
            else:
                amount_item.setForeground(QColor('#FF3B30'))  # macOS red

            self.transactions_table.setItem(row, 4, amount_item)

            # Type with simple text styling
            type_item = QTableWidgetItem("Income" if tx.get('is_income') else "Expense")
            
            # Style based on transaction type
            if tx.get('is_income'):
                type_item.setForeground(QColor("green"))
            else:
                type_item.setForeground(QColor("red"))
                
            self.transactions_table.setItem(row, 5, type_item)

            # Merchant
            merchant_item = QTableWidgetItem(tx.get('merchant', ''))
            merchant_item.setForeground(QColor('#8E8E93'))  # Subdued color for less important info
            self.transactions_table.setItem(row, 6, merchant_item)

            # Actions - Add standard buttons for edit and delete
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(4, 0, 4, 0)
            actions_layout.setSpacing(8)

            # Edit button - standard QPushButton
            edit_button = QPushButton("Edit")
            edit_button.setProperty("tx_id", tx.get('id'))
            edit_button.clicked.connect(
                lambda checked, tx_id=tx.get('id'): self.edit_transaction(tx_id)
            )
            edit_button.setFixedSize(70, 28)
            actions_layout.addWidget(edit_button)
            
            # Delete button - standard QPushButton
            delete_button = QPushButton("Delete")
            delete_button.setProperty("tx_id", tx.get('id'))
            delete_button.clicked.connect(
                lambda checked, tx_id=tx.get('id'): self.delete_transaction(tx_id)
            )
            delete_button.setFixedSize(70, 28)
            actions_layout.addWidget(delete_button)

            self.transactions_table.setCellWidget(row, 7, actions_widget)

    def update_summary(self, transactions):
        """
        Update the summary section with transaction totals.

        Args:
            transactions: List of transaction dictionaries
        """
        # Debug the transaction totals
        self.logger.info(f"Calculating summary for {len(transactions)} transactions")
        
        # Count transactions by category
        categories = {}
        for tx in transactions:
            cat = tx.get('category_name', 'Uncategorized')
            if cat not in categories:
                categories[cat] = 0
            categories[cat] += 1
        
        # Log categories for debugging
        self.logger.info(f"Transaction categories: {categories}")
        
        # Count transactions by type
        income_count = sum(1 for tx in transactions if tx.get('is_income'))
        expense_count = sum(1 for tx in transactions if not tx.get('is_income'))
        self.logger.info(f"Income transactions: {income_count}, Expense transactions: {expense_count}")
        
        # Filter to exclude initial balance entries
        filtered_transactions = [tx for tx in transactions 
                               if tx.get('category_name') != 'Initial Balance']
        
        # Calculate totals with more detailed logging
        income_amounts = [tx['amount'] for tx in filtered_transactions if tx.get('is_income')]
        expense_amounts = [tx['amount'] for tx in filtered_transactions if not tx.get('is_income')]
        
        total_income = sum(income_amounts) if income_amounts else 0
        total_expense = sum(expense_amounts) if expense_amounts else 0
        balance = total_income - total_expense
        
        # Debug output
        self.logger.info(f"Total income: {total_income:.2f} €, Total expense: {total_expense:.2f} €, Balance: {balance:.2f} €")
        
        # Log largest transactions for verification
        if income_amounts:
            largest_income = max(income_amounts)
            self.logger.info(f"Largest income transaction: {largest_income:.2f} €")
            
            # Check if largest income is from Picnic (salary)
            salary_total = sum(tx['amount'] for tx in filtered_transactions 
                              if tx.get('is_income') and tx.get('category_name') == 'Salary')
            self.logger.info(f"Total salary income: {salary_total:.2f} €")
            
        if expense_amounts:
            largest_expense = max(expense_amounts)
            self.logger.info(f"Largest expense transaction: {largest_expense:.2f} €")
            
            # Check housing expenses
            housing_total = sum(tx['amount'] for tx in filtered_transactions 
                               if not tx.get('is_income') and tx.get('category_name') == 'Housing')
            self.logger.info(f"Total housing expenses: {housing_total:.2f} €")
        
        # Update labels
        self.income_label.setText(f"{total_income:.2f} €")
        self.income_label.setStyleSheet("color: #2ecc71;")  # Green

        self.expense_label.setText(f"{total_expense:.2f} €")
        self.expense_label.setStyleSheet("color: #e74c3c;")  # Red

        self.balance_label.setText(f"{balance:.2f} €")
        if balance >= 0:
            self.balance_label.setStyleSheet("color: #2ecc71;")  # Green
        else:
            self.balance_label.setStyleSheet("color: #e74c3c;")  # Red

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
            checkbox_widget = self.transactions_table.cellWidget(row, 0)
            if checkbox_widget:
                checkbox = checkbox_widget.findChild(QCheckBox)
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
        Show context menu for the transactions table.

        Args:
            position: Menu position
        """
        # Get the row under the cursor
        row = self.transactions_table.rowAt(position.y())
        if row < 0:
            # Show context menu for empty area
            empty_menu = QMenu(self)
            
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

        # Add actions
        edit_action = QAction("Edit Transaction", self)
        edit_action.triggered.connect(lambda: self.edit_transaction(tx_id))
        menu.addAction(edit_action)

        delete_action = QAction("Delete Transaction", self)
        delete_action.triggered.connect(lambda: self.delete_transaction(tx_id))
        menu.addAction(delete_action)

        # Add duplicate detection and deletion
        find_duplicates_action = QAction("Find Duplicates", self)
        find_duplicates_action.triggered.connect(lambda: self.find_duplicate_transactions(tx_id))
        menu.addAction(find_duplicates_action)

        # Add separator
        menu.addSeparator()

        # Add category actions
        categories_menu = QMenu("Change Category", self)
        for category in self.db_manager.get_categories():
            category_action = QAction(category['name'], self)
            category_action.triggered.connect(
                lambda checked, cat_id=category['id']: self.change_transaction_category(tx_id, cat_id)
            )
            categories_menu.addAction(category_action)

        menu.addMenu(categories_menu)
        
        # Add batch selection options if in batch mode
        if self.batch_mode:
            menu.addSeparator()
            
            # Get the checkbox widget for this row
            checkbox_widget = self.transactions_table.cellWidget(row, 0)
            if checkbox_widget:
                checkbox = checkbox_widget.findChild(QCheckBox)
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

    def show_add_transaction_dialog(self):
        """Show the add transaction dialog."""
        dialog = AddTransactionDialog(self.db_manager)
        if dialog.exec_():
            # Reload transactions
            self.load_transactions()

    def edit_transaction(self, tx_id):
        """
        Show the edit transaction dialog.

        Args:
            tx_id: Transaction ID to edit
        """
        # Get transaction data
        tx = self.db_manager.get_transaction_by_id(tx_id)
        if not tx:
            self.logger.error(f"Transaction {tx_id} not found")
            return

        # Show edit dialog
        dialog = AddTransactionDialog(self.db_manager, tx)
        if dialog.exec_():
            # Reload transactions
            self.load_transactions()

    def delete_transaction(self, tx_id):
        """
        Delete a transaction after confirmation.

        Args:
            tx_id: Transaction ID to delete
        """
        # Get transaction data for confirmation message
        tx = self.db_manager.get_transaction_by_id(tx_id)
        if not tx:
            self.logger.error(f"Transaction {tx_id} not found")
            return

        # Confirm deletion
        reply = QMessageBox.question(
            self, "Confirm Deletion",
            f"Are you sure you want to delete this transaction?\n\n"
            f"Date: {tx.get('date')}\n"
            f"Description: {tx.get('description')}\n"
            f"Amount: {tx.get('amount'):.2f} €",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply == QMessageBox.Yes:
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
        """
        Change the category of a transaction.

        Args:
            tx_id: Transaction ID
            category_id: New category ID
        """
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
            
    def toggle_batch_mode(self):
        """Toggle batch selection mode."""
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
            state: Checkbox state (Qt.Checked or Qt.Unchecked)
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
        self.batch_categorize_button.setEnabled(has_selection)
    
    def cancel_batch_selection(self):
        """Cancel the current batch selection."""
        self.selected_transactions.clear()
        self.update_selection_count()
        
        # Uncheck all checkboxes
        for row in range(self.transactions_table.rowCount()):
            checkbox_widget = self.transactions_table.cellWidget(row, 0)
            if checkbox_widget:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox:
                    checkbox.setChecked(False)
    
    def delete_selected_transactions(self):
        """Delete all selected transactions."""
        count = len(self.selected_transactions)
        
        if count == 0:
            return
            
        # Confirm deletion
        reply = QMessageBox.question(
            self, "Confirm Batch Deletion",
            f"Are you sure you want to delete {count} selected transactions?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
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
    
    def select_all_transactions(self):
        """Select all transactions currently visible in the table."""
        # Check all checkboxes
        for row in range(self.transactions_table.rowCount()):
            checkbox_widget = self.transactions_table.cellWidget(row, 0)
            if checkbox_widget:
                checkbox = checkbox_widget.findChild(QCheckBox)
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
        
        # Show confirmation
        count = len(self.selected_transactions)
        self.logger.info(f"Selected all {count} transactions")
        
    def change_category_for_selected(self):
        """Change category for all selected transactions."""
        if not self.selected_transactions:
            return
            
        # Get all categories
        categories = self.db_manager.get_categories()
        
        # Create dialog
        category_dialog = QDialog(self)
        category_dialog.setWindowTitle("Change Category for Selected Transactions")
        dialog_layout = QVBoxLayout(category_dialog)
        
        # Category selection
        form_layout = QFormLayout()
        category_combo = QComboBox()
        
        for category in categories:
            category_combo.addItem(category['name'], category['id'])
            
        form_layout.addRow("New Category:", category_combo)
        dialog_layout.addLayout(form_layout)
        
        # Buttons
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
                self, "Batch Update",
                f"Successfully updated {updated_count} transactions."
            )
            
    def find_duplicate_transactions(self, tx_id):
        """
        Find potential duplicate transactions based on amount and date range.
        
        Args:
            tx_id: Reference transaction ID
        """
        # Get the reference transaction
        ref_tx = self.db_manager.get_transaction_by_id(tx_id)
        if not ref_tx:
            self.logger.error(f"Transaction {tx_id} not found")
            return
            
        # Get all transactions with the same amount
        ref_amount = ref_tx.get('amount', 0)
        ref_date = ref_tx.get('date', '')
        
        # Parse the reference date
        try:
            ref_date_obj = datetime.datetime.strptime(ref_date, "%Y-%m-%d").date()
            
            # Date range (7 days before and after the reference date)
            start_date = (ref_date_obj - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
            end_date = (ref_date_obj + datetime.timedelta(days=7)).strftime("%Y-%m-%d")
            
            # Find potential duplicates
            all_transactions = self.db_manager.get_transactions(
                start_date=start_date,
                end_date=end_date
            )
            
            # Filter transactions with the same amount and different ID
            potential_duplicates = [
                tx for tx in all_transactions
                if abs(tx.get('amount', 0) - ref_amount) < 0.01 and tx.get('id') != tx_id
            ]
            
            if potential_duplicates:
                self.show_duplicate_dialog(ref_tx, potential_duplicates)
            else:
                QMessageBox.information(
                    self, "No Duplicates Found",
                    "No potential duplicate transactions were found."
                )
            
        except (ValueError, TypeError) as e:
            self.logger.error(f"Error finding duplicates: {e}")
            QMessageBox.warning(
                self, "Error",
                "An error occurred while searching for duplicates."
            )
    
    def show_duplicate_dialog(self, reference_tx, duplicates):
        """
        Show a dialog with potential duplicate transactions.
        
        Args:
            reference_tx: Reference transaction
            duplicates: List of potential duplicate transactions
        """
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Potential Duplicate Transactions")
        
        # Create message
        message = f"The following transactions have the same amount as the selected transaction:\n\n"
        message += f"Reference: {reference_tx.get('date')} - {reference_tx.get('description')} - {reference_tx.get('amount'):.2f} €\n\n"
        message += "Potential duplicates:\n"
        
        # Add duplicates to message with checkboxes
        for i, tx in enumerate(duplicates):
            message += f"{i+1}. {tx.get('date')} - {tx.get('description')} - {tx.get('amount'):.2f} €\n"
        
        message += "\nDo you want to delete any of these potential duplicates?"
        dialog.setText(message)
        
        # Add buttons
        dialog.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        dialog.setDefaultButton(QMessageBox.No)
        
        if dialog.exec_() == QMessageBox.Yes:
            # Show a new dialog with checkboxes for each duplicate
            self.show_delete_duplicates_dialog(duplicates)
    
    def show_delete_duplicates_dialog(self, duplicates):
        """
        Show a dialog to select duplicate transactions to delete.
        
        Args:
            duplicates: List of potential duplicate transactions
        """
        # Create dialog
        delete_dialog = QMessageBox(self)
        delete_dialog.setWindowTitle("Delete Duplicate Transactions")
        delete_dialog.setText("Select the transactions to delete:")
        
        # Create checkbox for each duplicate
        checkboxes = []
        for i, tx in enumerate(duplicates):
            checkbox = QCheckBox(f"{tx.get('date')} - {tx.get('description')} - {tx.get('amount'):.2f} €")
            checkbox.setProperty("tx_id", tx.get('id'))
            checkboxes.append(checkbox)
            
            # Add checkbox to dialog
            delete_dialog.layout().addWidget(checkbox)
        
        # Add buttons
        delete_dialog.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        delete_dialog.setDefaultButton(QMessageBox.Cancel)
        
        if delete_dialog.exec_() == QMessageBox.Ok:
            # Get selected transactions
            to_delete = [cb.property("tx_id") for cb in checkboxes if cb.isChecked()]
            
            if to_delete:
                # Confirm deletion
                confirm = QMessageBox.question(
                    self, "Confirm Deletion",
                    f"Are you sure you want to delete {len(to_delete)} transactions?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                )
                
                if confirm == QMessageBox.Yes:
                    # Delete selected transactions
                    deleted_count = 0
                    for tx_id in to_delete:
                        if self.db_manager.delete_transaction(tx_id):
                            deleted_count += 1
                    
                    # Reload transactions
                    self.load_transactions()
                    
                    QMessageBox.information(
                        self, "Transactions Deleted",
                        f"Successfully deleted {deleted_count} transactions."
                    )