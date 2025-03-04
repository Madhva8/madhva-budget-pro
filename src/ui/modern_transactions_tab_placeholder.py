#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modern Transactions Tab Placeholder Module

This provides a temporary implementation for the transactions tab that can actually show 
imported transactions. It's a simplified version of the full implementation that's coming.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame, QTabWidget,
    QCheckBox, QDialogButtonBox, QDialog, QComboBox
)
from PySide6.QtCore import Qt, QSize, QDate
from PySide6.QtGui import QFont, QColor, QIcon, QBrush, QAction


class ModernTransactionsTabPlaceholder(QWidget):
    """Temporary transactions tab with basic functionality."""

    def __init__(self, db_manager=None):
        """Initialize the transactions tab widget."""
        super().__init__()
        self.db_manager = db_manager
        
        # State for batch operations
        self.batch_mode = False
        self.selected_transactions = set()  # Set of selected transaction IDs
        
        self.init_ui()
        
        # Load transactions if database manager is available
        if self.db_manager:
            self.load_transactions()

    def init_ui(self):
        """Initialize the user interface."""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        
        # Header section
        header_layout = QHBoxLayout()
        
        # Create a title
        title_label = QLabel("Transactions")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title_label.setFont(title_font)
        header_layout.addWidget(title_label, 1)  # Stretch factor 1
        
        # Batch mode button
        self.batch_mode_button = QPushButton("Batch Select")
        self.batch_mode_button.setFixedWidth(120)
        self.batch_mode_button.setStyleSheet("""
            QPushButton {
                border: 1px solid #007AFF;
                color: #007AFF;
                background-color: white;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: rgba(0, 122, 255, 0.1);
            }
        """)
        self.batch_mode_button.clicked.connect(self.toggle_batch_mode)
        header_layout.addWidget(self.batch_mode_button)
        
        # Refresh button
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setFixedWidth(120)
        self.refresh_button.setStyleSheet("""
            QPushButton {
                background-color: #007AFF;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #0062CC;
            }
        """)
        self.refresh_button.clicked.connect(self.load_transactions)
        header_layout.addWidget(self.refresh_button)
        
        layout.addLayout(header_layout)
        
        # ---- BATCH OPERATIONS BAR ----
        # Create batch operations bar (initially hidden)
        self.batch_bar = QFrame()
        self.batch_bar.setFrameShape(QFrame.StyledPanel)
        self.batch_bar.setObjectName("batchBar")
        self.batch_bar.setStyleSheet("""
            #batchBar {
                background-color: rgba(255, 255, 255, 0.7);
                border-radius: 10px;
                border: 1px solid #E5E5E5;
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
        self.select_all_button = QPushButton("Select All")
        self.select_all_button.setStyleSheet("""
            QPushButton {
                border: 1px solid #007AFF;
                color: #007AFF;
                background-color: white;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: rgba(0, 122, 255, 0.1);
            }
        """)
        self.select_all_button.clicked.connect(self.select_all_transactions)
        batch_layout.addWidget(self.select_all_button)
        
        self.select_none_button = QPushButton("Select None")
        self.select_none_button.setStyleSheet("""
            QPushButton {
                border: 1px solid #007AFF;
                color: #007AFF;
                background-color: white;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: rgba(0, 122, 255, 0.1);
            }
        """)
        self.select_none_button.clicked.connect(self.cancel_batch_selection)
        batch_layout.addWidget(self.select_none_button)
        
        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("color: #DDDDDD; max-width: 1px;")
        batch_layout.addWidget(separator)
        
        self.batch_delete_button = QPushButton("Delete Selected")
        self.batch_delete_button.setStyleSheet("""
            QPushButton {
                background-color: #FF3B30;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #E02D26;
            }
            QPushButton:disabled {
                background-color: #FFBAB6;
            }
        """)
        self.batch_delete_button.setEnabled(False)
        self.batch_delete_button.clicked.connect(self.delete_selected_transactions)
        batch_layout.addWidget(self.batch_delete_button)
        
        self.batch_categorize_button = QPushButton("Change Category")
        self.batch_categorize_button.setStyleSheet("""
            QPushButton {
                background-color: #007AFF;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #0062CC;
            }
            QPushButton:disabled {
                background-color: #AACCFF;
            }
        """)
        self.batch_categorize_button.setEnabled(False)
        self.batch_categorize_button.clicked.connect(self.change_category_for_selected)
        batch_layout.addWidget(self.batch_categorize_button)
        
        # Add stretch to push buttons to the left
        batch_layout.addStretch()
        
        layout.addWidget(self.batch_bar)
        
        # Add note about the new UI
        note_frame = QFrame()
        note_frame.setFrameShape(QFrame.StyledPanel)
        note_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(0, 122, 255, 0.1);
                border: 1px solid rgba(0, 122, 255, 0.3);
                border-radius: 8px;
            }
        """)
        
        note_layout = QVBoxLayout(note_frame)
        note_layout.setContentsMargins(16, 12, 16, 12)
        
        note_title = QLabel("✨ New UI Coming Soon")
        note_title.setStyleSheet("font-weight: bold; color: #007AFF; font-size: 14px;")
        note_layout.addWidget(note_title)
        
        note_text = QLabel(
            "We're building a beautiful new transactions interface with smooth animations "
            "and better filtering. This is a simplified view during the transition."
        )
        note_text.setWordWrap(True)
        note_layout.addWidget(note_text)
        
        layout.addWidget(note_frame)
        
        # Transaction table
        self.transactions_table = QTableWidget()
        self.transactions_table.setColumnCount(7)
        self.transactions_table.setHorizontalHeaderLabels([
            "Select", "Date", "Description", "Category", "Amount", "Type", "Merchant"
        ])
        
        # Modern table styling
        self.transactions_table.setAlternatingRowColors(True)
        self.transactions_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.transactions_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.transactions_table.verticalHeader().setVisible(False)
        self.transactions_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #E5E5E5;
                border-radius: 6px;
                background-color: white;
                alternate-background-color: #F9F9F9;
            }
            QHeaderView::section {
                background-color: #F5F5F5;
                border: none;
                border-bottom: 1px solid #E0E0E0;
                padding: 4px;
            }
        """)
        
        # Connect signals
        self.transactions_table.cellClicked.connect(self.on_transaction_cell_clicked)
        
        # Configure columns
        header = self.transactions_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        
        # Fixed width columns
        header.setSectionResizeMode(0, QHeaderView.Fixed)  # Select
        self.transactions_table.setColumnWidth(0, 50)
        
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Date
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Amount
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # Type
        
        layout.addWidget(self.transactions_table, 1)  # Stretch factor 1
        
        # Add info button at the bottom
        info_layout = QHBoxLayout()
        info_layout.addStretch(1)
        
        info_button = QPushButton("Learn More")
        info_button.setFixedWidth(150)
        info_button.setStyleSheet("""
            QPushButton {
                border: 1px solid #007AFF;
                color: #007AFF;
                background-color: white;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: rgba(0, 122, 255, 0.1);
            }
        """)
        info_button.clicked.connect(self.show_info)
        info_layout.addWidget(info_button)
        
        layout.addLayout(info_layout)
        
    def load_transactions(self):
        """Load and display transactions from the database."""
        if not self.db_manager:
            # Show a message if no database manager is available
            self.transactions_table.setRowCount(0)
            QMessageBox.information(
                self, 
                "Database Connection", 
                "No database connection available. Please restart the application."
            )
            return
        
        # Get transactions for the last 3 months
        end_date = QDate.currentDate().toString("yyyy-MM-dd")
        start_date = QDate.currentDate().addMonths(-3).toString("yyyy-MM-dd")
        
        try:
            # Get transactions from database
            transactions = self.db_manager.get_transactions(
                start_date=start_date,
                end_date=end_date,
                exclude_categories=['Initial Balance']  # Exclude initial balance entries
            )
            
            # Clear existing data
            self.transactions_table.setRowCount(0)
            
            # Clear selection when updating table
            self.selected_transactions.clear()
            self.update_selection_count()
            
            # Add transactions to table
            for row, tx in enumerate(transactions):
                self.transactions_table.insertRow(row)
                
                # Select checkbox column
                checkbox = QCheckBox()
                checkbox.setProperty("tx_id", tx.get('id'))
                checkbox.stateChanged.connect(
                    lambda state, tx_id=tx.get('id'): self.on_transaction_checked(state, tx_id)
                )
                # Center checkbox in cell
                checkbox_container = QWidget()
                checkbox_layout = QHBoxLayout(checkbox_container)
                checkbox_layout.addWidget(checkbox)
                checkbox_layout.setAlignment(Qt.AlignCenter)
                checkbox_layout.setContentsMargins(0, 0, 0, 0)
                
                self.transactions_table.setCellWidget(row, 0, checkbox_container)
                
                # Date column
                date_item = QTableWidgetItem(tx.get('date', ''))
                date_item.setData(Qt.UserRole, tx.get('id'))  # Store transaction ID
                self.transactions_table.setItem(row, 1, date_item)
                
                # Description column
                description_item = QTableWidgetItem(tx.get('description', ''))
                self.transactions_table.setItem(row, 2, description_item)
                
                # Category column
                category_item = QTableWidgetItem(tx.get('category_name', ''))
                self.transactions_table.setItem(row, 3, category_item)
                
                # Amount column
                amount = tx.get('amount', 0)
                amount_item = QTableWidgetItem(f"{amount:.2f} €")
                amount_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                
                # Set color based on transaction type
                if tx.get('is_income'):
                    amount_item.setForeground(QBrush(QColor('#34C759')))  # macOS green
                else:
                    amount_item.setForeground(QBrush(QColor('#FF3B30')))  # macOS red
                    
                self.transactions_table.setItem(row, 4, amount_item)
                
                # Type column
                type_item = QTableWidgetItem("Income" if tx.get('is_income') else "Expense")
                # Set text color based on type using macOS system colors
                if tx.get('is_income'):
                    type_item.setForeground(QBrush(QColor('#34C759')))  # Green
                else:
                    type_item.setForeground(QBrush(QColor('#FF3B30')))  # Red
                self.transactions_table.setItem(row, 5, type_item)
                
                # Merchant column
                merchant_item = QTableWidgetItem(tx.get('merchant', ''))
                self.transactions_table.setItem(row, 6, merchant_item)
                
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Error Loading Transactions", 
                f"An error occurred loading transactions: {str(e)}"
            )
        
    def on_transaction_cell_clicked(self, row, column):
        """
        Handle cell click event in the transactions table.
        
        Args:
            row: Clicked row index
            column: Clicked column index
        """
        # Ignore clicks on the checkbox column
        if column == 0:
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
        """Change category for all selected transactions."""
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
    
    def show_info(self):
        """Show information about the upcoming feature."""
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Coming Soon")
        msg_box.setText(
            "The Transactions Tab is being rebuilt with PySide6!\n\n"
            "New features will include:\n"
            "• Seamless macOS integration\n"
            "• Smooth animations and transitions\n"
            "• Better filtering and sorting options\n"
            "• Improved batch operations\n"
            "• Enhanced visualization of income and expenses\n"
            "• Native macOS dialogs and controls"
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