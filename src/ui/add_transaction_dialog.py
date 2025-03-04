#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Add Transaction Dialog Module

This module defines a dialog for adding or editing transactions.
"""

import logging
import datetime
from typing import Dict, Any, Optional
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QLineEdit,
    QDateEdit, QComboBox, QCheckBox, QTextEdit, QPushButton, QMessageBox,
    QGraphicsDropShadowEffect, QFrame, QWidget
)
from PyQt5.QtCore import Qt, QDate, QRegExp
from PyQt5.QtGui import QRegExpValidator, QFont, QColor

from ui.styled_button import StyledButton


class AddTransactionDialog(QDialog):
    """Dialog for adding or editing transactions."""

    def __init__(self, db_manager, transaction=None, parent=None):
        """
        Initialize the add transaction dialog.

        Args:
            db_manager: Database manager instance
            transaction: Transaction data for editing (None for new transaction)
            parent: Parent widget
        """
        super().__init__(parent)

        self.db_manager = db_manager
        self.transaction = transaction
        self.logger = logging.getLogger(__name__)

        # Initialize UI
        self.init_ui()

        # Set data if editing
        if self.transaction:
            self.set_transaction_data()

    def init_ui(self):
        """Initialize the user interface with macOS styling."""
        # Set window properties
        self.setWindowTitle("Add Transaction" if not self.transaction else "Edit Transaction")
        self.setMinimumWidth(500)
        self.setMinimumHeight(550)
        
        # Standard dialog styling
        
        # Main layout with margins
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)
        
        # Title with standard styling
        title_label = QLabel(self.windowTitle())
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        # Form container with simple frame
        form_container = QFrame()
        form_container.setFrameShape(QFrame.StyledPanel)
        
        # Form layout with better spacing
        form_layout = QFormLayout(form_container)
        form_layout.setContentsMargins(24, 24, 24, 24)
        form_layout.setSpacing(16)
        form_layout.setLabelAlignment(Qt.AlignLeft)
        form_layout.setVerticalSpacing(16)

        # Date input with macOS styling
        date_label = QLabel("Date:")
        date_label.setStyleSheet("font-weight: medium;")
        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDate(QDate.currentDate())
        self.date_edit.setMinimumHeight(32)
        form_layout.addRow(date_label, self.date_edit)

        # Transaction type selector styled as segmented control
        type_label = QLabel("Type:")
        type_label.setStyleSheet("font-weight: medium;")
        
        type_container = QWidget()
        type_layout = QHBoxLayout(type_container)
        type_layout.setContentsMargins(0, 0, 0, 0)
        type_layout.setSpacing(0)
        
        self.type_combo = QComboBox()
        self.type_combo.addItem("Expense", 0)
        self.type_combo.addItem("Income", 1)
        self.type_combo.currentIndexChanged.connect(self.on_type_changed)
        self.type_combo.setMinimumHeight(32)
        self.type_combo.setStyleSheet("""
            QComboBox {
                border-radius: 6px;
                padding: 5px 10px;
                background-color: #F1F1F3;
                selection-background-color: #0070F5;
            }
        """)
        type_layout.addWidget(self.type_combo)
        
        # Add spacing on the right
        type_layout.addStretch(1)
        
        form_layout.addRow(type_label, type_container)

        # Amount input with currency styling
        amount_label = QLabel("Amount (€):")
        amount_label.setStyleSheet("font-weight: medium;")
        
        amount_container = QWidget()
        amount_layout = QHBoxLayout(amount_container)
        amount_layout.setContentsMargins(0, 0, 0, 0)
        amount_layout.setSpacing(0)
        
        self.amount_edit = QLineEdit()
        self.amount_edit.setPlaceholderText("0.00")
        # Set validator for decimal input
        self.amount_edit.setValidator(QRegExpValidator(QRegExp("^\\d*\\.?\\d+$")))
        self.amount_edit.setAlignment(Qt.AlignRight)
        self.amount_edit.setMinimumHeight(32)
        self.amount_edit.setStyleSheet("""
            QLineEdit {
                font-size: 16px;
                font-weight: bold;
                padding-right: 10px;
            }
        """)
        amount_layout.addWidget(self.amount_edit)
        
        # Add spacing on the right
        amount_layout.addStretch(1)
        
        form_layout.addRow(amount_label, amount_container)

        # Description input
        description_label = QLabel("Description:")
        description_label.setStyleSheet("font-weight: medium;")
        self.description_edit = QLineEdit()
        self.description_edit.setPlaceholderText("Enter description")
        self.description_edit.setMinimumHeight(32)
        form_layout.addRow(description_label, self.description_edit)

        # Category dropdown with macOS styling
        category_label = QLabel("Category:")
        category_label.setStyleSheet("font-weight: medium;")
        self.category_combo = QComboBox()
        self.category_combo.setMinimumHeight(32)
        self.populate_categories()
        form_layout.addRow(category_label, self.category_combo)

        # Merchant input (optional)
        merchant_label = QLabel("Merchant:")
        merchant_label.setStyleSheet("font-weight: medium;")
        self.merchant_edit = QLineEdit()
        self.merchant_edit.setPlaceholderText("Enter merchant (optional)")
        self.merchant_edit.setMinimumHeight(32)
        form_layout.addRow(merchant_label, self.merchant_edit)

        # Recurring transaction checkbox with macOS styling
        recurring_label = QLabel("Recurring:")
        recurring_label.setStyleSheet("font-weight: medium;")
        
        recurring_container = QWidget()
        recurring_layout = QVBoxLayout(recurring_container)
        recurring_layout.setContentsMargins(0, 0, 0, 0)
        recurring_layout.setSpacing(8)
        
        self.recurring_check = QCheckBox("This is a recurring transaction")
        self.recurring_check.setChecked(False)
        self.recurring_check.stateChanged.connect(self.on_recurring_changed)
        recurring_layout.addWidget(self.recurring_check)
        
        # Recurring details (initially hidden)
        self.recurring_widget = QComboBox()
        self.recurring_widget.addItem("Daily", "daily")
        self.recurring_widget.addItem("Weekly", "weekly")
        self.recurring_widget.addItem("Monthly", "monthly")
        self.recurring_widget.addItem("Quarterly", "quarterly")
        self.recurring_widget.addItem("Yearly", "yearly")
        self.recurring_widget.setVisible(False)
        self.recurring_widget.setMinimumHeight(32)
        recurring_layout.addWidget(self.recurring_widget)
        
        form_layout.addRow(recurring_label, recurring_container)

        # Notes input with macOS styling
        notes_label = QLabel("Notes:")
        notes_label.setStyleSheet("font-weight: medium;")
        self.notes_edit = QTextEdit()
        self.notes_edit.setPlaceholderText("Enter notes (optional)")
        self.notes_edit.setMinimumHeight(80)
        self.notes_edit.setStyleSheet("""
            QTextEdit {
                border-radius: 6px;
                padding: 8px;
            }
        """)
        form_layout.addRow(notes_label, self.notes_edit)

        # Add form container to main layout
        layout.addWidget(form_container)

        # Buttons with macOS styling
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)

        # Cancel button
        self.cancel_button = StyledButton("Cancel", is_secondary=True)
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)

        button_layout.addStretch()

        # Save button
        self.save_button = SuccessButton("Save Transaction")
        self.save_button.clicked.connect(self.save_transaction)
        button_layout.addWidget(self.save_button)

        layout.addLayout(button_layout)

    def populate_categories(self):
        """Populate the categories dropdown based on transaction type."""
        # Clear current items
        self.category_combo.clear()

        # Get transaction type
        is_income = bool(self.type_combo.currentData())

        # Get appropriate categories
        categories = self.db_manager.get_categories("income" if is_income else "expense")

        # Add to dropdown
        for category in categories:
            self.category_combo.addItem(category['name'], category['id'])

    def on_type_changed(self, index):
        """
        Handle transaction type change.

        Args:
            index: New index
        """
        # Update categories
        self.populate_categories()

    def on_recurring_changed(self, state):
        """
        Handle recurring checkbox state change.

        Args:
            state: New state
        """
        # Show/hide recurring details
        self.recurring_widget.setVisible(state == Qt.Checked)

    def set_transaction_data(self):
        """Set form values from transaction data when editing."""
        if not self.transaction:
            return

        # Set date
        try:
            date = QDate.fromString(self.transaction.get('date', ''), "yyyy-MM-dd")
            self.date_edit.setDate(date)
        except:
            # Use current date as fallback
            self.date_edit.setDate(QDate.currentDate())

        # Set type
        is_income = self.transaction.get('is_income', False)
        index = 1 if is_income else 0
        self.type_combo.setCurrentIndex(index)

        # Set amount
        self.amount_edit.setText(str(self.transaction.get('amount', '')))

        # Set description
        self.description_edit.setText(self.transaction.get('description', ''))

        # Set merchant if available
        if 'merchant' in self.transaction and self.transaction['merchant']:
            self.merchant_edit.setText(self.transaction['merchant'])

        # Set category (after populating based on type)
        category_id = self.transaction.get('category_id')
        if category_id:
            index = self.category_combo.findData(category_id)
            if index >= 0:
                self.category_combo.setCurrentIndex(index)

        # Set recurring status
        is_recurring = bool(self.transaction.get('recurring', False))
        self.recurring_check.setChecked(is_recurring)

        # Set recurring period if applicable
        if is_recurring and 'recurring_period' in self.transaction and self.transaction['recurring_period']:
            index = self.recurring_widget.findData(self.transaction['recurring_period'])
            if index >= 0:
                self.recurring_widget.setCurrentIndex(index)

        # Set notes if available
        if 'notes' in self.transaction and self.transaction['notes']:
            self.notes_edit.setText(self.transaction['notes'])

    def validate_inputs(self):
        """
        Validate form inputs.

        Returns:
            True if all inputs are valid, False otherwise
        """
        # Validate amount
        if not self.amount_edit.text():
            QMessageBox.warning(self, "Invalid Input", "Please enter an amount.")
            self.amount_edit.setFocus()
            return False

        try:
            amount = float(self.amount_edit.text())
            if amount <= 0:
                QMessageBox.warning(self, "Invalid Input", "Amount must be greater than zero.")
                self.amount_edit.setFocus()
                return False
        except:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number for amount.")
            self.amount_edit.setFocus()
            return False

        # Validate description
        if not self.description_edit.text():
            QMessageBox.warning(self, "Invalid Input", "Please enter a description.")
            self.description_edit.setFocus()
            return False

        return True

    def save_transaction(self):
        """Save the transaction data."""
        # Validate inputs
        if not self.validate_inputs():
            return

        # Gather form data
        date = self.date_edit.date().toString("yyyy-MM-dd")
        is_income = bool(self.type_combo.currentData())
        amount = float(self.amount_edit.text())
        description = self.description_edit.text()
        category_id = self.category_combo.currentData()
        merchant = self.merchant_edit.text() or None
        recurring = self.recurring_check.isChecked()
        recurring_period = self.recurring_widget.currentData() if recurring else None
        notes = self.notes_edit.toPlainText() or None

        try:
            if self.transaction:
                # Update existing transaction
                tx_id = self.transaction.get('id')
                success = self.db_manager.update_transaction(
                    tx_id,
                    date=date,
                    amount=amount,
                    description=description,
                    category_id=category_id,
                    is_income=is_income,
                    merchant=merchant,
                    recurring=recurring,
                    recurring_period=recurring_period,
                    notes=notes
                )

                if not success:
                    QMessageBox.warning(
                        self, "Update Failed",
                        "Failed to update the transaction. Please try again."
                    )
                    return
            else:
                # Add new transaction
                tx_id = self.db_manager.add_transaction(
                    date=date,
                    amount=amount,
                    description=description,
                    category_id=category_id,
                    is_income=is_income,
                    merchant=merchant,
                    recurring=recurring,
                    recurring_period=recurring_period,
                    notes=notes
                )

                if not tx_id:
                    QMessageBox.warning(
                        self, "Add Failed",
                        "Failed to add the transaction. Please try again."
                    )
                    return

            # Close dialog on success
            self.accept()

        except Exception as e:
            self.logger.error(f"Error saving transaction: {e}")
            QMessageBox.critical(
                self, "Error",
                f"An error occurred while saving the transaction: {str(e)}"
            )