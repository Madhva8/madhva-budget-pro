#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modern Transaction Dialog Module

This module defines a dialog for adding or editing transactions with PySide6.
"""

import logging
import datetime
from typing import Dict, Any, Optional
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QLineEdit,
    QDateEdit, QComboBox, QCheckBox, QTextEdit, QPushButton, QMessageBox,
    QGraphicsDropShadowEffect, QFrame, QWidget
)
from PySide6.QtCore import Qt, QDate, QRegularExpression, Signal, Slot
from PySide6.QtGui import QRegularExpressionValidator, QFont, QColor

from ui.modern_styled_button import ModernStyledButton, ModernSuccessButton


class ModernTransactionDialog(QDialog):
    """Dialog for adding or editing transactions with modern UI."""

    def __init__(self, db_manager, transaction=None, parent=None):
        """
        Initialize the transaction dialog.

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
        """Initialize the user interface with modern macOS styling."""
        # Set window properties
        self.setWindowTitle("Add Transaction" if not self.transaction else "Edit Transaction")
        self.setMinimumWidth(500)
        self.setMinimumHeight(550)
        
        # Set dialog background to match the theme
        self.setStyleSheet("""
            QDialog {
                background-color: var(--bg-color, white);
            }
        """)
        
        # Main layout with proper margins for macOS style
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)
        
        # Title with SF-style typography
        title_label = QLabel(self.windowTitle())
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: var(--text-color, black);")
        layout.addWidget(title_label)

        # Form container with modern styling
        form_container = QFrame()
        form_container.setFrameShape(QFrame.StyledPanel)
        form_container.setStyleSheet("""
            QFrame {
                background-color: var(--card-bg-color, white);
                border: 1px solid var(--border-color, #E5E5E5);
                border-radius: 10px;
            }
        """)
        
        # Add shadow effect for depth
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 20))
        shadow.setOffset(0, 2)
        form_container.setGraphicsEffect(shadow)
        
        # Form layout with better spacing for macOS style
        form_layout = QFormLayout(form_container)
        form_layout.setContentsMargins(24, 24, 24, 24)
        form_layout.setSpacing(16)
        form_layout.setLabelAlignment(Qt.AlignLeft)
        form_layout.setVerticalSpacing(16)

        # Date input with macOS-style calendar popup
        date_label = QLabel("Date:")
        date_label.setStyleSheet("font-weight: medium; font-size: 14px; color: var(--text-color, black);")
        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDate(QDate.currentDate())
        self.date_edit.setMinimumHeight(32)
        self.date_edit.setStyleSheet("""
            QDateEdit {
                border: 1px solid var(--border-color, #E5E5E5);
                border-radius: 6px;
                padding: 4px 8px;
                background-color: var(--input-bg-color, #F9F9F9);
                color: var(--text-color, black);
            }
            QDateEdit::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: center right;
                width: 24px;
                border-left: none;
            }
        """)
        form_layout.addRow(date_label, self.date_edit)

        # Transaction type selector styled as a modern dropdown
        type_label = QLabel("Type:")
        type_label.setStyleSheet("font-weight: medium; font-size: 14px; color: var(--text-color, black);")
        
        type_container = QWidget()
        type_container.setStyleSheet("background-color: transparent;")
        type_layout = QHBoxLayout(type_container)
        type_layout.setContentsMargins(0, 0, 0, 0)
        type_layout.setSpacing(0)
        
        self.type_combo = QComboBox()
        self.type_combo.addItem("Expense", 0)
        self.type_combo.addItem("Income", 1)
        self.type_combo.addItem("Savings", 2)
        self.type_combo.currentIndexChanged.connect(self.on_type_changed)
        self.type_combo.setMinimumHeight(32)
        self.type_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid var(--border-color, #E5E5E5);
                border-radius: 6px;
                padding: 4px 8px;
                background-color: var(--input-bg-color, #F9F9F9);
                color: var(--text-color, black);
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: center right;
                width: 24px;
                border-left: none;
            }
            QComboBox QAbstractItemView {
                background-color: var(--card-bg-color, white);
                color: var(--text-color, black);
                border: 1px solid var(--border-color, #E5E5E5);
            }
        """)
        type_layout.addWidget(self.type_combo)
        
        # Add spacing on the right
        type_layout.addStretch(1)
        
        form_layout.addRow(type_label, type_container)

        # Amount input with currency styling
        amount_label = QLabel("Amount (€):")
        amount_label.setStyleSheet("font-weight: medium; font-size: 14px; color: var(--text-color, black);")
        
        amount_container = QWidget()
        amount_container.setStyleSheet("background-color: transparent;")
        amount_layout = QHBoxLayout(amount_container)
        amount_layout.setContentsMargins(0, 0, 0, 0)
        amount_layout.setSpacing(0)
        
        self.amount_edit = QLineEdit()
        self.amount_edit.setPlaceholderText("0.00")
        # Set validator for decimal input
        self.amount_edit.setValidator(QRegularExpressionValidator(QRegularExpression("^\\d*\\.?\\d+$")))
        self.amount_edit.setAlignment(Qt.AlignRight)
        self.amount_edit.setMinimumHeight(32)
        self.amount_edit.setStyleSheet("""
            QLineEdit {
                border: 1px solid var(--border-color, #E5E5E5);
                border-radius: 6px;
                padding: 4px 8px;
                background-color: var(--input-bg-color, #F9F9F9);
                color: var(--text-color, black);
                font-size: 16px;
                font-weight: bold;
                padding-right: 10px;
            }
        """)
        amount_layout.addWidget(self.amount_edit)
        
        # Add spacing on the right
        amount_layout.addStretch(1)
        
        form_layout.addRow(amount_label, amount_container)

        # Description input with modern styling
        description_label = QLabel("Description:")
        description_label.setStyleSheet("font-weight: medium; font-size: 14px; color: var(--text-color, black);")
        self.description_edit = QLineEdit()
        self.description_edit.setPlaceholderText("Enter description")
        self.description_edit.setMinimumHeight(32)
        self.description_edit.setStyleSheet("""
            QLineEdit {
                border: 1px solid var(--border-color, #E5E5E5);
                border-radius: 6px;
                padding: 4px 8px;
                background-color: var(--input-bg-color, #F9F9F9);
                color: var(--text-color, black);
            }
        """)
        form_layout.addRow(description_label, self.description_edit)

        # Category dropdown with macOS styling
        category_label = QLabel("Category:")
        category_label.setStyleSheet("font-weight: medium; font-size: 14px; color: var(--text-color, black);")
        self.category_combo = QComboBox()
        self.category_combo.setMinimumHeight(32)
        self.category_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid var(--border-color, #E5E5E5);
                border-radius: 6px;
                padding: 4px 8px;
                background-color: var(--input-bg-color, #F9F9F9);
                color: var(--text-color, black);
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: center right;
                width: 24px;
                border-left: none;
            }
            QComboBox QAbstractItemView {
                background-color: var(--card-bg-color, white);
                color: var(--text-color, black);
                border: 1px solid var(--border-color, #E5E5E5);
            }
        """)
        self.populate_categories()
        form_layout.addRow(category_label, self.category_combo)

        # Merchant input with modern styling
        merchant_label = QLabel("Merchant:")
        merchant_label.setStyleSheet("font-weight: medium; font-size: 14px; color: var(--text-color, black);")
        self.merchant_edit = QLineEdit()
        self.merchant_edit.setPlaceholderText("Enter merchant (optional)")
        self.merchant_edit.setMinimumHeight(32)
        self.merchant_edit.setStyleSheet("""
            QLineEdit {
                border: 1px solid var(--border-color, #E5E5E5);
                border-radius: 6px;
                padding: 4px 8px;
                background-color: var(--input-bg-color, #F9F9F9);
                color: var(--text-color, black);
            }
        """)
        form_layout.addRow(merchant_label, self.merchant_edit)

        # Recurring transaction checkbox with macOS styling
        recurring_label = QLabel("Recurring:")
        recurring_label.setStyleSheet("font-weight: medium; font-size: 14px; color: var(--text-color, black);")
        
        recurring_container = QWidget()
        recurring_container.setStyleSheet("background-color: transparent;")
        recurring_layout = QVBoxLayout(recurring_container)
        recurring_layout.setContentsMargins(0, 0, 0, 0)
        recurring_layout.setSpacing(8)
        
        self.recurring_check = QCheckBox("This is a recurring transaction")
        self.recurring_check.setChecked(False)
        self.recurring_check.stateChanged.connect(self.on_recurring_changed)
        self.recurring_check.setStyleSheet("""
            QCheckBox {
                spacing: 8px;
                color: var(--text-color, black);
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 1px solid var(--border-color, #BBBBBB);
                background-color: var(--input-bg-color, white);
            }
            QCheckBox::indicator:checked {
                background-color: var(--button-primary-bg, #007AFF);
                border-color: var(--button-primary-bg, #007AFF);
                image: url(:/images/check.png);  /* In a real app, you'd use a real checkmark image */
            }
        """)
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
        self.recurring_widget.setStyleSheet("""
            QComboBox {
                border: 1px solid var(--border-color, #E5E5E5);
                border-radius: 6px;
                padding: 4px 8px;
                background-color: var(--input-bg-color, #F9F9F9);
                color: var(--text-color, black);
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: center right;
                width: 24px;
                border-left: none;
            }
            QComboBox QAbstractItemView {
                background-color: var(--card-bg-color, white);
                color: var(--text-color, black);
                border: 1px solid var(--border-color, #E5E5E5);
            }
        """)
        recurring_layout.addWidget(self.recurring_widget)
        
        form_layout.addRow(recurring_label, recurring_container)

        # Notes input with macOS styling
        notes_label = QLabel("Notes:")
        notes_label.setStyleSheet("font-weight: medium; font-size: 14px; color: var(--text-color, black);")
        self.notes_edit = QTextEdit()
        self.notes_edit.setPlaceholderText("Enter notes (optional)")
        self.notes_edit.setMinimumHeight(80)
        self.notes_edit.setStyleSheet("""
            QTextEdit {
                border: 1px solid var(--border-color, #E5E5E5);
                border-radius: 6px;
                padding: 8px;
                background-color: var(--input-bg-color, #F9F9F9);
                color: var(--text-color, black);
            }
        """)
        form_layout.addRow(notes_label, self.notes_edit)

        # Add form container to main layout
        layout.addWidget(form_container)

        # Buttons with macOS styling in a separate card
        button_container = QFrame()
        button_container.setFrameShape(QFrame.StyledPanel)
        button_container.setStyleSheet("""
            QFrame {
                background-color: var(--card-bg-color, white);
                border: 1px solid var(--border-color, #E5E5E5);
                border-radius: 10px;
            }
        """)
        
        # Add subtle shadow
        button_shadow = QGraphicsDropShadowEffect()
        button_shadow.setBlurRadius(10)
        button_shadow.setColor(QColor(0, 0, 0, 15))
        button_shadow.setOffset(0, 1)
        button_container.setGraphicsEffect(button_shadow)
        
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(16, 16, 16, 16)
        button_layout.setSpacing(12)

        # Cancel button
        self.cancel_button = ModernStyledButton("Cancel", is_secondary=True)
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)

        button_layout.addStretch()

        # Save button
        self.save_button = ModernSuccessButton("Save Transaction")
        self.save_button.clicked.connect(self.save_transaction)
        button_layout.addWidget(self.save_button)

        layout.addWidget(button_container)

    def populate_categories(self):
        """Populate the categories dropdown based on transaction type."""
        # Clear current items
        self.category_combo.clear()

        # Get transaction type (0 = expense, 1 = income, 2 = savings)
        transaction_type = self.type_combo.currentData()
        
        if transaction_type == 2:  # Savings
            # For savings, we specifically want the Savings category
            categories = self.db_manager.get_categories("expense")
            for category in categories:
                if category['name'] == "Savings":
                    self.category_combo.addItem(category['name'], category['id'])
        else:
            # For regular income or expense
            is_income = bool(transaction_type)
            categories = self.db_manager.get_categories("income" if is_income else "expense")
            # Add to dropdown
            for category in categories:
                self.category_combo.addItem(category['name'], category['id'])

    @Slot(int)
    def on_type_changed(self, index):
        """
        Handle transaction type change.

        Args:
            index: New index
        """
        # Update categories
        self.populate_categories()

    @Slot(int)
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
        description = self.transaction.get('description', '')
        # Check if this is a savings transaction (contains "um" for transfers)
        if "um " in description or self.transaction.get('category_id') == 13:  # ID for Savings category
            index = 2  # Savings
        else:
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
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Invalid Input")
            msg_box.setText("Please enter an amount.")
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: var(--bg-color, white);
                    color: var(--text-color, black);
                }
                QLabel {
                    color: var(--text-color, black);
                }
                QPushButton {
                    background-color: var(--button-primary-bg, #007AFF);
                    color: var(--button-primary-text, white);
                    border: none;
                    border-radius: 4px;
                    padding: 6px 12px;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: var(--button-primary-hover-bg, #0062CC);
                }
            """)
            msg_box.exec_()
            self.amount_edit.setFocus()
            return False

        try:
            amount = float(self.amount_edit.text())
            if amount <= 0:
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("Invalid Input")
                msg_box.setText("Amount must be greater than zero.")
                msg_box.setIcon(QMessageBox.Warning)
                msg_box.setStyleSheet("""
                    QMessageBox {
                        background-color: var(--bg-color, white);
                        color: var(--text-color, black);
                    }
                    QLabel {
                        color: var(--text-color, black);
                    }
                    QPushButton {
                        background-color: var(--button-primary-bg, #007AFF);
                        color: var(--button-primary-text, white);
                        border: none;
                        border-radius: 4px;
                        padding: 6px 12px;
                        min-width: 80px;
                    }
                    QPushButton:hover {
                        background-color: var(--button-primary-hover-bg, #0062CC);
                    }
                """)
                msg_box.exec_()
                self.amount_edit.setFocus()
                return False
        except:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Invalid Input")
            msg_box.setText("Please enter a valid number for amount.")
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: var(--bg-color, white);
                    color: var(--text-color, black);
                }
                QLabel {
                    color: var(--text-color, black);
                }
                QPushButton {
                    background-color: var(--button-primary-bg, #007AFF);
                    color: var(--button-primary-text, white);
                    border: none;
                    border-radius: 4px;
                    padding: 6px 12px;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: var(--button-primary-hover-bg, #0062CC);
                }
            """)
            msg_box.exec_()
            self.amount_edit.setFocus()
            return False

        # Validate description
        if not self.description_edit.text():
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Invalid Input")
            msg_box.setText("Please enter a description.")
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: var(--bg-color, white);
                    color: var(--text-color, black);
                }
                QLabel {
                    color: var(--text-color, black);
                }
                QPushButton {
                    background-color: var(--button-primary-bg, #007AFF);
                    color: var(--button-primary-text, white);
                    border: none;
                    border-radius: 4px;
                    padding: 6px 12px;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: var(--button-primary-hover-bg, #0062CC);
                }
            """)
            msg_box.exec_()
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
        transaction_type = self.type_combo.currentData()
        is_income = bool(transaction_type) if transaction_type != 2 else False  # Savings is not income
        amount = float(self.amount_edit.text())
        
        # For savings transactions, add "um" to the description for proper identification
        description = self.description_edit.text()
        if transaction_type == 2 and "um" not in description:
            # Add "um" to description to mark as savings transfer
            description = f"{description} um {date}"
        
        category_id = self.category_combo.currentData()
        merchant = self.merchant_edit.text() or None
        recurring = self.recurring_check.isChecked()
        recurring_period = self.recurring_widget.currentData() if recurring else None
        notes = self.notes_edit.toPlainText() or None

        try:
            if self.transaction:
                # Update existing transaction
                tx_id = self.transaction.get('id')
                
                # Make sure the category is set to Savings if it's a Savings transaction type
                if transaction_type == 2:
                    # Find the Savings category ID if needed
                    if category_id != 13:  # if not already set to Savings
                        savings_categories = [cat for cat in self.db_manager.get_categories("expense") 
                                            if cat['name'] == "Savings"]
                        if savings_categories:
                            category_id = savings_categories[0]['id']
                            self.logger.info(f"Setting category to Savings (ID: {category_id}) for Savings transaction")
                        else:
                            self.logger.warning("Savings category not found in database")
                
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
                    msg_box = QMessageBox(self)
                    msg_box.setWindowTitle("Update Failed")
                    msg_box.setText("Failed to update the transaction. Please try again.")
                    msg_box.setIcon(QMessageBox.Warning)
                    msg_box.setStyleSheet("""
                        QMessageBox {
                            background-color: var(--bg-color, white);
                            color: var(--text-color, black);
                        }
                        QLabel {
                            color: var(--text-color, black);
                        }
                        QPushButton {
                            background-color: var(--button-primary-bg, #007AFF);
                            color: var(--button-primary-text, white);
                            border: none;
                            border-radius: 4px;
                            padding: 6px 12px;
                            min-width: 80px;
                        }
                        QPushButton:hover {
                            background-color: var(--button-primary-hover-bg, #0062CC);
                        }
                    """)
                    msg_box.exec_()
                    return
            else:
                # Add new transaction
                
                # Make sure the category is set to Savings if it's a Savings transaction type
                if transaction_type == 2:
                    # Find the Savings category ID if needed
                    if category_id != 13:  # if not already set to Savings
                        savings_categories = [cat for cat in self.db_manager.get_categories("expense") 
                                            if cat['name'] == "Savings"]
                        if savings_categories:
                            category_id = savings_categories[0]['id']
                            self.logger.info(f"Setting category to Savings (ID: {category_id}) for new Savings transaction")
                        else:
                            self.logger.warning("Savings category not found in database")
                
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
                    msg_box = QMessageBox(self)
                    msg_box.setWindowTitle("Add Failed")
                    msg_box.setText("Failed to add the transaction. Please try again.")
                    msg_box.setIcon(QMessageBox.Warning)
                    msg_box.setStyleSheet("""
                        QMessageBox {
                            background-color: var(--bg-color, white);
                            color: var(--text-color, black);
                        }
                        QLabel {
                            color: var(--text-color, black);
                        }
                        QPushButton {
                            background-color: var(--button-primary-bg, #007AFF);
                            color: var(--button-primary-text, white);
                            border: none;
                            border-radius: 4px;
                            padding: 6px 12px;
                            min-width: 80px;
                        }
                        QPushButton:hover {
                            background-color: var(--button-primary-hover-bg, #0062CC);
                        }
                    """)
                    msg_box.exec_()
                    return

            # Close dialog on success
            self.accept()

        except Exception as e:
            self.logger.error(f"Error saving transaction: {e}")
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Error")
            msg_box.setText(f"An error occurred while saving the transaction: {str(e)}")
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: var(--bg-color, white);
                    color: var(--text-color, black);
                }
                QLabel {
                    color: var(--text-color, black);
                }
                QPushButton {
                    background-color: var(--button-primary-bg, #007AFF);
                    color: var(--button-primary-text, white);
                    border: none;
                    border-radius: 4px;
                    padding: 6px 12px;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: var(--button-primary-hover-bg, #0062CC);
                }
            """)
            msg_box.exec_()