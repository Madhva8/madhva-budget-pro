#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Login Dialog for Financial Planner

A modern macOS styled login dialog with support for password, Touch ID, and FaceID.
"""

import os
import sys
import logging
import hashlib
import secrets
import base64
from pathlib import Path
from functools import partial

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QPushButton, QMessageBox, QCheckBox, QFrame, QApplication
)
from PySide6.QtCore import Qt, QSettings, Signal, QSize
from PySide6.QtGui import QPixmap, QIcon, QColor, QPainter, QPainterPath

# For macOS Biometric authentication
try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False

try:
    import PyQt6.QtMacExtras as QtMacExtras
    TOUCHID_AVAILABLE = True
except ImportError:
    TOUCHID_AVAILABLE = False

class ModernLoginDialog(QDialog):
    """Modern styled login dialog with biometric authentication support."""
    
    loginSuccessful = Signal(str)  # Signal emitted when login is successful
    
    def __init__(self, db_manager=None, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        
        # Set up dialog properties
        self.setWindowTitle("Madhva Budget Pro - Login")
        self.setFixedSize(450, 520)  # Slightly larger size
        self.setWindowFlags(Qt.WindowType.Dialog | 
                           Qt.WindowType.WindowCloseButtonHint | 
                           Qt.WindowType.WindowStaysOnTopHint |
                           Qt.WindowType.MSWindowsFixedSizeDialogHint)  # Better window behavior
        
        # Set application style
        self.setStyleSheet("""
            QDialog {
                background-color: white;
                border: 1px solid #d1d1d6;
                border-radius: 10px;
            }
            QLabel {
                color: #1d1d1f;
                font-size: 14px;
                margin-bottom: 5px;
                border: none;
            }
            QLabel#titleLabel {
                font-size: 22px;
                font-weight: bold;
                color: #1d1d1f;
                margin-bottom: 10px;
                border: none;
            }
            QLabel#logoLabel {
                margin-bottom: 20px;
                border: none;
            }
            QLineEdit {
                border: 1px solid #d1d1d6;
                border-radius: 6px;
                padding: 10px;
                background-color: white;
                font-size: 14px;
                min-height: 20px;
                margin-bottom: 15px;
            }
            QLineEdit:focus {
                border: 1px solid #0071e3;
            }
            QPushButton {
                background-color: #0071e3;
                border: none;
                border-radius: 6px;
                color: white;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
                margin-top: 5px;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #0077ed;
            }
            QPushButton:pressed {
                background-color: #005bbf;
            }
            QPushButton#bioButton {
                background-color: #34c759;
            }
            QPushButton#bioButton:hover {
                background-color: #30d158;
            }
            QPushButton#bioButton:pressed {
                background-color: #2db84c;
            }
            QPushButton#registerButton {
                background-color: white;
                border: 1px solid #d1d1d6;
                color: #1d1d1f;
            }
            QPushButton#registerButton:hover {
                background-color: #f2f2f7;
            }
            QPushButton#registerButton:pressed {
                background-color: #e5e5ea;
            }
            QCheckBox {
                color: #1d1d1f;
                font-size: 13px;
                border: none;
            }
        """)
        
        # Create UI elements
        self.init_ui()
        
        # Set up biometric authentication if available
        self.setup_biometric_auth()
    
    def init_ui(self):
        """Initialize the UI components."""
        # Main layout with proper spacing
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(10)  # Increased spacing between elements
        
        # Logo
        logo_layout = QHBoxLayout()
        logo_label = QLabel()
        logo_label.setObjectName("logoLabel")
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Try multiple paths to find the logo
        possible_paths = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logo.png"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "logo.png"),
            os.path.join(os.path.dirname(__file__), "logo.png"),
            "logo.png"
        ]
        
        logo_found = False
        for logo_path in possible_paths:
            if os.path.exists(logo_path):
                try:
                    pixmap = QPixmap(logo_path)
                    if not pixmap.isNull():
                        logo_label.setPixmap(pixmap.scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                        logo_found = True
                        break
                except Exception:
                    pass
                    
        if not logo_found:
            # Fallback to emoji if no logo is found
            logo_label.setText("💰")
            logo_label.setStyleSheet("font-size: 48px; color: #0071e3; border: none;")
        
        logo_layout.addStretch()
        logo_layout.addWidget(logo_label)
        logo_layout.addStretch()
        layout.addLayout(logo_layout)
        
        # Title
        title_label = QLabel("Welcome to Madhva Budget Pro")
        title_label.setObjectName("titleLabel")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Subtitle
        subtitle_label = QLabel("Please sign in to access your financial data")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label.setStyleSheet("color: #6e6e73; margin-bottom: 25px;")
        layout.addWidget(subtitle_label)
        
        # Username field
        username_label = QLabel("Username")
        username_label.setStyleSheet("margin-top: 10px;")
        layout.addWidget(username_label)
        
        self.username_edit = QLineEdit()
        self.username_edit.setPlaceholderText("Enter your username")
        self.username_edit.setMinimumHeight(36)  # Taller fields for better touch
        layout.addWidget(self.username_edit)
        
        # Password field
        password_label = QLabel("Password")
        password_label.setStyleSheet("margin-top: 10px;")
        layout.addWidget(password_label)
        
        self.password_edit = QLineEdit()
        self.password_edit.setPlaceholderText("Enter your password")
        self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_edit.setMinimumHeight(36)  # Taller fields for better touch
        layout.addWidget(self.password_edit)
        
        # Remember me checkbox
        self.remember_checkbox = QCheckBox("Remember me")
        layout.addWidget(self.remember_checkbox)
        
        # Add spacer
        layout.addSpacing(10)

        # Login button
        self.login_button = QPushButton("Sign In")
        self.login_button.clicked.connect(self.authenticate)
        self.login_button.setMinimumHeight(44)  # Taller button for better touch
        self.login_button.setCursor(Qt.CursorShape.PointingHandCursor)  # Hand cursor on hover
        layout.addWidget(self.login_button)
        
        # Biometric auth button (will be shown only if available)
        self.bio_button = QPushButton("Use Touch ID")
        self.bio_button.setObjectName("bioButton")
        self.bio_button.setVisible(False)  # Hidden by default
        self.bio_button.setMinimumHeight(44)  # Taller button
        self.bio_button.setCursor(Qt.CursorShape.PointingHandCursor)  # Hand cursor on hover
        self.bio_button.clicked.connect(self.authenticate_with_biometric)
        layout.addWidget(self.bio_button)
        
        # Add spacer
        layout.addSpacing(5)
        
        # Register button
        self.register_button = QPushButton("Create New Account")
        self.register_button.setObjectName("registerButton")
        self.register_button.setMinimumHeight(44)  # Taller button
        self.register_button.setCursor(Qt.CursorShape.PointingHandCursor)  # Hand cursor on hover
        self.register_button.clicked.connect(self.register_user)
        layout.addWidget(self.register_button)
        
        # Version info
        version_label = QLabel("Version 1.0.0")
        version_label.setStyleSheet("color: #86868b; font-size: 12px; margin-top: 15px;")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version_label)
        
        self.setLayout(layout)
        
        # Set default focus
        self.username_edit.setFocus()
        
        # Load saved credentials if any
        self.load_saved_credentials()
    
    def setup_biometric_auth(self):
        """Configure biometric authentication if available on the system."""
        # On macOS, check if Touch ID or Face ID is available
        if sys.platform == "darwin" and KEYRING_AVAILABLE and TOUCHID_AVAILABLE:
            # Show biometric button and update its label
            self.bio_button.setVisible(True)
            
            # Try to determine if Touch ID or Face ID is available
            # This is a simplified approach - in a real app you'd check what's actually available
            self.bio_button.setText("Use Touch ID")  # Default to Touch ID as it's more common
            
            # On newer Macs with Face ID, change the label
            if os.path.exists("/Library/Preferences/com.apple.faceid.plist"):
                self.bio_button.setText("Use Face ID")
    
    def authenticate(self):
        """Authenticate the user with username and password."""
        username = self.username_edit.text().strip()
        password = self.password_edit.text()
        
        if not username or not password:
            QMessageBox.warning(
                self, 
                "Login Error", 
                "Please enter both username and password."
            )
            return
        
        # If we have a database manager, verify the credentials
        if self.db_manager:
            if self.verify_credentials(username, password):
                self.save_credentials_if_requested(username, password)
                self.loginSuccessful.emit(username)
                self.accept()
            else:
                QMessageBox.warning(
                    self, 
                    "Login Error", 
                    "Invalid username or password. Please try again."
                )
        else:
            # Demo mode - accept any login (for testing)
            logging.warning("No database manager available. Using demo mode authentication.")
            
            # Hardcoded demo credentials
            if username == "demo" and password == "demo":
                self.save_credentials_if_requested(username, password)
                self.loginSuccessful.emit(username)
                self.accept()
            else:
                QMessageBox.warning(
                    self, 
                    "Login Error", 
                    "Invalid username or password. In demo mode, use 'demo' for both."
                )
    
    def authenticate_with_biometric(self):
        """Authenticate the user with Touch ID or Face ID."""
        if not KEYRING_AVAILABLE:
            QMessageBox.warning(
                self, 
                "Biometric Authentication", 
                "Biometric authentication is not available. Please install the keyring package."
            )
            return
            
        # Try to get saved credentials from keyring
        try:
            # The actual Touch ID/Face ID prompt is triggered here
            saved_username = keyring.get_password("MadhvaBudgetPro", "LastUsername")
            if not saved_username:
                QMessageBox.warning(
                    self, 
                    "Biometric Authentication", 
                    "No saved credentials found. Please login with username and password first."
                )
                return
                
            saved_password = keyring.get_password("MadhvaBudgetPro", saved_username)
            if not saved_password:
                QMessageBox.warning(
                    self, 
                    "Biometric Authentication", 
                    "No saved credentials found. Please login with username and password first."
                )
                return
                
            # Verify the credentials
            if self.db_manager and self.verify_credentials(saved_username, saved_password):
                self.loginSuccessful.emit(saved_username)
                self.accept()
            else:
                # In demo mode or if db_manager is not available
                if saved_username == "demo" and saved_password == "demo":
                    self.loginSuccessful.emit(saved_username)
                    self.accept()
                else:
                    QMessageBox.warning(
                        self, 
                        "Login Error", 
                        "Saved credentials are invalid. Please login with username and password."
                    )
        except Exception as e:
            QMessageBox.warning(
                self, 
                "Biometric Authentication", 
                f"Biometric authentication failed: {str(e)}"
            )
    
    def verify_credentials(self, username, password):
        """Verify the provided credentials against the database.
        
        Args:
            username (str): The provided username
            password (str): The provided password
            
        Returns:
            bool: True if credentials are valid, False otherwise
        """
        try:
            # Query the users table to check credentials
            query = "SELECT salt, password_hash FROM users WHERE username = ?"
            params = (username,)
            
            result = self.db_manager.execute_query(query, params, fetch_one=True)
            
            if not result:
                return False
                
            salt, stored_hash = result
            
            # Hash the provided password with the stored salt
            hashed_password = self.hash_password(password, salt)
            
            # Compare with stored hash
            return hashed_password == stored_hash
        except Exception as e:
            logging.error(f"Error verifying credentials: {e}")
            return False
    
    def register_user(self):
        """Show a dialog to register a new user."""
        # Create a registration dialog
        registration_dialog = QDialog(self)
        registration_dialog.setWindowTitle("Create Account")
        registration_dialog.setFixedSize(350, 250)
        registration_dialog.setStyleSheet(self.styleSheet())
        
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Username field
        username_label = QLabel("Username")
        layout.addWidget(username_label)
        
        new_username_edit = QLineEdit()
        new_username_edit.setPlaceholderText("Choose a username")
        layout.addWidget(new_username_edit)
        
        # Password field
        password_label = QLabel("Password")
        layout.addWidget(password_label)
        
        new_password_edit = QLineEdit()
        new_password_edit.setPlaceholderText("Choose a password")
        new_password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(new_password_edit)
        
        # Confirm password field
        confirm_label = QLabel("Confirm Password")
        layout.addWidget(confirm_label)
        
        confirm_password_edit = QLineEdit()
        confirm_password_edit.setPlaceholderText("Confirm your password")
        confirm_password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(confirm_password_edit)
        
        # Register button
        register_button = QPushButton("Create Account")
        layout.addWidget(register_button)
        
        registration_dialog.setLayout(layout)
        
        # Connect the register button
        register_button.clicked.connect(lambda: self.create_user(
            registration_dialog,
            new_username_edit.text().strip(),
            new_password_edit.text(),
            confirm_password_edit.text()
        ))
        
        registration_dialog.exec()
    
    def create_user(self, dialog, username, password, confirm_password):
        """Create a new user account.
        
        Args:
            dialog (QDialog): The registration dialog
            username (str): The chosen username
            password (str): The chosen password
            confirm_password (str): The confirmation password
        """
        # Validate inputs
        if not username or not password:
            QMessageBox.warning(
                dialog, 
                "Registration Error", 
                "Please enter both username and password."
            )
            return
            
        if password != confirm_password:
            QMessageBox.warning(
                dialog, 
                "Registration Error", 
                "Passwords do not match."
            )
            return
            
        if len(password) < 6:
            QMessageBox.warning(
                dialog, 
                "Registration Error", 
                "Password must be at least 6 characters long."
            )
            return
            
        # Check if username already exists
        if self.db_manager:
            query = "SELECT COUNT(*) FROM users WHERE username = ?"
            params = (username,)
            
            try:
                result = self.db_manager.execute_query(query, params, fetch_one=True)
                
                if result and result[0] > 0:
                    QMessageBox.warning(
                        dialog, 
                        "Registration Error", 
                        "Username already exists. Please choose another."
                    )
                    return
                    
                # Generate a salt and hash the password
                salt = secrets.token_hex(16)
                hashed_password = self.hash_password(password, salt)
                
                # Insert the new user
                insert_query = "INSERT INTO users (username, salt, password_hash, created_at) VALUES (?, ?, ?, datetime('now'))"
                insert_params = (username, salt, hashed_password)
                
                self.db_manager.execute_query(insert_query, insert_params)
                
                QMessageBox.information(
                    dialog, 
                    "Registration Successful", 
                    "Your account has been created successfully. You can now log in."
                )
                dialog.accept()
                
            except Exception as e:
                logging.error(f"Error creating user: {e}")
                QMessageBox.warning(
                    dialog, 
                    "Registration Error", 
                    f"Failed to create account: {e}"
                )
        else:
            # Demo mode - just show a success message
            QMessageBox.information(
                dialog, 
                "Registration Successful", 
                "Your account has been created successfully. You can now log in."
            )
            dialog.accept()
    
    def hash_password(self, password, salt):
        """Hash a password using PBKDF2 with HMAC-SHA256.
        
        Args:
            password (str): The password to hash
            salt (str): The salt to use
            
        Returns:
            str: The hashed password
        """
        # Convert the password and salt to bytes
        password_bytes = password.encode('utf-8')
        salt_bytes = bytes.fromhex(salt)
        
        # Hash the password
        hash_bytes = hashlib.pbkdf2_hmac(
            'sha256',
            password_bytes,
            salt_bytes,
            100000  # Number of iterations
        )
        
        # Convert the hash to a hexadecimal string
        return hash_bytes.hex()
    
    def save_credentials_if_requested(self, username, password):
        """Save the credentials if the 'Remember me' checkbox is checked.
        
        Args:
            username (str): The username to save
            password (str): The password to save
        """
        if self.remember_checkbox.isChecked() and KEYRING_AVAILABLE:
            try:
                keyring.set_password("MadhvaBudgetPro", "LastUsername", username)
                keyring.set_password("MadhvaBudgetPro", username, password)
                
                logging.info(f"Credentials saved for user: {username}")
            except Exception as e:
                logging.error(f"Error saving credentials: {e}")
    
    def load_saved_credentials(self):
        """Load saved credentials from the keyring."""
        if KEYRING_AVAILABLE:
            try:
                saved_username = keyring.get_password("MadhvaBudgetPro", "LastUsername")
                
                if saved_username:
                    self.username_edit.setText(saved_username)
                    self.remember_checkbox.setChecked(True)
                    
                    # Don't load the password into the field for security reasons
                    # Instead, just focus on the password field
                    self.password_edit.setFocus()
            except Exception as e:
                logging.error(f"Error loading saved credentials: {e}")
                
    def keyPressEvent(self, event):
        """Handle key press events."""
        # Press Enter/Return to login
        if event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            self.authenticate()
        else:
            super().keyPressEvent(event)


# Test function for the login dialog
def test_login_dialog():
    app = QApplication(sys.argv)
    
    login_dialog = ModernLoginDialog()
    login_dialog.loginSuccessful.connect(lambda username: print(f"Login successful for user: {username}"))
    
    if login_dialog.exec() == QDialog.DialogCode.Accepted:
        print("Login accepted")
    else:
        print("Login cancelled")


if __name__ == "__main__":
    test_login_dialog()