#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Database Manager Module

This module provides a database interface for the Financial Planner application.
It handles all database operations including creation, migrations, and queries.
"""

import os
import sqlite3
import logging
import datetime
import json
from typing import List, Dict, Any, Optional, Tuple, Union


class DatabaseManager:
    """Manages database operations for the Financial Planner app."""

    def __init__(self, db_path: str = 'financial_planner.db'):
        """
        Initialize database connection and setup tables.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.logger = logging.getLogger(__name__)

        # Connect to database
        self._connect()

        # Setup database schema
        self.setup_database()

    def _connect(self):
        """Establish connection to the SQLite database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
            self.cursor = self.conn.cursor()
            self.logger.info(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            self.logger.error(f"Error connecting to database: {e}")
            raise

    def setup_database(self):
        """Create necessary tables if they don't exist."""
        try:
            # Users Table
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT,
                full_name TEXT,
                is_admin BOOLEAN DEFAULT 0,
                is_active BOOLEAN DEFAULT 1,
                last_login TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Login Attempts Table (for security)
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS login_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                ip_address TEXT,
                success BOOLEAN NOT NULL,
                attempt_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Categories Table
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                type TEXT NOT NULL,
                color TEXT NOT NULL,
                icon TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')

            # Transactions Table
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                amount REAL NOT NULL,
                description TEXT,
                category_id INTEGER,
                is_income BOOLEAN NOT NULL,
                recurring BOOLEAN DEFAULT 0,
                recurring_period TEXT,
                notes TEXT,
                merchant TEXT,
                bank TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (category_id) REFERENCES categories (id)
            )
            ''')

            # Budgets Table
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS budgets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_id INTEGER,
                amount REAL NOT NULL,
                month INTEGER NOT NULL,
                year INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (category_id) REFERENCES categories (id),
                UNIQUE (category_id, month, year)
            )
            ''')

            # Financial Goals Table
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                target_amount REAL NOT NULL,
                current_amount REAL DEFAULT 0,
                start_date TEXT NOT NULL,
                target_date TEXT NOT NULL,
                priority INTEGER,
                status TEXT DEFAULT 'active',
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')

            # Settings Table
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')

            # AI Tips Table
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_tips (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tip_text TEXT NOT NULL,
                context TEXT,
                is_read BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')

            # Import History Table
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS import_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                bank TEXT,
                import_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                transaction_count INTEGER,
                status TEXT,
                error_message TEXT
            )
            ''')

            # Populate initial categories if table is empty
            self.cursor.execute("SELECT COUNT(*) FROM categories")
            if self.cursor.fetchone()[0] == 0:
                self._populate_default_categories()

            # Populate initial settings if table is empty
            self.cursor.execute("SELECT COUNT(*) FROM settings")
            if self.cursor.fetchone()[0] == 0:
                self._populate_default_settings()
                
            # Create default admin user if no users exist
            self._create_default_admin_user()

            self.conn.commit()
            self.logger.info("Database setup completed successfully")

        except sqlite3.Error as e:
            self.logger.error(f"Error setting up database: {e}")
            self.conn.rollback()
            raise

    def _populate_default_categories(self):
        """Add default categories to the categories table."""
        self.logger.info("Populating default categories")

        expense_categories = [
            ("Housing", "expense", "#FF5733", "home"),
            ("Food", "expense", "#33FF57", "food"),
            ("Transportation", "expense", "#3357FF", "transport"),
            ("Utilities", "expense", "#F3FF33", "flash"),
            ("Entertainment", "expense", "#FF33F3", "music"),
            ("Education", "expense", "#33FFF3", "book"),
            ("Health", "expense", "#FF3333", "heart"),
            ("Shopping", "expense", "#33FFAA", "cart"),
            ("Personal Care", "expense", "#AA33FF", "user"),
            ("Travel", "expense", "#FFAA33", "plane"),
            ("Insurance", "expense", "#33AAFF", "shield"),
            ("Debt Payments", "expense", "#FF3399", "credit-card"),
            ("Savings", "expense", "#99FF33", "piggy-bank"),
            ("Gifts & Donations", "expense", "#3399FF", "gift"),
            ("Subscriptions", "expense", "#FF9933", "repeat"),
            ("Other Expenses", "expense", "#999999", "question")
        ]

        income_categories = [
            ("Salary", "income", "#33CC33", "briefcase"),
            ("Scholarships", "income", "#33CCCC", "graduation-cap"),
            ("Grants", "income", "#CC33CC", "hand-holding-usd"),
            ("Gifts Received", "income", "#CCCC33", "gift"),
            ("Side Jobs", "income", "#CC3333", "tools"),
            ("Investments", "income", "#3333CC", "chart-line"),
            ("Refunds", "income", "#33CC99", "undo"),
            ("Other Income", "income", "#999999", "plus-circle")
        ]

        for name, type_, color, icon in expense_categories + income_categories:
            self.cursor.execute(
                "INSERT INTO categories (name, type, color, icon) VALUES (?, ?, ?, ?)",
                (name, type_, color, icon)
            )

    def _populate_default_settings(self):
        """Add default settings to the settings table."""
        self.logger.info("Populating default settings")

        default_settings = [
            ("theme", "light"),
            ("currency", "EUR"),
            ("language", "en"),
            ("first_day_of_week", "1"),  # Monday
            ("notification_enabled", "1"),
            ("backup_frequency", "weekly"),
            ("student_budget_profile", "enabled"),
            ("ai_assistant_enabled", "1"),
            ("dashboard_widgets", json.dumps(["income_expense", "categories", "budget", "recent_transactions"])),
            ("show_cents", "1"),
            ("date_format", "dd/MM/yyyy"),
            ("enable_touch_id", "1"),
            ("login_required", "1"),
            ("password_expiry_days", "90")
        ]

        for key, value in default_settings:
            self.cursor.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?)",
                (key, value)
            )
            
    def _create_default_admin_user(self):
        """Create a default admin user if no users exist in the database."""
        try:
            # Check if any users exist
            try:
                self.cursor.execute("SELECT COUNT(*) FROM users")
                users_count = self.cursor.fetchone()[0]
            except sqlite3.OperationalError:
                # Users table might not exist yet
                users_count = 0
                
            if users_count == 0:
                import hashlib
                import secrets
                
                # Generate a strong salt
                salt = secrets.token_hex(16)
                
                # Default credentials - should be changed on first login
                username = "admin"
                password = "admin"  # This is intentionally simple for first login
                
                # Hash the password with the salt
                password_bytes = password.encode('utf-8')
                salt_bytes = bytes.fromhex(salt)
                
                password_hash = hashlib.pbkdf2_hmac(
                    'sha256',
                    password_bytes,
                    salt_bytes,
                    100000  # Number of iterations
                ).hex()
                
                # Insert the admin user
                self.cursor.execute(
                    '''INSERT INTO users 
                    (username, salt, password_hash, is_admin, is_active) 
                    VALUES (?, ?, ?, 1, 1)''',
                    (username, salt, password_hash)
                )
                
                self.conn.commit()
                self.logger.info("Created default admin user")
                
                # Also create a demo user for testing
                demo_salt = secrets.token_hex(16)
                demo_password = "demo"
                
                demo_password_bytes = demo_password.encode('utf-8')
                demo_salt_bytes = bytes.fromhex(demo_salt)
                
                demo_password_hash = hashlib.pbkdf2_hmac(
                    'sha256',
                    demo_password_bytes,
                    demo_salt_bytes,
                    100000
                ).hex()
                
                self.cursor.execute(
                    '''INSERT INTO users 
                    (username, salt, password_hash, is_admin, is_active) 
                    VALUES (?, ?, ?, 0, 1)''',
                    ("demo", demo_salt, demo_password_hash)
                )
                
                self.conn.commit()
                self.logger.info("Created demo user")
        except sqlite3.Error as e:
            self.logger.error(f"Error creating default admin user: {e}")
            self.conn.rollback()

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.logger.info("Database connection closed")

    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Get a setting value by key.

        Args:
            key: Setting key
            default: Default value if setting not found

        Returns:
            Setting value or default
        """
        try:
            self.cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
            result = self.cursor.fetchone()
            return result[0] if result else default
        except sqlite3.Error as e:
            self.logger.error(f"Error getting setting {key}: {e}")
            return default

    def update_setting(self, key: str, value: Any) -> bool:
        """
        Update a setting value.

        Args:
            key: Setting key
            value: New setting value

        Returns:
            True if successful, False otherwise
        """
        try:
            self.cursor.execute(
                "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
                (key, value)
            )
            self.conn.commit()
            self.logger.debug(f"Updated setting {key} to {value}")
            return True
        except sqlite3.Error as e:
            self.logger.error(f"Error updating setting {key}: {e}")
            self.conn.rollback()
            return False

    # Transaction Methods

    def add_transaction(self, date: str, amount: float, description: str,
                        category_id: int, is_income: bool, recurring: bool = False,
                        recurring_period: Optional[str] = None, notes: Optional[str] = None,
                        merchant: Optional[str] = None, bank: Optional[str] = None,
                        is_savings: bool = False) -> Optional[int]:
        """
        Add a new transaction to the database.

        Args:
            date: Transaction date (YYYY-MM-DD)
            amount: Transaction amount
            description: Transaction description
            category_id: Category ID
            is_income: Whether this is income (True) or expense (False)
            recurring: Whether this is a recurring transaction
            recurring_period: Recurrence period (e.g., "monthly", "weekly")
            notes: Additional notes
            merchant: Merchant name
            bank: Bank name
            is_savings: Whether this is a savings balance record (not a regular transaction)

        Returns:
            Transaction ID if successful, None otherwise
        """
        try:
            # Check if this is a special savings balance entry (from "Kontostand am" in bank statements)
            if is_savings or description and "Kontostand am" in description:
                # Make sure it's categorized as Savings
                savings_category_id = category_id
                
                # Try to find the Savings category ID if not specified
                if category_id == 1 or category_id is None:  # Default or unspecified category
                    self.cursor.execute("SELECT id FROM categories WHERE name = 'Savings'")
                    savings_cat = self.cursor.fetchone()
                    if savings_cat:
                        savings_category_id = savings_cat['id']
                    else:
                        # Add a savings category if it doesn't exist
                        savings_category_id = self.add_category("Savings", "expense", "#99FF33", "piggy-bank")
                
                # Force this to be marked as income for display purposes
                is_income = True
                
                self.logger.info(f"Processing savings balance entry: {amount} € on {date}, {description}")
            
            # Insert the transaction with the appropriate category and flags
            self.cursor.execute(
                '''INSERT INTO transactions 
                   (date, amount, description, category_id, is_income, recurring, 
                   recurring_period, notes, merchant, bank)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (date, amount, description, category_id, is_income, recurring,
                 recurring_period, notes, merchant, bank)
            )
            self.conn.commit()
            self.logger.debug(f"Added transaction: {amount} € on {date}, {description}")
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            self.logger.error(f"Error adding transaction: {e}")
            self.conn.rollback()
            return None

    def get_transactions(self, start_date: Optional[str] = None, end_date: Optional[str] = None,
                         category_id: Optional[int] = None, is_income: Optional[bool] = None,
                         limit: Optional[int] = None, offset: Optional[int] = None,
                         exclude_categories: List[str] = None) -> List[Dict[str, Any]]:
        """
        Get transactions filtered by parameters.

        Args:
            start_date: Filter by start date (inclusive)
            end_date: Filter by end date (inclusive)
            category_id: Filter by category ID
            is_income: Filter by transaction type (income/expense)
            limit: Limit the number of results
            offset: Result offset (for pagination)
            exclude_categories: List of category names to exclude

        Returns:
            List of transaction dictionaries
        """
        try:
            query = '''
            SELECT t.*, c.name as category_name, c.color as category_color, c.icon as category_icon 
            FROM transactions t
            LEFT JOIN categories c ON t.category_id = c.id
            WHERE 1=1
            '''
            params = []

            if start_date:
                query += " AND t.date >= ?"
                params.append(start_date)

            if end_date:
                query += " AND t.date <= ?"
                params.append(end_date)

            if category_id is not None:
                query += " AND t.category_id = ?"
                params.append(category_id)

            if is_income is not None:
                query += " AND t.is_income = ?"
                params.append(is_income)
                
            # Exclude specific categories (like 'Initial Balance')
            if exclude_categories and len(exclude_categories) > 0:
                placeholders = ','.join(['?'] * len(exclude_categories))
                query += f" AND (c.name IS NULL OR c.name NOT IN ({placeholders}))"
                params.extend(exclude_categories)

            query += " ORDER BY t.date DESC"

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            if offset:
                query += " OFFSET ?"
                params.append(offset)

            self.cursor.execute(query, params)

            # Convert to list of dictionaries
            transactions = []
            for row in self.cursor.fetchall():
                transactions.append(dict(row))

            return transactions

        except sqlite3.Error as e:
            self.logger.error(f"Error getting transactions: {e}")
            return []

    def update_transaction(self, transaction_id: int, **kwargs) -> bool:
        """
        Update a transaction by ID.

        Args:
            transaction_id: Transaction ID to update
            **kwargs: Fields to update (date, amount, description, etc.)

        Returns:
            True if successful, False otherwise
        """
        valid_fields = ["date", "amount", "description", "category_id", "is_income",
                        "recurring", "recurring_period", "notes", "merchant", "bank"]

        # Filter valid fields
        fields = [f"{key} = ?" for key in kwargs.keys() if key in valid_fields]

        if not fields:
            self.logger.warning("No valid fields provided for transaction update")
            return False

        try:
            query = f"UPDATE transactions SET {', '.join(fields)} WHERE id = ?"
            params = [kwargs[key] for key in kwargs.keys() if key in valid_fields]
            params.append(transaction_id)

            self.cursor.execute(query, params)
            self.conn.commit()

            if self.cursor.rowcount > 0:
                self.logger.debug(f"Updated transaction {transaction_id}")
                return True
            else:
                self.logger.warning(f"Transaction {transaction_id} not found")
                return False

        except sqlite3.Error as e:
            self.logger.error(f"Error updating transaction {transaction_id}: {e}")
            self.conn.rollback()
            return False

    def delete_transaction(self, transaction_id: int) -> bool:
        """
        Delete a transaction by ID.

        Args:
            transaction_id: Transaction ID to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            self.cursor.execute("DELETE FROM transactions WHERE id = ?", (transaction_id,))
            self.conn.commit()

            if self.cursor.rowcount > 0:
                self.logger.debug(f"Deleted transaction {transaction_id}")
                return True
            else:
                self.logger.warning(f"Transaction {transaction_id} not found")
                return False

        except sqlite3.Error as e:
            self.logger.error(f"Error deleting transaction {transaction_id}: {e}")
            self.conn.rollback()
            return False

    def get_transaction_by_id(self, transaction_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a transaction by ID.

        Args:
            transaction_id: Transaction ID

        Returns:
            Transaction dictionary or None if not found
        """
        try:
            self.cursor.execute(
                '''SELECT t.*, c.name as category_name, c.color as category_color, c.icon as category_icon 
                   FROM transactions t
                   LEFT JOIN categories c ON t.category_id = c.id
                   WHERE t.id = ?''',
                (transaction_id,)
            )
            row = self.cursor.fetchone()
            return dict(row) if row else None

        except sqlite3.Error as e:
            self.logger.error(f"Error getting transaction {transaction_id}: {e}")
            return None

    # Category Methods

    def get_categories(self, type_: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all categories or filtered by type.

        Args:
            type_: Category type ('income' or 'expense')

        Returns:
            List of category dictionaries
        """
        try:
            query = "SELECT * FROM categories"
            params = []

            if type_:
                query += " WHERE type = ?"
                params.append(type_)

            query += " ORDER BY name"
            self.cursor.execute(query, params)

            # Convert to list of dictionaries
            categories = []
            for row in self.cursor.fetchall():
                categories.append(dict(row))

            return categories

        except sqlite3.Error as e:
            self.logger.error(f"Error getting categories: {e}")
            return []

    def add_category(self, name: str, type_: str = 'expense', color: str = "#888888", icon: Optional[str] = None) -> Optional[int]:
        """
        Add a new category.

        Args:
            name: Category name
            type_: Category type ('income' or 'expense'), defaults to 'expense'
            color: Category color (hex code), defaults to gray
            icon: Category icon name

        Returns:
            Category ID if successful, None otherwise
        """
        try:
            # Check if category already exists (case-insensitive)
            self.cursor.execute("SELECT id FROM categories WHERE LOWER(name) = LOWER(?)", (name,))
            existing = self.cursor.fetchone()
            if existing:
                self.logger.info(f"Category '{name}' already exists with ID {existing['id']}")
                return existing['id']
                
            # Determine type based on name (simple heuristic)
            if not type_:
                if name.lower() in ['salary', 'income', 'revenue', 'bonus', 'refund', 'scholarship']:
                    type_ = 'income'
                else:
                    type_ = 'expense'
                    
            # Generate a default color if none provided
            if not color:
                # Simple hash of category name to get a consistent color
                hash_val = sum(ord(c) for c in name)
                r = (hash_val * 123) % 200 + 55  # Avoid too dark or too light
                g = (hash_val * 45) % 200 + 55
                b = (hash_val * 67) % 200 + 55
                color = f"#{r:02x}{g:02x}{b:02x}"
                
            self.cursor.execute(
                "INSERT INTO categories (name, type, color, icon) VALUES (?, ?, ?, ?)",
                (name, type_, color, icon)
            )
            self.conn.commit()
            self.logger.debug(f"Added category: {name} ({type_})")
            return self.cursor.lastrowid

        except sqlite3.IntegrityError:
            self.logger.warning(f"Category '{name}' already exists, but couldn't get ID")
            self.conn.rollback()
            
            # Try to get the ID again
            try:
                self.cursor.execute("SELECT id FROM categories WHERE LOWER(name) = LOWER(?)", (name,))
                existing = self.cursor.fetchone()
                if existing:
                    return existing['id']
            except:
                pass
                
            return None

        except sqlite3.Error as e:
            self.logger.error(f"Error adding category: {e}")
            self.conn.rollback()
            return None

    def update_category(self, category_id: int, **kwargs) -> bool:
        """
        Update a category by ID.

        Args:
            category_id: Category ID to update
            **kwargs: Fields to update (name, type, color, icon)

        Returns:
            True if successful, False otherwise
        """
        valid_fields = ["name", "type", "color", "icon"]
        fields = [f"{key} = ?" for key in kwargs.keys() if key in valid_fields]

        if not fields:
            self.logger.warning("No valid fields provided for category update")
            return False

        try:
            query = f"UPDATE categories SET {', '.join(fields)} WHERE id = ?"
            params = [kwargs[key] for key in kwargs.keys() if key in valid_fields]
            params.append(category_id)

            self.cursor.execute(query, params)
            self.conn.commit()

            if self.cursor.rowcount > 0:
                self.logger.debug(f"Updated category {category_id}")
                return True
            else:
                self.logger.warning(f"Category {category_id} not found")
                return False

        except sqlite3.IntegrityError:
            self.logger.warning(f"Category update failed: name already exists")
            self.conn.rollback()
            return False

        except sqlite3.Error as e:
            self.logger.error(f"Error updating category {category_id}: {e}")
            self.conn.rollback()
            return False

    def delete_category(self, category_id: int) -> bool:
        """
        Delete a category by ID.

        Args:
            category_id: Category ID to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if category is used in transactions
            self.cursor.execute("SELECT COUNT(*) FROM transactions WHERE category_id = ?", (category_id,))
            if self.cursor.fetchone()[0] > 0:
                self.logger.warning(f"Cannot delete category {category_id}: in use by transactions")
                return False

            self.cursor.execute("DELETE FROM categories WHERE id = ?", (category_id,))
            self.conn.commit()

            if self.cursor.rowcount > 0:
                self.logger.debug(f"Deleted category {category_id}")
                return True
            else:
                self.logger.warning(f"Category {category_id} not found")
                return False

        except sqlite3.Error as e:
            self.logger.error(f"Error deleting category {category_id}: {e}")
            self.conn.rollback()
            return False

    # Budget Methods

    def set_budget(self, category_id: int, amount: float, month: int, year: int) -> Optional[int]:
        """
        Set or update a budget for a category and month/year.

        Args:
            category_id: Category ID
            amount: Budget amount
            month: Month (1-12)
            year: Year

        Returns:
            Budget ID if successful, None otherwise
        """
        try:
            self.cursor.execute(
                '''INSERT OR REPLACE INTO budgets 
                   (category_id, amount, month, year)
                   VALUES (?, ?, ?, ?)''',
                (category_id, amount, month, year)
            )
            self.conn.commit()
            self.logger.debug(f"Set budget for category {category_id}: {amount} € ({month}/{year})")
            return self.cursor.lastrowid

        except sqlite3.Error as e:
            self.logger.error(f"Error setting budget: {e}")
            self.conn.rollback()
            return None

    def get_budgets(self, month: Optional[int] = None, year: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get budgets, optionally filtered by month and year.

        Args:
            month: Filter by month (1-12)
            year: Filter by year

        Returns:
            List of budget dictionaries
        """
        try:
            query = '''
            SELECT b.*, c.name as category_name, c.color as category_color, c.icon as category_icon
            FROM budgets b
            JOIN categories c ON b.category_id = c.id
            WHERE 1=1
            '''
            params = []

            if month is not None:
                query += " AND b.month = ?"
                params.append(month)

            if year is not None:
                query += " AND b.year = ?"
                params.append(year)

            self.cursor.execute(query, params)

            # Convert to list of dictionaries
            budgets = []
            for row in self.cursor.fetchall():
                budgets.append(dict(row))

            return budgets

        except sqlite3.Error as e:
            self.logger.error(f"Error getting budgets: {e}")
            return []

    # Financial Reports Methods

    def get_monthly_summary(self, month: int, year: int) -> Dict[str, Any]:
        """
        Get income, expense and savings summary for a month.

        Args:
            month: Month (1-12)
            year: Year

        Returns:
            Dictionary with monthly summary data
        """
        try:
            # Format date strings for SQLite comparison
            start_date = f"{year}-{month:02d}-01"
            if month == 12:
                next_month = 1
                next_year = year + 1
            else:
                next_month = month + 1
                next_year = year
            end_date = f"{next_year}-{next_month:02d}-01"

            # Get all transactions for this month
            self.cursor.execute(
                "SELECT id, amount, description, is_income, category_id, date FROM transactions WHERE date >= ? AND date < ?",
                (start_date, end_date)
            )
            transactions = self.cursor.fetchall()
            
            # Process transactions properly
            total_income = 0
            total_expenses = 0
            savings_transactions = 0
            
            for tx in transactions:
                amount = tx['amount']
                is_income = tx['is_income']
                description = tx['description'] if tx['description'] else ''
                
                # Check if this is a savings transaction based on various indicators in the description
                is_savings = (
                    "um " in description or 
                    "Kontostand am" in description or
                    "Total Savings Balance" in description or
                    "Account Balance" in description or
                    tx['category_id'] == 13  # 13 is the Savings category ID
                )
                
                # Log the transaction details for debugging
                self.logger.info(f"Processing transaction: ID={tx['id']}, Amount={amount:.2f}, Category_ID={tx['category_id']}, " +
                                f"Is_Income={is_income}, Description={description[:30]}, Is_Savings={is_savings}")
                
                if is_savings:
                    # Count this in a separate category - we'll consider it neutral
                    savings_transactions += amount
                    self.logger.info(f"Added to savings_transactions: {tx['id']} - {amount:.2f} - {description[:30]}")
                elif is_income:
                    total_income += amount
                else:
                    total_expenses += amount
            
            self.logger.info(f"Monthly summary - Income: {total_income:.2f}, Expenses: {total_expenses:.2f}, Savings transfers: {savings_transactions:.2f}")

            # Get category breakdown with proper filtering for savings transactions
            self.cursor.execute(
                '''SELECT c.id, c.name, c.color, c.type, SUM(t.amount) as total
                   FROM transactions t
                   JOIN categories c ON t.category_id = c.id
                   WHERE t.date >= ? AND t.date < ?
                       AND (t.description NOT LIKE '%um %' 
                           AND t.description NOT LIKE '%Kontostand am%'
                           AND t.description NOT LIKE '%Total Savings Balance%'
                           AND t.description NOT LIKE '%Account Balance%'
                           OR t.description IS NULL)
                   GROUP BY c.id
                   ORDER BY total DESC''',
                (start_date, end_date)
            )

            # Convert to list of dictionaries
            category_breakdown = []
            for row in self.cursor.fetchall():
                category_breakdown.append(dict(row))

            # Calculate savings - now excluding the "um" transfer transactions
            balance = total_income - total_expenses
            savings_rate = (balance / total_income * 100) if total_income > 0 else 0
            
            # Add the savings records to category breakdown if they exist
            if savings_transactions > 0:
                # Find the Savings category ID
                savings_cat_id = 13  # Default based on observed data
                savings_cat_name = "Savings"
                savings_cat_color = "#99FF33"  # Default green color
                
                # Try to find the actual savings category
                self.cursor.execute("SELECT id, name, color FROM categories WHERE name LIKE '%saving%'")
                savings_cat = self.cursor.fetchone()
                if savings_cat:
                    savings_cat_id = savings_cat['id']
                    savings_cat_name = savings_cat['name']
                    savings_cat_color = savings_cat['color']
                
                # Add a separate entry for savings transfers
                savings_entry = {
                    'id': savings_cat_id,
                    'name': savings_cat_name,
                    'color': savings_cat_color,
                    'type': 'savings',
                    'total': savings_transactions
                }
                category_breakdown.append(savings_entry)
            
            return {
                'total_income': total_income,
                'total_expenses': total_expenses,
                'savings_transfers': savings_transactions,
                'savings': balance,
                'savings_rate': savings_rate,
                'category_breakdown': category_breakdown
            }

        except sqlite3.Error as e:
            self.logger.error(f"Error getting monthly summary: {e}")
            return {
                'total_income': 0,
                'total_expenses': 0,
                'savings_transfers': 0, 
                'savings': 0,
                'savings_rate': 0,
                'category_breakdown': []
            }

    def get_yearly_summary(self, year: int) -> List[Dict[str, Any]]:
        """
        Get monthly summaries for an entire year.

        Args:
            year: Year

        Returns:
            List of monthly summary dictionaries
        """
        try:
            monthly_data = []

            for month in range(1, 13):
                monthly_summary = self.get_monthly_summary(month, year)
                monthly_data.append({
                    'month': month,
                    'income': monthly_summary['total_income'],
                    'expenses': monthly_summary['total_expenses'],
                    'savings_transfers': monthly_summary.get('savings_transfers', 0),
                    'savings': monthly_summary['savings'],
                    'savings_rate': monthly_summary['savings_rate']
                })

            return monthly_data

        except Exception as e:
            self.logger.error(f"Error getting yearly summary: {e}")
            return []

    # Goals Methods

    def add_goal(self, name: str, target_amount: float, start_date: str,
                 target_date: str, priority: int = 1, notes: Optional[str] = None) -> Optional[int]:
        """
        Add a new financial goal.

        Args:
            name: Goal name
            target_amount: Target amount
            start_date: Start date (YYYY-MM-DD)
            target_date: Target date (YYYY-MM-DD)
            priority: Priority level (1-3)
            notes: Additional notes

        Returns:
            Goal ID if successful, None otherwise
        """
        try:
            self.cursor.execute(
                '''INSERT INTO goals
                   (name, target_amount, start_date, target_date, priority, notes)
                   VALUES (?, ?, ?, ?, ?, ?)''',
                (name, target_amount, start_date, target_date, priority, notes)
            )
            self.conn.commit()
            self.logger.debug(f"Added goal: {name} ({target_amount} €)")
            return self.cursor.lastrowid

        except sqlite3.Error as e:
            self.logger.error(f"Error adding goal: {e}")
            self.conn.rollback()
            return None

    def update_goal_progress(self, goal_id: int, current_amount: float) -> bool:
        """
        Update progress towards a financial goal.

        Args:
            goal_id: Goal ID
            current_amount: Current amount saved

        Returns:
            True if successful, False otherwise
        """
        try:
            self.cursor.execute(
                "UPDATE goals SET current_amount = ? WHERE id = ?",
                (current_amount, goal_id)
            )
            self.conn.commit()

            if self.cursor.rowcount > 0:
                self.logger.debug(f"Updated goal {goal_id} progress: {current_amount} €")
                return True
            else:
                self.logger.warning(f"Goal {goal_id} not found")
                return False

        except sqlite3.Error as e:
            self.logger.error(f"Error updating goal progress: {e}")
            self.conn.rollback()
            return False

    def get_goals(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all financial goals or filter by status.

        Args:
            status: Goal status ('active', 'completed', etc.)

        Returns:
            List of goal dictionaries
        """
        try:
            query = "SELECT * FROM goals"
            params = []

            if status:
                query += " WHERE status = ?"
                params.append(status)

            query += " ORDER BY priority DESC, target_date ASC"
            self.cursor.execute(query, params)

            # Convert to list of dictionaries
            goals = []
            for row in self.cursor.fetchall():
                goals.append(dict(row))

            return goals

        except sqlite3.Error as e:
            self.logger.error(f"Error getting goals: {e}")
            return []

    # AI Assistant Methods

    def add_ai_tip(self, tip_text: str, context: Optional[str] = None) -> Optional[int]:
        """
        Add a new AI-generated tip.

        Args:
            tip_text: Tip text
            context: Tip context/category

        Returns:
            Tip ID if successful, None otherwise
        """
        try:
            self.cursor.execute(
                "INSERT INTO ai_tips (tip_text, context) VALUES (?, ?)",
                (tip_text, context)
            )
            self.conn.commit()
            self.logger.debug(f"Added AI tip: {tip_text[:30]}...")
            return self.cursor.lastrowid

        except sqlite3.Error as e:
            self.logger.error(f"Error adding AI tip: {e}")
            self.conn.rollback()
            return None

    def get_unread_ai_tips(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get unread AI tips.

        Args:
            limit: Maximum number of tips to return

        Returns:
            List of AI tip dictionaries
        """
        try:
            self.cursor.execute(
                "SELECT * FROM ai_tips WHERE is_read = 0 ORDER BY created_at DESC LIMIT ?",
                (limit,)
            )

            # Convert to list of dictionaries
            tips = []
            for row in self.cursor.fetchall():
                tips.append(dict(row))

            return tips

        except sqlite3.Error as e:
            self.logger.error(f"Error getting unread AI tips: {e}")
            return []

    def mark_tip_as_read(self, tip_id: int) -> bool:
        """
        Mark an AI tip as read.

        Args:
            tip_id: Tip ID

        Returns:
            True if successful, False otherwise
        """
        try:
            self.cursor.execute(
                "UPDATE ai_tips SET is_read = 1 WHERE id = ?",
                (tip_id,)
            )
            self.conn.commit()

            if self.cursor.rowcount > 0:
                self.logger.debug(f"Marked AI tip {tip_id} as read")
                return True
            else:
                self.logger.warning(f"AI tip {tip_id} not found")
                return False

        except sqlite3.Error as e:
            self.logger.error(f"Error marking AI tip as read: {e}")
            self.conn.rollback()
            return False

    # Import History Methods

    def add_import_history(self, file_name: str, file_hash: str, bank: Optional[str] = None,
                           transaction_count: int = 0, status: str = "success",
                           error_message: Optional[str] = None) -> Optional[int]:
        """
        Add a new import history record.

        Args:
            file_name: Imported file name
            file_hash: File hash for deduplication
            bank: Bank name
            transaction_count: Number of transactions imported
            status: Import status ('success', 'error', etc.)
            error_message: Error message if status is 'error'

        Returns:
            Import history ID if successful, None otherwise
        """
        try:
            self.cursor.execute(
                '''INSERT INTO import_history
                   (file_name, file_hash, bank, transaction_count, status, error_message)
                   VALUES (?, ?, ?, ?, ?, ?)''',
                (file_name, file_hash, bank, transaction_count, status, error_message)
            )
            self.conn.commit()
            self.logger.debug(f"Added import history: {file_name} ({status})")
            return self.cursor.lastrowid

        except sqlite3.Error as e:
            self.logger.error(f"Error adding import history: {e}")
            self.conn.rollback()
            return None

    def check_file_imported(self, file_hash: str) -> bool:
        """
        Check if a file has already been imported.

        Args:
            file_hash: File hash to check

        Returns:
            True if file has been imported, False otherwise
        """
        try:
            self.cursor.execute(
                "SELECT COUNT(*) FROM import_history WHERE file_hash = ?",
                (file_hash,)
            )
            count = self.cursor.fetchone()[0]
            return count > 0

        except sqlite3.Error as e:
            self.logger.error(f"Error checking file import: {e}")
            return False

    def get_import_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get import history records.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of import history dictionaries
        """
        try:
            self.cursor.execute(
                "SELECT * FROM import_history ORDER BY import_date DESC LIMIT ?",
                (limit,)
            )

            # Convert to list of dictionaries
            history = []
            for row in self.cursor.fetchall():
                history.append(dict(row))

            return history

        except sqlite3.Error as e:
            self.logger.error(f"Error getting import history: {e}")
            return []
            
    def find_duplicate_transactions(self, date: str, amount: float, time_window_days: int = 7) -> List[Dict[str, Any]]:
        """
        Find potential duplicate transactions based on date and amount.
        
        Args:
            date: Transaction date (YYYY-MM-DD)
            amount: Transaction amount
            time_window_days: Number of days before and after the date to search
            
        Returns:
            List of potential duplicate transactions
        """
        try:
            # Convert date string to datetime
            date_obj = datetime.datetime.strptime(date, "%Y-%m-%d").date()
            
            # Calculate date range
            start_date = (date_obj - datetime.timedelta(days=time_window_days)).strftime("%Y-%m-%d")
            end_date = (date_obj + datetime.timedelta(days=time_window_days)).strftime("%Y-%m-%d")
            
            # Query with small tolerance for floating point comparison
            query = '''
            SELECT t.*, c.name as category_name, c.color as category_color
            FROM transactions t
            LEFT JOIN categories c ON t.category_id = c.id
            WHERE t.date BETWEEN ? AND ? 
            AND ABS(t.amount - ?) < 0.01
            ORDER BY t.date
            '''
            
            self.cursor.execute(query, (start_date, end_date, amount))
            
            # Convert to list of dictionaries
            duplicates = []
            for row in self.cursor.fetchall():
                duplicates.append(dict(row))
                
            return duplicates
            
        except Exception as e:
            self.logger.error(f"Error finding duplicate transactions: {e}")
            return []
            
    def delete_transactions_by_properties(self, amount: float, date_range: Tuple[str, str], 
                                          description_pattern: Optional[str] = None) -> int:
        """
        Delete transactions matching specific properties.
        
        Args:
            amount: Transaction amount
            date_range: Tuple of (start_date, end_date) strings in YYYY-MM-DD format
            description_pattern: Optional SQL LIKE pattern for transaction description
            
        Returns:
            Number of transactions deleted
        """
        try:
            query = '''
            DELETE FROM transactions 
            WHERE ABS(amount - ?) < 0.01
            AND date BETWEEN ? AND ?
            '''
            params = [amount, date_range[0], date_range[1]]
            
            if description_pattern:
                query += " AND description LIKE ?"
                params.append(f"%{description_pattern}%")
                
            self.cursor.execute(query, params)
            self.conn.commit()
            
            deleted_count = self.cursor.rowcount
            self.logger.info(f"Deleted {deleted_count} transactions matching criteria")
            return deleted_count
            
        except sqlite3.Error as e:
            self.logger.error(f"Error deleting transactions by properties: {e}")
            self.conn.rollback()
            return 0
            
    # User Authentication and Management Methods
    
    def execute_query(self, query: str, params: tuple = (), fetch_one: bool = False):
        """
        Execute a SQL query with parameters.
        
        Args:
            query: SQL query string
            params: Query parameters
            fetch_one: Whether to fetch one result only
            
        Returns:
            Query results (single row or all rows)
        """
        try:
            self.cursor.execute(query, params)
            
            if fetch_one:
                return self.cursor.fetchone()
            else:
                return self.cursor.fetchall()
                
        except sqlite3.Error as e:
            self.logger.error(f"Error executing query: {e}")
            return None
    
    def authenticate_user(self, username: str, password: str, ip_address: Optional[str] = None) -> Dict[str, Any]:
        """
        Authenticate a user with username and password.
        
        Args:
            username: Username to authenticate
            password: Password to verify
            ip_address: Optional IP address for tracking login attempts
            
        Returns:
            Dictionary with authentication result and user info (if successful)
        """
        try:
            # Get user credentials
            self.cursor.execute(
                "SELECT id, username, salt, password_hash, is_admin, is_active FROM users WHERE username = ?",
                (username,)
            )
            user = self.cursor.fetchone()
            
            # Record login attempt
            login_success = False
            
            if not user:
                self.logger.warning(f"Authentication failed: User {username} not found")
                self._record_login_attempt(username, ip_address, False)
                return {"success": False, "message": "Invalid username or password"}
            
            if not user['is_active']:
                self.logger.warning(f"Authentication failed: User {username} is inactive")
                self._record_login_attempt(username, ip_address, False)
                return {"success": False, "message": "Account is inactive"}
            
            # Get salt and hash from database
            salt = user['salt']
            stored_hash = user['password_hash']
            
            # Hash the provided password with the stored salt
            import hashlib
            password_bytes = password.encode('utf-8')
            salt_bytes = bytes.fromhex(salt)
            
            hashed_password = hashlib.pbkdf2_hmac(
                'sha256',
                password_bytes,
                salt_bytes,
                100000  # Number of iterations
            ).hex()
            
            # Compare hashes
            if hashed_password != stored_hash:
                self.logger.warning(f"Authentication failed: Invalid password for user {username}")
                self._record_login_attempt(username, ip_address, False)
                return {"success": False, "message": "Invalid username or password"}
            
            # Update last login time
            self.cursor.execute(
                "UPDATE users SET last_login = datetime('now') WHERE id = ?",
                (user['id'],)
            )
            self.conn.commit()
            
            # Record successful login attempt
            self._record_login_attempt(username, ip_address, True)
            
            # Return success and user info
            return {
                "success": True,
                "user_id": user['id'],
                "username": user['username'],
                "is_admin": bool(user['is_admin']),
                "message": "Authentication successful"
            }
            
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            self._record_login_attempt(username, ip_address, False)
            return {"success": False, "message": f"Authentication error: {e}"}
    
    def _record_login_attempt(self, username: str, ip_address: Optional[str], success: bool) -> None:
        """
        Record a login attempt for security monitoring.
        
        Args:
            username: Username that attempted to login
            ip_address: IP address of the login attempt
            success: Whether the login was successful
        """
        try:
            self.cursor.execute(
                "INSERT INTO login_attempts (username, ip_address, success) VALUES (?, ?, ?)",
                (username, ip_address, success)
            )
            self.conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"Error recording login attempt: {e}")
            # Don't rollback here - this is a non-critical operation
    
    def check_login_attempts(self, username: str, minutes: int = 10, max_attempts: int = 5) -> bool:
        """
        Check if a user has exceeded maximum login attempts within a time window.
        
        Args:
            username: Username to check
            minutes: Time window in minutes
            max_attempts: Maximum number of failed attempts allowed
            
        Returns:
            True if user is locked out (exceeded max attempts), False otherwise
        """
        try:
            # Get number of failed attempts in time window
            self.cursor.execute(
                '''SELECT COUNT(*) FROM login_attempts 
                   WHERE username = ? AND success = 0 AND 
                   attempt_time > datetime('now', ?) AND 
                   attempt_time <= datetime('now')''',
                (username, f"-{minutes} minutes")
            )
            failed_attempts = self.cursor.fetchone()[0]
            
            return failed_attempts >= max_attempts
            
        except sqlite3.Error as e:
            self.logger.error(f"Error checking login attempts: {e}")
            return False  # Default to allowing login if we can't check
    
    def create_user(self, username: str, password: str, email: Optional[str] = None, 
                   full_name: Optional[str] = None, is_admin: bool = False) -> Dict[str, Any]:
        """
        Create a new user account.
        
        Args:
            username: Username for the new account
            password: Password for the new account
            email: Optional email address
            full_name: Optional full name
            is_admin: Whether this is an admin account
            
        Returns:
            Dictionary with creation result and user info (if successful)
        """
        try:
            # Check if username already exists
            self.cursor.execute("SELECT COUNT(*) FROM users WHERE username = ?", (username,))
            if self.cursor.fetchone()[0] > 0:
                return {"success": False, "message": "Username already exists"}
            
            # Generate salt and hash password
            import hashlib
            import secrets
            
            salt = secrets.token_hex(16)
            password_bytes = password.encode('utf-8')
            salt_bytes = bytes.fromhex(salt)
            
            password_hash = hashlib.pbkdf2_hmac(
                'sha256',
                password_bytes,
                salt_bytes,
                100000  # Number of iterations
            ).hex()
            
            # Insert new user
            self.cursor.execute(
                '''INSERT INTO users 
                   (username, salt, password_hash, email, full_name, is_admin, is_active)
                   VALUES (?, ?, ?, ?, ?, ?, 1)''',
                (username, salt, password_hash, email, full_name, is_admin)
            )
            self.conn.commit()
            
            user_id = self.cursor.lastrowid
            
            return {
                "success": True,
                "user_id": user_id,
                "username": username,
                "message": "User created successfully"
            }
            
        except sqlite3.Error as e:
            self.logger.error(f"Error creating user: {e}")
            self.conn.rollback()
            return {"success": False, "message": f"Error creating user: {e}"}
    
    def change_password(self, user_id: int, current_password: str, new_password: str) -> Dict[str, Any]:
        """
        Change a user's password.
        
        Args:
            user_id: User ID
            current_password: Current password for verification
            new_password: New password to set
            
        Returns:
            Dictionary with change result
        """
        try:
            # Get user credentials
            self.cursor.execute(
                "SELECT salt, password_hash FROM users WHERE id = ?",
                (user_id,)
            )
            user = self.cursor.fetchone()
            
            if not user:
                return {"success": False, "message": "User not found"}
            
            # Verify current password
            import hashlib
            
            salt = user['salt']
            stored_hash = user['password_hash']
            
            current_password_bytes = current_password.encode('utf-8')
            salt_bytes = bytes.fromhex(salt)
            
            current_hashed = hashlib.pbkdf2_hmac(
                'sha256',
                current_password_bytes,
                salt_bytes,
                100000
            ).hex()
            
            if current_hashed != stored_hash:
                return {"success": False, "message": "Current password is incorrect"}
            
            # Generate new salt and hash for new password
            import secrets
            
            new_salt = secrets.token_hex(16)
            new_password_bytes = new_password.encode('utf-8')
            new_salt_bytes = bytes.fromhex(new_salt)
            
            new_hashed = hashlib.pbkdf2_hmac(
                'sha256',
                new_password_bytes,
                new_salt_bytes,
                100000
            ).hex()
            
            # Update password
            self.cursor.execute(
                "UPDATE users SET salt = ?, password_hash = ?, updated_at = datetime('now') WHERE id = ?",
                (new_salt, new_hashed, user_id)
            )
            self.conn.commit()
            
            return {"success": True, "message": "Password changed successfully"}
            
        except sqlite3.Error as e:
            self.logger.error(f"Error changing password: {e}")
            self.conn.rollback()
            return {"success": False, "message": f"Error changing password: {e}"}
    
    def get_users(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """
        Get all users or only active users.
        
        Args:
            active_only: Whether to get only active users
            
        Returns:
            List of user dictionaries (without sensitive info)
        """
        try:
            query = '''SELECT id, username, email, full_name, is_admin, is_active, 
                      last_login, created_at, updated_at FROM users'''
            
            if active_only:
                query += " WHERE is_active = 1"
                
            query += " ORDER BY username"
            
            self.cursor.execute(query)
            
            # Convert to list of dictionaries
            users = []
            for row in self.cursor.fetchall():
                users.append(dict(row))
                
            return users
            
        except sqlite3.Error as e:
            self.logger.error(f"Error getting users: {e}")
            return []
    
    def update_user(self, user_id: int, **kwargs) -> Dict[str, Any]:
        """
        Update user information.
        
        Args:
            user_id: User ID to update
            **kwargs: Fields to update (email, full_name, is_admin, is_active)
            
        Returns:
            Dictionary with update result
        """
        valid_fields = ["email", "full_name", "is_admin", "is_active"]
        fields = [f"{key} = ?" for key in kwargs.keys() if key in valid_fields]
        
        if not fields:
            return {"success": False, "message": "No valid fields provided for update"}
        
        try:
            # Add updated_at timestamp
            fields.append("updated_at = datetime('now')")
            
            query = f"UPDATE users SET {', '.join(fields)} WHERE id = ?"
            params = [kwargs[key] for key in kwargs.keys() if key in valid_fields]
            params.append(user_id)
            
            self.cursor.execute(query, params)
            self.conn.commit()
            
            if self.cursor.rowcount > 0:
                return {"success": True, "message": "User updated successfully"}
            else:
                return {"success": False, "message": "User not found or no changes made"}
                
        except sqlite3.Error as e:
            self.logger.error(f"Error updating user: {e}")
            self.conn.rollback()
            return {"success": False, "message": f"Error updating user: {e}"}


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)

    # Create database manager
    db_manager = DatabaseManager("financial_planner_test.db")

    # Add a test category
    category_id = db_manager.add_category(
        name="Test Category",
        type_="expense",
        color="#FF0000",
        icon="test"
    )

    if category_id:
        print(f"Added category with ID: {category_id}")

        # Add a test transaction
        transaction_id = db_manager.add_transaction(
            date="2025-01-01",
            amount=100.0,
            description="Test Transaction",
            category_id=category_id,
            is_income=False
        )

        if transaction_id:
            print(f"Added transaction with ID: {transaction_id}")

            # Get the transaction
            transaction = db_manager.get_transaction_by_id(transaction_id)
            print(f"Transaction: {transaction}")

            # Get monthly summary
            summary = db_manager.get_monthly_summary(1, 2025)
            print(f"Monthly summary: {summary}")

    # Close the database connection
    db_manager.close()