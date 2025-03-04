import datetime
import sqlite3
import logging
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass, field


@dataclass
class Transaction:
    """Class representing a financial transaction."""

    # Required attributes
    date: str
    amount: float
    description: str
    is_income: bool

    # Optional attributes
    id: Optional[int] = None
    category_id: Optional[int] = None
    category_name: Optional[str] = None
    category_color: Optional[str] = None
    category_icon: Optional[str] = None
    recurring: bool = False
    recurring_period: Optional[str] = None
    notes: Optional[str] = None
    merchant: Optional[str] = None
    bank: Optional[str] = None
    
    # AI-related attributes
    user_corrected_category: bool = False  # Flag indicating user has manually corrected category
    ai_confidence_score: Optional[float] = None  # AI confidence in categorization
    ai_suggested_category: Optional[str] = None  # Category suggested by AI

    # Metadata
    _modified: bool = field(default=False, repr=False)
    _db_path: str = field(default="financial_planner.db", repr=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Transaction':
        data_copy = data.copy()
        valid_attrs = [attr for attr in cls.__annotations__ if not attr.startswith('_')]
        transaction_data = {k: v for k, v in data_copy.items() if k in valid_attrs}
        return cls(**transaction_data)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def validate(self) -> bool:
        """Validates transaction data."""
        # Basic validation
        if not self.description or not self.date:
            return False
        
        try:
            # Check date format
            if isinstance(self.date, str):
                datetime.datetime.strptime(self.date, "%Y-%m-%d")
            
            # Amount should be a number
            float(self.amount)
            
            return True
        except (ValueError, TypeError):
            return False
            
    def mark_as_user_corrected(self, category_name: str) -> bool:
        """
        Mark this transaction as having a user-corrected category that should be used for AI training.
        
        Args:
            category_name: The correct category name provided by the user
            
        Returns:
            Success flag
        """
        try:
            if not category_name:
                return False
                
            self.category_name = category_name
            self.user_corrected_category = True
            self._modified = True
            
            # Update in database if we have an ID
            if self.id:
                conn = sqlite3.connect(self._db_path)
                cursor = conn.cursor()
                
                # First update the category
                cursor.execute(
                    "UPDATE transactions SET category = ?, user_corrected_category = 1 WHERE id = ?",
                    (category_name, self.id)
                )
                
                # Check if we need to add AI training data column
                cursor.execute("PRAGMA table_info(transactions)")
                columns = [col[1] for col in cursor.fetchall()]
                
                if 'user_corrected_category' not in columns:
                    # Add the column if it doesn't exist
                    cursor.execute("ALTER TABLE transactions ADD COLUMN user_corrected_category INTEGER DEFAULT 0")
                    
                conn.commit()
                conn.close()
                
            return True
            
        except Exception as e:
            logging.error(f"Error marking transaction as user-corrected: {e}")
            return False
    
    @classmethod
    def get_all_transactions(cls) -> List['Transaction']:
        """
        Get all transactions from the database.
        
        Returns:
            List of Transaction objects
        """
        transactions = []
        try:
            conn = sqlite3.connect(cls._db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, date, amount, description, is_income, category, 
                       recurring, recurring_period, notes, merchant, bank,
                       user_corrected_category, ai_confidence_score, ai_suggested_category
                FROM transactions
                ORDER BY date DESC
            """)
            
            for row in cursor.fetchall():
                # Handle the case where not all columns exist (backward compatibility)
                tx_data = {
                    'id': row[0],
                    'date': row[1],
                    'amount': row[2],
                    'description': row[3],
                    'is_income': bool(row[4]),
                    'category_name': row[5],
                    'recurring': bool(row[6]) if row[6] is not None else False,
                    'recurring_period': row[7],
                    'notes': row[8],
                    'merchant': row[9],
                    'bank': row[10]
                }
                
                # Add AI-related fields if they exist
                if len(row) > 11:
                    tx_data['user_corrected_category'] = bool(row[11])
                if len(row) > 12:
                    tx_data['ai_confidence_score'] = row[12]
                if len(row) > 13:
                    tx_data['ai_suggested_category'] = row[13]
                
                transactions.append(cls.from_dict(tx_data))
                
            conn.close()
            
        except Exception as e:
            logging.error(f"Error getting all transactions: {e}")
            
        return transactions
        
    @classmethod
    def get_training_transactions(cls) -> List['Transaction']:
        """
        Get transactions that have been marked as user-corrected for AI training.
        
        Returns:
            List of Transaction objects with user-corrected categories
        """
        transactions = []
        try:
            conn = sqlite3.connect(cls._db_path)
            cursor = conn.cursor()
            
            # Check if the user_corrected_category column exists
            cursor.execute("PRAGMA table_info(transactions)")
            columns = [col[1] for col in cursor.fetchall()]
            
            if 'user_corrected_category' not in columns:
                # If the column doesn't exist, no training data is available
                conn.close()
                return []
                
            cursor.execute("""
                SELECT id, date, amount, description, is_income, category, 
                       recurring, recurring_period, notes, merchant, bank
                FROM transactions
                WHERE user_corrected_category = 1
                ORDER BY date DESC
            """)
            
            for row in cursor.fetchall():
                tx_data = {
                    'id': row[0],
                    'date': row[1],
                    'amount': row[2],
                    'description': row[3],
                    'is_income': bool(row[4]),
                    'category_name': row[5],
                    'recurring': bool(row[6]) if row[6] is not None else False,
                    'recurring_period': row[7],
                    'notes': row[8],
                    'merchant': row[9],
                    'bank': row[10],
                    'user_corrected_category': True
                }
                
                transactions.append(cls.from_dict(tx_data))
                
            conn.close()
            
        except Exception as e:
            logging.error(f"Error getting training transactions: {e}")
            
        return transactions