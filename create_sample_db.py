#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create a sample database for Madhva Budget Pro
This script creates a clean database with sample data for GitHub
"""

import os
import sys
import sqlite3
import datetime
import random
from pathlib import Path

# Determine the path for the new sample database
DB_PATH = "sample_financial_planner.db"

# Sample transaction descriptions
SAMPLE_DESCRIPTIONS = [
    "Monthly Salary",
    "Rent Payment",
    "Grocery Shopping - Supermarket",
    "Restaurant Dinner",
    "Coffee Shop",
    "Phone Bill Payment",
    "Internet Service",
    "Electricity Bill",
    "Gas Bill",
    "Water Bill",
    "Public Transport",
    "Taxi/Uber Ride",
    "Clothing Purchase",
    "Movie Tickets",
    "Gym Membership",
    "Health Insurance",
    "Doctor Visit Co-pay",
    "Pharmacy - Medication",
    "Home Improvement Store",
    "Electronics Purchase",
    "Book Store",
    "Online Subscription",
    "Savings Deposit",
    "Investment Contribution",
    "Loan Repayment",
    "Credit Card Payment",
    "Charity Donation",
    "Gift Purchase",
    "Haircut/Salon",
    "Car Maintenance"
]

# Sample merchants
SAMPLE_MERCHANTS = [
    "Employer Inc.",
    "Apartment Management",
    "Safeway",
    "Whole Foods",
    "Trader Joe's",
    "Olive Garden",
    "Starbucks",
    "AT&T",
    "Comcast",
    "PG&E",
    "Water Utility Co.",
    "Uber",
    "Public Transit Authority",
    "Macy's",
    "H&M",
    "AMC Theaters",
    "24 Hour Fitness",
    "Blue Cross Insurance",
    "Kaiser Permanente",
    "CVS Pharmacy",
    "Home Depot",
    "Best Buy",
    "Amazon",
    "Barnes & Noble",
    "Netflix",
    "Spotify",
    "Bank of America",
    "Fidelity Investments",
    "Student Loan Servicer",
    "Visa",
    "American Red Cross",
    "Great Clips"
]

# Categories
CATEGORIES = [
    (1, "Uncategorized", False),
    (2, "Housing", False),
    (3, "Food", False),
    (4, "Transportation", False),
    (5, "Entertainment", False),
    (6, "Utilities", False),
    (7, "Healthcare", False),
    (8, "Insurance", False),
    (9, "Debt", False),
    (10, "Savings", False),
    (11, "Income", True),
    (12, "Shopping", False),
    (13, "Education", False),
    (14, "Personal Care", False),
    (15, "Gifts & Donations", False)
]

# Category mapping
CATEGORY_MAPPING = {
    "Monthly Salary": 11,
    "Rent Payment": 2,
    "Grocery Shopping - Supermarket": 3,
    "Restaurant Dinner": 3,
    "Coffee Shop": 3,
    "Phone Bill Payment": 6,
    "Internet Service": 6,
    "Electricity Bill": 6,
    "Gas Bill": 6,
    "Water Bill": 6,
    "Public Transport": 4,
    "Taxi/Uber Ride": 4,
    "Clothing Purchase": 12,
    "Movie Tickets": 5,
    "Gym Membership": 14,
    "Health Insurance": 8,
    "Doctor Visit Co-pay": 7,
    "Pharmacy - Medication": 7,
    "Home Improvement Store": 2,
    "Electronics Purchase": 12,
    "Book Store": 13,
    "Online Subscription": 5,
    "Savings Deposit": 10,
    "Investment Contribution": 10,
    "Loan Repayment": 9,
    "Credit Card Payment": 9,
    "Charity Donation": 15,
    "Gift Purchase": 15,
    "Haircut/Salon": 14,
    "Car Maintenance": 4
}

def create_sample_database():
    """Create a sample database with tables and sample data."""
    print(f"Creating sample database at {DB_PATH}...")
    
    # Delete existing database if it exists
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    
    # Create a new database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS categories (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        is_income BOOLEAN NOT NULL
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        amount REAL NOT NULL,
        description TEXT,
        category_id INTEGER,
        is_income BOOLEAN,
        merchant TEXT,
        FOREIGN KEY (category_id) REFERENCES categories (id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        email TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS settings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        value TEXT
    )
    ''')
    
    # Insert sample categories
    cursor.executemany(
        "INSERT INTO categories (id, name, is_income) VALUES (?, ?, ?)",
        CATEGORIES
    )
    
    # Insert sample transactions
    # Generate sample transactions for the last 6 months
    today = datetime.date.today()
    sample_transactions = []
    
    # Create sample transactions
    for i in range(100):  # 100 sample transactions
        # Random date in the last 6 months
        days_ago = random.randint(0, 180)
        transaction_date = today - datetime.timedelta(days=days_ago)
        date_str = transaction_date.strftime("%Y-%m-%d")
        
        # Pick a random description
        description = random.choice(SAMPLE_DESCRIPTIONS)
        
        # Determine category and if it's income
        category_id = CATEGORY_MAPPING.get(description, 1)
        is_income = category_id == 11  # Income category
        
        # Set amount (income positive, expenses negative)
        if is_income:
            amount = random.randint(1000, 5000)  # Typical salary range
        else:
            amount = -random.randint(5, 500)  # Expense range
        
        # Pick a merchant related to the transaction
        merchant_index = min(SAMPLE_DESCRIPTIONS.index(description), len(SAMPLE_MERCHANTS) - 1)
        merchant = SAMPLE_MERCHANTS[merchant_index]
        
        sample_transactions.append(
            (date_str, amount, description, category_id, is_income, merchant)
        )
    
    cursor.executemany(
        "INSERT INTO transactions (date, amount, description, category_id, is_income, merchant) VALUES (?, ?, ?, ?, ?, ?)",
        sample_transactions
    )
    
    # Add demo users (password hashes for 'admin' and 'demo')
    # Note: In a real application, use proper password hashing
    cursor.execute(
        "INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)",
        ("admin", "pbkdf2:sha256:600000$7NEqcpD7L12l43Uy$4daf1b4d09bb21e2a1de37ab4f656424787e5ff29ecff5af304a4fe53c94aa10", "admin@example.com")
    )
    
    cursor.execute(
        "INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)",
        ("demo", "pbkdf2:sha256:600000$g7RL9sKSGRx24INb$6d9f5db9f1c9322f5f09cf9aa5518dec9f8ac7cd3e7f13e764855a0c16be83c2", "demo@example.com")
    )
    
    # Add default settings
    cursor.execute(
        "INSERT INTO settings (name, value) VALUES (?, ?)",
        ("theme", "light")
    )
    
    cursor.execute(
        "INSERT INTO settings (name, value) VALUES (?, ?)",
        ("login_required", "1")
    )
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print(f"Sample database created with {len(sample_transactions)} transactions and {len(CATEGORIES)} categories.")

if __name__ == "__main__":
    create_sample_database()