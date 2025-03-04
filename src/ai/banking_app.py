#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Banking App Integration Module

This module provides integration with banking apps and services
for automatic transaction imports into the Financial Planner.
"""

import os
import json
import time
import logging
import datetime
import requests
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum


class BankingSupportLevel(Enum):
    """Enum representing the level of support for each bank."""
    FULL = "full"  # Full API support with transaction download
    PARTIAL = "partial"  # Partial support (may require file uploads)
    STATEMENT_ONLY = "statement_only"  # Only PDF/CSV statement support
    UNSUPPORTED = "unsupported"  # No support yet


class BankingAppConnector:
    """
    Connector for banking apps and services to import transactions.

    Note: This is a simulation since real banking APIs would require
    proper authentication, API keys, and data handling that is outside
    the scope of this example.
    """

    # German banks and their support status
    SUPPORTED_BANKS = {
        "Sparkasse": {
            "support_level": BankingSupportLevel.STATEMENT_ONLY,
            "statement_formats": ["PDF", "CSV"],
            "api_available": False,
            "website": "https://www.sparkasse.de/"
        },
        "Deutsche Bank": {
            "support_level": BankingSupportLevel.STATEMENT_ONLY,
            "statement_formats": ["PDF", "CSV"],
            "api_available": False,
            "website": "https://www.deutsche-bank.de/"
        },
        "Commerzbank": {
            "support_level": BankingSupportLevel.STATEMENT_ONLY,
            "statement_formats": ["PDF", "CSV"],
            "api_available": False,
            "website": "https://www.commerzbank.de/"
        },
        "DKB": {
            "support_level": BankingSupportLevel.STATEMENT_ONLY,
            "statement_formats": ["PDF", "CSV"],
            "api_available": False,
            "website": "https://www.dkb.de/"
        },
        "ING": {
            "support_level": BankingSupportLevel.STATEMENT_ONLY,
            "statement_formats": ["PDF", "CSV"],
            "api_available": False,
            "website": "https://www.ing.de/"
        },
        "N26": {
            "support_level": BankingSupportLevel.STATEMENT_ONLY,
            "statement_formats": ["PDF", "CSV"],
            "api_available": False,
            "website": "https://n26.com/"
        },
        "comdirect": {
            "support_level": BankingSupportLevel.STATEMENT_ONLY,
            "statement_formats": ["PDF", "CSV"],
            "api_available": False,
            "website": "https://www.comdirect.de/"
        }
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the banking connector.

        Args:
            config_path: Optional path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config = {}
        self.api_tokens = {}

        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                    self.logger.info(f"Loaded banking configuration from {config_path}")
            except Exception as e:
                self.logger.error(f"Error loading banking configuration: {e}")

    def get_supported_banks(self) -> Dict[str, Dict[str, Any]]:
        """
        Get list of supported banks and their capabilities.

        Returns:
            Dictionary of supported banks
        """
        return self.SUPPORTED_BANKS

    def check_bank_support(self, bank_name: str) -> Dict[str, Any]:
        """
        Check support status for a specific bank.

        Args:
            bank_name: Name of the bank to check

        Returns:
            Dictionary with support information
        """
        if bank_name in self.SUPPORTED_BANKS:
            return self.SUPPORTED_BANKS[bank_name]
        else:
            return {
                "support_level": BankingSupportLevel.UNSUPPORTED,
                "statement_formats": [],
                "api_available": False,
                "website": None
            }

    def parse_csv_statement(self, bank_name: str, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse a CSV statement from a supported bank.

        Args:
            bank_name: Name of the bank
            file_path: Path to the CSV file

        Returns:
            List of transaction dictionaries
        """
        import csv

        if bank_name not in self.SUPPORTED_BANKS:
            self.logger.error(f"Bank {bank_name} is not supported")
            return []

        if not os.path.exists(file_path):
            self.logger.error(f"File {file_path} does not exist")
            return []

        try:
            transactions = []

            with open(file_path, 'r', encoding='utf-8') as f:
                # Try to detect CSV dialect
                sample = f.read(1024)
                f.seek(0)

                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample)
                has_header = sniffer.has_header(sample)

                # Read CSV file
                reader = csv.reader(f, dialect)

                # Skip header if present
                if has_header:
                    next(reader)

                # Parse rows based on bank format
                for row in reader:
                    if bank_name == "Sparkasse":
                        transaction = self._parse_sparkasse_csv_row(row)
                    elif bank_name == "DKB":
                        transaction = self._parse_dkb_csv_row(row)
                    elif bank_name == "N26":
                        transaction = self._parse_n26_csv_row(row)
                    elif bank_name == "ING":
                        transaction = self._parse_ing_csv_row(row)
                    else:
                        # Generic CSV parser as fallback
                        transaction = self._parse_generic_csv_row(row)

                    if transaction:
                        transactions.append(transaction)

            self.logger.info(f"Parsed {len(transactions)} transactions from {bank_name} CSV statement")
            return transactions

        except Exception as e:
            self.logger.error(f"Error parsing CSV statement: {e}")
            return []

    def _parse_sparkasse_csv_row(self, row: List[str]) -> Optional[Dict[str, Any]]:
        """
        Parse a CSV row from Sparkasse format.

        Args:
            row: CSV row as list of strings

        Returns:
            Transaction dictionary or None if parsing failed
        """
        # Check if row has enough columns
        if len(row) < 5:
            return None

        try:
            # Assume Sparkasse CSV format (simplified example):
            # Date | Description | Amount | Currency | Type
            date_str = row[0]
            description = row[1]
            amount_str = row[2].replace('.', '').replace(',', '.')  # Convert German number format
            amount = float(amount_str)

            # Determine if income or expense
            is_income = amount > 0

            # Parse date
            try:
                date = datetime.datetime.strptime(date_str, "%d.%m.%Y").strftime("%Y-%m-%d")
            except ValueError:
                # Try alternative date format
                date = datetime.datetime.strptime(date_str, "%d.%m.%y").strftime("%Y-%m-%d")

            return {
                "date": date,
                "description": description,
                "amount": abs(amount),
                "is_income": is_income,
                "bank": "Sparkasse",
                "currency": "EUR",
                "type": "Transfer" if "Überweisung" in description else "Card Payment" if "Karte" in description else "Other"
            }

        except Exception as e:
            self.logger.debug(f"Error parsing Sparkasse CSV row: {e}")
            return None

    def _parse_dkb_csv_row(self, row: List[str]) -> Optional[Dict[str, Any]]:
        """
        Parse a CSV row from DKB format.

        Args:
            row: CSV row as list of strings

        Returns:
            Transaction dictionary or None if parsing failed
        """
        # Check if row has enough columns
        if len(row) < 5:
            return None

        try:
            # Assume DKB CSV format (simplified example):
            # Booking date | Value date | Description | Amount | Credit/Debit
            booking_date_str = row[0]
            description = row[2]
            amount_str = row[3].replace('.', '').replace(',', '.')  # Convert German number format
            amount = abs(float(amount_str))

            # Determine if income or expense
            is_income = row[4].lower() == "credit" or row[4].lower() == "haben"

            # Parse date
            try:
                date = datetime.datetime.strptime(booking_date_str, "%d.%m.%Y").strftime("%Y-%m-%d")
            except ValueError:
                # Try alternative date format
                date = datetime.datetime.strptime(booking_date_str, "%d.%m.%y").strftime("%Y-%m-%d")

            return {
                "date": date,
                "description": description,
                "amount": amount,
                "is_income": is_income,
                "bank": "DKB",
                "currency": "EUR",
                "type": "Transfer" if "Überweisung" in description else "Card Payment" if "Karte" in description else "Other"
            }

        except Exception as e:
            self.logger.debug(f"Error parsing DKB CSV row: {e}")
            return None

    def _parse_n26_csv_row(self, row: List[str]) -> Optional[Dict[str, Any]]:
        """
        Parse a CSV row from N26 format.

        Args:
            row: CSV row as list of strings

        Returns:
            Transaction dictionary or None if parsing failed
        """
        # Check if row has enough columns
        if len(row) < 6:
            return None

        try:
            # Assume N26 CSV format (simplified example):
            # Date | Payee | Account number | Transaction type | Payment reference | Amount (EUR) | Amount (Foreign Currency) | Exchange Rate
            date_str = row[0]
            payee = row[1]
            transaction_type = row[3]
            reference = row[4]
            amount_str = row[5].replace('.', '').replace(',', '.')  # Convert German number format
            amount = float(amount_str)

            # Determine if income or expense
            is_income = amount > 0

            # Combine payee and reference for description
            description = f"{payee}: {reference}" if reference else payee

            # Parse date
            try:
                date = datetime.datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
            except ValueError:
                # Try alternative date format
                date = datetime.datetime.strptime(date_str, "%d.%m.%Y").strftime("%Y-%m-%d")

            return {
                "date": date,
                "description": description,
                "amount": abs(amount),
                "is_income": is_income,
                "bank": "N26",
                "currency": "EUR",
                "type": transaction_type
            }

        except Exception as e:
            self.logger.debug(f"Error parsing N26 CSV row: {e}")
            return None

    def _parse_ing_csv_row(self, row: List[str]) -> Optional[Dict[str, Any]]:
        """
        Parse a CSV row from ING format.

        Args:
            row: CSV row as list of strings

        Returns:
            Transaction dictionary or None if parsing failed
        """
        # Check if row has enough columns
        if len(row) < 6:
            return None

        try:
            # Assume ING CSV format (simplified example):
            # Booking | Value date | Payee/Payer | Description | IBAN | BIC | Amount | Currency
            date_str = row[0]
            counterparty = row[2]
            description = row[3]
            amount_str = row[6].replace('.', '').replace(',', '.')  # Convert German number format
            amount = float(amount_str)

            # Determine if income or expense
            is_income = amount > 0

            # Combine counterparty and description
            full_description = f"{counterparty}: {description}" if counterparty else description

            # Parse date
            try:
                date = datetime.datetime.strptime(date_str, "%d.%m.%Y").strftime("%Y-%m-%d")
            except ValueError:
                # Try alternative date format
                date = datetime.datetime.strptime(date_str, "%d.%m.%y").strftime("%Y-%m-%d")

            return {
                "date": date,
                "description": full_description,
                "amount": abs(amount),
                "is_income": is_income,
                "bank": "ING",
                "currency": "EUR",
                "type": "Transfer" if "Überweisung" in description else "Card Payment" if "Karte" in description else "Other"
            }

        except Exception as e:
            self.logger.debug(f"Error parsing ING CSV row: {e}")
            return None

    def _parse_generic_csv_row(self, row: List[str]) -> Optional[Dict[str, Any]]:
        """
        Parse a generic CSV row trying to detect fields.

        Args:
            row: CSV row as list of strings

        Returns:
            Transaction dictionary or None if parsing failed
        """
        # Check if row has enough columns
        if len(row) < 3:
            return None

        try:
            # Try to identify date, amount, and description columns
            date_col = -1
            amount_col = -1
            description_col = -1

            # Look for date column
            for i, cell in enumerate(row):
                # Check for date-like format
                if re.match(r'\d{2}[./-]\d{2}[./-]\d{2,4}', cell):
                    date_col = i
                    break

            # Look for amount column
            for i, cell in enumerate(row):
                # Check for amount-like format with German number formatting
                if re.match(r'-?\d{1,3}(?:\.\d{3})*,\d{2}', cell) or re.match(r'-?\d+,\d{2}', cell):
                    amount_col = i
                    break

            # If we found date and amount, let's assume the longest text field is the description
            if date_col >= 0 and amount_col >= 0:
                max_length = 0
                for i, cell in enumerate(row):
                    if i != date_col and i != amount_col and len(cell) > max_length:
                        max_length = len(cell)
                        description_col = i

            # If we couldn't identify columns, return None
            if date_col < 0 or amount_col < 0 or description_col < 0:
                return None

            # Parse identified columns
            date_str = row[date_col]
            description = row[description_col]
            amount_str = row[amount_col].replace('.', '').replace(',', '.')  # Convert German number format
            amount = float(amount_str)

            # Determine if income or expense
            is_income = amount > 0

            # Parse date (try common formats)
            date = None
            for date_format in ["%d.%m.%Y", "%d.%m.%y", "%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y"]:
                try:
                    date = datetime.datetime.strptime(date_str, date_format).strftime("%Y-%m-%d")
                    break
                except ValueError:
                    continue

            if not date:
                return None

            return {
                "date": date,
                "description": description,
                "amount": abs(amount),
                "is_income": is_income,
                "bank": "Unknown",
                "currency": "EUR",
                "type": "Unknown"
            }

        except Exception as e:
            self.logger.debug(f"Error parsing generic CSV row: {e}")
            return None

    def get_bank_statement_import_instructions(self, bank_name: str) -> Dict[str, Any]:
        """
        Get instructions for importing statements from a specific bank.

        Args:
            bank_name: Name of the bank

        Returns:
            Dictionary with import instructions
        """
        # Check if bank is supported
        if bank_name not in self.SUPPORTED_BANKS:
            return {
                "supported": False,
                "message": f"Bank {bank_name} is not currently supported",
                "alternative": "You can try using the generic CSV or PDF import"
            }

        # Get bank info
        bank_info = self.SUPPORTED_BANKS[bank_name]

        # Return instructions based on support level
        if bank_info["support_level"] == BankingSupportLevel.STATEMENT_ONLY:
            instructions = {
                "supported": True,
                "support_level": "statement_only",
                "message": f"{bank_name} is supported through statement uploads",
                "statement_formats": bank_info["statement_formats"],
                "steps": []
            }

            # Add bank-specific instructions
            if bank_name == "Sparkasse":
                instructions["steps"] = [
                    "1. Log in to your Sparkasse online banking",
                    "2. Go to 'Konto & Umsätze' (Account & Transactions)",
                    "3. Select the account and date range you want to export",
                    "4. Click on 'Umsätze exportieren' (Export Transactions)",
                    "5. Select CSV or PDF format and download the file",
                    "6. Import the downloaded file into the Financial Planner"
                ]
            elif bank_name == "DKB":
                instructions["steps"] = [
                    "1. Log in to your DKB online banking",
                    "2. Click on your account to view transactions",
                    "3. Set the desired date range",
                    "4. Click on 'Export' and select CSV or PDF format",
                    "5. Download the file",
                    "6. Import the downloaded file into the Financial Planner"
                ]
            elif bank_name == "N26":
                instructions["steps"] = [
                    "1. Log in to your N26 web interface or app",
                    "2. Go to 'My Account' > 'Statements & Reports'",
                    "3. Select 'Export Transactions'",
                    "4. Choose the date range and CSV format",
                    "5. Download the file",
                    "6. Import the downloaded file into the Financial Planner"
                ]
            else:
                # Generic instructions
                instructions["steps"] = [
                    f"1. Log in to your {bank_name} online banking",
                    "2. Go to account transactions or statements section",
                    "3. Select the account and date range you want to export",
                    "4. Look for an export or download option",
                    "5. Choose CSV or PDF format",
                    "6. Import the downloaded file into the Financial Planner"
                ]

            return instructions
        else:
            return {
                "supported": False,
                "message": f"{bank_name} integration is not currently available",
                "alternative": "You can export your transactions as CSV or PDF and import them manually"
            }


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create banking connector
    connector = BankingAppConnector()

    # Get supported banks
    supported_banks = connector.get_supported_banks()
    print(f"Supported banks: {', '.join(supported_banks.keys())}")

    # Get import instructions for Sparkasse
    instructions = connector.get_bank_statement_import_instructions("Sparkasse")
    print("\nImport instructions for Sparkasse:")
    for step in instructions["steps"]:
        print(step)