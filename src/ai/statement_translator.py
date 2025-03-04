#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bank Statement Translator

This module provides functionality to translate German banking terms and
descriptions to English for better processing and categorization.
"""

import re
from typing import Dict, List, Optional, Any


class StatementTranslator:
    """Translator for German banking terms."""

    # Common German banking terms and their English translations
    BANKING_TERMS = {
        # Transaction types
        'Lastschrift Einzug': 'Direct Debit',
        'Gutschr. Überweisung': 'Credit Transfer',
        'Kartenzahlung': 'Card Payment',
        'Bargeldausz.Debitk.GA': 'ATM Withdrawal',
        'GutschrÜberwLohnRente': 'Salary/Pension Credit',
        'Kartenzahlung Fremdw.': 'Foreign Card Payment',

        # Common terms
        'Kontoauszug': 'Account Statement',
        'Privatgirokonto': 'Private Current Account',
        'Kontostand': 'Account Balance',
        'Betrag': 'Amount',
        'Datum': 'Date',
        'Erläuterung': 'Description',
        'Auszug': 'Statement',
        'Wert': 'Value',
        'SEPA-Basislastschrift': 'SEPA Direct Debit',
        'SEPA-Überweisung': 'SEPA Transfer',
        'Entgelt': 'Fee',
        'Zinsen': 'Interest',
        'Zahlung': 'Payment',
        'Karte': 'Card',
        'Dauerauftrag': 'Standing Order',
        'Überweisung': 'Transfer',
        'Gehalt': 'Salary',
        'Lohn': 'Wages',
        'Rente': 'Pension',
        'Abbuchung': 'Debit',
        'Gutschrift': 'Credit',

        # Common merchant terms
        'Lebensmittel': 'Groceries',
        'Versicherung': 'Insurance',
        'Miete': 'Rent',
        'Telefon': 'Phone',
        'Apotheke': 'Pharmacy',
        'Restaurant': 'Restaurant',
        'Tankstelle': 'Gas Station',
        'Drogerie': 'Drugstore',
        'Einzelhandel': 'Retail',
        'Supermarkt': 'Supermarket',
        'Sparkasse': 'Savings Bank',
        'Fitnessstudio': 'Gym',
        'Wohnen': 'Housing',
        'Nebenkosten': 'Utilities',
        'Strom': 'Electricity',
        'Gas': 'Gas',
        'Wasser': 'Water',
        'Internet': 'Internet',
        'Rundfunk': 'Broadcasting',
        'Steuer': 'Tax',
        'Finanzamt': 'Tax Office',
        'Spende': 'Donation',
        'Arzt': 'Doctor',
        'Krankenhaus': 'Hospital',
        'Studierendenwerk': 'Student Services',
    }

    # German month names
    MONTH_NAMES = {
        'Januar': 'January',
        'Februar': 'February',
        'März': 'March',
        'April': 'April',
        'Mai': 'May',
        'Juni': 'June',
        'Juli': 'July',
        'August': 'August',
        'September': 'September',
        'Oktober': 'October',
        'November': 'November',
        'Dezember': 'December',
    }

    # Abbreviated German month names
    MONTH_ABBR = {
        'Jan': 'Jan',
        'Feb': 'Feb',
        'Mär': 'Mar',
        'Apr': 'Apr',
        'Mai': 'May',
        'Jun': 'Jun',
        'Jul': 'Jul',
        'Aug': 'Aug',
        'Sep': 'Sep',
        'Okt': 'Oct',
        'Nov': 'Nov',
        'Dez': 'Dec',
    }

    def __init__(self):
        """Initialize the translator."""
        pass

    def translate_term(self, term: str) -> str:
        """
        Translate a single banking term from German to English.

        Args:
            term: German banking term

        Returns:
            English translation or the original term if not found
        """
        return self.BANKING_TERMS.get(term, term)

    def translate_month(self, month: str) -> str:
        """
        Translate a German month name to English.

        Args:
            month: German month name

        Returns:
            English month name or the original if not found
        """
        # Check for full month names
        if month in self.MONTH_NAMES:
            return self.MONTH_NAMES[month]

        # Check for abbreviated month names
        if month in self.MONTH_ABBR:
            return self.MONTH_ABBR[month]

        return month

    def translate_date(self, date_str: str) -> str:
        """
        Translate a German date format to ISO format.

        Args:
            date_str: Date string in German format (DD.MM.YYYY)

        Returns:
            Date string in ISO format (YYYY-MM-DD)
        """
        # Match German date format (DD.MM.YYYY)
        date_match = re.match(r'(\d{2})\.(\d{2})\.(\d{4})', date_str)
        if date_match:
            day, month, year = date_match.groups()
            return f"{year}-{month}-{day}"

        # Try other common formats
        date_match = re.match(r'(\d{2})\s+([A-Za-zäöüÄÖÜß]{3,})\s+(\d{4})', date_str)
        if date_match:
            day, month_german, year = date_match.groups()
            month_english = self.translate_month(month_german)
            # Convert month name to number
            month_dict = {
                'January': '01', 'February': '02', 'March': '03', 'April': '04',
                'May': '05', 'June': '06', 'July': '07', 'August': '08',
                'September': '09', 'October': '10', 'November': '11', 'December': '12',
                'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05',
                'Jun': '06', 'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10',
                'Nov': '11', 'Dec': '12'
            }
            month_num = month_dict.get(month_english, '01')
            return f"{year}-{month_num}-{day}"

        # Return original if format not recognized
        return date_str

    def translate_description(self, description: str) -> str:
        """
        Translate a transaction description from German to English.

        Args:
            description: German transaction description

        Returns:
            Translated description
        """
        translated = description

        # Translate individual words and phrases
        for german, english in self.BANKING_TERMS.items():
            # Use word boundaries to avoid partial matches
            translated = re.sub(r'\b' + re.escape(german) + r'\b', english, translated, flags=re.IGNORECASE)

        return translated

    def translate_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate a transaction dictionary from German to English.

        Args:
            transaction: Transaction dictionary with German terms

        Returns:
            Transaction dictionary with English terms
        """
        translated = transaction.copy()

        # Translate description
        if 'description' in translated:
            translated['description'] = self.translate_description(translated['description'])

        # Translate date if needed
        if 'date' in translated and '.' in translated['date']:
            translated['date'] = self.translate_date(translated['date'])

        # Translate transaction type
        if 'type' in translated:
            translated['type'] = self.translate_term(translated['type'])

        return translated

    def translate_transactions(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Translate a list of transactions from German to English.

        Args:
            transactions: List of transaction dictionaries

        Returns:
            List of translated transaction dictionaries
        """
        return [self.translate_transaction(tx) for tx in transactions]


# Example usage
if __name__ == "__main__":
    translator = StatementTranslator()

    # Example translation
    german_desc = "Lastschrift Einzug Studierendenwerk Münster Miete"
    english_desc = translator.translate_description(german_desc)
    print(f"German: {german_desc}")
    print(f"English: {english_desc}")

    # Example date translation
    german_date = "31.01.2025"
    iso_date = translator.translate_date(german_date)
    print(f"German date: {german_date}")
    print(f"ISO date: {iso_date}")