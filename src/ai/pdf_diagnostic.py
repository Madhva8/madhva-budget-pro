#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PDF Diagnostic Tool

This module provides utilities to analyze PDF bank statements,
extract their structure, and diagnose issues with parsing.
"""

import os
import re
import io
import hashlib
import pdfplumber
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict, Counter


class PDFDiagnostic:
    """Tools for analyzing and diagnosing PDF structure and content."""

    def __init__(self):
        """Initialize the PDF diagnostic tool."""
        pass

    def analyze_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Analyze a PDF file structure and content.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary with analysis results
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Basic PDF information
                info = {
                    'filename': os.path.basename(pdf_path),
                    'filepath': pdf_path,
                    'page_count': len(pdf.pages),
                    'metadata': pdf.metadata,
                    'filesize': os.path.getsize(pdf_path) / 1024,  # KB
                    'pages': []
                }

                # Analyze each page
                for i, page in enumerate(pdf.pages):
                    page_info = self._analyze_page(page, i + 1)
                    info['pages'].append(page_info)

                # Find potential table structures
                info['potential_tables'] = self._find_potential_tables(info['pages'])

                # Identify statement specific information
                info['statement_info'] = self._extract_statement_info(info['pages'])

                # Identify transactions
                info['transactions'] = self._identify_transaction_patterns(info['pages'])

                return info

        except Exception as e:
            return {
                'error': str(e),
                'filename': os.path.basename(pdf_path),
                'filepath': pdf_path
            }

    def analyze_pdf_from_bytes(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """
        Analyze a PDF from bytes data.

        Args:
            pdf_bytes: PDF content as bytes

        Returns:
            Dictionary with analysis results
        """
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                # Basic PDF information
                info = {
                    'filename': 'memory_buffer',
                    'filesize': len(pdf_bytes) / 1024,  # KB
                    'page_count': len(pdf.pages),
                    'metadata': pdf.metadata,
                    'pages': []
                }

                # Analyze each page
                for i, page in enumerate(pdf.pages):
                    page_info = self._analyze_page(page, i + 1)
                    info['pages'].append(page_info)

                # Find potential table structures
                info['potential_tables'] = self._find_potential_tables(info['pages'])

                # Identify statement specific information
                info['statement_info'] = self._extract_statement_info(info['pages'])

                # Identify transactions
                info['transactions'] = self._identify_transaction_patterns(info['pages'])

                return info

        except Exception as e:
            return {
                'error': str(e),
                'filename': 'memory_buffer'
            }

    def _analyze_page(self, page, page_number: int) -> Dict[str, Any]:
        """
        Analyze a single page of the PDF.

        Args:
            page: PDF page object
            page_number: Page number

        Returns:
            Dictionary with page analysis
        """
        # Extract text
        text = page.extract_text()

        # Extract raw text lines
        lines = text.split('\n') if text else []

        # Extract words with their positions
        words = page.extract_words()

        # Extract tables (if any)
        tables = page.extract_tables()

        # Look for images
        images = page.images

        # Calculate text statistics
        char_count = len(text) if text else 0
        word_count = len(words)
        line_count = len(lines)

        # Identify potential data patterns
        date_patterns = self._find_date_patterns(text)
        amount_patterns = self._find_amount_patterns(text)
        account_patterns = self._find_account_patterns(text)

        # Check for columns
        columns = self._identify_columns(words) if words else []

        return {
            'page_number': page_number,
            'text': text,
            'lines': lines,
            'char_count': char_count,
            'word_count': word_count,
            'line_count': line_count,
            'tables': [self._table_to_dict(table) for table in tables] if tables else [],
            'table_count': len(tables) if tables else 0,
            'columns': columns,
            'column_count': len(columns),
            'has_images': len(images) > 0,
            'image_count': len(images),
            'date_patterns': date_patterns,
            'amount_patterns': amount_patterns,
            'account_patterns': account_patterns
        }

    def _find_date_patterns(self, text: str) -> List[str]:
        """
        Find date patterns in text.

        Args:
            text: Text to search for date patterns

        Returns:
            List of found date patterns
        """
        # Common date formats (German/European)
        patterns = [
            r'\d{2}\.\d{2}\.\d{4}',  # DD.MM.YYYY
            r'\d{2}/\d{2}/\d{4}',  # DD/MM/YYYY
            r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
            r'\d{2}\.\d{2}\.\d{2}',  # DD.MM.YY
            # Common month/year formats
            r'(Jan|Feb|Mär|Apr|Mai|Jun|Jul|Aug|Sep|Okt|Nov|Dez)\.?\s+\d{4}',
            r'(Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s+\d{4}'
        ]

        found_dates = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                found_dates.extend(matches)

        return found_dates

    def _find_amount_patterns(self, text: str) -> List[str]:
        """
        Find monetary amount patterns in text.

        Args:
            text: Text to search for amount patterns

        Returns:
            List of found amount patterns
        """
        # Common amount formats (European/German)
        patterns = [
            r'\d{1,3}(?:\.\d{3})*,\d{2}',  # 1.234,56
            r'\d+,\d{2}',  # 1234,56
            r'(?:EUR|€)\s*\d{1,3}(?:\.\d{3})*,\d{2}',  # EUR 1.234,56
            r'\d{1,3}(?:\.\d{3})*,\d{2}\s*(?:EUR|€)',  # 1.234,56 EUR
            r'[-+]\s*\d{1,3}(?:\.\d{3})*,\d{2}'  # -1.234,56 or +1.234,56
        ]

        found_amounts = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                found_amounts.extend(matches)

        return found_amounts

    def _find_account_patterns(self, text: str) -> Dict[str, List[str]]:
        """
        Find banking account number patterns in text.

        Args:
            text: Text to search for account patterns

        Returns:
            Dictionary mapping account pattern types to matches
        """
        patterns = {
            'IBAN': r'[A-Z]{2}\d{2}\s*(?:\d{4}\s*){4,7}\d{0,2}',  # IBAN format
            'BIC': r'[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?',  # BIC/SWIFT format
            'Account': r'Konto\s*(?:nr\.?|nummer)\s*:?\s*\d+',  # Kontonummer
            'Bank': r'BLZ\s*:?\s*\d{8}'  # Bankleitzahl
        }

        found_accounts = {}
        for key, pattern in patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                found_accounts[key] = matches

        return found_accounts

    def _identify_columns(self, words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify potential columns in the page based on word positions.

        Args:
            words: List of words with their positions

        Returns:
            List of identified columns
        """
        if not words:
            return []

        # Extract x positions
        x_positions = [word["x0"] for word in words]

        # Use histogram to identify clusters
        hist, bin_edges = np.histogram(x_positions, bins='auto')

        # Find peaks in histogram (column starts)
        column_starts = []
        for i in range(1, len(hist)):
            if hist[i] > hist[i - 1] and hist[i] > 3:  # Threshold for column detection
                column_starts.append(bin_edges[i])

        # Convert to column objects
        columns = []
        for i, start_x in enumerate(column_starts):
            # Define column end as either next column start or page edge
            end_x = column_starts[i + 1] if i + 1 < len(column_starts) else max(w["x1"] for w in words)

            # Get words in this column
            column_words = [w for w in words if w["x0"] >= start_x and w["x0"] < end_x]

            if column_words:
                columns.append({
                    "index": i,
                    "x_start": start_x,
                    "x_end": end_x,
                    "width": end_x - start_x,
                    "word_count": len(column_words),
                    "sample_words": [w["text"] for w in column_words[:5]]
                })

        return columns

    def _table_to_dict(self, table: List[List[str]]) -> Dict[str, Any]:
        """
        Convert a table to a dictionary representation.

        Args:
            table: Table as a list of rows

        Returns:
            Dictionary with table information
        """
        if not table:
            return {"rows": 0, "cols": 0, "data": []}

        # Count rows and columns
        rows = len(table)
        cols = max(len(row) for row in table)

        # Check if first row might be headers
        has_headers = False
        if rows > 1:
            first_row = table[0]

            # Check if first row has different formatting or contains typical header keywords
            header_keywords = ["date", "datum", "description", "beschreibung", "amount", "betrag",
                               "balance", "saldo", "kontostand", "type", "typ"]

            # Convert first row to lowercase for comparison
            first_row_lower = [str(cell).lower() if cell else "" for cell in first_row]

            # Check if any header keywords exist in the first row
            if any(keyword in "".join(first_row_lower) for keyword in header_keywords):
                has_headers = True

        return {
            "rows": rows,
            "cols": cols,
            "data": table,
            "has_headers": has_headers,
            "headers": table[0] if has_headers else None
        }

    def _find_potential_tables(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find potential table structures in the PDF that might not be detected as tables.

        Args:
            pages: List of page information dictionaries

        Returns:
            List of potential table structures
        """
        potential_tables = []

        for page_info in pages:
            lines = page_info.get('lines', [])

            # Look for consistent patterns in line structure
            pattern_groups = self._group_lines_by_pattern(lines)

            for pattern, group_lines in pattern_groups.items():
                if len(group_lines) >= 3:  # Minimum rows for a table
                    # Check if these lines have consistent date and amount patterns
                    dates_count = sum(
                        1 for line in group_lines if any(date in line for date in page_info.get('date_patterns', [])))
                    amounts_count = sum(1 for line in group_lines if
                                        any(amount in line for amount in page_info.get('amount_patterns', [])))

                    if dates_count >= len(group_lines) * 0.7 or amounts_count >= len(group_lines) * 0.7:
                        potential_tables.append({
                            'page': page_info['page_number'],
                            'pattern': pattern,
                            'line_count': len(group_lines),
                            'sample_lines': group_lines[:3],
                            'has_dates': dates_count >= len(group_lines) * 0.7,
                            'has_amounts': amounts_count >= len(group_lines) * 0.7
                        })

        return potential_tables

    def _group_lines_by_pattern(self, lines: List[str]) -> Dict[str, List[str]]:
        """
        Group lines by their structural pattern.

        Args:
            lines: List of text lines

        Returns:
            Dictionary mapping patterns to lists of lines
        """
        pattern_groups = defaultdict(list)

        for line in lines:
            # Create a simplified pattern representation of the line
            pattern = self._get_line_pattern(line)
            if pattern:
                pattern_groups[pattern].append(line)

        return pattern_groups

    def _get_line_pattern(self, line: str) -> str:
        """
        Get a simplified pattern representation of a line.

        Args:
            line: Text line

        Returns:
            Pattern string
        """
        # Replace dates with D
        line = re.sub(r'\d{2}[./-]\d{2}[./-]\d{2,4}', 'D', line)

        # Replace amounts with A
        line = re.sub(r'[-+]?\d{1,3}(?:\.\d{3})*,\d{2}', 'A', line)

        # Replace numbers with N
        line = re.sub(r'\d+', 'N', line)

        # Replace words with W
        line = re.sub(r'[A-Za-z]+', 'W', line)

        # Replace whitespace sequences with a single space
        line = re.sub(r'\s+', ' ', line)

        return line.strip()

    def _extract_statement_info(self, pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract banking statement specific information.

        Args:
            pages: List of page information dictionaries

        Returns:
            Dictionary with statement information
        """
        info = {
            'bank_name': None,
            'statement_date': None,
            'account_number': None,
            'account_holder': None,
            'iban': None,
            'bic': None,
            'period': None,
            'opening_balance': None,
            'closing_balance': None
        }

        # Common patterns for statement information
        bank_patterns = [
            r'(Sparkasse\s+\w+)',
            r'(Deutsche\s+Bank)',
            r'(Commerzbank)',
            r'(DKB)',
            r'(Volksbank\s+\w+)',
            r'(Postbank)'
        ]

        date_patterns = [
            r'(Kontoauszug|Statement)\s+(?:vom|from|date)?\s*:?\s*(\d{2}[./-]\d{2}[./-]\d{2,4})',
            r'(Datum|Date)\s*:?\s*(\d{2}[./-]\d{2}[./-]\d{2,4})'
        ]

        account_holder_patterns = [
            r'(Kontoinhaber|Account holder)\s*:?\s*([A-Za-z0-9\s]+)',
            r'(?:Herr|Frau|Mr\.|Mrs\.|Ms\.)\s+([A-Za-z0-9\s]+)'
        ]

        period_patterns = [
            r'(Zeitraum|Period)\s*:?\s*(\d{2}[./-]\d{2}[./-]\d{2,4})\s*(?:bis|to)\s*(\d{2}[./-]\d{2}[./-]\d{2,4})',
            r'(\d{2}[./-]\d{2}[./-]\d{2,4})\s*(?:bis|to)\s*(\d{2}[./-]\d{2}[./-]\d{2,4})'
        ]

        balance_patterns = [
            r'(Kontostand|Balance)(?:\s+am|\s+vom)?\s*:?\s*(\d{2}[./-]\d{2}[./-]\d{2,4})?\s*:?\s*([-+]?\d{1,3}(?:\.\d{3})*,\d{2})',
            r'(Anfangssaldo|Opening balance|Saldo\s+alt)\s*:?\s*([-+]?\d{1,3}(?:\.\d{3})*,\d{2})',
            r'(Endsaldo|Closing balance|Saldo\s+neu)\s*:?\s*([-+]?\d{1,3}(?:\.\d{3})*,\d{2})'
        ]

        # Combine text from first page (usually contains statement info)
        first_page_text = pages[0]['text'] if pages else ""

        # Extract bank name
        for pattern in bank_patterns:
            match = re.search(pattern, first_page_text)
            if match:
                info['bank_name'] = match.group(1)
                break

        # Extract statement date
        for pattern in date_patterns:
            match = re.search(pattern, first_page_text)
            if match:
                info['statement_date'] = match.group(2)
                break

        # Extract account holder
        for pattern in account_holder_patterns:
            match = re.search(pattern, first_page_text)
            if match:
                holder = match.group(2) if len(match.groups()) > 1 else match.group(1)
                info['account_holder'] = holder.strip()
                break

        # Extract period
        for pattern in period_patterns:
            match = re.search(pattern, first_page_text)
            if match:
                if len(match.groups()) > 2:
                    info['period'] = f"{match.group(2)} to {match.group(3)}"
                else:
                    info['period'] = f"{match.group(1)} to {match.group(2)}"
                break

        # Extract account numbers
        for page_info in pages:
            account_patterns = page_info.get('account_patterns', {})

            if 'IBAN' in account_patterns and not info['iban']:
                info['iban'] = account_patterns['IBAN'][0]

            if 'BIC' in account_patterns and not info['bic']:
                info['bic'] = account_patterns['BIC'][0]

            if 'Account' in account_patterns and not info['account_number']:
                info['account_number'] = account_patterns['Account'][0]

        # Extract balance information
        for pattern in balance_patterns:
            if "opening" in pattern.lower() or "anfangssaldo" in pattern.lower() or "alt" in pattern.lower():
                match = re.search(pattern, first_page_text)
                if match:
                    balance = match.group(2) if len(match.groups()) > 1 else match.group(1)
                    info['opening_balance'] = balance

            if "closing" in pattern.lower() or "endsaldo" in pattern.lower() or "neu" in pattern.lower():
                match = re.search(pattern, first_page_text)
                if match:
                    balance = match.group(2) if len(match.groups()) > 1 else match.group(1)
                    info['closing_balance'] = balance

            # Look for generic balance pattern
            match = re.search(pattern, first_page_text)
            if match:
                balance = match.group(3) if len(match.groups()) > 2 else match.group(2)
                if "opening" not in info or not info['opening_balance']:
                    info['opening_balance'] = balance
                elif "closing" not in info or not info['closing_balance']:
                    info['closing_balance'] = balance

        return info

    def _identify_transaction_patterns(self, pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Identify transaction patterns in the statement.

        Args:
            pages: List of page information dictionaries

        Returns:
            Dictionary with transaction pattern information
        """
        # Combine lines from all pages
        all_lines = []
        for page in pages:
            all_lines.extend(page.get('lines', []))

        # Find lines containing dates and amounts
        transaction_lines = []
        for line in all_lines:
            date_match = any(re.search(r'\d{2}[./-]\d{2}[./-]\d{2,4}', line))
            amount_match = any(re.search(r'[-+]?\d{1,3}(?:\.\d{3})*,\d{2}', line))

            if date_match and amount_match:
                transaction_lines.append(line)

        # Group by pattern
        pattern_groups = self._group_lines_by_pattern(transaction_lines)

        # Filter likely transaction patterns (must have at least 3 matching lines)
        transaction_patterns = [
            {
                'pattern': pattern,
                'count': len(lines),
                'samples': lines[:3]
            }
            for pattern, lines in pattern_groups.items()
            if len(lines) >= 3
        ]

        # Sort by frequency
        transaction_patterns.sort(key=lambda x: x['count'], reverse=True)

        # Identify potential headers
        headers = self._identify_transaction_headers(all_lines, transaction_patterns)

        return {
            'pattern_count': len(transaction_patterns),
            'patterns': transaction_patterns,
            'total_transactions': sum(p['count'] for p in transaction_patterns),
            'headers': headers
        }

    def _identify_transaction_headers(self, all_lines: List[str], transaction_patterns: List[Dict[str, Any]]) -> List[
        str]:
        """
        Identify potential transaction table headers.

        Args:
            all_lines: All lines from the statement
            transaction_patterns: Identified transaction patterns

        Returns:
            List of potential header lines
        """
        headers = []

        # Common header terms in German and English
        header_terms = [
            "datum", "date", "buchung", "booking",
            "beschreibung", "description", "verwendungszweck", "purpose",
            "betrag", "amount", "saldo", "balance",
            "valuta", "wert", "value", "belastung", "gutschrift"
        ]

        # Look for lines containing multiple header terms
        for i, line in enumerate(all_lines):
            line_lower = line.lower()

            if sum(1 for term in header_terms if term in line_lower) >= 2:
                # Check if this line appears before transaction lines
                # Check next few lines to see if they match transaction patterns
                if i + 1 < len(all_lines) and i + 2 < len(all_lines):
                    next_lines = all_lines[i + 1:i + 4]
                    for pattern in transaction_patterns:
                        # Check if next lines match a transaction pattern
                        if any(self._get_line_pattern(next_line) == pattern['pattern'] for next_line in next_lines):
                            headers.append(line)
                            break

        return headers


# Example usage
if __name__ == "__main__":
    diagnostic = PDFDiagnostic()

    # Example: Analyze a PDF file
    # result = diagnostic.analyze_pdf("path/to/statement.pdf")
    # print(f"PDF has {result['page_count']} pages")
    # print(f"Found {result['transactions']['total_transactions']} potential transactions")