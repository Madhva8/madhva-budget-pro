#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sparkasse Bank Statement Parser

This module extracts transaction data from Sparkasse bank statements (PDF format)
and converts them into structured data for the Financial Planner application.
"""

import re
import io
import os
import logging
import traceback
import pandas as pd
import pdfplumber
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple


class SparkasseParser:
    """Parser for Sparkasse bank statements in PDF format."""

    # Common transaction identifiers in Sparkasse statements
    TRANSACTION_TYPES = {
        'Lastschrift Einzug': 'Direct Debit',
        'Gutschr. Überweisung': 'Credit Transfer',
        'GutschrÜberwLohnRente': 'Salary/Pension Credit',
        'Kartenzahlung': 'Card Payment',
        'Bargeldausz.Debitk.GA': 'ATM Withdrawal',
        'Kartenzahlung Fremdw.': 'Foreign Card Payment',
    }

    # Common merchant patterns for categorization
    MERCHANT_PATTERNS = {
        # Housing/Accommodation
        'Housing': [
            r'Studierendenwerk\s+M[üu]nster',
            r'STW[-\s]M[üu]nster',
            r'Miete',
            r'Nebenkosten',
            r'Wohnung',
            r'Immobilien',
            r'WOHN',
            r'RENT',
            r'STUD\.WERK',
            r'STUDENTENWERK',
        ],

        # Telecom/Mobile
        'Telecommunications': [
            r'Telefonica',
            r'O2',
            r'Telekom',
            r'Vodafone',
            r'1und1',
            r'Internet',
            r'Mobile',
            r'DSL',
            r'TEL\.',
        ],

        # Subscriptions
        'Subscriptions': [
            r'APPLE\.COM\.BILL',
            r'iTunes',
            r'Netflix',
            r'Spotify',
            r'Disney',
            r'Youtube',
            r'Klarna',
            r'Google',
            r'EUROBILL',
            r'OpenAI',
            r'ExpressVPN',
            r'Prime',
            r'Amazon\s+Prime',
            r'AMZN',
            r'Abo',
            r'Abonnement',
            r'Subscription',
            r'ADOBE',
            r'MICROSOFT',
            r'OFFICE',
        ],

        # Food
        'Food': [
            r'REWE',
            r'EDEKA',
            r'LIDL',
            r'ALDI',
            r'NETTO',
            r'Supermarkt',
            r'Restaurant',
            r'UBER[\.\s]EATS',
            r'Lieferando',
            r'KFC',
            r'Burger King',
            r'BK\s+\d+',
            r'McDonald',
            r'Subway',
            r'Ditsch',
            r'Le Crobag',
            r'BACKEREI',
            r'Bäckerei',
            r'Baeckerei',
            r'BAECKEREI',
            r'RESTAURANT',
            r'PIZZERIA',
            r'CAFE',
            r'Café',
            r'IMBISS',
            r'MARKT',
            r'FOOD',
            r'PENNY',
            r'KAUFLAND',
            r'NAHKAUF',
            r'TEGUT',
            r'REAL',
            r'KONSUM',
            r'NORMA',
            r'SPAR',
            r'DENN',
            r'EAT',
            r'MENU',
            r'MENÜ',
            r'MAHLZEIT',
            r'KANTINE',
            r'MENSA',
        ],

        # Transportation
        'Transportation': [
            r'DB Vertrieb',
            r'UBER\s+B\.V',
            r'BOLT\.EU',
            r'Bahn',
            r'Bus',
            r'Flixbus',
            r'Taxi',
            r'VERKEHR',
            r'FAHRKARTE',
            r'TICKET',
            r'MOBILITY',
            r'FLIX',
            r'FAHRT',
            r'REISE',
            r'TRAVEL',
            r'METRO',
            r'STADTWERKE',
            r'BAHN',
            r'BUS',
            r'DB\s+',
            r'DEUTSCHE\s+BAHN',
            r'TANKSTELLE',
            r'TANKEN',
            r'GAS',
            r'BENZIN',
            r'DIESEL',
            r'ARAL',
            r'SHELL',
            r'ESSO',
            r'JET\s+',
            r'OMV',
            r'TOTAL',
        ],

        # Healthcare
        'Healthcare': [
            r'Techniker Krankenkasse',
            r'TK-',
            r'Apotheke',
            r'Arzt',
            r'Krankenversicherung',
            r'GESUNDHEIT',
            r'HEALTH',
            r'DOCTOR',
            r'DOKTOR',
            r'ZAHNARZT',
            r'DENTIST',
            r'PHARMACY',
            r'FARMACIA',
            r'APOTHE',
            r'KLINIK',
            r'SANITÄTS',
            r'MEDICAL',
            r'MEDIZIN',
            r'PRAXIS',
            r'PHYSIOTHERAPIE',
            r'MASSAGE',
            r'OPTIKER',
            r'OPTIK',
            r'OPTICIAN',
            r'KRANKEN',
        ],

        # Shopping
        'Shopping': [
            r'DM Drogerie',
            r'Rossmann',
            r'Amazon',
            r'Zalando',
            r'H&M',
            r'Paypal',
            r'KAUFHAUS',
            r'SHOP',
            r'STORE',
            r'MARKT(?!KAUF)',  # Exclude "marktkauf" which is food
            r'MALL',
            r'KAUFHOF',
            r'KARSTADT',
            r'GALERIA',
            r'SATURN',
            r'MEDIAMARKT',
            r'MEDIA\s+MARKT',
            r'IKEA',
            r'MÖBEL',
            r'FURNITURE',
            r'FASHION',
            r'WEAR',
            r'KLEIDUNG',
            r'CLOTHES',
            r'BEKLEIDUNG',
            r'TEXTIL',
            r'SCHUH',
            r'SHOE',
            r'BOOTS',
            r'DEICHMANN',
            r'SPORT',
            r'INTERSPORT',
            r'DECATHLON',
            r'ELECTRONIC',
            r'ELEKTRONIK',
            r'BOOK',
            r'BUCH',
            r'THALIA',
            r'HUGENDUBEL',
            r'BUECHER',
            r'BÜCHER',
            r'PARFUEM',
            r'PARFÜM',
            r'BEAUTY',
            r'DROGERIE',
            r'MÜLLER',
            r'MUELLER',
            r'BUDNI',
            r'ALDI\s+',  # For ALDI purchases with other words after
            r'LIDL\s+',  # For LIDL purchases with other words after
            r'REWE\s+',  # For REWE purchases with other words after
            r'EDEKA\s+', # For EDEKA purchases with other words after
            r'KAUFLAND\s+',
            r'NETTO\s+',
        ],

        # Entertainment
        'Entertainment': [
            r'Kino',
            r'Cinema',
            r'Coursera',
            r'Blossomup',
            r'THEATER',
            r'KONZERT',
            r'CONCERT',
            r'FESTIVAL',
            r'EVENT',
            r'TICKET',
            r'EINTRITTSKARTE',
            r'VERANSTALTUNG',
            r'SHOW',
            r'MUSEUM',
            r'AUSSTELLUNG',
            r'EXHIBITION',
            r'FREIZEIT',
            r'LEISURE',
            r'SPIEL',
            r'GAME',
            r'STEAM',
            r'PLAYSTATION',
            r'XBOX',
            r'NINTENDO',
            r'EPIC',
            r'UBISOFT',
            r'GAMING',
            r'TWITCH',
            r'DISCORD',
            r'AUDIBLE',
            r'KINDLE',
            r'EBOOK',
            r'BOOK',
            r'BUCH',
            r'BUCHUNG',
            r'BOOKING',
            r'NIGHT',
            r'NACHT',
            r'CLUB',
            r'BAR',
            r'PUB',
            r'DISCO',
            r'DISCO(?:THEK)?',
            r'KNEIPE',
        ],

        # Insurance
        'Insurance': [
            r'Versicherung',
            r'GETSAFE',
            r'INSURANCE',
            r'VERSICHER',
            r'ALLIANZ',
            r'AXA',
            r'HUK',
            r'DA DIREKT',
            r'GENERALI',
            r'ERGO',
            r'ZURICH',
            r'HDI',
            r'LVM',
            r'GOTHAER',
            r'DBV',
            r'POLICE',
            r'ASSURANCE',
            r'ASSEKURANZ',
            r'SCHUTZ',
            r'PROTECTION',
            r'ABSICHERUNG',
        ],

        # Donations
        'Donations': [
            r'Plan International',
            r'Spende',
            r'DONATION',
            r'SPENDEN',
            r'CHARITY',
            r'STIFTUNG',
            r'FOUNDATION',
            r'UNICEF',
            r'ROTES KREUZ',
            r'RED CROSS',
            r'CARITAS',
            r'DIAKONIE',
            r'AMNESTY',
            r'GREENPEACE',
            r'WWF',
            r'BUND',
            r'NABU',
            r'HILFSWERK',
            r'HILFSAKTION',
            r'ÄRZTE OHNE GRENZEN',
            r'DOCTORS WITHOUT BORDERS',
        ],

        # Fitness
        'Fitness': [
            r'RSG Group',
            r'McFit',
            r'Fitness',
            r'Gym',
            r'FITNESSSTUDIO',
            r'WORKOUT',
            r'SPORT',
            r'TRAINING',
            r'SPORTS',
            r'FITX',
            r'CLEVER FIT',
            r'JOHN REED',
            r'JOHN\s+REED',
            r'KIESER',
            r'HOLMES PLACE',
            r'HOLMES\s+PLACE',
            r'ELEMENTS',
            r'PILATES',
            r'YOGA',
            r'CROSSFIT',
            r'SCHWIMMBAD',
            r'SWIMMING',
            r'SWIM',
            r'POOL',
            r'SAUNA',
        ],

        # Education
        'Education': [
            r'SCHULE',
            r'SCHOOL',
            r'COLLEGE',
            r'UNIVERSITY',
            r'UNIVERSITÄT',
            r'HOCHSCHULE',
            r'SEMESTERBEITRAG',
            r'SEMESTER',
            r'TUITION',
            r'STUDIENGEBÜHR',
            r'STUDIENGEBUEHR',
            r'KURS',
            r'COURSE',
            r'TRAINING',
            r'WEITERBILDUNG',
            r'FORTBILDUNG',
            r'EDUCATION',
            r'BILDUNG',
            r'LERNEN',
            r'LEARN',
            r'STUDY',
            r'STUDIUM',
            r'AUSBILDUNG',
            r'APPRENTICESHIP',
            r'VORKURS',
            r'SEMINAR',
            r'WORKSHOP',
            r'UNTERRICHT',
            r'LESSON',
            r'INSTITUT',
            r'INSTITUTE',
            r'AKADEMIE',
            r'ACADEMY',
            r'CLASS',
            r'KLASSE',
            r'SCHOOLBOOK',
            r'SCHULBUCH',
            r'LEHRBUCH',
            r'TEXTBOOK',
            r'FH ',
            r'UNI ',
            r'TU ',
            r'TH ',
            r'LMU',
            r'WWU',
            r'RWTH',
        ],

        # Banking & Finance
        'Banking': [
            r'Bankgebuehr',
            r'Bankgebühr',
            r'KONTOFÜHRUNGSGEBÜHR',
            r'KONTOFUEHRUNGSGEBUEHR',
            r'ACCOUNT FEE',
            r'MAINTENANCE FEE',
            r'BANKING FEE',
            r'SERVICE FEE',
            r'GEBÜHR',
            r'GEBUEHR',
            r'FEE',
            r'ZINSEN',
            r'INTEREST',
            r'KREDIT',
            r'LOAN',
            r'DARLEHEN',
            r'MORTGAGE',
            r'HYPOTHEK',
            r'FINANZ',
            r'FINANCE',
            r'BANK',
            r'SPARKASSE',
            r'VOLKSBANK',
            r'RAIFFEISENBANK',
            r'DIREKTBANK',
            r'MAXBLUE',
            r'BARCLAYS',
            r'DEUTSCHE BANK',
            r'COMMERZBANK',
            r'HVB',
            r'HYPOVEREINSBANK',
            r'DKB',
            r'TARGO',
            r'SANTANDER',
            r'ING',
            r'POSTBANK',
            r'COMDIRECT',
            r'TARGOBANK',
            r'N26',
            r'TRANSFER',
            r'ÜBERWEISUNG',
            r'UEBERWEISUNG',
            r'GELDAUTOMAT',
            r'ATM',
            r'CASH',
            r'BARGELD',
            r'ABHEBUNG',
            r'WITHDRAWAL',
            r'WALLET',
            r'BÖRSE',
            r'BOERSE',
            r'STOCK',
            r'AKTIEN',
            r'SHARES',
            r'ANLAGE',
            r'INVESTMENT',
            r'DEPOT',
            r'PORTFOLIO',
            r'ETF',
            r'FONDS',
            r'FUND',
        ],
    }

    def __init__(self):
        """Initialize the parser."""
        self.logger = logging.getLogger(__name__)

    def parse_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Parse a Sparkasse PDF bank statement.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of transactions with parsed details
        """
        transactions = []

        try:
            self.logger.info(f"Attempting to parse PDF file: {pdf_path}")
            print(f"Attempting to parse PDF file: {pdf_path}")

            # Check if file exists
            if not os.path.exists(pdf_path):
                self.logger.error(f"PDF file does not exist: {pdf_path}")
                print(f"PDF file does not exist: {pdf_path}")
                return []

            # Try to open the file to check permissions
            try:
                with open(pdf_path, 'rb') as f:
                    self.logger.info("PDF file opened successfully")
                    print("PDF file opened successfully")
            except Exception as e:
                self.logger.error(f"Error opening PDF file: {e}")
                print(f"Error opening PDF file: {e}")
                return []

            with pdfplumber.open(pdf_path) as pdf:
                self.logger.info(f"PDF opened with pdfplumber, {len(pdf.pages)} pages found")
                print(f"PDF opened with pdfplumber, {len(pdf.pages)} pages found")

                for page_num, page in enumerate(pdf.pages):
                    self.logger.info(f"Processing page {page_num + 1}/{len(pdf.pages)}")
                    print(f"Processing page {page_num + 1}/{len(pdf.pages)}")

                    try:
                        text = page.extract_text()
                        if text:
                            self.logger.info(f"Extracted {len(text)} characters from page {page_num + 1}")
                            print(f"Extracted {len(text)} characters from page {page_num + 1}")
                            # Print a short sample for debugging
                            print(f"Sample text: {text[:100]}...")

                            page_transactions = self._extract_transactions_from_text(text)
                            if page_transactions:
                                self.logger.info(f"Found {len(page_transactions)} transactions on page {page_num + 1}")
                                print(f"Found {len(page_transactions)} transactions on page {page_num + 1}")
                                transactions.extend(page_transactions)
                            else:
                                self.logger.warning(f"No transactions found on page {page_num + 1}")
                                print(f"No transactions found on page {page_num + 1}")
                        else:
                            self.logger.warning(f"No text extracted from page {page_num + 1}")
                            print(f"No text extracted from page {page_num + 1}")
                    except Exception as e:
                        self.logger.error(f"Error processing page {page_num + 1}: {e}")
                        print(f"Error processing page {page_num + 1}: {e}")
                        traceback.print_exc()

            self.logger.info(f"Total transactions found: {len(transactions)}")
            print(f"Total transactions found: {len(transactions)}")

            if not transactions:
                self.logger.warning("No transactions were extracted from the PDF")
                print("No transactions were extracted from the PDF")

            return transactions

        except Exception as e:
            self.logger.error(f"Error parsing PDF: {e}")
            print(f"Error parsing PDF: {e}")
            traceback.print_exc()
            return []

    def parse_pdf_from_bytes(self, pdf_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Parse a Sparkasse PDF bank statement from bytes data.

        Args:
            pdf_bytes: PDF content as bytes

        Returns:
            List of transactions with parsed details
        """
        transactions = []

        try:
            self.logger.info(f"Attempting to parse PDF from bytes, size: {len(pdf_bytes)} bytes")
            print(f"Attempting to parse PDF from bytes, size: {len(pdf_bytes)} bytes")

            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                self.logger.info(f"PDF opened with pdfplumber, {len(pdf.pages)} pages found")
                print(f"PDF opened with pdfplumber, {len(pdf.pages)} pages found")

                for page_num, page in enumerate(pdf.pages):
                    self.logger.info(f"Processing page {page_num + 1}/{len(pdf.pages)}")
                    print(f"Processing page {page_num + 1}/{len(pdf.pages)}")

                    try:
                        text = page.extract_text()
                        if text:
                            self.logger.info(f"Extracted {len(text)} characters from page {page_num + 1}")
                            print(f"Extracted {len(text)} characters from page {page_num + 1}")
                            # Print a short sample for debugging
                            print(f"Sample text: {text[:100]}...")

                            page_transactions = self._extract_transactions_from_text(text)
                            if page_transactions:
                                self.logger.info(f"Found {len(page_transactions)} transactions on page {page_num + 1}")
                                print(f"Found {len(page_transactions)} transactions on page {page_num + 1}")
                                transactions.extend(page_transactions)
                            else:
                                self.logger.warning(f"No transactions found on page {page_num + 1}")
                                print(f"No transactions found on page {page_num + 1}")
                        else:
                            self.logger.warning(f"No text extracted from page {page_num + 1}")
                            print(f"No text extracted from page {page_num + 1}")
                    except Exception as e:
                        self.logger.error(f"Error processing page {page_num + 1}: {e}")
                        print(f"Error processing page {page_num + 1}: {e}")
                        traceback.print_exc()

            self.logger.info(f"Total transactions found: {len(transactions)}")
            print(f"Total transactions found: {len(transactions)}")

            return transactions

        except Exception as e:
            self.logger.error(f"Error parsing PDF bytes: {e}")
            print(f"Error parsing PDF bytes: {e}")
            traceback.print_exc()
            return []

    def _extract_transactions_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract transactions from the text content of a PDF page.

        Args:
            text: The extracted text from a PDF page

        Returns:
            List of transactions found on the page
        """
        transactions = []
        self.logger.info("Extracting transactions from text")
        print("Extracting transactions from text")
        
        # Dump the first few chars for debugging
        print(f"TEXT SAMPLE (first 200 chars): {text[:200]}...")
        
        # Detect and handle all account balance entries
        balance_entries = []
        
        # Improved pattern to better match "Kontostand am" entries with more flexible spacing
        balance_pattern = r'Kontostand\s+am\s+(\d{2}\.\d{2}\.\d{4})(?:.{0,80}?)(\d{1,3}(?:\.\d{3})*,\d{2})'
        
        # Additional pattern to catch other variations, including Auszug Nr. format
        alt_balance_pattern = r'Auszug\s+Nr\.\s+\d+.*?Kontostand(?:\s+vom|\s+am)?\s+(\d{2}\.\d{2}\.\d{4})\s+(\d{1,3}(?:\.\d{3})*,\d{2})'
        
        # Pattern to find dates following 'Auszug Nr.' with a nearby amount that represents account balance
        auszug_pattern = r'Auszug\s+Nr\.\s+\d+\s+(?:vom|per)?\s+(\d{2}\.\d{2}\.\d{4})(?:.{1,100}?)(\d{1,3}(?:\.\d{3})*,\d{2})'
        
        # Find all balance entries with the first pattern
        for match in re.finditer(balance_pattern, text):
            balance_date = match.group(1)
            balance_amount = self._parse_amount(match.group(2))
            balance_pos = match.start()
            match_context = text[max(0, match.start()-20):min(len(text), match.end()+20)]
            
            balance_entries.append({
                'date': balance_date,
                'amount': balance_amount,
                'position': balance_pos,
                'is_savings': True  # Flag this as a savings entry
            })
            
            self.logger.info(f"Found account balance (pattern 1): {balance_amount} on {balance_date}")
            self.logger.info(f"Context: {match_context}")
            print(f"Found account balance: {balance_amount} on {balance_date} - this will be treated as savings")
            
        # Find all balance entries with the alternate pattern
        for match in re.finditer(alt_balance_pattern, text):
            balance_date = match.group(1)
            balance_amount = self._parse_amount(match.group(2))
            balance_pos = match.start()
            match_context = text[max(0, match.start()-20):min(len(text), match.end()+20)]
            
            # Check if we already found this date to avoid duplicates
            if not any(entry['date'] == balance_date for entry in balance_entries):
                balance_entries.append({
                    'date': balance_date,
                    'amount': balance_amount,
                    'position': balance_pos,
                    'is_savings': True  # Flag this as a savings entry
                })
                
                self.logger.info(f"Found account balance (pattern 2): {balance_amount} on {balance_date}")
                self.logger.info(f"Context: {match_context}")
                print(f"Found account balance (alternate pattern): {balance_amount} on {balance_date} - this will be treated as savings")
                
        # Find balance entries with the Auszug Nr. pattern
        for match in re.finditer(auszug_pattern, text):
            balance_date = match.group(1)
            balance_amount = self._parse_amount(match.group(2))
            balance_pos = match.start()
            match_context = text[max(0, match.start()-20):min(len(text), match.end()+20)]
            
            # Check if we already found this date to avoid duplicates
            if not any(entry['date'] == balance_date for entry in balance_entries):
                balance_entries.append({
                    'date': balance_date,
                    'amount': balance_amount,
                    'position': balance_pos,
                    'is_savings': True  # Flag this as a savings entry
                })
                
                self.logger.info(f"Found account balance (Auszug pattern): {balance_amount} on {balance_date}")
                self.logger.info(f"Context: {match_context}")
                print(f"Found account balance (Auszug pattern): {balance_amount} on {balance_date} - this will be treated as savings")

        try:
            # Create savings transactions from balance entries
            for entry in balance_entries:
                # Convert date format from DD.MM.YYYY to YYYY-MM-DD
                date_obj = datetime.strptime(entry['date'], '%d.%m.%Y')
                formatted_date = date_obj.strftime('%Y-%m-%d')
                
                # Create a transaction record for this savings balance
                savings_tx = {
                    'date': formatted_date,
                    'description': f"Kontostand am {entry['date']} - Total Savings Balance",
                    'amount': abs(entry['amount']),  # Store absolute amount
                    'is_income': True,  # Treat as income for display purposes
                    'type': 'Savings Balance',
                    'category': 'Savings',
                    'merchant': 'Bank',
                    'raw_description': f"Kontostand am {entry['date']}",
                    'is_savings': True  # Special flag to identify as a savings balance
                }
                
                transactions.append(savings_tx)
                print(f"Added savings balance transaction: {formatted_date} | {entry['amount']} | Savings Balance")
                
                # Make the balance more prominently visible in the log for debugging
                self.logger.info(f"*** SAVINGS BALANCE: {formatted_date} | {entry['amount']:.2f} € | Kontostand am {entry['date']} ***")

            # Search for all "Kontostand" entries we might have missed
            additional_balance_pattern = r'(?:Kontostand|Saldo)(?:\s+vom|\s+am|\s+:|\s+)?\s*(\d{2}\.\d{2}\.\d{4})(?:.*?)(?:EUR\s*)?(\d{1,3}(?:\.\d{3})*,\d{2})'
            
            for additional_match in re.finditer(additional_balance_pattern, text):
                balance_date = additional_match.group(1)
                balance_amount = self._parse_amount(additional_match.group(2))
                match_context = text[max(0, additional_match.start()-20):min(len(text), additional_match.end()+20)]
                
                # Skip if this date is already in our balance entries
                if any(entry['date'] == balance_date for entry in balance_entries):
                    continue
                    
                self.logger.info(f"Found additional balance: {balance_amount} on {balance_date}")
                self.logger.info(f"Match context: {match_context}")
                print(f"Found additional balance: {balance_amount} on {balance_date}")
                
                # Add as a savings transaction
                date_obj = datetime.strptime(balance_date, '%d.%m.%Y')
                formatted_date = date_obj.strftime('%Y-%m-%d')
                
                savings_tx = {
                    'date': formatted_date,
                    'description': f"Kontostand am {balance_date} - Account Balance",
                    'amount': abs(balance_amount),
                    'is_income': True,
                    'type': 'Savings Balance',
                    'category': 'Savings',
                    'merchant': 'Bank',
                    'raw_description': f"Kontostand am {balance_date}",
                    'is_savings': True
                }
                
                transactions.append(savings_tx)
                self.logger.info(f"*** ADDITIONAL SAVINGS BALANCE: {formatted_date} | {balance_amount:.2f} € | Kontostand am {balance_date} ***")
                print(f"Added additional balance transaction: {formatted_date} | {balance_amount} | Savings Balance")

            # Find the final account balance if available
            final_balance_match = re.search(r'Kontostand am (\d{2}\.\d{2}\.\d{4}) um .*?(\d{1,3}(?:\.\d{3})*,\d{2})',
                                            text)
            if final_balance_match and not any(entry['date'] == final_balance_match.group(1) for entry in balance_entries):
                final_date = final_balance_match.group(1)
                final_balance = self._parse_amount(final_balance_match.group(2))
                self.logger.info(f"Found final balance: {final_balance} on {final_date}")
                print(f"Found final balance: {final_balance} on {final_date}")
                
                # Add as a savings transaction if not already captured
                date_obj = datetime.strptime(final_date, '%d.%m.%Y')
                formatted_date = date_obj.strftime('%Y-%m-%d')
                
                savings_tx = {
                    'date': formatted_date,
                    'description': f"Kontostand am {final_date} - Final Balance",
                    'amount': abs(final_balance),
                    'is_income': True,
                    'type': 'Savings Balance',
                    'category': 'Savings',
                    'merchant': 'Bank',
                    'raw_description': f"Kontostand am {final_date}",
                    'is_savings': True
                }
                
                transactions.append(savings_tx)
                print(f"Added final balance transaction: {formatted_date} | {final_balance} | Savings Balance")

            # Find all transactions in the text
            lines = text.split('\n')
            self.logger.info(f"Processing {len(lines)} lines of text")
            print(f"Processing {len(lines)} lines of text")

            # Print first 20 lines for debugging
            print("First 20 lines of the document:")
            for i, line in enumerate(lines[:20]):
                print(f"Line {i}: {line}")
                
            # Try to detect the transaction table structure by finding the header
            header_pattern = re.compile(r'Datum\s+Erläuterung\s+Betrag\s+EUR', re.IGNORECASE)
            header_indexes = []
            
            for i, line in enumerate(lines):
                if header_pattern.search(line):
                    header_indexes.append(i)
                    print(f"Found transaction table header at line {i}: {line}")
            
            # Process transactions using a more robust approach that doesn't depend on exact column positions
            if header_indexes:
                # Start processing from the line after the last header
                start_line = header_indexes[-1] + 1
                
                # Use direct pattern matching for each line to identify transactions
                i = start_line
                while i < len(lines):
                    line = lines[i]
                    
                    # Look for date pattern (dd.mm.yyyy) to identify start of transaction
                    date_match = re.search(r'(\d{2}\.\d{2}\.\d{4})', line)
                    if date_match:
                        date_str = date_match.group(1)
                        date_start_pos = line.find(date_str)
                        
                        # Check if this line contains an amount
                        amount_match = re.search(r'([-+]?\d{1,3}(?:\.\d{3})*,\d{2})', line)
                        
                        if amount_match:
                            # Transaction on a single line - extract all parts
                            amount_str = amount_match.group(1)
                            amount_start_pos = line.find(amount_str)
                            
                            # Description is between date and amount
                            if date_start_pos >= 0 and amount_start_pos > date_start_pos:
                                description = line[date_start_pos + len(date_str):amount_start_pos].strip()
                                transaction = self._parse_transaction(date_str, description, amount_str)
                                if transaction:
                                    transactions.append(transaction)
                                    print(f"Added single-line transaction: {transaction['date']} | {transaction['amount']} | {transaction['description'][:30]}...")
                        else:
                            # This could be a multi-line transaction - collect description across lines
                            description = line[date_start_pos + len(date_str):].strip()
                            
                            # Look ahead for the amount in subsequent lines
                            j = i + 1
                            amount_found = False
                            
                            while j < len(lines) and j < i + 5:  # Look at most 5 lines ahead
                                next_line = lines[j]
                                
                                # If we find a new date, break - we're at a new transaction
                                if re.search(r'(\d{2}\.\d{2}\.\d{4})', next_line):
                                    break
                                
                                # Look for an amount
                                amount_match = re.search(r'([-+]?\d{1,3}(?:\.\d{3})*,\d{2})', next_line)
                                if amount_match:
                                    amount_str = amount_match.group(1)
                                    amount_start_pos = next_line.find(amount_str)
                                    
                                    # Add text before the amount to description
                                    if amount_start_pos > 0:
                                        description += " " + next_line[:amount_start_pos].strip()
                                    
                                    transaction = self._parse_transaction(date_str, description, amount_str)
                                    if transaction:
                                        transactions.append(transaction)
                                        print(f"Added multi-line transaction: {transaction['date']} | {transaction['amount']} | {transaction['description'][:30]}...")
                                    
                                    amount_found = True
                                    i = j  # Skip to this line
                                    break
                                else:
                                    # Add this line to description
                                    description += " " + next_line.strip()
                                
                                j += 1
                            
                            # If we didn't find an amount but collected description, try to look ahead a bit more
                            if not amount_found and j < len(lines):
                                # Look a few more lines ahead for any amount
                                for k in range(j, min(j + 3, len(lines))):
                                    amount_match = re.search(r'([-+]?\d{1,3}(?:\.\d{3})*,\d{2})', lines[k])
                                    if amount_match:
                                        amount_str = amount_match.group(1)
                                        transaction = self._parse_transaction(date_str, description, amount_str)
                                        if transaction:
                                            transactions.append(transaction)
                                            print(f"Added extended multi-line transaction: {transaction['date']} | {transaction['amount']} | {transaction['description'][:30]}...")
                                        i = k  # Skip to this line
                                        break
                    
                    i += 1  # Move to next line
            
            # If we still have few transactions, try pattern matching as a fallback
            if len(transactions) < 5:
                print("Few structured transactions found, trying pattern matching...")
                
                # Enhanced patterns to handle different Sparkasse statement formats
                transaction_patterns = [
                    # Standard format: date, description, amount at end of line
                    r'(\d{2}\.\d{2}\.\d{4})\s+(.*?)\s+([-+]?\d{1,3}(?:\.\d{3})*,\d{2})\s*$',
                    
                    # Date at start, amount at end, with possible whitespace
                    r'^\s*(\d{2}\.\d{2}\.\d{4})\s+(.*?)\s+([-+]?\d{1,3}(?:\.\d{3})*,\d{2})\s*$',
                    
                    # Date followed by description and amount
                    r'(\d{2}\.\d{2}\.\d{4})\s+(.*?)\s+([-+]?\d{1,3}(?:\.\d{3})*,\d{2})',
                    
                    # Alternative format with negative amount
                    r'(\d{2}\.\d{2}\.\d{4})\s+(.*?)\s+([-]\d{1,3}(?:\.\d{3})*,\d{2})',
                ]

                # First pass: Extract transactions that match patterns directly
                for line_num, line in enumerate(lines):
                    # Try all transaction patterns
                    tx_found = False
                    for pattern in transaction_patterns:
                        tx_match = re.search(pattern, line)
                        if tx_match:
                            try:
                                date_str = tx_match.group(1)
                                description = tx_match.group(2).strip()
                                amount_str = tx_match.group(3)
                                
                                print(f"Line {line_num} - Matched: Date={date_str}, Desc={description[:30]}..., Amount={amount_str}")
                                
                                # Parse the transaction data
                                transaction = self._parse_transaction(date_str, description, amount_str)
                                if transaction:
                                    # Check if this transaction is already captured
                                    existing = False
                                    for tx in transactions:
                                        if tx['date'] == transaction['date'] and abs(tx['amount'] - transaction['amount']) < 0.01:
                                            existing = True
                                            break
                                            
                                    if not existing:
                                        transactions.append(transaction)
                                        print(f"Added pattern transaction: {transaction['date']} | {transaction['amount']} | {transaction['description'][:30]}...")
                                    tx_found = True
                                    break
                            except Exception as e:
                                print(f"Error parsing transaction in line {line_num}: {e}")
                    
                    # Debug output for lines with dates that didn't match
                    if not tx_found and line_num < 50:  # Only look at first 50 lines
                        date_match = re.search(r'\b(\d{2}\.\d{2}\.\d{4})\b', line)
                        if date_match:
                            print(f"Line {line_num} has date {date_match.group(1)} but didn't match transaction pattern: {line[:50]}...")
            
            # Extract date-amount pairs from entire text as a last resort
            if len(transactions) < 5:
                print("Still few transactions found, trying date-amount pairs...")
                date_amount_pairs = re.findall(r'(\d{2}\.\d{2}\.\d{4}).*?([-+]?\d{1,3}(?:\.\d{3})*,\d{2})', text)
                for date_str, amount_str in date_amount_pairs:
                    # Check if this date-amount pair is already in our transactions list
                    existing = False
                    for tx in transactions:
                        if tx['date'] == datetime.strptime(date_str, '%d.%m.%Y').strftime('%Y-%m-%d') and \
                           abs(tx['amount'] - abs(self._parse_amount(amount_str))) < 0.01:
                            existing = True
                            break
                    
                    if not existing:
                        # Extract some context around this pair
                        # Find the position of this date in the text
                        date_pos = text.find(date_str)
                        if date_pos > 0:
                            # Extract up to 150 chars after the date as description
                            context = text[date_pos:date_pos+150]
                            # Remove the date and amount from the context
                            description = context.replace(date_str, '').replace(amount_str, '').strip()
                            
                            # Parse the transaction
                            transaction = self._parse_transaction(date_str, description, amount_str)
                            if transaction:
                                transactions.append(transaction)
                                print(f"Added date-amount pair: {transaction['date']} | {transaction['amount']} | {transaction['description'][:30]}...")
            
            # Try to reconstruct multi-line transactions if we still have problems
            if len(transactions) < 5:
                print("Attempting to reconstruct multi-line transactions...")
                # Find all possible dates and amounts in the text
                date_matches = list(re.finditer(r'\b(\d{2}\.\d{2}\.\d{4})\b', text))
                amount_matches = list(re.finditer(r'\b([-+]?\d{1,3}(?:\.\d{3})*,\d{2})\b', text))
                
                print(f"Found {len(date_matches)} dates and {len(amount_matches)} amounts in the text")
                
                # For each date, find the corresponding amount and description
                for i, date_match in enumerate(date_matches):
                    date_str = date_match.group(1)
                    date_pos = date_match.start()
                    
                    # Find next date position or end of text
                    next_date_pos = text.find(date_matches[i+1].group(1)) if i < len(date_matches)-1 else len(text)
                    
                    # Find amounts between this date and the next date
                    section_amounts = [m for m in amount_matches if date_pos < m.start() < next_date_pos]
                    
                    if section_amounts:
                        amount_match = section_amounts[-1]  # Take the last amount in this section
                        amount_str = amount_match.group(1)
                        amount_pos = amount_match.start()
                        
                        # Extract description from after date to before amount
                        desc_start = date_pos + len(date_str)
                        desc_end = amount_pos
                        description = text[desc_start:desc_end].strip()
                        
                        # Clean up description (remove line breaks, excessive spaces)
                        description = re.sub(r'\s+', ' ', description)
                        
                        # Check if we already have this transaction
                        existing = False
                        formatted_date = datetime.strptime(date_str, '%d.%m.%Y').strftime('%Y-%m-%d')
                        amount_value = abs(self._parse_amount(amount_str))
                        
                        for tx in transactions:
                            if tx['date'] == formatted_date and abs(tx['amount'] - amount_value) < 0.01:
                                existing = True
                                break
                        
                        if not existing and description:
                            transaction = self._parse_transaction(date_str, description, amount_str)
                            if transaction:
                                transactions.append(transaction)
                                print(f"Added reconstructed transaction: {transaction['date']} | {transaction['amount']} | {transaction['description'][:30]}...")
            
            self.logger.info(f"Extracted {len(transactions)} transactions")
            print(f"Extracted {len(transactions)} transactions")
            
            # Remove duplicates based on date and amount
            unique_transactions = []
            seen = set()
            
            for tx in transactions:
                # Create a key using date and amount (rounded to 2 decimals)
                key = (tx['date'], round(tx['amount'], 2))
                if key not in seen:
                    seen.add(key)
                    unique_transactions.append(tx)
            
            print(f"After removing duplicates: {len(unique_transactions)} unique transactions")
            return unique_transactions

        except Exception as e:
            self.logger.error(f"Error extracting transactions from text: {e}")
            print(f"Error extracting transactions from text: {e}")
            traceback.print_exc()
            return []

    def _parse_transaction(self, date_str: str, description: str, amount_str: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single transaction from its components.

        Args:
            date_str: Transaction date as string
            description: Transaction description
            amount_str: Transaction amount as string

        Returns:
            Parsed transaction as a dictionary
        """
        try:
            # Parse date
            date = datetime.strptime(date_str, '%d.%m.%Y').strftime('%Y-%m-%d')

            # Parse amount
            amount = self._parse_amount(amount_str)

            # Determine transaction type and category
            transaction_type = self._determine_transaction_type(description)
            category = self._categorize_transaction(description)

            # Determine if it's income or expense
            is_income = amount > 0

            # Extract merchant name if possible
            merchant = self._extract_merchant_name(description)

            return {
                'date': date,
                'description': description,
                'amount': abs(amount),  # Store absolute amount
                'is_income': is_income,
                'type': transaction_type,
                'category': category,
                'merchant': merchant,
                'raw_description': description,
            }

        except Exception as e:
            self.logger.error(f"Error parsing transaction: {e}")
            print(f"Error parsing transaction: {e}")
            traceback.print_exc()
            return None

    def _parse_amount(self, amount_str: str) -> float:
        """
        Parse a German-formatted amount string to a float.

        Args:
            amount_str: Amount as string (e.g. "1.234,56")

        Returns:
            Amount as a float
        """
        try:
            # Remove any leading +/- signs
            cleaned_str = amount_str.strip()
            sign = -1 if cleaned_str.startswith('-') else 1
            cleaned_str = cleaned_str.lstrip('+-')

            # Replace German number formatting
            cleaned_str = cleaned_str.replace('.', '')  # Remove thousands separator
            cleaned_str = cleaned_str.replace(',', '.')  # Replace decimal comma with point

            return sign * float(cleaned_str)
        except Exception as e:
            self.logger.error(f"Error parsing amount '{amount_str}': {e}")
            print(f"Error parsing amount '{amount_str}': {e}")
            raise

    def _determine_transaction_type(self, description: str) -> str:
        """
        Determine the transaction type from its description.

        Args:
            description: Transaction description

        Returns:
            Translated transaction type
        """
        for german_type, english_type in self.TRANSACTION_TYPES.items():
            if german_type in description:
                return english_type

        # Default type if not recognized
        return 'Other'

    def _categorize_transaction(self, description: str) -> str:
        """
        Categorize the transaction based on its description.

        Args:
            description: Transaction description

        Returns:
            Transaction category
        """
        description_upper = description.upper()
        
        # Check if this is a savings balance entry
        if "KONTOSTAND AM" in description_upper:
            return 'Savings'
        
        # Check if this is an initial balance entry or carry-forward
        if "ÜBERTRAG" in description_upper or "UEBERTRAG" in description_upper:
            return 'Initial Balance'
        
        # Special case for Picnic salary/income
        if "PICNIC" in description_upper and ("LOHN" in description_upper or 
                                             "SALARY" in description_upper or 
                                             "GEHALT" in description_upper):
            return 'Salary'
            
        # Alternative: If any Picnic transaction is a credit/incoming payment, assume it's salary
        if "PICNIC" in description_upper and "GUTSCHR" in description_upper:
            return 'Salary'
        
        # Special case for online payment services (from friends)
        if ("PAYPAL" in description_upper or "GIROPAY" in description_upper) and "GUTSCHR" in description_upper:
            return 'Income/Transfers'
        
        # Special case for Studierendenwerk rent payments
        # Only exact matches for Studierendenwerk and related terms should be classified as Housing
        studierendenwerk_terms = [
            "STUDIERENDENWERK", "STW MÜNSTER", "STW MUNSTER", 
            "STUDIERENDENWERK MÜNSTER", "STUDENTENWERK"
        ]
        
        for term in studierendenwerk_terms:
            if term in description_upper:
                # Make sure it's actually a rent payment by checking for specific terms
                if (("MIETE" in description_upper) or 
                    ("RENT" in description_upper) or 
                    ("WOHNHEIM" in description_upper) or 
                    ("LEASING" in description_upper)):
                    return 'Housing'
               
        # Special case: Transaction contains an exact amount of €346 (typical student housing cost)
        # and is a bank transfer or direct debit
        if ("346,00" in description or "346.00" in description or 
            "345,00" in description or "347,00" in description):
            
            # Only if it's specifically for housing-related purposes
            housing_terms = ["MIETE", "RENT", "WOHNUNG", "FLAT", "ZIMMER", "ROOM", 
                           "KAUTION", "DEPOSIT", "WOHNHEIM", "DORMITORY"]
            
            for term in housing_terms:
                if term in description_upper:
                    return 'Housing'
                
        # First identify the transaction type
        transaction_type = None
        
        # Identify bank transfers and direct debits
        if "LASTSCHRIFT" in description_upper:
            transaction_type = "direct_debit"
        elif "ÜBERWEISUNG" in description_upper or "UEBERWEISUNG" in description_upper:
            transaction_type = "transfer"
        elif "GUTSCHR" in description_upper:
            transaction_type = "credit"
        elif "DAUERAUFTRAG" in description_upper:
            transaction_type = "standing_order"
        elif "KARTENZAHLUNG" in description_upper:
            transaction_type = "card_payment"
        
        # Card payments with location markers
        if transaction_type == "card_payment":
            # Food and grocery stores
            food_markers = ["REWE", "EDEKA", "LIDL", "ALDI", "KAUFLAND", "NETTO", "PENNY", 
                           "RESTAURANT", "BACKEREI", "BÄCKEREI", "CAFE", "CAFÉ", "PIZZERIA",
                           "MENSA", "KANTINE", "SUPERMARKT", "LEBENSMITTEL"]
            
            for marker in food_markers:
                if marker in description_upper:
                    return 'Food'
            
            # Transportation
            transport_markers = ["TANKSTELLE", "ARAL", "SHELL", "ESSO", "JET", "TOTAL", 
                                "DB ", "BAHN", "BUS", "TICKET", "DEUTSCHE BAHN", "FLIXBUS",
                                "TAXI", "UBER", "BENZIN", "FAHRKARTE"]
            
            for marker in transport_markers:
                if marker in description_upper:
                    return 'Transportation'
            
            # Shopping
            shopping_markers = ["AMAZON", "ZALANDO", "H&M", "SATURN", "MEDIAMARKT", "DM", 
                               "ROSSMANN", "MÜLLER", "MUELLER", "IKEA", "KAUFHAUS",
                               "GALERIA", "KARSTADT", "ZARA", "PRIMARK", "DECATHLON"]
            
            for marker in shopping_markers:
                if marker in description_upper:
                    return 'Shopping'
            
            # Entertainment
            entertainment_markers = ["KINO", "CINEMA", "MOVIE", "THEATER", "KONZERT", "CONCERT",
                                    "CLUB", "BAR", "PUB", "DISCO", "EVENT", "TICKET", "NETFLIX",
                                    "SPOTIFY", "STEAM", "PLAYSTATION", "NINTENDO", "XBOX"]
            
            for marker in entertainment_markers:
                if marker in description_upper:
                    return 'Entertainment'
            
            # Education
            education_markers = ["BUCHHANDLUNG", "BIBLIOTHEK", "LIBRARY", "UNI ", "UNIVERSITÄT", 
                                "SCHULE", "SCHOOL", "LEHRBUCH", "TEXTBOOK", "SEMINAR", "KURS", 
                                "COURSE", "BILDUNG", "EDUCATION"]
            
            for marker in education_markers:
                if marker in description_upper:
                    return 'Education'
        
        # Special handling for bank transfers and standing orders
        if transaction_type in ["transfer", "standing_order", "direct_debit"]:
            # Check categories based on keywords in description
            
            # Housing related transfers
            housing_terms = ["MIETE", "RENT", "WOHNUNG", "WOHNHEIM", "STUDENTENWOHNHEIM", 
                           "APARTMENT", "ZIMMER", "ROOM", "KAUTION", "DEPOSIT", "IMMOBILIEN"]
            
            for term in housing_terms:
                if term in description_upper:
                    return 'Housing'
            
            # Insurance
            insurance_terms = ["VERSICHERUNG", "INSURANCE", "ALLIANZ", "AXA", "GETSAFE", 
                             "HAFTPFLICHT", "KRANKENVERSICHERUNG", "TK", "AOK", "BARMER"]
            
            for term in insurance_terms:
                if term in description_upper:
                    return 'Insurance'
            
            # Subscriptions
            subscription_terms = ["NETFLIX", "SPOTIFY", "AMAZON PRIME", "DISNEY", "YOUTUBE",
                                "ABO", "ABONNEMENT", "SUBSCRIPTION", "MITGLIEDSCHAFT",
                                "MEMBERSHIP", "APPLE", "GOOGLE"]
            
            for term in subscription_terms:
                if term in description_upper:
                    return 'Subscriptions'
        
        # Standard pattern matching as a fallback
        for category, patterns in self.MERCHANT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, description, re.IGNORECASE):
                    return category
        
        # Special case: Transactions without match but with known amounts
        # e.g., rent is typically a fixed amount each month
        amount_match = re.search(r'([-+]?\d{1,3}(?:\.\d{3})*,\d{2})', description)
        if amount_match:
            amount_str = amount_match.group(1)
            try:
                amount = self._parse_amount(amount_str)
                # If very large amounts (>1000€), likely transfers/deposits which are uncategorized
                if abs(amount) > 1000:
                    return 'Banking'
            except Exception:
                pass
        
        # Default category if not recognized
        return 'Uncategorized'

    def _extract_merchant_name(self, description: str) -> str:
        """
        Extract the merchant name from the transaction description.

        Args:
            description: Transaction description

        Returns:
            Extracted merchant name or empty string
        """
        # Common patterns in Sparkasse statements
        patterns = [
            r'([A-Za-z0-9\.\s]+?)(?:/|//|\.\.|$)',  # Match text before // or ..
            r'([A-Za-z0-9\.\s]+?)\s+(?:GmbH|AG|KG|e\.V\.)',  # Match company names
        ]

        for pattern in patterns:
            match = re.search(pattern, description)
            if match:
                merchant = match.group(1).strip()
                if merchant:
                    return merchant

        # If no pattern matches, return a portion of the description
        parts = description.split()
        if len(parts) >= 2:
            return ' '.join(parts[:2])
        elif len(parts) == 1:
            return parts[0]
        else:
            return ''

    def transactions_to_dataframe(self, transactions: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert a list of transactions to a pandas DataFrame.

        Args:
            transactions: List of transaction dictionaries

        Returns:
            DataFrame with transaction data
        """
        return pd.DataFrame(transactions)

    def export_to_csv(self, transactions: List[Dict[str, Any]], output_path: str) -> None:
        """
        Export parsed transactions to a CSV file.

        Args:
            transactions: List of transaction dictionaries
            output_path: Path where to save the CSV file
        """
        df = self.transactions_to_dataframe(transactions)
        df.to_csv(output_path, index=False)


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    parser = SparkasseParser()

    # Test with a PDF file if one is provided as argument
    import sys

    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        print(f"Processing PDF file: {pdf_path}")
        transactions = parser.parse_pdf(pdf_path)

        print(f"Found {len(transactions)} transactions")
        for i, tx in enumerate(transactions[:5]):  # Print first 5 for sample
            print(f"{i + 1}. {tx['date']} | {tx['merchant']} | {tx['amount']:.2f} € | {tx['category']}")

        if len(transactions) > 0:
            csv_path = pdf_path.replace('.pdf', '.csv')
            parser.export_to_csv(transactions, csv_path)
            print(f"Exported transactions to {csv_path}")