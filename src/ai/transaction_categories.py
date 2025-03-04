#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transaction Categories

This module provides advanced AI and NLP functionality for categorizing financial transactions
based on their descriptions, merchants, and other attributes.
"""

import re
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from collections import defaultdict

# Set up logging
logger = logging.getLogger(__name__)

# Advanced NLP and ML capabilities with fallbacks
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
    # Download required NLTK data if not already downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
        
    NLP_AVAILABLE = True
    logger.info("NLTK components successfully loaded")
    
except ImportError:
    NLP_AVAILABLE = False
    logger.warning("NLTK libraries not available. Falling back to rule-based categorization.")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestClassifier
    
    ML_AVAILABLE = True
    logger.info("Scikit-learn machine learning components successfully loaded")
except ImportError:
    ML_AVAILABLE = False
    logger.warning("Scikit-learn not available. Falling back to rule-based categorization.")


class TransactionCategorizer:
    """
    Advanced AI-powered categorizer for financial transactions using NLP and ML techniques.
    Features:
    - NLP-based text processing for improved understanding of transaction descriptions
    - Machine learning classification for accurate categorization
    - Sentiment analysis to determine necessary vs. discretionary spending
    - Anomaly detection to identify unusual transactions
    - Self-improving through active learning from user corrections
    """

    # Default category definitions with pattern matching rules
    DEFAULT_CATEGORIES = {
        'Housing': [
            r'(?i)(miete|rent|studierendenwerk|studentenwerk|stw|wohnung|apartment|housing|accommodation|nebenkosten|utilities)',
            r'(?i)(elec|strom|gas|water|wasser|internet|wifi|broadband)',
            r'(?i)(hausmeister|caretaker|building|maintenance|hausverwaltung|property)',
            r'(?i)(mietkaution|deposit|security|damage|insurance|hausrat)',
            r'(?i)(wohn|living|home|furniture|möbel|domestic|haushalt)',
        ],

        'Food': [
            r'(?i)(rewe|edeka|lidl|aldi|netto|kaufland|supermarkt|grocery|lebensmittel)',
            r'(?i)(restaurant|cafe|coffee|kaffee|imbiss|food|essen|lieferando|pizza|burger|mcdonald|subway)',
            r'(?i)(uber\s*eats|grubhub|doordash|wolt|deliveroo|takeaway|foodora)',
            r'(?i)(backerei|bakery|konditorei|patisserie|kiosk)',
            r'(?i)(kfc|kentucky|burger king|bk\s+\d+|ditsch|le crobag)',
            r'(?i)(starbucks|dunkin|espresso|latte|brunch|breakfast|dinner|lunch)',
            r'(?i)(sushi|pasta|cuisine|bistro|grill|bar|lounge|pub|restaurant)',
            r'(?i)(markt|market|frisch|fresh|organic|bio|getränk|drink|beverage)',
        ],

        'Transportation': [
            r'(?i)(db|deutsche bahn|flixbus|blablacar|bus|train|zug|ticket|fahrkarte)',
            r'(?i)(uber|bolt|taxi|cab|ride|fahrt|lyft|sharing)',
            r'(?i)(tankstelle|fuel|gas station|parking|parken|parkplatz)',
            r'(?i)(auto|car|vehicle|service|repair|maintenance|inspection)',
            r'(?i)(fahrrad|bike|bicycle|cycling|rad|ebike|rental)',
            r'(?i)(sbahn|ubahn|tram|metro|subway|underground|public transport)',
            r'(?i)(fahrzeug|vehicle|reifen|tire|werkstatt|garage|inspektion)',
            r'(?i)(fahrschein|ticket|fahrkarte|bahn|mvv|mvg|bvg|hvv)',
        ],

        'Telecommunications': [
            r'(?i)(telefonica|o2|telekom|vodafone|phone|handy|mobile|sim|prepaid)',
            r'(?i)(internet|dsl|fiber|glasfaser|broadband|router|modem)',
            r'(?i)(rebtel|call|international|roaming)',
        ],

        'Subscriptions': [
            r'(?i)(netflix|prime|amazon|disney|hulu|hbo|paramount|streaming|youtube)',
            r'(?i)(spotify|apple|itunes|music|audio|podcast|audible)',
            r'(?i)(newspaper|magazine|zeit|spiegel|focus|subscription|abo)',
            r'(?i)(adobe|office|microsoft|google|cloud|storage|software)',
            r'(?i)(apple\.com\.bill|apple\.com|itunes\.com)',
            r'(?i)(klarna|eurobill|openai|chatgpt|expressvpn|vpn)',
        ],

        'Shopping': [
            r'(?i)(amazon|ebay|otto|zalando|zara|h&m|primark|karstadt|galeria)',
            r'(?i)(mediamarkt|saturn|electronic|computer|laptop|phone|smartphone)',
            r'(?i)(ikea|moebel|furniture|home|bauhaus|baumarkt|hardware)',
            r'(?i)(dm|rossmann|drogerie|drugstore|cosmetic|beauty|pflege)',
            r'(?i)(clothing|kleidung|shoe|schuh|accessories|accessoire)',
            r'(?i)(book|buch|thalia|hugendubel|mayersche)',
        ],

        'Health': [
            r'(?i)(krankenversicherung|health insurance|techniker|tk-|barmer|aok|dak)',
            r'(?i)(doctor|arzt|hospital|krankenhaus|emergency|notfall|ambulance)',
            r'(?i)(pharmacy|apotheke|medicine|medication|prescription|rezept)',
            r'(?i)(dentist|zahnarzt|optician|optiker|physiotherapy|massage)',
            r'(?i)(fitness|gym|sport|exercise|workout|training)',
            r'(?i)(mcfit|clever fit|fitness first|rsg group)',
        ],

        'Entertainment': [
            r'(?i)(cinema|kino|movie|film|concert|konzert|festival|theater|theatre)',
            r'(?i)(club|disco|bar|pub|lounge|event|party|veranstaltung)',
            r'(?i)(museum|exhibition|ausstellung|gallery|galerie|zoo|park)',
            r'(?i)(game|spiel|playstation|xbox|nintendo|steam|epic|gaming)',
            r'(?i)(hobby|craft|kunst|art|musical|show|performance)',
        ],

        'Education': [
            r'(?i)(university|universität|uni|college|school|schule|bildung|education)',
            r'(?i)(course|kurs|seminar|workshop|training|class|lesson|unterricht)',
            r'(?i)(book|buch|ebook|textbook|lehrbuch|material|stationery)',
            r'(?i)(language|sprache|certificate|zertifikat|exam|prüfung)',
            r'(?i)(coursera|udemy|edx|skillshare|blossomup|masterclass|linkedin learning)',
        ],

        'Financial': [
            r'(?i)(bank|sparkasse|fee|gebühr|interest|zinsen|charge|tax|steuer)',
            r'(?i)(insurance|versicherung|getsafe|allianz|ergo|axa|huk)',
            r'(?i)(investment|investition|etf|fund|fonds|stock|aktie|depot)',
            r'(?i)(loan|kredit|mortgage|hypothek|finance|financing|leasing)',
        ],

        'Donations': [
            r'(?i)(donation|spende|charity|wohltätigkeit|organization|organisation)',
            r'(?i)(plan international|unicef|wwf|greenpeace|amnesty|médecins|ärzte)',
        ],

        'Income': [
            r'(?i)(salary|gehalt|lohn|wage|income|einkommen|payroll)',
            r'(?i)(bonus|award|prize|prämie|commission|provision)',
            r'(?i)(refund|erstattung|reimbursement|rückerstattung|return)',
            r'(?i)(dividend|dividende|interest|zins|yield|rendite)',
            r'(?i)(grant|stipend|scholarship|bafög|student finance)',
            r'(?i)(gift|geschenk|present|transfer|überweisung|received)',
            r'(?i)(picnic|ila solution|gmbh lohn)',
        ],

        'Travel': [
            r'(?i)(hotel|accommodation|unterkunft|airbnb|booking|hostel)',
            r'(?i)(flight|flug|airline|lufthansa|easyjet|ryanair|eurowings)',
            r'(?i)(rail|railway|bahn|train|zug|ticket|fahrkarte)',
            r'(?i)(travel|reise|tour|trip|vacation|urlaub|holiday|ferien)',
            r'(?i)(rental|mietwagen|car hire|taxi|uber|transport)',
            r'(?i)(suitcase|koffer|luggage|gepäck|accessory|zubehör)',
            r'(?i)(bounce|usebounce)',
        ],
    }

    def __init__(self, custom_categories: Optional[Dict[str, List[str]]] = None):
        """
        Initialize the categorizer with default or custom categories.

        Args:
            custom_categories: Optional custom category patterns to use
                               instead of the defaults
        """
        self.categories = custom_categories if custom_categories else self.DEFAULT_CATEGORIES

        # Compile regex patterns for efficiency
        self.compiled_patterns = {}
        for category, patterns in self.categories.items():
            self.compiled_patterns[category] = [re.compile(p) for p in patterns]
            
        # Initialize NLP components if available
        self.nlp_enabled = NLP_AVAILABLE
        if self.nlp_enabled:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            logger.info("NLP preprocessing components initialized")
            
        # Initialize ML model if available
        self.ml_enabled = ML_AVAILABLE
        self.ml_model = None
        self.vectorizer = None
        self.classifier = None
        self.training_data = []
        self.training_labels = []
        self.confidence_threshold = 0.65  # Confidence threshold for ML predictions
        
        if self.ml_enabled:
            self._initialize_ml_model()
            logger.info("Machine learning model initialized")
            
        # Anomaly detection
        self.transaction_history = []
        self.anomaly_detector = None
        if self.ml_enabled:
            self._initialize_anomaly_detector()
            
        # Sentiment analysis for spending classification
        self.necessary_keywords = [
            'rent', 'mortgage', 'electric', 'utilities', 'grocery', 
            'insurance', 'medicine', 'doctor', 'healthcare', 'transport',
            'food', 'education', 'tuition', 'loan', 'tax', 'heat', 'water'
        ]
        
        self.discretionary_keywords = [
            'restaurant', 'entertainment', 'shopping', 'clothing', 'vacation', 
            'travel', 'luxury', 'game', 'hobby', 'subscription', 'streaming',
            'bar', 'alcohol', 'gift', 'donation', 'beauty', 'jewelry'
        ]
        
    def _initialize_ml_model(self):
        """Initialize the machine learning pipeline for transaction categorization."""
        if not self.ml_enabled:
            return
            
        try:
            # Create TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),  # Include both unigrams and bigrams
                stop_words='english'
            )
            
            # Use Random Forest as the main classifier
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42
            )
            
            # Create the pipeline
            self.ml_model = Pipeline([
                ('vectorizer', self.vectorizer),
                ('classifier', self.classifier)
            ])
            
            # Pre-train with category patterns
            self._pretrain_with_patterns()
            
        except Exception as e:
            logger.error(f"Error initializing ML model: {e}")
            self.ml_enabled = False
            
    def _initialize_anomaly_detector(self):
        """Initialize the anomaly detection model."""
        if not self.ml_enabled:
            return
            
        try:
            # Using KMeans for anomaly detection
            self.anomaly_detector = KMeans(n_clusters=10, random_state=42)
        except Exception as e:
            logger.error(f"Error initializing anomaly detector: {e}")
            
    def _pretrain_with_patterns(self):
        """Pre-train the ML model with synthetic data from patterns."""
        if not self.ml_enabled:
            return
            
        training_data = []
        labels = []
        
        # Generate training examples from patterns
        for category, patterns in self.categories.items():
            # Extract keywords from patterns
            for pattern in patterns:
                # Extract likely words from the pattern
                words = re.findall(r'\(\?i\)\(([^)]+)\)', pattern)
                if words:
                    keywords = []
                    for word_group in words:
                        keywords.extend(word_group.split('|'))
                    
                    # Create synthetic transaction descriptions
                    for keyword in keywords:
                        if len(keyword) < 3 or '\\' in keyword:
                            continue  # Skip very short keywords or regex special chars
                            
                        examples = [
                            f"Payment to {keyword.upper()}",
                            f"Purchase at {keyword.title()}",
                            f"{keyword.upper()} transaction",
                            f"Payment for {keyword}",
                            f"{keyword.title()}"
                        ]
                        
                        for example in examples:
                            training_data.append(example)
                            labels.append(category)
        
        if training_data:
            # Fit the vectorizer
            X = self.vectorizer.fit_transform(training_data)
            # Train the classifier
            self.classifier.fit(X, labels)
            
            # Save the training data for future reference
            self.training_data = training_data
            self.training_labels = labels
            
            logger.info(f"ML model pre-trained with {len(training_data)} synthetic examples")
        else:
            logger.warning("No training data could be generated from patterns")

    def _preprocess_text(self, text: str) -> str:
        """
        Apply NLP preprocessing to transaction text.
        
        Args:
            text: Raw text string
            
        Returns:
            Preprocessed text
        """
        if not self.nlp_enabled:
            return text.lower()
        
        try:
            # Tokenize
            tokens = word_tokenize(text.lower())
            
            # Remove stopwords and lemmatize
            processed_tokens = [
                self.lemmatizer.lemmatize(token)
                for token in tokens
                if token not in self.stop_words and token.isalpha()
            ]
            
            # Join back into a string
            processed_text = " ".join(processed_tokens)
            return processed_text
            
        except Exception as e:
            logger.warning(f"Error in text preprocessing: {e}. Using basic preprocessing.")
            return text.lower()
    
    def _predict_with_ml(self, text_to_check: str) -> Tuple[str, float]:
        """
        Predict category using ML model.
        
        Args:
            text_to_check: Preprocessed text
            
        Returns:
            Tuple of (predicted_category, confidence_score)
        """
        if not self.ml_enabled or not self.ml_model:
            return None, 0.0
            
        try:
            # Make prediction
            # Transform the text using the vectorizer
            X = self.vectorizer.transform([text_to_check])
            
            # Get predicted probabilities
            probabilities = self.classifier.predict_proba(X)[0]
            
            # Get the highest probability and its index
            max_prob = max(probabilities)
            max_idx = list(probabilities).index(max_prob)
            
            # Get the predicted class
            predicted_category = self.classifier.classes_[max_idx]
            
            return predicted_category, max_prob
            
        except Exception as e:
            logger.warning(f"Error in ML prediction: {e}")
            return None, 0.0
            
    def analyze_spending_type(self, text: str) -> str:
        """
        Analyze if a transaction is necessary or discretionary spending.
        
        Args:
            text: Transaction description
            
        Returns:
            'NECESSARY', 'DISCRETIONARY', or 'NEUTRAL'
        """
        if not self.nlp_enabled:
            return 'NEUTRAL'
            
        try:
            # Preprocess the text
            processed_text = self._preprocess_text(text)
            tokens = processed_text.split()
            
            # Count matches for necessary and discretionary keywords
            necessary_count = sum(1 for token in tokens if token in self.necessary_keywords)
            discretionary_count = sum(1 for token in tokens if token in self.discretionary_keywords)
            
            if necessary_count > discretionary_count:
                return 'NECESSARY'
            elif discretionary_count > necessary_count:
                return 'DISCRETIONARY'
            else:
                return 'NEUTRAL'
                
        except Exception as e:
            logger.warning(f"Error in spending analysis: {e}")
            return 'NEUTRAL'
    
    def detect_anomalies(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect anomalous transactions using ML.
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            List of transactions flagged as anomalous
        """
        if not self.ml_enabled or not self.anomaly_detector or len(transactions) < 10:
            # Basic anomaly detection as fallback
            return self._basic_anomaly_detection(transactions)
            
        try:
            # Extract features: amount and day of week
            features = []
            for tx in transactions:
                amount = tx.get('amount', 0)
                date = pd.to_datetime(tx.get('date', pd.Timestamp.now()))
                day_of_week = date.dayofweek
                hour = date.hour if hasattr(date, 'hour') else 12
                features.append([amount, day_of_week, hour])
                
            # Convert to numpy array
            features = np.array(features)
            
            # Fit and predict
            self.anomaly_detector.fit(features)
            distances = self.anomaly_detector.transform(features)
            
            # Flag anomalies (transactions far from their cluster center)
            threshold = np.percentile(distances.min(axis=1), 95)  # Top 5% as anomalies
            anomaly_indices = np.where(distances.min(axis=1) > threshold)[0]
            
            # Extract anomalous transactions
            anomalies = [transactions[i] for i in anomaly_indices]
            
            return anomalies
            
        except Exception as e:
            logger.warning(f"Error in ML anomaly detection: {e}. Using basic detection.")
            return self._basic_anomaly_detection(transactions)
    
    def _basic_anomaly_detection(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Basic rule-based anomaly detection as fallback."""
        anomalies = []
        
        # Group by category
        category_txs = defaultdict(list)
        for tx in transactions:
            category = tx.get('category', 'Other')
            category_txs[category].append(tx)
            
        # For each category, find outliers by amount
        for category, txs in category_txs.items():
            if len(txs) < 5:  # Skip categories with too few transactions
                continue
                
            amounts = [tx.get('amount', 0) for tx in txs]
            mean = sum(amounts) / len(amounts)
            std_dev = (sum((x - mean) ** 2 for x in amounts) / len(amounts)) ** 0.5
            
            # Flag transactions more than 2 standard deviations from the mean
            for tx in txs:
                amount = tx.get('amount', 0)
                if abs(amount - mean) > 2 * std_dev:
                    tx['anomaly_reason'] = f"Amount (€{amount}) significantly different from average (€{mean:.2f})"
                    anomalies.append(tx)
                    
        return anomalies
            
    def categorize_transaction(self, transaction: Dict[str, Any]) -> str:
        """
        Categorize a single transaction with advanced AI techniques.

        Args:
            transaction: Transaction dictionary

        Returns:
            Category name as string
        """
        # Check if a category is already assigned
        if 'category' in transaction and transaction['category']:
            # If it's a valid category, keep it
            if transaction['category'] in self.categories:
                return transaction['category']

        # Get description and other fields to check
        description = transaction.get('description', '')
        merchant = transaction.get('merchant', '')
        transaction_type = transaction.get('type', '')
        amount = transaction.get('amount', 0)

        # Text to check for patterns
        text_to_check = f"{description} {merchant} {transaction_type}"
        
        # Save to transaction history for future anomaly detection
        if hasattr(self, 'transaction_history'):
            self.transaction_history.append(transaction.copy())
            # Keep history at a reasonable size
            if len(self.transaction_history) > 1000:
                self.transaction_history = self.transaction_history[-1000:]

        # Special case for income
        if transaction.get('is_income', False):
            for pattern in self.compiled_patterns.get('Income', []):
                if pattern.search(text_to_check.lower()):
                    return 'Income'
            # Default income category if no specific match
            return 'Income'
        
        # Try ML-based categorization if available
        if self.ml_enabled and self.ml_model:
            # Preprocess text for ML
            preprocessed_text = self._preprocess_text(text_to_check)
            
            # Get ML prediction and confidence
            ml_category, confidence = self._predict_with_ml(preprocessed_text)
            
            # If confidence is high enough, use ML prediction
            if ml_category and confidence >= self.confidence_threshold:
                logger.debug(f"ML categorization: '{text_to_check}' → {ml_category} (confidence: {confidence:.2f})")
                # Add to training data for future improvement
                if preprocessed_text not in self.training_data:
                    self.training_data.append(preprocessed_text)
                    self.training_labels.append(ml_category)
                return ml_category
            else:
                logger.debug(f"ML confidence too low ({confidence:.2f}), using rule-based approach")
            
        # Advanced rule-based categorization as fallback
        text_to_check = text_to_check.lower()
        
        # 1. Utility payments tend to be regular and fixed amounts
        if re.search(r'(?i)(rechnung|abrechnung|invoice|bill|payment)', text_to_check):
            # Check for telecommunications terms
            if re.search(r'(?i)(phone|mobile|handy|telefon|internet|dsl|wifi)', text_to_check):
                return 'Telecommunications'
            # Check for utility terms
            elif re.search(r'(?i)(strom|power|gas|wasser|water|heizung|heating)', text_to_check):
                return 'Housing'

        # 2. Transportation patterns
        if re.search(r'(?i)(ticket|fare|fahrkarte|bahn|bus|train|ride|fahrt)', text_to_check):
            return 'Transportation'
            
        # 3. Financial services
        if re.search(r'(?i)(versicherung|insurance|bank|konto|account|credit|kredit|debit)', text_to_check):
            return 'Financial'
            
        # 4. Food delivery services
        if re.search(r'(?i)(lieferd|deliver|bestell|order|takeaway)', text_to_check):
            return 'Food'
            
        # 5. Common payment patterns for quick identification
        if re.search(r'(?i)(amazon)', text_to_check):
            # Amazon could be various categories, check for clues
            if re.search(r'(?i)(prime|video|music|subscription)', text_to_check):
                return 'Subscriptions'
            else:
                return 'Shopping'
                
        if re.search(r'(?i)(paypal|venmo|transferwise|wise)', text_to_check):
            # These are payment methods, not categories
            # Try to look deeper at the description
            if re.search(r'(?i)(shop|store|retail|boutique|market)', text_to_check):
                return 'Shopping'
            elif re.search(r'(?i)(eat|food|restaurant|cafe|bar|drink)', text_to_check):
                return 'Food'
                
        # Check each category's patterns (standard approach)
        match_scores = {}
        for category, patterns in self.compiled_patterns.items():
            category_score = 0
            for pattern in patterns:
                matches = pattern.findall(text_to_check)
                category_score += len(matches)
                
                # Increase score for more specific matches
                if matches and len(text_to_check) < 50:
                    category_score += 1  # Shorter descriptions with matches are more reliable
                    
            if category_score > 0:
                match_scores[category] = category_score
                
        # Find category with highest match score if any
        if match_scores:
            best_category = max(match_scores.items(), key=lambda x: x[1])[0]
            
            # Learn from this categorization if ML is enabled
            if self.ml_enabled and preprocessed_text not in self.training_data:
                self.training_data.append(preprocessed_text)
                self.training_labels.append(best_category)
                
            return best_category

        # Default category if no match found
        return 'Other'

    def categorize_transactions(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Categorize a list of transactions using AI techniques.

        Args:
            transactions: List of transaction dictionaries

        Returns:
            List of transactions with categories and insights assigned
        """
        categorized = []
        
        # First pass: basic categorization
        for tx in transactions:
            tx_copy = tx.copy()
            # Only categorize if no valid category is already assigned
            if 'category' not in tx_copy or tx_copy['category'] not in self.categories:
                tx_copy['category'] = self.categorize_transaction(tx_copy)
                
            # Add spending type analysis
            if not tx_copy.get('is_income', False):
                description = f"{tx_copy.get('description', '')} {tx_copy.get('merchant', '')}"
                tx_copy['spending_type'] = self.analyze_spending_type(description)
            
            categorized.append(tx_copy)
            
        # Second pass: enhanced analysis with full context
        if len(categorized) > 5:  # Only do advanced analysis if we have enough data
            # Detect anomalies
            anomalies = self.detect_anomalies(categorized)
            
            # Mark anomalous transactions
            anomaly_ids = {tx.get('id') for tx in anomalies if 'id' in tx}
            for tx in categorized:
                if tx.get('id') in anomaly_ids:
                    tx['is_anomaly'] = True
                    # Find matching anomaly to get the reason
                    for anomaly in anomalies:
                        if anomaly.get('id') == tx.get('id'):
                            tx['anomaly_reason'] = anomaly.get('anomaly_reason', 'Unusual transaction')
                            break
                else:
                    tx['is_anomaly'] = False
            
            # Train ML model with new data if enabled
            if self.ml_enabled and self.ml_model and len(self.training_data) > len(categorized):
                try:
                    X = self.vectorizer.transform(self.training_data)
                    self.classifier.fit(X, self.training_labels)
                    logger.info(f"ML model updated with {len(self.training_data)} examples")
                except Exception as e:
                    logger.warning(f"Error updating ML model: {e}")
                    
        return categorized

    def get_category_summary(self, transactions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Get a summary of spending by category.

        Args:
            transactions: List of categorized transaction dictionaries

        Returns:
            Dictionary mapping categories to total amounts
        """
        category_totals = defaultdict(float)

        for tx in transactions:
            category = tx.get('category', 'Other')
            amount = tx.get('amount', 0)
            is_income = tx.get('is_income', False)

            # Skip income for spending summary
            if not is_income:
                category_totals[category] += amount

        return dict(category_totals)

    def get_monthly_category_summary(self, transactions: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Get a monthly summary of spending by category.

        Args:
            transactions: List of categorized transaction dictionaries

        Returns:
            DataFrame with categories as rows and months as columns
        """
        # Ensure transactions have date and are categorized
        for tx in transactions:
            if 'category' not in tx:
                tx['category'] = self.categorize_transaction(tx)

        # Create DataFrame
        df = pd.DataFrame(transactions)

        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Extract month and year
        df['month_year'] = df['date'].dt.strftime('%Y-%m')

        # Filter out income transactions
        expense_df = df[~df['is_income']]

        # Group by month, year, and category
        summary = expense_df.pivot_table(
            index='category',
            columns='month_year',
            values='amount',
            aggfunc='sum',
            fill_value=0
        )

        return summary

    def _save_training_data(self) -> bool:
        """Save training data to persistent storage."""
        # Skip if we can't create required tables
        try:
            if not hasattr(self, 'db_manager') or self.db_manager is None:
                # Try to import database_manager module
                try:
                    from ..database.database_manager import DatabaseManager
                    self.db_manager = DatabaseManager()
                except ImportError:
                    logger.warning("Could not import DatabaseManager for training data persistence")
                    return False
            
            # Check if tables exist, create them if not
            self.db_manager.execute_query("""
            CREATE TABLE IF NOT EXISTS category_classifier_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trained_date TEXT,
                version INTEGER,
                metrics TEXT
            )
            """)
            
            self.db_manager.execute_query("""
            CREATE TABLE IF NOT EXISTS category_training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_text TEXT,
                original_category TEXT,
                corrected_category TEXT,
                confidence REAL,
                date_added TEXT
            )
            """)
            
            # Set version
            if not hasattr(self, 'model_version'):
                self.model_version = 1
            else:
                self.model_version += 1
            
            # Clear current training data
            self.db_manager.execute_query("DELETE FROM category_training_data")
            
            # Save training data
            added_count = 0
            for text, category in self.category_corrections.items():
                self.db_manager.execute_query("""
                INSERT INTO category_training_data (
                    transaction_text, corrected_category, date_added
                ) VALUES (?, ?, ?)
                """, (text, category, datetime.now().isoformat()))
                added_count += 1
            
            # Save model state
            metrics = {
                'training_examples': len(self.training_data),
                'categories': len(set(self.training_labels)),
                'corrections': len(self.category_corrections)
            }
            
            self.db_manager.execute_query("""
            INSERT INTO category_classifier_state (
                trained_date, version, metrics
            ) VALUES (?, ?, ?)
            """, (datetime.now().isoformat(), self.model_version, str(metrics)))
            
            logger.info(f"Saved training data: {added_count} examples, model version {self.model_version}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving training data: {e}")
            return False
        
    def _load_training_data(self) -> bool:
        """Load training data from persistent storage."""
        try:
            if not hasattr(self, 'db_manager') or self.db_manager is None:
                # Try to import database_manager module
                try:
                    from ..database.database_manager import DatabaseManager
                    self.db_manager = DatabaseManager()
                except ImportError:
                    logger.warning("Could not import DatabaseManager for loading training data")
                    return False
            
            # Check if tables exist
            result = self.db_manager.execute_query("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='category_training_data'
            """)
            
            if not result:
                logger.info("No training data tables found in database")
                return False
            
            # Get latest model version
            model_state = self.db_manager.execute_query("""
            SELECT version FROM category_classifier_state
            ORDER BY trained_date DESC LIMIT 1
            """)
            
            if model_state:
                self.model_version = model_state[0][0]
            
            # Load training data
            training_data = self.db_manager.execute_query("""
            SELECT transaction_text, corrected_category
            FROM category_training_data
            """)
            
            # Process training data
            for text, category in training_data:
                # Add to training data
                if text not in self.training_data:
                    self.training_data.append(text)
                    self.training_labels.append(category)
                
                # Add to corrections
                self.category_corrections[text] = category
            
            if training_data:
                logger.info(f"Loaded {len(training_data)} training examples, model version {self.model_version}")
                return True
            else:
                logger.info("No training data found in database")
                return False
                
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return False
    
    def learn_from_user_correction(self, transaction_text: str, correct_category: str) -> bool:
        """
        Update the model with user-provided category correction.
        
        Args:
            transaction_text: The transaction description
            correct_category: The correct category provided by the user
            
        Returns:
            Success flag
        """
        if not transaction_text or not correct_category:
            return False
            
        try:
            # Preprocess the text
            processed_text = self._preprocess_text(transaction_text)
            
            # Add to our corrections dictionary
            self.category_corrections[processed_text] = correct_category
            
            # Add to training data if not already present
            if processed_text not in self.training_data:
                self.training_data.append(processed_text)
                self.training_labels.append(correct_category)
            else:
                # Update existing label
                idx = self.training_data.index(processed_text)
                self.training_labels[idx] = correct_category
            
            # Retrain the model if we have ML capabilities
            if self.ml_enabled and self.ml_model:
                try:
                    X = self.vectorizer.transform(self.training_data)
                    self.classifier.fit(X, self.training_labels)
                    logger.info(f"Retrained model with {len(self.training_data)} examples")
                except Exception as e:
                    logger.warning(f"Could not retrain model: {e}")
            
            # Save the training data
            self._save_training_data()
            
            logger.info(f"Learned correction: '{transaction_text}' → {correct_category}")
            return True
            
        except Exception as e:
            logger.error(f"Error learning correction: {e}")
            return False
    
    def suggest_category_improvements(self, transactions: List[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
        """
        Analyze transactions to suggest improvements using AI techniques.

        Args:
            transactions: List of transaction dictionaries

        Returns:
            List of tuples (description, current_category, suggested_category)
        """
        suggestions = []

        # Count transactions per merchant
        merchant_counts = defaultdict(int)
        merchant_categories = {}
        merchant_texts = defaultdict(list)

        for tx in transactions:
            merchant = tx.get('merchant', '').lower()
            description = tx.get('description', '').lower()
            if merchant:
                merchant_counts[merchant] += 1
                current_category = tx.get('category', 'Other')
                
                # Store the transaction text for NLP analysis
                merchant_texts[merchant].append(description)

                if merchant in merchant_categories:
                    if merchant_categories[merchant] != current_category:
                        # Merchant has inconsistent categorization
                        suggestions.append((
                            tx.get('description', ''),
                            current_category,
                            merchant_categories[merchant]
                        ))
                else:
                    merchant_categories[merchant] = current_category

        # Find common merchants without specific rules
        for merchant, count in merchant_counts.items():
            if count >= 5:  # Merchant appears frequently
                found_pattern = False
                category = merchant_categories.get(merchant, 'Other')

                if category != 'Other':
                    # Check if any existing pattern matches this merchant
                    for pattern in self.categories.get(category, []):
                        if re.search(pattern, merchant, re.IGNORECASE):
                            found_pattern = True
                            break

                    if not found_pattern:
                        # Suggest adding a pattern for this merchant
                        suggestions.append((
                            f"Frequent merchant: {merchant}",
                            "No pattern match",
                            f"Add pattern to {category}"
                        ))
        
        # Use NLP to identify misclassified transactions if available
        if self.nlp_enabled and len(transactions) > 10:
            # Process transaction descriptions with NLP
            processed_descriptions = []
            categories = []
            transaction_ids = []
            
            for i, tx in enumerate(transactions):
                description = f"{tx.get('description', '')} {tx.get('merchant', '')}"
                if description:
                    processed_descriptions.append(self._preprocess_text(description))
                    categories.append(tx.get('category', 'Other'))
                    transaction_ids.append(i)
            
            # Check for user corrections that could be applied
            if self.category_corrections:
                for i, desc in enumerate(processed_descriptions):
                    if desc in self.category_corrections:
                        correct_category = self.category_corrections[desc]
                        if correct_category != categories[i]:
                            tx_idx = transaction_ids[i]
                            tx = transactions[tx_idx]
                            suggestions.append((
                                f"{tx.get('description', '')} ({tx.get('merchant', '')})",
                                categories[i],
                                f"User previously categorized as {correct_category}"
                            ))
            
            # Find potential misclassifications using clustering or similarity
            if self.ml_enabled and len(processed_descriptions) > 10:
                try:
                    # Convert to TF-IDF features
                    vectorizer = TfidfVectorizer(max_features=100)
                    X = vectorizer.fit_transform(processed_descriptions)
                    
                    # Use K-means to find clusters of similar transactions
                    kmeans = KMeans(n_clusters=min(10, len(processed_descriptions) // 5), random_state=42)
                    clusters = kmeans.fit_predict(X)
                    
                    # Check each cluster for category consistency
                    cluster_categories = defaultdict(list)
                    for i, cluster_id in enumerate(clusters):
                        cluster_categories[cluster_id].append(categories[i])
                    
                    # Find clusters with inconsistent categories
                    for cluster_id, cluster_cats in cluster_categories.items():
                        # If cluster has multiple categories
                        if len(set(cluster_cats)) > 1:
                            # Find the dominant category
                            category_counts = Counter(cluster_cats)
                            dominant_category = category_counts.most_common(1)[0][0]
                            
                            # Find transactions in this cluster with different categories
                            for i, (cluster_i, category) in enumerate(zip(clusters, categories)):
                                if cluster_i == cluster_id and category != dominant_category:
                                    # This transaction might be misclassified
                                    tx_idx = transaction_ids[i]
                                    tx = transactions[tx_idx]
                                    
                                    # Check if we already have a suggestion for this transaction
                                    if not any(sugg[0] == f"{tx.get('description', '')} ({tx.get('merchant', '')})" for sugg in suggestions):
                                        suggestions.append((
                                            f"{tx.get('description', '')} ({tx.get('merchant', '')})",
                                            category,
                                            f"Consider {dominant_category} (similar transactions in this cluster)"
                                        ))
                                    
                except Exception as e:
                    logger.warning(f"Error in clustering analysis: {e}")

        # Use trained model to find misclassifications
        if self.ml_enabled and self.ml_model and len(self.training_data) >= 10:
            try:
                for i, tx in enumerate(transactions):
                    description = f"{tx.get('description', '')} {tx.get('merchant', '')}"
                    if description:
                        processed_text = self._preprocess_text(description)
                        current_category = tx.get('category', 'Other')
                        
                        # Use model to predict
                        ml_category, confidence = self._predict_with_ml(processed_text)
                        
                        # If prediction differs from current category with high confidence
                        if ml_category and ml_category != current_category and confidence >= 0.8:
                            # Check if we already have this suggestion
                            if not any(sugg[0] == description and sugg[2].startswith(f"Consider {ml_category}") for sugg in suggestions):
                                suggestions.append((
                                    description,
                                    current_category,
                                    f"Consider {ml_category} (ML confidence: {confidence:.1%})"
                                ))
            except Exception as e:
                logger.warning(f"Error using ML for category suggestions: {e}")

        return suggestions
        
    def generate_smart_insights(self, transactions: List[Dict[str, Any]]) -> List[Dict]:
        """
        Generate smart insights from transaction data using AI.
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            List of insight dictionaries with type, description, and importance
        """
        insights = []
        
        if not transactions:
            return insights
            
        try:
            # 1. Categorize transactions if not already categorized
            categorized = []
            for tx in transactions:
                if 'category' not in tx or tx['category'] not in self.categories:
                    tx_copy = tx.copy()
                    tx_copy['category'] = self.categorize_transaction(tx_copy)
                    categorized.append(tx_copy)
                else:
                    categorized.append(tx.copy())
                    
            # Create DataFrame for analysis
            df = pd.DataFrame(categorized)
            
            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Extract month and year
            df['month_year'] = df['date'].dt.strftime('%Y-%m')
            df['month'] = df['date'].dt.month
            
            # 2. Identify spending trends
            if len(df['month_year'].unique()) > 1:
                # Calculate spending by category and month
                category_monthly = df.pivot_table(
                    index='category', 
                    columns='month_year',
                    values='amount',
                    aggfunc='sum',
                    fill_value=0
                )
                
                # Get the last two months
                if len(category_monthly.columns) >= 2:
                    last_month = category_monthly.columns[-1]
                    previous_month = category_monthly.columns[-2]
                    
                    # Compare each category
                    for category in category_monthly.index:
                        current = category_monthly.loc[category, last_month]
                        previous = category_monthly.loc[category, previous_month]
                        
                        # Skip categories with very small amounts
                        if current < 10 and previous < 10:
                            continue
                            
                        # Calculate percentage change
                        if previous > 0:
                            change_pct = ((current - previous) / previous) * 100
                            
                            # Significant increase
                            if change_pct > 25 and (current - previous) > 20:
                                insights.append({
                                    'type': 'spending_increase',
                                    'category': category,
                                    'description': f"Spending in {category} increased by {change_pct:.1f}% from {previous:.2f} to {current:.2f}",
                                    'importance': min(abs(change_pct) / 20, 10),  # Scale importance by percentage change
                                    'action': f"Review {category} expenses for opportunities to save"
                                })
                            
                            # Significant decrease
                            elif change_pct < -25 and (previous - current) > 20:
                                insights.append({
                                    'type': 'spending_decrease',
                                    'category': category,
                                    'description': f"Spending in {category} decreased by {abs(change_pct):.1f}% from {previous:.2f} to {current:.2f}",
                                    'importance': min(abs(change_pct) / 30, 5),  # Lower importance for decreases
                                    'action': "Keep up the good work!"
                                })
            
            # 3. Find recurring transactions
            # Group by similar amounts and merchants
            df['amount_rounded'] = df['amount'].round(2)
            recurring = df.groupby(['merchant', 'amount_rounded']).agg({
                'date': 'count',
                'description': 'first',
                'category': 'first',
                'amount': 'mean'
            }).reset_index()
            
            # Filter for potential subscriptions
            recurring = recurring[recurring['date'] >= 2]  # At least 2 occurrences
            
            for _, row in recurring.iterrows():
                if row['date'] >= 3:  # Highly recurring (3+ times)
                    insights.append({
                        'type': 'recurring_payment',
                        'category': row['category'],
                        'description': f"Recurring payment of {row['amount']:.2f} to {row['merchant']} ({row['date']} times)",
                        'importance': min(row['date'], 8),  # More occurrences = higher importance
                        'action': "Review if this subscription is still needed"
                    })
            
            # 4. Anomaly detection
            anomalies = self.detect_anomalies(categorized)
            for anomaly in anomalies:
                insights.append({
                    'type': 'anomaly',
                    'category': anomaly.get('category', 'Unknown'),
                    'description': f"Unusual transaction: {anomaly.get('description', '')} ({anomaly.get('amount', 0):.2f})",
                    'importance': 9,  # Anomalies are high importance
                    'action': "Verify this transaction is legitimate",
                    'reason': anomaly.get('anomaly_reason', 'Unusual pattern')
                })
                
            # 5. Budget analysis
            # Calculate total spending by necessity
            spending_types = {}
            total_spending = 0
            
            for tx in categorized:
                if not tx.get('is_income', False):
                    amount = tx.get('amount', 0)
                    total_spending += amount
                    
                    # Analyze if it's necessary or discretionary
                    description = f"{tx.get('description', '')} {tx.get('merchant', '')}"
                    spending_type = self.analyze_spending_type(description)
                    
                    spending_types[spending_type] = spending_types.get(spending_type, 0) + amount
            
            # Calculate percentages
            if total_spending > 0:
                necessary_pct = (spending_types.get('NECESSARY', 0) / total_spending) * 100
                discretionary_pct = (spending_types.get('DISCRETIONARY', 0) / total_spending) * 100
                
                # Generate insights based on spending distribution
                if discretionary_pct > 40:
                    insights.append({
                        'type': 'budget_insight',
                        'description': f"Discretionary spending is {discretionary_pct:.1f}% of your total expenses",
                        'importance': 7,
                        'action': "Consider reducing non-essential expenses to save more"
                    })
                elif necessary_pct > 80:
                    insights.append({
                        'type': 'budget_insight',
                        'description': f"Necessary expenses make up {necessary_pct:.1f}% of your spending",
                        'importance': 6,
                        'action': "Your budget is focused on essentials, which is good fiscal discipline"
                    })
            
            # Sort insights by importance
            insights.sort(key=lambda x: x.get('importance', 0), reverse=True)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return []

    def add_custom_category_pattern(self, category: str, pattern: str) -> None:
        """
        Add a custom pattern to a category.

        Args:
            category: Category name
            pattern: Regex pattern string
        """
        # Create category if it doesn't exist
        if category not in self.categories:
            self.categories[category] = []
            self.compiled_patterns[category] = []

        # Add pattern
        self.categories[category].append(pattern)
        self.compiled_patterns[category].append(re.compile(pattern))


# Example usage
if __name__ == "__main__":
    categorizer = TransactionCategorizer()

    # Example transaction
    transaction = {
        "description": "REWE SAGT DANKE. 58652545",
        "merchant": "REWE",
        "amount": 25.67,
        "is_income": False
    }

    category = categorizer.categorize_transaction(transaction)
    print(f"Transaction categorized as: {category}")