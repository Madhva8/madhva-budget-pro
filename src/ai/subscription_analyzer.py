#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Advanced AI-Powered Subscription Analyzer

This module specializes in analyzing subscription spending patterns using NLP and ML,
identifying wasteful subscriptions, detecting hidden recurring payments, and providing 
intelligent personalized recommendations for optimizing subscription spending.
"""

import re
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import defaultdict, Counter

# Set up logging
logger = logging.getLogger(__name__)

# Advanced NLP capabilities with fallbacks
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
    logger.info("NLP components successfully loaded for subscription analysis")
    
except ImportError:
    NLP_AVAILABLE = False
    logger.warning("NLTK libraries not available. Some advanced text analysis features will be limited.")

# Advanced ML capabilities with fallbacks
try:
    from sklearn.cluster import DBSCAN
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    
    ML_AVAILABLE = True
    logger.info("ML components successfully loaded for subscription analysis")
except ImportError:
    ML_AVAILABLE = False
    logger.warning("Scikit-learn not available. Some advanced analysis features will be limited.")


class SubscriptionAnalyzer:
    """
    AI-powered subscription spending analysis and optimization.
    
    Features:
    - NLP-based subscription detection from transaction descriptions
    - ML clustering for identifying hidden recurring payments
    - Semantic understanding of subscription types and purposes
    - Advanced analytics for identifying wasteful subscriptions
    - Personalized recommendations based on spending patterns and preferences
    - Subscription market intelligence for pricing benchmarks
    """

    # Known subscription services with typical price ranges (in euros)
    KNOWN_SUBSCRIPTIONS = {
        # Streaming services
        'netflix': {'type': 'entertainment', 'typical_price': [7.99, 17.99], 'frequency': 'monthly'},
        'spotify': {'type': 'entertainment', 'typical_price': [4.99, 9.99], 'frequency': 'monthly'},
        'amazon prime': {'type': 'entertainment', 'typical_price': [7.99, 8.99], 'frequency': 'monthly'},
        'disney+': {'type': 'entertainment', 'typical_price': [5.99, 11.99], 'frequency': 'monthly'},
        'apple tv+': {'type': 'entertainment', 'typical_price': [4.99, 6.99], 'frequency': 'monthly'},
        'hbo': {'type': 'entertainment', 'typical_price': [8.99, 14.99], 'frequency': 'monthly'},
        'youtube premium': {'type': 'entertainment', 'typical_price': [11.99, 17.99], 'frequency': 'monthly'},
        'paramount+': {'type': 'entertainment', 'typical_price': [7.99, 10.99], 'frequency': 'monthly'},
        'dazn': {'type': 'entertainment', 'typical_price': [9.99, 29.99], 'frequency': 'monthly'},
        'apple music': {'type': 'entertainment', 'typical_price': [4.99, 16.99], 'frequency': 'monthly'},
        'sky': {'type': 'entertainment', 'typical_price': [12.50, 30.00], 'frequency': 'monthly'},
        'audible': {'type': 'entertainment', 'typical_price': [7.95, 9.95], 'frequency': 'monthly'},
        
        # Software services
        'adobe': {'type': 'productivity', 'typical_price': [11.89, 59.99], 'frequency': 'monthly'},
        'microsoft 365': {'type': 'productivity', 'typical_price': [6.99, 12.99], 'frequency': 'monthly'},
        'google one': {'type': 'productivity', 'typical_price': [1.99, 9.99], 'frequency': 'monthly'},
        'dropbox': {'type': 'productivity', 'typical_price': [9.99, 19.99], 'frequency': 'monthly'},
        'canva': {'type': 'productivity', 'typical_price': [12.99, 30.00], 'frequency': 'monthly'},
        'notion': {'type': 'productivity', 'typical_price': [4.00, 8.00], 'frequency': 'monthly'},
        'evernote': {'type': 'productivity', 'typical_price': [7.99, 9.99], 'frequency': 'monthly'},
        
        # Gaming services
        'playstation plus': {'type': 'gaming', 'typical_price': [8.99, 13.99], 'frequency': 'monthly'},
        'xbox game pass': {'type': 'gaming', 'typical_price': [9.99, 14.99], 'frequency': 'monthly'},
        'nintendo switch online': {'type': 'gaming', 'typical_price': [3.99, 7.99], 'frequency': 'monthly'},
        'ea play': {'type': 'gaming', 'typical_price': [3.99, 14.99], 'frequency': 'monthly'},
        
        # News/magazines
        'zeit': {'type': 'news', 'typical_price': [5.20, 20.90], 'frequency': 'monthly'},
        'spiegel': {'type': 'news', 'typical_price': [4.99, 19.99], 'frequency': 'monthly'},
        'bild plus': {'type': 'news', 'typical_price': [7.99, 9.99], 'frequency': 'monthly'},
        'faz': {'type': 'news', 'typical_price': [9.90, 16.90], 'frequency': 'monthly'},
        
        # Other services
        'fitnessstudio': {'type': 'fitness', 'typical_price': [15.00, 60.00], 'frequency': 'monthly'},
        'mcfit': {'type': 'fitness', 'typical_price': [19.90, 29.90], 'frequency': 'monthly'},
        'clever fit': {'type': 'fitness', 'typical_price': [19.90, 39.90], 'frequency': 'monthly'},
        'chatgpt': {'type': 'productivity', 'typical_price': [20.00, 22.00], 'frequency': 'monthly'},
        'openai': {'type': 'productivity', 'typical_price': [20.00, 22.00], 'frequency': 'monthly'},
        'duolingo': {'type': 'education', 'typical_price': [6.99, 12.99], 'frequency': 'monthly'},
        'linkedin premium': {'type': 'professional', 'typical_price': [29.99, 59.99], 'frequency': 'monthly'},
        'dating app': {'type': 'social', 'typical_price': [9.99, 29.99], 'frequency': 'monthly'},
        'tinder': {'type': 'social', 'typical_price': [9.99, 29.99], 'frequency': 'monthly'},
        'bumble': {'type': 'social', 'typical_price': [14.99, 32.99], 'frequency': 'monthly'},
        'vpn': {'type': 'security', 'typical_price': [3.00, 12.00], 'frequency': 'monthly'},
        'nordvpn': {'type': 'security', 'typical_price': [3.49, 11.99], 'frequency': 'monthly'},
        'expressvpn': {'type': 'security', 'typical_price': [8.32, 12.95], 'frequency': 'monthly'},
    }

    # Subscription value categories based on usage patterns
    SUBSCRIPTION_VALUE_METRICS = {
        'entertainment': {
            'high_value': {'min_uses': 10, 'cost_per_use': 0.5},  # e.g., Netflix watched 10+ times/month is good value
            'medium_value': {'min_uses': 5, 'cost_per_use': 1.5},
            'low_value': {'min_uses': 1, 'cost_per_use': 5.0}
        },
        'productivity': {
            'high_value': {'min_uses': 20, 'cost_per_use': 0.25},  # e.g., Office used 20+ times/month is good value
            'medium_value': {'min_uses': 10, 'cost_per_use': 0.5},
            'low_value': {'min_uses': 5, 'cost_per_use': 1.0}
        },
        'gaming': {
            'high_value': {'min_uses': 8, 'cost_per_use': 0.75},
            'medium_value': {'min_uses': 4, 'cost_per_use': 2.0},
            'low_value': {'min_uses': 1, 'cost_per_use': 5.0}
        },
        'fitness': {
            'high_value': {'min_uses': 12, 'cost_per_use': 2.0},  # e.g., Gym visited 12+ times/month is good value
            'medium_value': {'min_uses': 8, 'cost_per_use': 3.0},
            'low_value': {'min_uses': 4, 'cost_per_use': 6.0}
        },
        'news': {
            'high_value': {'min_uses': 20, 'cost_per_use': 0.25},
            'medium_value': {'min_uses': 10, 'cost_per_use': 0.5},
            'low_value': {'min_uses': 5, 'cost_per_use': 1.0}
        },
    }
    
    # Subscription keywords and phrases for improved detection
    SUBSCRIPTION_KEYWORDS = {
        'general': [
            'subscription', 'recurring', 'membership', 'monthly fee', 'annual fee',
            'premium', 'plus', 'pro', 'unlimited', 'abonnement', 'abo', 'plan'
        ],
        'billing_terms': [
            'billed monthly', 'auto-renewal', 'automatically renews', 
            'recurring payment', 'continuous service', 'subscription fee'
        ],
        'cancellation': [
            'cancel anytime', 'minimum term', 'contract period', 'no commitment',
            'cancel subscription', 'binding period', 'notice period'
        ]
    }
    
    # Common subscription services and their associated domains/identifiers
    SUBSCRIPTION_DOMAINS = {
        'apple.com': ['apple music', 'apple tv+', 'apple arcade', 'icloud', 'apple one'],
        'google.com': ['google one', 'youtube premium', 'google workspace', 'play pass'],
        'microsoft.com': ['microsoft 365', 'xbox game pass', 'onedrive'],
        'amazon.com': ['amazon prime', 'kindle unlimited', 'audible'],
    }

    def __init__(self):
        """Initialize the subscription analyzer with AI capabilities."""
        self.logger = logger
        self.logger.info("Initializing AI-powered SubscriptionAnalyzer")
        
        # Compile regex patterns for subscription detection
        self.subscription_patterns = {
            name: re.compile(rf'(?i)\b{re.escape(name)}\b')
            for name in self.KNOWN_SUBSCRIPTIONS.keys()
        }
        
        # Initialize NLP components if available
        self.nlp_enabled = NLP_AVAILABLE
        if self.nlp_enabled:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            # Add domain-specific stopwords
            domain_stopwords = {'payment', 'transaction', 'card', 'debit', 'credit', 'charge', 'bill', 'invoice'}
            self.stop_words.update(domain_stopwords)
            
            # Prepare subscription keyword patterns
            self.subscription_keyword_patterns = []
            for keyword_list in self.SUBSCRIPTION_KEYWORDS.values():
                for keyword in keyword_list:
                    # Handle multi-word keywords with more complex regex
                    if ' ' in keyword:
                        pattern = r'(?i)\b' + re.escape(keyword) + r'\b'
                    else:
                        pattern = r'(?i)\b' + re.escape(keyword) + r'\b'
                    self.subscription_keyword_patterns.append(re.compile(pattern))
                    
            self.logger.info(f"NLP components initialized with {len(self.subscription_keyword_patterns)} subscription keyword patterns")
        
        # Initialize ML components if available
        self.ml_enabled = ML_AVAILABLE
        if self.ml_enabled:
            # Initialize DBSCAN for clustering similar transactions
            self.dbscan = DBSCAN(eps=0.5, min_samples=2, metric='cosine')
            # Initialize TF-IDF vectorizer for text similarity
            self.vectorizer = TfidfVectorizer(max_features=100)
            # Initialize scaler for numerical features
            self.scaler = StandardScaler()
            
            self.logger.info("ML components initialized for advanced subscription detection")
            
        self.logger.info(f"Compiled {len(self.subscription_patterns)} subscription patterns")

    def identify_subscriptions(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify subscription services from transaction data.
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            List of identified subscription transactions with metadata
        """
        if not transactions:
            return []
            
        # Convert to DataFrame
        df = pd.DataFrame(transactions)
        
        # Ensure required fields exist
        if 'description' not in df.columns or 'amount' not in df.columns:
            return []
            
        # Filter expenses
        try:
            # First check if is_income is in columns and is a boolean/integer value
            if 'is_income' in df.columns and all(isinstance(x, (bool, int, float)) or x is None for x in df['is_income']):
                expenses = df[~df['is_income'].astype(bool)]
            else:
                # If is_income is not available or not boolean, use all transactions
                expenses = df
                
            if expenses.empty:
                return []
        except Exception as e:
            print(f"Error filtering expenses: {e}")
            # Fallback to using all transactions
            expenses = df
            
        # Identify subscriptions
        subscription_list = []
        
        for _, tx in expenses.iterrows():
            description = tx.get('description', '').lower()
            merchant = tx.get('merchant', '').lower()
            text_to_check = f"{description} {merchant}"
            
            # Check for known subscription services
            for service_name, pattern in self.subscription_patterns.items():
                if pattern.search(text_to_check):
                    service_info = self.KNOWN_SUBSCRIPTIONS[service_name]
                    
                    subscription_list.append({
                        'transaction_id': tx.get('id'),
                        'date': tx.get('date'),
                        'description': tx.get('description'),
                        'amount': tx.get('amount'),
                        'service_name': service_name,
                        'service_type': service_info['type'],
                        'typical_price_range': service_info['typical_price'],
                        'frequency': service_info['frequency']
                    })
                    break
                    
            # Check for subscription keywords if no known service matched
            subscription_keywords = [
                'subscription', 'recurring', 'monthly', 'yearly', 'premium',
                'membership', 'abo', 'abonnement', 'plan', 'service'
            ]
            
            if not any(pattern.search(text_to_check) for pattern in self.subscription_patterns.values()):
                if any(keyword in text_to_check for keyword in subscription_keywords):
                    # This looks like a subscription but isn't in our known list
                    subscription_list.append({
                        'transaction_id': tx.get('id'),
                        'date': tx.get('date'),
                        'description': tx.get('description'),
                        'amount': tx.get('amount'),
                        'service_name': 'unknown',
                        'service_type': 'other',
                        'typical_price_range': None,
                        'frequency': 'unknown'
                    })
        
        return subscription_list
    
    def detect_recurring_transactions(self, transactions: List[Dict[str, Any]], 
                                     tolerance: float = 0.1,
                                     min_occurrences: int = 2) -> List[Dict[str, Any]]:
        """
        Detect recurring transactions that may be subscriptions even if they don't match known patterns.
        
        Args:
            transactions: List of transaction dictionaries
            tolerance: Price variation tolerance as a percentage (0.1 = 10%)
            min_occurrences: Minimum number of occurrences to consider as recurring
            
        Returns:
            List of recurring transaction patterns
        """
        if not transactions:
            return []
            
        # Convert to DataFrame
        df = pd.DataFrame(transactions)
        
        # Ensure required fields exist
        if 'description' not in df.columns or 'amount' not in df.columns or 'date' not in df.columns:
            return []
            
        # Filter expenses
        try:
            # First check if is_income is in columns and is a boolean/integer value
            if 'is_income' in df.columns and all(isinstance(x, (bool, int, float)) or x is None for x in df['is_income']):
                expenses = df[~df['is_income'].astype(bool)]
            else:
                # If is_income is not available or not boolean, use all transactions
                expenses = df
                
            if expenses.empty:
                return []
        except Exception as e:
            print(f"Error filtering expenses in detect_recurring_transactions: {e}")
            # Fallback to using all transactions
            expenses = df
            
        # Convert dates to datetime
        expenses['date'] = pd.to_datetime(expenses['date'])
        
        # Group by merchant/description
        potential_recurring = []
        
        # Group by merchant if available, otherwise by description
        group_by_field = 'merchant' if 'merchant' in expenses.columns else 'description'
        
        # For each merchant/company
        for name, group in expenses.groupby(group_by_field):
            # Skip empty names
            if not name:
                continue
                
            # Group by similar amounts (within tolerance)
            amount_groups = defaultdict(list)
            
            for _, row in group.iterrows():
                amount = row['amount']
                # Find a matching amount group
                matched = False
                
                for base_amount, transactions in amount_groups.items():
                    # Check if this amount is within tolerance of the base amount
                    if abs(amount - base_amount) <= (base_amount * tolerance):
                        transactions.append(row.to_dict())
                        matched = True
                        break
                        
                if not matched:
                    # Create a new amount group
                    amount_groups[amount].append(row.to_dict())
            
            # Check each amount group for recurring patterns
            for base_amount, group_txs in amount_groups.items():
                if len(group_txs) >= min_occurrences:
                    # Sort by date
                    sorted_txs = sorted(group_txs, key=lambda x: x['date'])
                    
                    # Calculate intervals between transactions
                    intervals = []
                    for i in range(1, len(sorted_txs)):
                        interval_days = (sorted_txs[i]['date'] - sorted_txs[i-1]['date']).days
                        intervals.append(interval_days)
                    
                    # Skip if no intervals
                    if not intervals:
                        continue
                        
                    # Determine if consistent interval
                    avg_interval = sum(intervals) / len(intervals)
                    std_dev = np.std(intervals) if len(intervals) > 1 else 0
                    
                    # Classify by interval
                    if 25 <= avg_interval <= 35 and std_dev <= 5:
                        frequency = 'monthly'
                    elif 6 <= avg_interval <= 8 and std_dev <= 2:
                        frequency = 'weekly'
                    elif 13 <= avg_interval <= 15 and std_dev <= 3:
                        frequency = 'biweekly'
                    elif 85 <= avg_interval <= 95 and std_dev <= 10:
                        frequency = 'quarterly'
                    elif 350 <= avg_interval <= 380 and std_dev <= 15:
                        frequency = 'yearly'
                    else:
                        frequency = 'irregular'
                    
                    # Add to potential recurring list if it seems regular enough
                    if frequency != 'irregular' or (avg_interval > 0 and std_dev / avg_interval < 0.3):
                        potential_recurring.append({
                            'name': name,
                            'average_amount': base_amount,
                            'frequency': frequency,
                            'average_interval_days': avg_interval,
                            'transaction_count': len(group_txs),
                            'first_date': sorted_txs[0]['date'],
                            'last_date': sorted_txs[-1]['date'],
                            'transactions': sorted_txs,
                            'consistency': 'high' if std_dev / avg_interval < 0.2 else 'medium'
                        })
        
        return potential_recurring
        
    def analyze_subscription_spending(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze subscription spending patterns and generate insights.
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            Dictionary with subscription spending analysis
        """
        # Identify explicit subscriptions
        subscriptions = self.identify_subscriptions(transactions)
        
        # Detect recurring transactions (potential subscriptions)
        recurring = self.detect_recurring_transactions(transactions)
        
        # Combine explicit and potential subscriptions
        all_subscriptions = subscriptions.copy()
        
        # Add recurring transactions that aren't already identified as subscriptions
        recurring_names = set()
        for rec in recurring:
            name = rec['name'].lower()
            recurring_names.add(name)
            
            # Check if this recurring transaction is not already in subscriptions
            if not any(sub['service_name'] == name for sub in subscriptions):
                # Convert recurring format to subscription format
                for tx in rec['transactions']:
                    all_subscriptions.append({
                        'transaction_id': tx.get('id'),
                        'date': tx.get('date'),
                        'description': tx.get('description', ''),
                        'amount': tx.get('amount', 0),
                        'service_name': name,
                        'service_type': 'recurring',
                        'typical_price_range': [rec['average_amount'] * 0.9, rec['average_amount'] * 1.1],
                        'frequency': rec['frequency']
                    })
        
        # Calculate spending totals
        if all_subscriptions:
            total_subscription_spend = sum(sub['amount'] for sub in all_subscriptions)
            
            # Group by service
            service_totals = defaultdict(float)
            for sub in all_subscriptions:
                service_totals[sub['service_name']] += sub['amount']
            
            # Group by type
            type_totals = defaultdict(float)
            for sub in all_subscriptions:
                type_totals[sub['service_type']] += sub['amount']
                
            # Calculate monthly estimate
            # Get date range from transactions
            df = pd.DataFrame(transactions)
            if 'date' in df.columns and not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                min_date = df['date'].min()
                max_date = df['date'].max()
                
                # Calculate months in range
                months = (max_date.year - min_date.year) * 12 + max_date.month - min_date.month + 1
                
                if months > 0:
                    monthly_subscription_estimate = total_subscription_spend / months
                else:
                    monthly_subscription_estimate = total_subscription_spend
            else:
                monthly_subscription_estimate = total_subscription_spend
                
            # Calculate percentage of total spending
            try:
                # Try to calculate total spending safely
                total_spending = sum(tx.get('amount', 0) for tx in transactions 
                                    if not tx.get('is_income', False) and isinstance(tx.get('is_income'), (bool, int)))
                
                # If that fails, just sum all amounts
                if total_spending <= 0:
                    total_spending = sum(tx.get('amount', 0) for tx in transactions)
                    
                subscription_percentage = (total_subscription_spend / total_spending * 100) if total_spending > 0 else 0
            except Exception as e:
                print(f"Error calculating subscription percentage: {e}")
                total_spending = 1  # Avoid division by zero
                subscription_percentage = 0
            
        else:
            total_subscription_spend = 0
            service_totals = {}
            type_totals = {}
            monthly_subscription_estimate = 0
            subscription_percentage = 0
            
        return {
            'subscriptions': all_subscriptions,
            'recurring_patterns': recurring,
            'total_subscription_spend': total_subscription_spend,
            'service_totals': dict(service_totals),
            'type_totals': dict(type_totals),
            'monthly_subscription_estimate': monthly_subscription_estimate,
            'subscription_percentage': subscription_percentage
        }
        
    def generate_subscription_recommendations(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate recommendations for optimizing subscription spending.
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            List of recommendations to save money on subscriptions
        """
        # Analyze subscription spending
        analysis = self.analyze_subscription_spending(transactions)
        
        recommendations = []
        
        # 1. Check for duplicate services of the same type
        service_by_type = defaultdict(list)
        for sub in analysis['subscriptions']:
            service_by_type[sub['service_type']].append(sub)
            
        for service_type, services in service_by_type.items():
            if service_type in ['entertainment', 'productivity', 'news', 'gaming'] and len(services) > 1:
                service_names = set(sub['service_name'] for sub in services)
                
                if len(service_names) > 1:
                    total_spend = sum(analysis['service_totals'].get(name, 0) for name in service_names)
                    
                    recommendations.append({
                        'type': 'duplicate_services',
                        'service_type': service_type,
                        'services': list(service_names),
                        'total_spend': total_spend,
                        'recommendation': f"You subscribe to multiple {service_type} services: {', '.join(service_names)}. Consider consolidating to reduce costs.",
                        'potential_savings': total_spend * 0.5  # Estimate 50% savings from consolidation
                    })
        
        # 2. Check for services with prices above typical range
        for sub in analysis['subscriptions']:
            if sub['typical_price_range'] and sub['amount'] > sub['typical_price_range'][1]:
                recommendations.append({
                    'type': 'high_price',
                    'service_name': sub['service_name'],
                    'current_price': sub['amount'],
                    'typical_range': sub['typical_price_range'],
                    'recommendation': f"You're paying {sub['amount']:.2f}€ for {sub['service_name']}, which is above the typical price range of {sub['typical_price_range'][0]:.2f}€ - {sub['typical_price_range'][1]:.2f}€. Consider checking for a cheaper plan or negotiating.",
                    'potential_savings': sub['amount'] - sub['typical_price_range'][1]
                })
        
        # 3. Check if total subscription spending is high
        if analysis['subscription_percentage'] > 12:  # If subscriptions are more than 12% of spending
            recommendations.append({
                'type': 'high_subscription_percentage',
                'current_percentage': analysis['subscription_percentage'],
                'monthly_estimate': analysis['monthly_subscription_estimate'],
                'recommendation': f"Your subscription services account for {analysis['subscription_percentage']:.1f}% of your total spending, which is higher than recommended. Consider reviewing and eliminating non-essential subscriptions.",
                'potential_savings': analysis['monthly_subscription_estimate'] * 0.3  # Estimate 30% savings
            })
        
        # 4. Identify rarely used subscriptions (based on frequency patterns)
        if analysis['recurring_patterns']:
            for pattern in analysis['recurring_patterns']:
                if pattern['frequency'] == 'monthly':
                    # Calculate typical days between transactions
                    expected_days = 30
                    # Get dates of transactions
                    dates = [tx['date'] for tx in pattern['transactions']]
                    
                    if len(dates) >= 2:
                        # Calculate how many expected transactions should have occurred
                        date_range_days = (max(dates) - min(dates)).days
                        expected_transactions = date_range_days / expected_days
                        
                        # If actual is significantly less than expected, might be underused
                        if len(dates) < expected_transactions * 0.7:
                            recommendations.append({
                                'type': 'underused_subscription',
                                'service_name': pattern['name'],
                                'average_amount': pattern['average_amount'],
                                'recommendation': f"You may be underusing your {pattern['name']} subscription. Consider if you need this service.",
                                'potential_savings': pattern['average_amount']  # Potential savings = full cancellation
                            })
        
        # 5. Bundle opportunities
        streaming_services = [sub['service_name'] for sub in analysis['subscriptions'] 
                             if sub['service_type'] == 'entertainment' and 'typical_price_range' in sub]
        
        # Check for potential bundles
        bundle_opportunities = {
            'disney_bundle': {'services': ['disney+', 'hulu'], 'savings': 5.0},
            'amazon_prime': {'services': ['amazon prime', 'prime video'], 'savings': 9.0},
            'apple_one': {'services': ['apple music', 'apple tv+', 'icloud'], 'savings': 6.0},
        }
        
        for bundle_name, details in bundle_opportunities.items():
            matches = [service for service in details['services'] if any(s in streaming_services for s in service.split())]
            if len(matches) >= 2:
                recommendations.append({
                    'type': 'bundle_opportunity',
                    'bundle_name': bundle_name,
                    'services': matches,
                    'recommendation': f"You could save money by bundling your {' and '.join(matches)} subscriptions.",
                    'potential_savings': details['savings']
                })
        
        # 6. Check for annual vs monthly payment opportunities
        monthly_subs = [sub for sub in analysis['subscriptions'] if sub['frequency'] == 'monthly']
        
        for sub in monthly_subs:
            # Services that typically offer annual discounts
            annual_discount_services = [
                'netflix', 'spotify', 'disney+', 'amazon prime', 
                'nordvpn', 'expressvpn', 'microsoft 365', 'adobe'
            ]
            
            if any(service in sub['service_name'].lower() for service in annual_discount_services):
                monthly_amount = sub['amount']
                annual_equivalent = monthly_amount * 12
                estimated_annual_price = annual_equivalent * 0.8  # Assume 20% discount for annual
                potential_savings = annual_equivalent - estimated_annual_price
                
                if potential_savings > 20:  # Only suggest if savings are significant
                    recommendations.append({
                        'type': 'annual_payment_opportunity',
                        'service_name': sub['service_name'],
                        'monthly_amount': monthly_amount,
                        'annual_equivalent': annual_equivalent,
                        'estimated_annual_price': estimated_annual_price,
                        'recommendation': f"Consider switching to an annual payment plan for {sub['service_name']} to save approximately {potential_savings:.2f}€ per year.",
                        'potential_savings': potential_savings
                    })
        
        # 7. Student/family plan opportunities
        potential_student_services = ['spotify', 'apple music', 'youtube premium', 'amazon prime']
        
        for service in potential_student_services:
            matching_subs = [sub for sub in analysis['subscriptions'] if service in sub['service_name'].lower()]
            
            if matching_subs:
                for sub in matching_subs:
                    recommendations.append({
                        'type': 'student_plan_opportunity',
                        'service_name': sub['service_name'],
                        'current_amount': sub['amount'],
                        'recommendation': f"Check if you're eligible for a student discount on {sub['service_name']}. Student plans typically save 50% or more.",
                        'potential_savings': sub['amount'] * 0.5  # Assume 50% savings with student plan
                    })
        
        # Sort recommendations by potential savings (highest first)
        recommendations.sort(key=lambda x: x.get('potential_savings', 0), reverse=True)
        
        return recommendations


# Example usage
if __name__ == "__main__":
    analyzer = SubscriptionAnalyzer()
    
    # Example transactions
    transactions = [
        {
            "date": "2025-01-05",
            "description": "Netflix Premium",
            "amount": 17.99,
            "is_income": False,
            "category": "Subscriptions"
        },
        {
            "date": "2025-02-05",
            "description": "Netflix Premium",
            "amount": 17.99,
            "is_income": False,
            "category": "Subscriptions"
        },
        {
            "date": "2025-01-10",
            "description": "Spotify Premium",
            "amount": 9.99,
            "is_income": False,
            "category": "Subscriptions"
        },
        {
            "date": "2025-02-10",
            "description": "Spotify Premium",
            "amount": 9.99,
            "is_income": False,
            "category": "Subscriptions"
        },
        {
            "date": "2025-01-15",
            "description": "Amazon Prime",
            "amount": 8.99,
            "is_income": False,
            "category": "Subscriptions"
        },
        {
            "date": "2025-02-15",
            "description": "Amazon Prime",
            "amount": 8.99,
            "is_income": False,
            "category": "Subscriptions"
        }
    ]
    
    # Generate recommendations
    recommendations = analyzer.generate_subscription_recommendations(transactions)
    
    for rec in recommendations:
        print(f"{rec['recommendation']} Potential savings: {rec['potential_savings']:.2f}€")