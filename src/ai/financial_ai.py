#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Financial AI Module

This module provides advanced AI-powered analysis of financial data, spending patterns,
and personalized recommendations for the Financial Planner application.
Leverages NLP and machine learning for deeper insights.
"""

import datetime
import calendar
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import defaultdict

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
    logger.info("NLP components successfully loaded for financial insights")
    
except ImportError:
    NLP_AVAILABLE = False
    logger.warning("NLTK libraries not available. Some advanced text analysis features will be limited.")

# Advanced ML capabilities with fallbacks
try:
    from sklearn.cluster import KMeans
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    ML_AVAILABLE = True
    logger.info("ML components successfully loaded for financial analysis")
except ImportError:
    ML_AVAILABLE = False
    logger.warning("Scikit-learn not available. Some advanced analysis features will be limited.")


class FinancialAI:
    """
    Advanced AI assistant for analyzing financial data and providing personalized recommendations.
    Features:
    - Natural language processing for understanding transaction contexts
    - Machine learning for anomaly detection and pattern recognition
    - Predictive analytics for forecasting financial trends
    - Semantic clustering of similar transactions
    - Tailored financial insights and recommendations
    """

    def __init__(self, db_manager=None):
        """
        Initialize the Financial AI with advanced ML and NLP capabilities.

        Args:
            db_manager: Optional database manager to fetch transaction data
        """
        self.db_manager = db_manager
        
        # Model training state
        self.is_trained = False
        self.training_history = []
        self.model_version = 0
        self.last_trained_timestamp = None
        
        # Initialize NLP components
        self.nlp_enabled = NLP_AVAILABLE
        if self.nlp_enabled:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            # Add financial domain-specific stopwords
            financial_stopwords = {'eur', 'euro', 'payment', 'transaction', 'card', 'debit', 
                                  'credit', 'amount', 'total', 'purchase', 'fee'}
            self.stop_words.update(financial_stopwords)
            logger.info("NLP components initialized with financial domain knowledge")
        
        # Initialize ML components
        self.ml_enabled = ML_AVAILABLE
        if self.ml_enabled:
            # Isolation Forest for anomaly detection
            self.anomaly_detector = IsolationForest(
                n_estimators=100,
                contamination=0.05,  # Assume about 5% of transactions are anomalous
                random_state=42
            )
            
            # Regression model for forecasting
            self.forecast_model = LinearRegression()
            
            # Clustering model for transaction grouping
            self.clustering_model = KMeans(n_clusters=8, random_state=42)
            
            # Data scaler for preprocessing
            self.scaler = StandardScaler()
            
            # Transaction description vectorizer
            self.description_vectorizer = TfidfVectorizer(
                max_features=200,
                ngram_range=(1, 2),
                stop_words='english'
            )
            
            # Transaction embedding cache
            self.transaction_embeddings = {}
            
            # Incremental learning data
            self.known_transactions = []
            self.category_corrections = {}  # User corrections to learn from
            self.sentiment_labels = {}     # User-provided sentiment labels
            
            logger.info("ML models initialized for financial analysis")
            
        # Define semantic categories for transaction understanding
        self.semantic_categories = {
            'essentials': ['rent', 'mortgage', 'electricity', 'water', 'heating', 'grocery', 
                          'insurance', 'medical', 'healthcare', 'prescription', 'doctor'],
            'transportation': ['bus', 'train', 'taxi', 'uber', 'fuel', 'gas', 'car', 'maintenance',
                              'repair', 'ticket', 'transport', 'metro', 'subway'],
            'dining': ['restaurant', 'cafe', 'coffee', 'takeout', 'delivery', 'bar', 'bistro',
                      'food', 'dinner', 'lunch', 'breakfast'],
            'entertainment': ['movie', 'cinema', 'concert', 'theater', 'show', 'netflix', 'spotify',
                             'streaming', 'subscription', 'game', 'gaming'],
            'shopping': ['clothing', 'apparel', 'electronics', 'furniture', 'shoes', 'accessory',
                        'beauty', 'cosmetics', 'department', 'store', 'mall', 'amazon'],
            'education': ['tuition', 'book', 'course', 'class', 'training', 'workshop', 'seminar',
                         'university', 'college', 'school', 'supplies'],
            'debt': ['loan', 'interest', 'credit', 'payment', 'mortgage', 'minimum']
        }
        
        # Transaction sentiment words (positive/negative financial implications)
        self.positive_words = ['save', 'discount', 'refund', 'cashback', 'investment', 'return', 'income',
                              'dividend', 'bonus', 'reward', 'profit', 'reimbursement']
        self.negative_words = ['fee', 'penalty', 'interest', 'charge', 'debt', 'overdue', 'expensive',
                              'withdrawal', 'overdraft', 'declined', 'rejected']
                              
        # Try to load previous training data if db_manager is available
        if self.db_manager:
            self._load_model_state()
            
        # Automatic training from historical data
        if self.db_manager and not self.is_trained:
            try:
                self.auto_train_from_historical_data()
            except Exception as e:
                logger.warning(f"Could not auto-train from historical data: {e}")

    def _preprocess_text(self, text: str) -> str:
        """
        Apply NLP preprocessing to transaction descriptions.
        
        Args:
            text: Raw transaction description
            
        Returns:
            Preprocessed text
        """
        if not self.nlp_enabled or not text:
            return text.lower() if text else ""
            
        try:
            # Tokenize
            tokens = word_tokenize(text.lower())
            
            # Remove stopwords and lemmatize
            processed_tokens = [
                self.lemmatizer.lemmatize(token) 
                for token in tokens 
                if token not in self.stop_words and token.isalpha()
            ]
            
            return " ".join(processed_tokens)
        except Exception as e:
            logger.warning(f"Error preprocessing text: {e}")
            return text.lower() if text else ""
            
    def _extract_semantic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract semantic features from transaction descriptions using NLP.
        
        Args:
            df: Transaction DataFrame
            
        Returns:
            DataFrame with added semantic features
        """
        if not self.nlp_enabled or 'description' not in df.columns or df.empty:
            return df
            
        try:
            # Create a copy to avoid modifying the original
            df_enriched = df.copy()
            
            # Preprocess descriptions
            df_enriched['processed_desc'] = df_enriched['description'].fillna('').apply(self._preprocess_text)
            
            # Extract semantic category scores
            for category, keywords in self.semantic_categories.items():
                df_enriched[f'sem_{category}'] = df_enriched['processed_desc'].apply(
                    lambda x: sum(1 for keyword in keywords if keyword in x) / len(keywords)
                )
            
            # Calculate sentiment scores
            df_enriched['positive_sentiment'] = df_enriched['processed_desc'].apply(
                lambda x: sum(1 for word in self.positive_words if word in x) / len(self.positive_words)
            )
            
            df_enriched['negative_sentiment'] = df_enriched['processed_desc'].apply(
                lambda x: sum(1 for word in self.negative_words if word in x) / len(self.negative_words)
            )
            
            df_enriched['sentiment_score'] = df_enriched['positive_sentiment'] - df_enriched['negative_sentiment']
            
            return df_enriched
        except Exception as e:
            logger.warning(f"Error extracting semantic features: {e}")
            return df
            
    def analyze_spending_patterns(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze spending patterns and generate insights using AI techniques.

        Args:
            transactions: List of transaction dictionaries

        Returns:
            Dictionary with advanced analysis results
        """
        if not transactions:
            return {
                'summary': {},
                'categories': {},
                'trends': {},
                'anomalies': [],
                'insights': [],
                'semantic_analysis': {},
                'forecast': {}
            }
            
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(transactions)

        # Ensure date is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
        # Enrich with semantic features if NLP is available
        if self.nlp_enabled and 'description' in df.columns:
            df = self._extract_semantic_features(df)

        # Create comprehensive analysis dictionary
        analysis = {
            'summary': self._get_summary_stats(df),
            'categories': self._analyze_categories(df),
            'trends': self._analyze_trends(df),
            'anomalies': self._detect_anomalies(df),
            'insights': self._generate_insights(df),
            'semantic_analysis': self._analyze_semantics(df) if self.nlp_enabled else {},
            'forecast': self._generate_forecast(df) if self.ml_enabled else {}
        }

        return analysis

    def _get_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics from transaction data.

        Args:
            df: DataFrame with transaction data

        Returns:
            Dictionary with summary statistics
        """
        # Filter income and expense transactions
        expenses = df[~df['is_income']] if 'is_income' in df.columns else df
        income = df[df['is_income']] if 'is_income' in df.columns else pd.DataFrame()

        # Calculate time range
        date_min = df['date'].min() if 'date' in df.columns and not df.empty else None
        date_max = df['date'].max() if 'date' in df.columns and not df.empty else None

        # Calculate basic stats
        total_expense = expenses['amount'].sum() if not expenses.empty else 0
        total_income = income['amount'].sum() if not income.empty else 0
        average_expense = expenses['amount'].mean() if not expenses.empty else 0

        # Calculate monthly averages
        if 'date' in df.columns and not df.empty:
            df['month'] = df['date'].dt.strftime('%Y-%m')

            monthly_expenses = expenses.groupby('month')['amount'].sum().mean() if not expenses.empty else 0
            monthly_income = income.groupby('month')['amount'].sum().mean() if not income.empty else 0

            # Get most recent month's data
            latest_month = df['month'].max()
            latest_expenses = expenses[expenses['month'] == latest_month]['amount'].sum() if not expenses.empty else 0
            latest_income = income[income['month'] == latest_month]['amount'].sum() if not income.empty else 0

            # Calculate month-over-month changes
            months = sorted(df['month'].unique())
            if len(months) >= 2:
                current_month = months[-1]
                previous_month = months[-2]

                current_expense = expenses[expenses['month'] == current_month][
                    'amount'].sum() if not expenses.empty else 0
                previous_expense = expenses[expenses['month'] == previous_month][
                    'amount'].sum() if not expenses.empty else 0

                expense_change = (
                                             current_expense - previous_expense) / previous_expense * 100 if previous_expense > 0 else 0
            else:
                expense_change = 0
        else:
            monthly_expenses = 0
            monthly_income = 0
            latest_expenses = 0
            latest_income = 0
            expense_change = 0

        # Return summary stats
        return {
            'period_start': date_min,
            'period_end': date_max,
            'total_transactions': len(df),
            'total_expense': total_expense,
            'total_income': total_income,
            'net_cash_flow': total_income - total_expense,
            'average_expense': average_expense,
            'monthly_average_expense': monthly_expenses,
            'monthly_average_income': monthly_income,
            'latest_month_expense': latest_expenses,
            'latest_month_income': latest_income,
            'latest_month_savings': latest_income - latest_expenses,
            'expense_change_percent': expense_change
        }

    def _analyze_categories(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze spending by category.

        Args:
            df: DataFrame with transaction data

        Returns:
            Dictionary with category analysis
        """
        if 'category' not in df.columns or df.empty:
            return {
                'top_categories': [],
                'category_breakdown': {}
            }

        # Filter expenses
        expenses = df[~df['is_income']] if 'is_income' in df.columns else df

        # Calculate spending by category
        if not expenses.empty:
            category_totals = expenses.groupby('category')['amount'].sum().to_dict()

            # Calculate percentages
            total_expense = sum(category_totals.values())
            category_percentages = {cat: (amt / total_expense) * 100 for cat, amt in category_totals.items()}

            # Get top categories
            top_categories = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)

            # Get monthly breakdown by category
            if 'date' in df.columns:
                df['month'] = df['date'].dt.strftime('%Y-%m')
                monthly_by_category = expenses.pivot_table(
                    index='month',
                    columns='category',
                    values='amount',
                    aggfunc='sum',
                    fill_value=0
                ).to_dict()
            else:
                monthly_by_category = {}

            return {
                'top_categories': top_categories,
                'category_breakdown': category_totals,
                'category_percentages': category_percentages,
                'monthly_by_category': monthly_by_category
            }
        else:
            return {
                'top_categories': [],
                'category_breakdown': {}
            }

    def _analyze_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze spending trends over time.

        Args:
            df: DataFrame with transaction data

        Returns:
            Dictionary with trend analysis
        """
        if 'date' not in df.columns or df.empty:
            return {
                'monthly_trend': {},
                'weekday_trend': {},
                'trend_direction': 'stable'
            }

        # Filter expenses
        expenses = df[~df['is_income']] if 'is_income' in df.columns else df

        if not expenses.empty:
            # Calculate monthly trends
            expenses['month'] = expenses['date'].dt.strftime('%Y-%m')
            monthly_expense = expenses.groupby('month')['amount'].sum().to_dict()

            # Calculate day of week trends
            expenses['weekday'] = expenses['date'].dt.weekday
            weekday_expense = expenses.groupby('weekday')['amount'].mean().to_dict()

            # Convert weekday numbers to names
            weekday_names = {
                0: 'Monday',
                1: 'Tuesday',
                2: 'Wednesday',
                3: 'Thursday',
                4: 'Friday',
                5: 'Saturday',
                6: 'Sunday'
            }
            weekday_trend = {weekday_names[day]: amount for day, amount in weekday_expense.items()}

            # Determine trend direction (increasing, decreasing, stable)
            months = sorted(monthly_expense.keys())
            if len(months) >= 3:
                recent_months = months[-3:]
                values = [monthly_expense[m] for m in recent_months]

                if values[2] > values[0] * 1.1:  # 10% increase
                    trend_direction = 'increasing'
                elif values[2] < values[0] * 0.9:  # 10% decrease
                    trend_direction = 'decreasing'
                else:
                    trend_direction = 'stable'
            else:
                trend_direction = 'insufficient_data'

            return {
                'monthly_trend': monthly_expense,
                'weekday_trend': weekday_trend,
                'trend_direction': trend_direction
            }
        else:
            return {
                'monthly_trend': {},
                'weekday_trend': {},
                'trend_direction': 'stable'
            }

    def _analyze_semantics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform semantic analysis on transaction descriptions using NLP.
        
        Args:
            df: DataFrame with transaction data and semantic features
            
        Returns:
            Dictionary with semantic analysis results
        """
        if not self.nlp_enabled or df.empty or 'processed_desc' not in df.columns:
            return {}
            
        try:
            # Get the most common semantic categories for transactions
            semantic_columns = [col for col in df.columns if col.startswith('sem_')]
            if not semantic_columns:
                return {}
                
            # Calculate average semantic scores by category
            semantic_scores = {}
            for sem_col in semantic_columns:
                category = sem_col.replace('sem_', '')
                semantic_scores[category] = df[sem_col].mean()
            
            # Find dominant semantic category for each transaction
            df['dominant_semantic'] = df[semantic_columns].idxmax(axis=1).apply(lambda x: x.replace('sem_', ''))
            
            # Get counts by dominant semantic category
            semantic_counts = df['dominant_semantic'].value_counts().to_dict()
            
            # Calculate sentiment statistics
            if 'sentiment_score' in df.columns:
                sentiment_stats = {
                    'mean_sentiment': df['sentiment_score'].mean(),
                    'positive_transactions': (df['sentiment_score'] > 0).sum(),
                    'negative_transactions': (df['sentiment_score'] < 0).sum(),
                    'neutral_transactions': (df['sentiment_score'] == 0).sum(),
                }
            else:
                sentiment_stats = {}
            
            # Calculate semantic category by transaction category
            semantic_by_category = {}
            if 'category' in df.columns:
                for category in df['category'].unique():
                    category_df = df[df['category'] == category]
                    if not category_df.empty:
                        category_semantic = {}
                        for sem_col in semantic_columns:
                            sem_name = sem_col.replace('sem_', '')
                            category_semantic[sem_name] = category_df[sem_col].mean()
                        semantic_by_category[category] = category_semantic
            
            return {
                'semantic_scores': semantic_scores,
                'dominant_semantics': semantic_counts,
                'sentiment_stats': sentiment_stats,
                'semantic_by_category': semantic_by_category
            }
            
        except Exception as e:
            logger.error(f"Error in semantic analysis: {e}")
            return {}
    
    def _generate_forecast(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate financial forecasts using machine learning.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            Dictionary with forecast results
        """
        if not self.ml_enabled or df.empty or 'date' not in df.columns or 'amount' not in df.columns:
            return {}
            
        try:
            # Create date features for prediction
            df_forecast = df.copy()
            df_forecast['date'] = pd.to_datetime(df_forecast['date'])
            df_forecast['day_of_month'] = df_forecast['date'].dt.day
            df_forecast['month'] = df_forecast['date'].dt.month
            df_forecast['year'] = df_forecast['date'].dt.year
            df_forecast['day_of_week'] = df_forecast['date'].dt.dayofweek
            
            # Separate expenses and income
            expenses = df_forecast[~df_forecast['is_income']] if 'is_income' in df_forecast.columns else df_forecast
            income = df_forecast[df_forecast['is_income']] if 'is_income' in df_forecast.columns else pd.DataFrame()
            
            # Group by month for monthly forecasting
            expenses['month_year'] = expenses['date'].dt.strftime('%Y-%m')
            monthly_expenses = expenses.groupby('month_year')['amount'].sum().reset_index()
            monthly_expenses['month_idx'] = range(len(monthly_expenses))
            
            if not income.empty:
                income['month_year'] = income['date'].dt.strftime('%Y-%m')
                monthly_income = income.groupby('month_year')['amount'].sum().reset_index()
                monthly_income['month_idx'] = range(len(monthly_income))
            
            forecasts = {}
            
            # Generate expense forecast if we have enough data
            if len(monthly_expenses) >= 3:
                # Prepare data for the model
                X = monthly_expenses[['month_idx']].values
                y = monthly_expenses['amount'].values
                
                # Train the model
                self.forecast_model.fit(X, y)
                
                # Predict next 3 months
                future_months = np.array([[i + len(monthly_expenses)] for i in range(3)])
                expense_predictions = self.forecast_model.predict(future_months)
                
                # Create forecast results
                expense_forecast = []
                last_date = pd.to_datetime(monthly_expenses['month_year'].iloc[-1])
                
                for i, prediction in enumerate(expense_predictions):
                    next_month = last_date + pd.DateOffset(months=i+1)
                    expense_forecast.append({
                        'month': next_month.strftime('%Y-%m'),
                        'amount': float(max(0, prediction)),  # Ensure non-negative
                        'confidence': 0.9 - (i * 0.1)  # Confidence decreases with time
                    })
                    
                forecasts['expenses'] = expense_forecast
            
            # Generate income forecast if we have enough data
            if not income.empty and len(monthly_income) >= 3:
                # Prepare data for the model
                X = monthly_income[['month_idx']].values
                y = monthly_income['amount'].values
                
                # Train the model
                income_model = LinearRegression()
                income_model.fit(X, y)
                
                # Predict next 3 months
                future_months = np.array([[i + len(monthly_income)] for i in range(3)])
                income_predictions = income_model.predict(future_months)
                
                # Create forecast results
                income_forecast = []
                last_date = pd.to_datetime(monthly_income['month_year'].iloc[-1])
                
                for i, prediction in enumerate(income_predictions):
                    next_month = last_date + pd.DateOffset(months=i+1)
                    income_forecast.append({
                        'month': next_month.strftime('%Y-%m'),
                        'amount': float(max(0, prediction)),  # Ensure non-negative
                        'confidence': 0.9 - (i * 0.1)  # Confidence decreases with time
                    })
                    
                forecasts['income'] = income_forecast
            
            # Calculate savings forecast if we have both income and expense forecasts
            if 'income' in forecasts and 'expenses' in forecasts:
                savings_forecast = []
                
                for i in range(len(forecasts['income'])):
                    month = forecasts['income'][i]['month']
                    income_amount = forecasts['income'][i]['amount']
                    expense_amount = forecasts['expenses'][i]['amount']
                    savings = income_amount - expense_amount
                    
                    savings_forecast.append({
                        'month': month,
                        'amount': savings,
                        'confidence': min(forecasts['income'][i]['confidence'], forecasts['expenses'][i]['confidence'])
                    })
                    
                forecasts['savings'] = savings_forecast
            
            # Add category forecasts if possible
            if 'category' in expenses.columns and len(expenses) >= 20:
                category_forecasts = {}
                
                for category in expenses['category'].unique():
                    cat_expenses = expenses[expenses['category'] == category]
                    
                    if len(cat_expenses) >= 3:
                        cat_monthly = cat_expenses.groupby('month_year')['amount'].sum().reset_index()
                        
                        if len(cat_monthly) >= 3:
                            cat_monthly['month_idx'] = range(len(cat_monthly))
                            
                            # Train category model
                            X_cat = cat_monthly[['month_idx']].values
                            y_cat = cat_monthly['amount'].values
                            
                            cat_model = LinearRegression()
                            cat_model.fit(X_cat, y_cat)
                            
                            # Predict next 3 months
                            future_months = np.array([[i + len(cat_monthly)] for i in range(3)])
                            cat_predictions = cat_model.predict(future_months)
                            
                            # Create forecast
                            cat_forecast = []
                            last_date = pd.to_datetime(cat_monthly['month_year'].iloc[-1])
                            
                            for i, prediction in enumerate(cat_predictions):
                                next_month = last_date + pd.DateOffset(months=i+1)
                                cat_forecast.append({
                                    'month': next_month.strftime('%Y-%m'),
                                    'amount': float(max(0, prediction)),
                                    'confidence': 0.85 - (i * 0.1)
                                })
                                
                            category_forecasts[category] = cat_forecast
                
                if category_forecasts:
                    forecasts['categories'] = category_forecasts
            
            return {
                'forecast_periods': len(forecasts.get('expenses', [])),
                'forecasts': forecasts
            }
            
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            return {}
            
    def _detect_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect unusual spending patterns or anomalies using advanced ML techniques.

        Args:
            df: DataFrame with transaction data

        Returns:
            List of detected anomalies with explanation
        """
        anomalies = []

        if df.empty or 'amount' not in df.columns:
            return anomalies

        # Filter expenses
        expenses = df[~df['is_income']] if 'is_income' in df.columns else df

        if not expenses.empty:
            # Try ML-based anomaly detection if available
            if self.ml_enabled and len(expenses) >= 10:
                try:
                    # Prepare features for anomaly detection
                    features = ['amount']
                    
                    # Add day of week and month if date is available
                    if 'date' in expenses.columns:
                        if 'day_of_week' not in expenses.columns:
                            expenses['day_of_week'] = expenses['date'].dt.dayofweek
                        if 'day_of_month' not in expenses.columns:
                            expenses['day_of_month'] = expenses['date'].dt.day
                        if 'month' not in expenses.columns:
                            expenses['month'] = expenses['date'].dt.month
                            
                        features.extend(['day_of_week', 'day_of_month', 'month'])
                    
                    # Add semantic features if available
                    semantic_features = [col for col in expenses.columns if col.startswith('sem_')]
                    if semantic_features:
                        features.extend(semantic_features)
                    
                    # Prepare the data
                    X = expenses[features].fillna(0).values
                    
                    # Scale the features
                    X_scaled = self.scaler.fit_transform(X)
                    
                    # Fit the model and predict
                    predictions = self.anomaly_detector.fit_predict(X_scaled)
                    
                    # Find anomalies (predictions of -1)
                    anomaly_indices = np.where(predictions == -1)[0]
                    
                    # Add detected anomalies
                    for idx in anomaly_indices:
                        tx = expenses.iloc[idx]
                        # Calculate anomaly score (how far from normal)
                        anomaly_score = self.anomaly_detector.score_samples(X_scaled[idx].reshape(1, -1))[0]
                        
                        # Create explanation
                        explanation = "Transaction flagged by ML model as unusual "
                        
                        # Add amount context
                        mean_amount = expenses['amount'].mean()
                        if tx['amount'] > mean_amount * 2:
                            explanation += f"due to high amount (€{tx['amount']:.2f} vs avg €{mean_amount:.2f}). "
                        
                        # Add category context if available
                        if 'category' in tx and tx['category']:
                            cat_mean = expenses[expenses['category'] == tx['category']]['amount'].mean()
                            if tx['amount'] > cat_mean * 1.5:
                                explanation += f"Amount is {(tx['amount']/cat_mean):.1f}x higher than average for {tx['category']} category. "
                        
                        # Add semantic context if available
                        if 'sentiment_score' in tx and tx['sentiment_score'] < -0.3:
                            explanation += "Transaction has negative sentiment indicators. "
                        
                        anomalies.append({
                            'type': 'ml_anomaly',
                            'transaction_id': tx.get('id', None),
                            'date': tx.get('date', None),
                            'amount': tx.get('amount', 0),
                            'description': tx.get('description', ''),
                            'category': tx.get('category', ''),
                            'anomaly_score': float(anomaly_score),
                            'explanation': explanation
                        })
                except Exception as e:
                    logger.warning(f"Error in ML anomaly detection: {e}. Falling back to statistical approach.")
            
            # Statistical approach (as fallback or supplement)
            # Calculate statistics for anomaly detection
            mean_amount = expenses['amount'].mean()
            std_amount = expenses['amount'].std()

            # Detect large transactions (more than 2 standard deviations above mean)
            threshold = mean_amount + (2 * std_amount)
            large_transactions = expenses[expenses['amount'] > threshold]

            for _, tx in large_transactions.iterrows():
                # Skip if already detected by ML
                if any(a.get('transaction_id') == tx.get('id') for a in anomalies if 'transaction_id' in a):
                    continue
                    
                deviation_pct = ((tx['amount'] - mean_amount) / mean_amount) * 100
                
                anomalies.append({
                    'type': 'large_transaction',
                    'transaction_id': tx.get('id', None),
                    'date': tx.get('date', None),
                    'amount': tx.get('amount', 0),
                    'description': tx.get('description', ''),
                    'category': tx.get('category', ''),
                    'threshold': float(threshold),
                    'deviation_percent': float(deviation_pct),
                    'explanation': f"Transaction amount (€{tx['amount']:.2f}) is {deviation_pct:.1f}% higher than average (€{mean_amount:.2f})"
                })

            # Detect category anomalies
            if 'category' in expenses.columns and 'date' in expenses.columns:
                expenses['month'] = expenses['date'].dt.strftime('%Y-%m')

                # Get the last two months if available
                months = sorted(expenses['month'].unique())
                if len(months) >= 2:
                    current_month = months[-1]
                    previous_month = months[-2]

                    # Calculate spending by category for last two months
                    current = expenses[expenses['month'] == current_month].groupby('category')['amount'].sum()
                    previous = expenses[expenses['month'] == previous_month].groupby('category')['amount'].sum()

                    # Find categories with significant increases
                    for category in current.index:
                        if category in previous.index:
                            current_amount = current[category]
                            previous_amount = previous[category]

                            if previous_amount > 0 and current_amount > previous_amount * 1.5:  # 50% increase
                                increase_pct = ((current_amount - previous_amount) / previous_amount) * 100
                                
                                anomalies.append({
                                    'type': 'category_increase',
                                    'category': category,
                                    'current_month': current_month,
                                    'current_amount': float(current_amount),
                                    'previous_month': previous_month,
                                    'previous_amount': float(previous_amount),
                                    'increase_percent': float(increase_pct),
                                    'explanation': f"Spending in {category} increased by {increase_pct:.1f}% from {previous_month} to {current_month}"
                                })
            
            # Detect frequency anomalies (sudden changes in transaction frequency)
            if 'date' in expenses.columns and len(expenses) >= 10:
                # Count transactions by day
                expenses['date_only'] = expenses['date'].dt.date
                daily_counts = expenses['date_only'].value_counts().sort_index()
                
                if len(daily_counts) >= 5:
                    # Get average daily count and std dev
                    avg_count = daily_counts.mean()
                    std_count = daily_counts.std()
                    
                    # Find days with unusually high transaction counts
                    high_days = daily_counts[daily_counts > avg_count + (2 * std_count)]
                    
                    for day, count in high_days.items():
                        day_txs = expenses[expenses['date_only'] == day]
                        
                        anomalies.append({
                            'type': 'frequency_anomaly',
                            'date': day,
                            'transaction_count': int(count),
                            'average_count': float(avg_count),
                            'deviation': float((count - avg_count) / std_count),
                            'explanation': f"Unusual number of transactions ({count}) on {day} compared to average ({avg_count:.1f})",
                            'categories': day_txs['category'].value_counts().to_dict() if 'category' in day_txs.columns else {}
                        })

        return anomalies

    def _generate_insights(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate financial insights and recommendations.

        Args:
            df: DataFrame with transaction data

        Returns:
            List of insights and recommendations
        """
        insights = []

        if df.empty:
            insights.append({
                'type': 'general',
                'title': 'Start Tracking Your Expenses',
                'description': 'Add more transactions to get personalized financial insights and recommendations.'
            })
            return insights

        # Filter expenses and income
        expenses = df[~df['is_income']] if 'is_income' in df.columns else df
        income = df[df['is_income']] if 'is_income' in df.columns else pd.DataFrame()

        # Check if we have enough data
        if len(df) < 10:
            insights.append({
                'type': 'general',
                'title': 'Add More Transaction Data',
                'description': 'We need more transaction data to provide meaningful insights. Try adding at least a month of transactions.'
            })
            return insights

        # Insight 1: Savings rate
        if not income.empty and not expenses.empty and 'date' in df.columns:
            df['month'] = df['date'].dt.strftime('%Y-%m')
            months = df['month'].unique()

            for month in months:
                month_income = income[income['month'] == month]['amount'].sum()
                month_expenses = expenses[expenses['month'] == month]['amount'].sum()

                if month_income > 0:
                    savings_rate = ((month_income - month_expenses) / month_income) * 100

                    if savings_rate < 0:
                        insights.append({
                            'type': 'savings_alert',
                            'title': 'Negative Savings Rate',
                            'description': f'You spent more than you earned in {month}, resulting in a negative savings rate of {savings_rate:.1f}%.',
                            'recommendation': 'Review your expenses to identify areas where you can cut back.'
                        })
                    elif savings_rate < 20:
                        insights.append({
                            'type': 'savings_alert',
                            'title': 'Low Savings Rate',
                            'description': f'Your savings rate in {month} was {savings_rate:.1f}%, which is below the recommended 20%.',
                            'recommendation': 'Try to increase your savings rate to build financial security.'
                        })
                    elif savings_rate > 50:
                        insights.append({
                            'type': 'savings_positive',
                            'title': 'Excellent Savings Rate',
                            'description': f'Your savings rate in {month} was {savings_rate:.1f}%, which is excellent!',
                            'recommendation': 'Consider investing some of your savings for long-term growth.'
                        })

        # Insight 2: Top expense categories
        if 'category' in expenses.columns and not expenses.empty:
            category_totals = expenses.groupby('category')['amount'].sum()
            total_expense = category_totals.sum()

            if total_expense > 0:
                for category, amount in category_totals.items():
                    percentage = (amount / total_expense) * 100

                    if percentage > 40:
                        insights.append({
                            'type': 'category_alert',
                            'title': f'High {category} Spending',
                            'description': f'Your {category} expenses account for {percentage:.1f}% of your total spending.',
                            'recommendation': f'Look for ways to reduce your {category} expenses.'
                        })

        # Insight 3: Recurring subscriptions
        if 'category' in expenses.columns and 'description' in expenses.columns:
            subscriptions = expenses[expenses['category'] == 'Subscriptions']

            if not subscriptions.empty:
                subscription_total = subscriptions['amount'].sum()

                if subscription_total > 0 and total_expense > 0:
                    subscription_percentage = (subscription_total / total_expense) * 100

                    if subscription_percentage > 10:
                        insights.append({
                            'type': 'subscription_alert',
                            'title': 'High Subscription Costs',
                            'description': f'You spend {subscription_percentage:.1f}% of your budget on subscriptions.',
                            'recommendation': 'Review your subscriptions and cancel those you don\'t use regularly.'
                        })

        # Insight 4: Student-specific tips
        insights.append({
            'type': 'student_tip',
            'title': 'Student Discount Opportunities',
            'description': 'As an international student in Germany, you can access many discounts.',
            'recommendation': 'Check for student discounts on public transportation, cultural events, and software.'
        })

        insights.append({
            'type': 'student_tip',
            'title': 'Budget-Friendly Grocery Shopping',
            'description': 'German discount supermarkets offer excellent value for students.',
            'recommendation': 'Shop at stores like Aldi, Lidl, and Netto to save on groceries.'
        })

        # Add a few general financial tips
        general_tips = [
            {
                'type': 'general_tip',
                'title': 'Emergency Fund',
                'description': 'An emergency fund is essential for financial security.',
                'recommendation': 'Aim to save 3-6 months of expenses in an easily accessible account.'
            },
            {
                'type': 'general_tip',
                'title': 'The 50/30/20 Rule',
                'description': 'A simple budgeting guideline for financial health.',
                'recommendation': 'Allocate 50% for needs, 30% for wants, and 20% for savings and debt repayment.'
            },
            {
                'type': 'general_tip',
                'title': 'Track Every Euro',
                'description': 'Awareness is the first step to financial improvement.',
                'recommendation': 'Continue tracking all your expenses to identify saving opportunities.'
            }
        ]

        # Add some general tips
        insights.extend(general_tips)

        return insights

    def _load_model_state(self):
        """Load saved model state from database if available."""
        if not self.db_manager:
            return False
            
        try:
            # Check if we have a model state table
            query = """
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='ai_model_state'
            """
            result = self.db_manager.execute_query(query)
            
            if not result:
                logger.info("No model state found in database")
                return False
                
            # Load model state metadata
            query = """
            SELECT model_version, trained_timestamp, metrics 
            FROM ai_model_state 
            ORDER BY model_version DESC LIMIT 1
            """
            state_metadata = self.db_manager.execute_query(query)
            
            if not state_metadata:
                logger.info("No model state records found")
                return False
                
            # Load model data
            query = """
            SELECT transaction_text, category, sentiment, embedding
            FROM ai_training_data
            """
            training_data = self.db_manager.execute_query(query)
            
            if not training_data:
                logger.info("No training data found")
                return False
                
            # Populate our model state
            self.model_version = state_metadata[0][0]
            self.last_trained_timestamp = state_metadata[0][1]
            
            # Load the known transactions
            for tx_text, category, sentiment, embedding in training_data:
                self.known_transactions.append({
                    'text': tx_text,
                    'category': category,
                    'sentiment': sentiment
                })
                
                if tx_text and category:
                    self.category_corrections[tx_text] = category
                    
                if tx_text and sentiment:
                    self.sentiment_labels[tx_text] = sentiment
            
            # Set trained flag
            self.is_trained = True
            logger.info(f"Loaded model state version {self.model_version} with {len(self.known_transactions)} training examples")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model state: {e}")
            return False
            
    def _save_model_state(self):
        """Save the current model state to the database."""
        if not self.db_manager:
            return False
            
        try:
            # Create model state tables if they don't exist
            self.db_manager.execute_query("""
            CREATE TABLE IF NOT EXISTS ai_model_state (
                model_version INTEGER PRIMARY KEY,
                trained_timestamp TEXT,
                metrics TEXT
            )
            """)
            
            self.db_manager.execute_query("""
            CREATE TABLE IF NOT EXISTS ai_training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_text TEXT,
                category TEXT,
                sentiment TEXT,
                embedding TEXT,
                added_timestamp TEXT
            )
            """)
            
            # Increment model version
            self.model_version += 1
            
            # Get current timestamp
            current_time = datetime.datetime.now().isoformat()
            
            # Create metrics JSON
            metrics = {
                'known_transactions': len(self.known_transactions),
                'category_corrections': len(self.category_corrections),
                'sentiment_labels': len(self.sentiment_labels)
            }
            
            # Insert new model state record
            self.db_manager.execute_query("""
            INSERT INTO ai_model_state (model_version, trained_timestamp, metrics)
            VALUES (?, ?, ?)
            """, (self.model_version, current_time, str(metrics)))
            
            # Clear old training data
            self.db_manager.execute_query("DELETE FROM ai_training_data")
            
            # Insert all known transactions
            for tx in self.known_transactions:
                self.db_manager.execute_query("""
                INSERT INTO ai_training_data 
                (transaction_text, category, sentiment, embedding, added_timestamp)
                VALUES (?, ?, ?, ?, ?)
                """, (
                    tx.get('text', ''),
                    tx.get('category', ''),
                    tx.get('sentiment', ''),
                    '', # We don't store actual embeddings in DB for now
                    current_time
                ))
                
            self.last_trained_timestamp = current_time
            logger.info(f"Saved model state version {self.model_version} with {len(self.known_transactions)} examples")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model state: {e}")
            return False
    
    def auto_train_from_historical_data(self):
        """Automatically train the model from historical transaction data."""
        if not self.db_manager or not self.ml_enabled:
            return False
            
        try:
            # Get all historical transactions
            query = """
            SELECT id, date, description, merchant, amount, category, is_income
            FROM transactions
            ORDER BY date
            """
            transactions = self.db_manager.execute_query(query)
            
            if not transactions:
                logger.info("No historical transactions found for training")
                return False
                
            # Convert to list of dictionaries
            tx_list = []
            for tx in transactions:
                tx_dict = {
                    'id': tx[0],
                    'date': tx[1],
                    'description': tx[2],
                    'merchant': tx[3],
                    'amount': tx[4],
                    'category': tx[5],
                    'is_income': tx[6]
                }
                tx_list.append(tx_dict)
                
            # Skip if we don't have enough data
            if len(tx_list) < 10:
                logger.info(f"Not enough transactions for training ({len(tx_list)} found, need at least 10)")
                return False
                
            # Train the model
            result = self.train_from_transactions(tx_list)
            
            if result:
                logger.info(f"Successfully auto-trained model from {len(tx_list)} historical transactions")
                return True
            else:
                logger.warning("Failed to auto-train model")
                return False
                
        except Exception as e:
            logger.error(f"Error in auto-training: {e}")
            return False
    
    def train_from_transactions(self, transactions: List[Dict[str, Any]]) -> bool:
        """
        Train the ML models from a list of transactions.
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            Success flag
        """
        if not self.ml_enabled or not transactions:
            return False
            
        try:
            # Convert to DataFrame
            df = pd.DataFrame(transactions)
            
            # Skip if important columns are missing
            if 'description' not in df.columns or 'amount' not in df.columns:
                logger.warning("Missing required columns for training")
                return False
                
            # Preprocess text data
            descriptions = []
            for _, tx in df.iterrows():
                desc = tx.get('description', '')
                merchant = tx.get('merchant', '')
                
                # Combine description and merchant
                text = f"{desc} {merchant}".strip()
                
                if text:
                    # Preprocess
                    processed_text = self._preprocess_text(text)
                    descriptions.append(processed_text)
                    
                    # Add to known transactions
                    if processed_text not in [t.get('text') for t in self.known_transactions]:
                        new_tx = {
                            'text': processed_text,
                            'amount': tx.get('amount', 0),
                            'category': tx.get('category', '')
                        }
                        self.known_transactions.append(new_tx)
                        
                        # Add category correction if available
                        if tx.get('category'):
                            self.category_corrections[processed_text] = tx.get('category')
            
            # If we have enough data, train the vectorizer
            if len(descriptions) >= 10:
                self.description_vectorizer.fit(descriptions)
                logger.info(f"Trained vectorizer on {len(descriptions)} descriptions")
                
            # Train the anomaly detector if we have enough transactions
            if len(df) >= 20 and 'amount' in df.columns:
                # Prepare features for anomaly detection
                features = ['amount']
                
                # Add day of week and month if date is available
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df['day_of_week'] = df['date'].dt.dayofweek
                    df['day_of_month'] = df['date'].dt.day
                    df['month'] = df['date'].dt.month
                    
                    features.extend(['day_of_week', 'day_of_month', 'month'])
                
                # Train the anomaly detector
                X = df[features].fillna(0).values
                X_scaled = self.scaler.fit_transform(X)
                self.anomaly_detector.fit(X_scaled)
                logger.info(f"Trained anomaly detector on {len(df)} transactions")
            
            # Set trained flag and save model state
            self.is_trained = True
            self._save_model_state()
            
            return True
            
        except Exception as e:
            logger.error(f"Error training from transactions: {e}")
            return False
    
    def learn_from_user_feedback(self, transaction_text: str, correct_category: str, 
                               sentiment: Optional[str] = None) -> bool:
        """
        Learn from user feedback on transaction categorization.
        
        Args:
            transaction_text: The transaction description text
            correct_category: The correct category provided by user
            sentiment: Optional sentiment label
            
        Returns:
            Success flag
        """
        if not transaction_text or not correct_category:
            return False
            
        try:
            # Preprocess the text
            processed_text = self._preprocess_text(transaction_text)
            
            # Update category correction
            self.category_corrections[processed_text] = correct_category
            
            # Update sentiment if provided
            if sentiment:
                self.sentiment_labels[processed_text] = sentiment
            
            # Check if we already have this transaction
            for tx in self.known_transactions:
                if tx.get('text') == processed_text:
                    # Update existing transaction
                    tx['category'] = correct_category
                    if sentiment:
                        tx['sentiment'] = sentiment
                    break
            else:
                # Add new transaction
                self.known_transactions.append({
                    'text': processed_text,
                    'category': correct_category,
                    'sentiment': sentiment
                })
            
            # Save model state
            self._save_model_state()
            
            logger.info(f"Learned new category correction: '{transaction_text}' -> {correct_category}")
            return True
            
        except Exception as e:
            logger.error(f"Error learning from feedback: {e}")
            return False
    
    def generate_monthly_budget(self, transactions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Generate a recommended monthly budget based on previous spending patterns.
        Enhanced with ML capabilities and user-specific patterns.

        Args:
            transactions: List of transaction dictionaries

        Returns:
            Dictionary mapping categories to recommended budget amounts
        """
        if not transactions:
            return {}

        # Convert to DataFrame
        df = pd.DataFrame(transactions)

        # Ensure required columns exist
        if 'date' not in df.columns or 'category' not in df.columns or 'amount' not in df.columns:
            return {}

        # Filter expenses
        expenses = df[~df['is_income']] if 'is_income' in df.columns else df

        if expenses.empty:
            return {}

        # Convert date to datetime
        expenses['date'] = pd.to_datetime(expenses['date'])

        # Get data from last 3 months if available
        today = datetime.datetime.now()
        three_months_ago = today - datetime.timedelta(days=90)
        recent_expenses = expenses[expenses['date'] >= three_months_ago]

        # If not enough recent data, use all data
        if len(recent_expenses) < 10:
            recent_expenses = expenses

        # Calculate average monthly spending by category
        recent_expenses['month'] = recent_expenses['date'].dt.strftime('%Y-%m')
        monthly_by_category = recent_expenses.pivot_table(
            index='month',
            columns='category',
            values='amount',
            aggfunc='sum',
            fill_value=0
        )

        # Calculate average
        category_averages = monthly_by_category.mean()
        
        # Get spending trends for more intelligent recommendations
        category_trends = {}
        
        if len(monthly_by_category) >= 2:
            # Calculate month-over-month trends
            months = sorted(monthly_by_category.index)
            for category in category_averages.index:
                if category in monthly_by_category.columns:
                    values = monthly_by_category[category].values
                    if len(values) >= 2:
                        # Calculate average change
                        changes = []
                        for i in range(1, len(values)):
                            if values[i-1] > 0:
                                percent_change = (values[i] - values[i-1]) / values[i-1]
                                changes.append(percent_change)
                                
                        if changes:
                            avg_change = sum(changes) / len(changes)
                            category_trends[category] = avg_change

        # Create recommended budget with intelligent adjustments
        recommended_budget = {}

        for category, avg_amount in category_averages.items():
            # Dynamic adjustment based on spending trends
            trend_adjustment = 0
            if category in category_trends:
                # If spending is increasing, recommend tighter budget
                if category_trends[category] > 0.05:  # >5% increase trend
                    trend_adjustment = -0.05  # 5% reduction from trend
                # If spending is decreasing, support the trend
                elif category_trends[category] < -0.05:  # >5% decrease trend
                    trend_adjustment = -0.02  # 2% additional reduction to support good behavior
            
            # Apply category-specific logic and trend adjustments
            if category in ['Entertainment', 'Shopping', 'Dining Out', 'Subscriptions']:
                # Discretionary categories - more aggressive optimization
                recommended_budget[category] = avg_amount * (0.9 + trend_adjustment)
            elif category in ['Housing', 'Utilities', 'Healthcare', 'Transportation', 'Groceries']:
                # Essential categories - maintain levels but apply minimal trend adjustment
                recommended_budget[category] = avg_amount * (1.0 + trend_adjustment/2)
            else:
                # Other categories - moderate optimization
                recommended_budget[category] = avg_amount * (0.95 + trend_adjustment)
        
        # Apply user-specific patterns if model is trained
        if self.is_trained and self.ml_enabled:
            # Identify categories user has been consistent with
            consistent_categories = self._identify_consistent_categories(recent_expenses)
            
            # For consistent categories, stick closer to user's actual spending
            for category in consistent_categories:
                if category in recommended_budget:
                    current = recommended_budget[category]
                    # Adjust recommended budget to be closer to actual spending for consistent categories
                    recommended_budget[category] = current * 0.9 + category_averages[category] * 0.1

        return recommended_budget
        
    def _identify_consistent_categories(self, expenses_df: pd.DataFrame) -> List[str]:
        """Identify categories where user consistently spends similar amounts."""
        consistent_categories = []
        
        if 'category' not in expenses_df.columns or 'amount' not in expenses_df.columns:
            return consistent_categories
            
        # Group by category
        category_groups = expenses_df.groupby('category')
        
        for category, group in category_groups:
            if len(group) >= 3:  # Need at least 3 transactions to determine consistency
                amounts = group['amount'].values
                
                # Calculate coefficient of variation (normalized measure of dispersion)
                mean = np.mean(amounts)
                std = np.std(amounts)
                
                if mean > 0:
                    cv = std / mean
                    
                    # If coefficient of variation is low (consistent spending)
                    if cv < 0.2:  # 20% variation threshold
                        consistent_categories.append(category)
        
        return consistent_categories

    def get_semantic_insights(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate advanced semantic insights from transaction descriptions using NLP.
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            List of semantic insights
        """
        if not self.nlp_enabled or not transactions:
            return []
            
        try:
            # Convert to DataFrame
            df = pd.DataFrame(transactions)
            
            # Preprocess and extract features
            df = self._extract_semantic_features(df)
            
            insights = []
            
            # Analyze sentiment patterns
            if 'sentiment_score' in df.columns:
                # Get transactions with strongly negative sentiment
                negative_txs = df[df['sentiment_score'] < -0.3]
                if len(negative_txs) >= 2:
                    negative_categories = negative_txs['category'].value_counts().to_dict() if 'category' in negative_txs.columns else {}
                    
                    insights.append({
                        'type': 'sentiment_insight',
                        'title': 'Negative Financial Sentiment',
                        'description': f"Detected {len(negative_txs)} transactions with negative financial implications.",
                        'details': negative_categories,
                        'recommendation': "Review these transactions for potential issues or unnecessary fees."
                    })
                
                # Get transactions with strongly positive sentiment
                positive_txs = df[df['sentiment_score'] > 0.3]
                if len(positive_txs) >= 2:
                    positive_categories = positive_txs['category'].value_counts().to_dict() if 'category' in positive_txs.columns else {}
                    
                    insights.append({
                        'type': 'sentiment_insight',
                        'title': 'Positive Financial Decisions',
                        'description': f"Found {len(positive_txs)} transactions with positive financial indicators.",
                        'details': positive_categories,
                        'recommendation': "Continue these positive financial habits."
                    })
            
            # Analyze semantic category distribution
            semantic_cols = [col for col in df.columns if col.startswith('sem_')]
            if semantic_cols:
                # Get dominant semantic category
                if 'dominant_semantic' in df.columns:
                    dom_semantic = df['dominant_semantic'].value_counts().to_dict()
                    top_semantic = max(dom_semantic.items(), key=lambda x: x[1]) if dom_semantic else (None, 0)
                    
                    if top_semantic[0] and top_semantic[1] > len(df) * 0.25:  # If it represents more than 25% of transactions
                        insights.append({
                            'type': 'semantic_focus',
                            'title': f"Focus on {top_semantic[0].title()} Spending",
                            'description': f"{top_semantic[1]} transactions ({(top_semantic[1]/len(df))*100:.1f}%) are related to {top_semantic[0]}.",
                            'recommendation': self._get_recommendation_for_semantic(top_semantic[0])
                        })
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating semantic insights: {e}")
            return []
            
    def _get_recommendation_for_semantic(self, semantic_category: str) -> str:
        """Generate specific recommendations based on semantic category."""
        recommendations = {
            'essentials': "Your spending is focused on essentials, which is financially responsible. Look for opportunities to optimize these necessary expenses.",
            'transportation': "Consider whether there are more cost-effective transportation options available to you.",
            'dining': "Eating out contributes significantly to your expenses. Try meal planning to reduce food costs.",
            'entertainment': "Look for free or low-cost entertainment alternatives to reduce discretionary spending.",
            'shopping': "Consider implementing a waiting period (e.g., 48 hours) before making non-essential purchases.",
            'education': "Educational spending is an investment in your future. Look for scholarships or student discounts.",
            'debt': "Prioritize paying off high-interest debt first to reduce interest payments."
        }
        
        return recommendations.get(semantic_category, "Review your spending patterns in this area for potential savings.")
    
    def get_personalized_recommendation(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get detailed personalized financial recommendations based on transaction data.

        Args:
            transactions: List of transaction dictionaries

        Returns:
            Dictionary with personalized recommendations and confidence scores
        """
        if not transactions:
            return {
                "primary_recommendation": "Start tracking your expenses regularly to receive personalized financial insights.",
                "confidence": 0.5,
                "supporting_insights": [],
                "action_items": ["Add more transactions to get started"]
            }

        # Analyze spending patterns with advanced features
        analysis = self.analyze_spending_patterns(transactions)

        # Get insights from analysis
        insights = analysis['insights']
        semantic_insights = self.get_semantic_insights(transactions) if self.nlp_enabled else []
        anomalies = analysis['anomalies']
        
        # Combine all insights
        all_insights = insights + semantic_insights
        
        # Generate action items based on all available data
        action_items = []
        key_findings = []
        
        # Process anomalies
        if anomalies:
            action_items.append(f"Review {len(anomalies)} unusual transactions identified in your spending")
            key_findings.append({
                "type": "anomaly_alert",
                "title": "Unusual Transaction Patterns",
                "description": f"Detected {len(anomalies)} transactions that don't match your typical spending patterns."
            })
        
        # Process forecast if available
        if 'forecast' in analysis and analysis['forecast'] and 'forecasts' in analysis['forecast']:
            forecasts = analysis['forecast']['forecasts']
            
            if 'savings' in forecasts and forecasts['savings']:
                next_month_savings = forecasts['savings'][0]['amount']
                if next_month_savings < 0:
                    action_items.append("Your projected savings for next month are negative. Review your budget now.")
                    key_findings.append({
                        "type": "forecast_alert",
                        "title": "Negative Savings Forecast",
                        "description": f"You are projected to spend more than you earn next month by {abs(next_month_savings):.2f}."
                    })
        
        # Process semantic insights
        for insight in semantic_insights:
            if 'recommendation' in insight:
                action_items.append(insight['recommendation'])
            
            key_findings.append({
                "type": insight.get('type', 'semantic_insight'),
                "title": insight.get('title', 'Semantic Pattern'),
                "description": insight.get('description', '')
            })
        
        # Process regular insights
        for insight in insights:
            if insight.get('type', '').endswith('_alert'):
                action_items.append(insight.get('recommendation', ''))
                
                key_findings.append({
                    "type": insight.get('type', 'insight'),
                    "title": insight.get('title', ''),
                    "description": insight.get('description', '')
                })
        
        # Choose primary recommendation
        primary_rec = "Track your expenses regularly to receive more personalized financial insights."
        confidence = 0.5
        
        # Prioritize recommendations from alerts
        if action_items:
            primary_rec = action_items[0]
            confidence = 0.8
            
            # If we have ML-based anomalies or forecasts, increase confidence
            if any(a.get('type') == 'ml_anomaly' for a in anomalies) or 'forecast' in analysis and analysis['forecast']:
                confidence = 0.9
        
        # If we have semantic insights, provide more personalized recommendation
        elif semantic_insights:
            primary_rec = semantic_insights[0].get('recommendation', primary_rec)
            confidence = 0.75
        # Otherwise, fall back to a general insight if available
        elif insights:
            import random
            tip = random.choice(insights)
            primary_rec = f"{tip.get('recommendation', '')}"
            confidence = 0.7
        
        return {
            "primary_recommendation": primary_rec,
            "confidence": confidence,
            "supporting_insights": key_findings[:5],  # Limit to top 5 insights
            "action_items": list(set(action_items))[:3]  # Limit to top 3 unique actions
        }


# Example usage
if __name__ == "__main__":
    financial_ai = FinancialAI()

    # Example transactions
    transactions = [
        {
            "date": "2025-01-05",
            "description": "REWE SAGT DANKE. 58652545",
            "amount": 25.67,
            "is_income": False,
            "category": "Food"
        },
        {
            "date": "2025-01-10",
            "description": "Studierendenwerk Münster Miete",
            "amount": 346.84,
            "is_income": False,
            "category": "Housing"
        },
        {
            "date": "2025-01-15",
            "description": "PICNIC GMBH LOHN / GEHALT",
            "amount": 813.13,
            "is_income": True,
            "category": "Income"
        }
    ]

    # Get a recommendation
    recommendation = financial_ai.get_personalized_recommendation(transactions)
    print(recommendation)