#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Budget Optimizer with ML/NLP Capabilities

This module provides functionality to analyze spending patterns and
generate optimized budget recommendations for the Financial Planner application
using machine learning and natural language processing techniques.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import defaultdict
import datetime
import logging
import re
import json
import os

# Set up logging
logger = logging.getLogger(__name__)

# Advanced NLP capabilities with fallbacks
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    
    ML_AVAILABLE = True
    logger.info("ML components successfully loaded for budget optimization")
except ImportError:
    ML_AVAILABLE = False
    logger.warning("Scikit-learn not available. Basic budget optimization will be used.")


class BudgetOptimizer:
    """
    Analyzes spending patterns and recommends optimized budgets using ML/NLP.
    
    Features:
    - 50/30/20 rule implementation with adaptive learning
    - Personalized budget recommendations based on spending patterns
    - Machine learning for category spending prediction
    - Anomaly detection for unusual spending patterns
    - Budget adherence tracking and intelligent adjustments
    - Smart savings goal setting based on financial capacity
    - Self-learning from user behavior and adjustments
    """

    # Default budget allocation percentages based on 50/30/20 rule
    DEFAULT_ALLOCATION = {
        'Needs': 0.5,  # 50% for needs
        'Wants': 0.3,  # 30% for wants
        'Savings': 0.2  # 20% for savings
    }

    # Category classifications
    CATEGORY_TYPES = {
        'Needs': [
            'Housing',
            'Utilities',
            'Groceries',
            'Transportation',
            'Healthcare',
            'Telecommunications',
            'Insurance',
            'Education',
            'Debt'
        ],
        'Wants': [
            'Food',  # Restaurants, dining out
            'Shopping',
            'Entertainment',
            'Subscriptions',
            'Travel',
            'Fitness',
            'Personal Care'
        ],
        'Savings': [
            'Savings',
            'Investments',
            'Emergency Fund'
        ]
    }
    
    # Budget adherence thresholds
    ADHERENCE_THRESHOLDS = {
        'excellent': 0.05,  # Within 5% of budget
        'good': 0.10,       # Within 10% of budget
        'fair': 0.20,       # Within 20% of budget
        'poor': 0.30,       # Within 30% of budget
        'very_poor': float('inf')  # More than 30% off budget
    }
    
    # Model training parameters
    MODEL_PARAMS = {
        'min_training_samples': 10,
        'kmeans_clusters': 5,
        'prediction_horizon': 3,  # months ahead to predict
        'spending_anomaly_threshold': 2.0  # standard deviations
    }

    def __init__(self, db_manager=None):
        """
        Initialize the budget optimizer with ML capabilities.
        
        Args:
            db_manager: Optional database manager for persisting models and preferences
        """
        self.logger = logger
        self.db_manager = db_manager
        
        # Initialize ML components
        self.ml_enabled = ML_AVAILABLE
        self.models = {}
        self.category_predictors = {}
        self.user_preferences = {}
        self.training_data = {}
        
        # Set default user preferences
        self._initialize_user_preferences()
        
        # Load saved models and preferences if available
        self._load_saved_state()
        
        self.logger.info("Budget optimizer initialized with ML enabled: %s", self.ml_enabled)
    
    def _initialize_user_preferences(self):
        """Initialize default user budget preferences."""
        self.user_preferences = {
            'allocation_rule': '50/30/20',  # Default budget rule
            'savings_goal': 0.0,            # Monthly savings goal
            'max_category_percent': 0.3,    # No category should exceed 30% of budget
            'budget_period': 'monthly',     # Default budget period
            'custom_allocation': None,      # Custom allocation percentages
            'priority_categories': [],      # Categories to prioritize
            'exclude_categories': [],       # Categories to exclude from analysis
            'learning_rate': 0.1,           # How quickly to adapt to user behavior
            'risk_tolerance': 'medium',     # User's risk tolerance level
            'auto_adjust': True,            # Whether to automatically adjust budgets
            'favorite_categories': {}       # Categories the user values most
        }
        
    def _load_saved_state(self):
        """Load saved models, training data, and user preferences."""
        if not self.db_manager:
            return
            
        try:
            # Load user preferences
            preferences = self.db_manager.get_budget_preferences()
            if preferences:
                self.user_preferences.update(preferences)
                self.logger.info("Loaded user budget preferences")
                
            # Load training data
            training_data = self.db_manager.get_budget_training_data()
            if training_data:
                self.training_data = training_data
                self.logger.info("Loaded budget training data with %d samples", 
                                len(training_data.get('spending_history', [])))
                
            # Load ML models if ML is available
            if self.ml_enabled:
                model_data = self.db_manager.get_budget_models()
                if model_data:
                    self._deserialize_models(model_data)
                    self.logger.info("Loaded budget optimization models")
        except Exception as e:
            self.logger.error("Error loading budget optimization state: %s", str(e))
            
    def _save_state(self):
        """Save current models, training data, and preferences."""
        if not self.db_manager:
            return
            
        try:
            # Save user preferences
            self.db_manager.save_budget_preferences(self.user_preferences)
            
            # Save training data
            self.db_manager.save_budget_training_data(self.training_data)
            
            # Save ML models if available
            if self.ml_enabled and self.models:
                model_data = self._serialize_models()
                self.db_manager.save_budget_models(model_data)
                
            self.logger.info("Saved budget optimization state successfully")
        except Exception as e:
            self.logger.error("Error saving budget optimization state: %s", str(e))
            
    def _serialize_models(self):
        """Serialize ML models for storage."""
        if not self.ml_enabled:
            return None
            
        try:
            import pickle
            import base64
            
            serialized_models = {}
            
            # Serialize each model
            for model_name, model in self.models.items():
                model_bytes = pickle.dumps(model)
                serialized_models[model_name] = base64.b64encode(model_bytes).decode('utf-8')
                
            # Serialize category predictors
            for category, predictor in self.category_predictors.items():
                model_bytes = pickle.dumps(predictor)
                serialized_models[f"category_{category}"] = base64.b64encode(model_bytes).decode('utf-8')
                
            return serialized_models
        except Exception as e:
            self.logger.error("Error serializing models: %s", str(e))
            return None
            
    def _deserialize_models(self, model_data):
        """Deserialize ML models from storage."""
        if not self.ml_enabled or not model_data:
            return
            
        try:
            import pickle
            import base64
            
            # Deserialize main models
            for model_name, serialized_model in model_data.items():
                if model_name.startswith("category_"):
                    # Handle category predictors
                    category = model_name[9:]  # Remove "category_" prefix
                    model_bytes = base64.b64decode(serialized_model)
                    self.category_predictors[category] = pickle.loads(model_bytes)
                else:
                    # Handle main models
                    model_bytes = base64.b64decode(serialized_model)
                    self.models[model_name] = pickle.loads(model_bytes)
        except Exception as e:
            self.logger.error("Error deserializing models: %s", str(e))
            # Clear partial models to avoid inconsistent state
            self.models = {}
            self.category_predictors = {}

    def analyze_spending(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze spending patterns from transaction data with ML-enhanced insights.

        Args:
            transactions: List of transaction dictionaries

        Returns:
            Dictionary with spending analysis and ML insights
        """
        if not transactions:
            return {
                'total_income': 0,
                'total_expense': 0,
                'savings_rate': 0,
                'category_spending': {},
                'category_percentages': {},
                'needs_wants_savings': {
                    'Needs': 0,
                    'Wants': 0,
                    'Savings': 0
                },
                'insights': [],
                'anomalies': [],
                'forecast': {}
            }

        # Convert to DataFrame
        df = pd.DataFrame(transactions)

        # Separate income and expenses
        income_df = df[df['is_income']] if 'is_income' in df.columns else pd.DataFrame()
        expense_df = df[~df['is_income']] if 'is_income' in df.columns else df

        # Calculate totals
        total_income = income_df['amount'].sum() if not income_df.empty else 0
        total_expense = expense_df['amount'].sum() if not expense_df.empty else 0

        # Calculate savings rate
        savings_rate = ((total_income - total_expense) / total_income * 100) if total_income > 0 else 0

        # Calculate spending by category
        category_spending = defaultdict(float)
        category_field = 'category' if 'category' in expense_df.columns else 'category_name' if 'category_name' in expense_df.columns else None

        if category_field and not expense_df.empty:
            for _, row in expense_df.iterrows():
                category = row.get(category_field, 'Uncategorized')
                category_spending[category] += row['amount']

        # Calculate category percentages
        category_percentages = {}
        for category, amount in category_spending.items():
            category_percentages[category] = (amount / total_expense * 100) if total_expense > 0 else 0

        # Calculate spending in needs, wants, savings
        needs_wants_savings = {
            'Needs': 0,
            'Wants': 0,
            'Savings': 0
        }

        # Classify each category
        category_classifications = {}
        for category, amount in category_spending.items():
            classified = False
            for budget_type, categories in self.CATEGORY_TYPES.items():
                if category in categories:
                    needs_wants_savings[budget_type] += amount
                    category_classifications[category] = budget_type
                    classified = True
                    break

            # Use ML to classify unknown categories if available
            if not classified and self.ml_enabled and len(self.training_data.get('category_mappings', {})) > 0:
                predicted_type = self._predict_category_type(category)
                if predicted_type:
                    needs_wants_savings[predicted_type] += amount
                    category_classifications[category] = predicted_type
                    classified = True

            # If category still not classified, assign to Wants (default)
            if not classified:
                needs_wants_savings['Wants'] += amount
                category_classifications[category] = 'Wants'

        # Calculate percentages for needs, wants, savings
        needs_wants_savings_pct = {}
        for budget_type, amount in needs_wants_savings.items():
            needs_wants_savings_pct[budget_type] = (amount / total_expense * 100) if total_expense > 0 else 0

        # Generate ML insights if enabled
        insights = []
        anomalies = []
        forecast = {}
        
        if self.ml_enabled:
            # Add this transaction data to training data
            self._update_training_data(transactions)
            
            # Generate insights
            insights = self._generate_insights(transactions, category_spending, needs_wants_savings_pct)
            
            # Detect spending anomalies
            anomalies = self._detect_spending_anomalies(transactions, category_spending)
            
            # Generate spending forecast
            forecast = self._generate_spending_forecast(transactions, category_spending)
        
        # Update ML models periodically if enough data
        if self.ml_enabled and len(self.training_data.get('spending_history', [])) >= self.MODEL_PARAMS['min_training_samples']:
            self._train_models()
            
        # Store user's spending patterns for future reference
        if 'date' in df.columns:
            self._record_spending_patterns(df, category_spending)

        return {
            'total_income': total_income,
            'total_expense': total_expense,
            'savings_rate': savings_rate,
            'category_spending': dict(category_spending),
            'category_percentages': category_percentages,
            'needs_wants_savings': needs_wants_savings,
            'needs_wants_savings_pct': needs_wants_savings_pct,
            'category_classifications': category_classifications,
            'insights': insights,
            'anomalies': anomalies,
            'forecast': forecast
        }
        
    def _predict_category_type(self, category_name: str) -> Optional[str]:
        """Predict budget type for unknown category using ML."""
        if not self.ml_enabled or not category_name:
            return None
            
        # Handle direct matches in training data
        category_mappings = self.training_data.get('category_mappings', {})
        if category_name.lower() in category_mappings:
            return category_mappings[category_name.lower()]
            
        # Use text similarity for unknown categories
        try:
            # First check exact matches in known categories
            for budget_type, categories in self.CATEGORY_TYPES.items():
                if category_name in categories:
                    return budget_type
            
            # If we have a TF-IDF model, use it for text similarity
            if 'tfidf_model' in self.models and 'category_classifier' in self.models:
                # Extract features using TF-IDF
                vectorizer = self.models['tfidf_model']
                classifier = self.models['category_classifier']
                
                # Transform the category name to features
                features = vectorizer.transform([category_name.lower()])
                
                # Predict the budget type
                prediction = classifier.predict(features)[0]
                return prediction
                
            # Fallback to simple keyword matching
            keywords = {
                'Needs': ['rent', 'mortgage', 'utilities', 'grocery', 'groceries', 'electric', 'water', 
                          'gas', 'insurance', 'medical', 'healthcare', 'education', 'tuition', 'transport'],
                'Wants': ['restaurant', 'dining', 'bar', 'cafe', 'entertainment', 'subscription', 'game',
                          'shopping', 'travel', 'vacation', 'hobby', 'gym', 'fitness', 'clothing', 'beauty'],
                'Savings': ['savings', 'investment', 'retirement', 'fund', 'stock', 'bond', 'emergency', 'ira', '401k']
            }
            
            # Check for keyword matches
            category_lower = category_name.lower()
            for budget_type, words in keywords.items():
                for word in words:
                    if word in category_lower:
                        return budget_type
                        
        except Exception as e:
            self.logger.error("Error predicting category type: %s", str(e))
            
        # Default to Wants if we can't predict
        return 'Wants'
        
    def _update_training_data(self, transactions: List[Dict[str, Any]]):
        """Update training data with new transactions."""
        if not transactions:
            return
            
        # Initialize training data structure if empty
        if not self.training_data:
            self.training_data = {
                'spending_history': [],
                'category_mappings': {},
                'budget_adjustments': [],
                'monthly_patterns': {},
                'user_corrections': []
            }
            
        # Extract date information if available
        current_month = None
        current_year = None
        
        try:
            if transactions and 'date' in transactions[0]:
                date_parts = transactions[0]['date'].split('-')
                if len(date_parts) >= 2:
                    current_year = int(date_parts[0])
                    current_month = int(date_parts[1])
        except (ValueError, IndexError):
            pass
            
        # Add the transactions to spending history
        category_field = None
        for tx in transactions:
            # Determine which field contains the category
            if category_field is None:
                if 'category' in tx:
                    category_field = 'category'
                elif 'category_name' in tx:
                    category_field = 'category_name'
                else:
                    category_field = 'unknown'
            
            # Skip if it's not an expense or doesn't have a category
            if tx.get('is_income', False) or category_field == 'unknown':
                continue
                
            # Get the category
            category = tx.get(category_field, 'Uncategorized')
            if not category:
                continue
                
            # Record the category mapping if known
            found_in_types = False
            for budget_type, categories in self.CATEGORY_TYPES.items():
                if category in categories:
                    self.training_data['category_mappings'][category.lower()] = budget_type
                    found_in_types = True
                    break
                    
            # Add transaction to spending history
            if 'date' in tx and 'amount' in tx and not tx.get('is_income', False):
                self.training_data['spending_history'].append({
                    'date': tx['date'],
                    'amount': tx['amount'],
                    'category': category,
                    'year': current_year,
                    'month': current_month
                })
                
        # Update monthly patterns
        if current_month and current_year:
            month_key = f"{current_year}-{current_month:02d}"
            
            # Calculate total spending for the month
            month_spending = sum(tx['amount'] for tx in self.training_data['spending_history'] 
                               if tx.get('year') == current_year and tx.get('month') == current_month)
            
            # Update monthly patterns
            self.training_data['monthly_patterns'][month_key] = month_spending
            
        # Save the updated training data
        self._save_state()
        
    def _generate_insights(self, transactions: List[Dict[str, Any]], 
                          category_spending: Dict[str, float],
                          allocation_pct: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate ML-enhanced spending insights."""
        insights = []
        
        if not transactions or not category_spending:
            return insights
            
        # Insight 1: Budget allocation alignment
        # Check if current spending aligns with 50/30/20 rule
        target_allocation = self.DEFAULT_ALLOCATION
        if self.user_preferences.get('custom_allocation'):
            target_allocation = self.user_preferences['custom_allocation']
            
        # Calculate allocation differences
        allocation_diff = {}
        for budget_type, target_pct in target_allocation.items():
            current_pct = allocation_pct.get(budget_type, 0) / 100  # Convert from percentage to decimal
            diff = current_pct - target_pct
            allocation_diff[budget_type] = diff
            
            # Generate insight for significant differences
            if abs(diff) > 0.1:  # More than 10% off target
                direction = "high" if diff > 0 else "low"
                severity = "significantly" if abs(diff) > 0.2 else "somewhat"
                
                insights.append({
                    'type': f"allocation_{direction}",
                    'category': budget_type,
                    'title': f"{budget_type} allocation is {severity} {direction}",
                    'description': f"Your {budget_type.lower()} spending is at {current_pct*100:.1f}% of total budget, vs. recommended {target_pct*100:.1f}%.",
                    'recommendation': f"Consider {'reducing' if diff > 0 else 'increasing'} your {budget_type.lower()} allocation by about {abs(diff)*100:.1f}%.",
                    'impact': "medium" if abs(diff) > 0.2 else "low"
                })
                
        # Insight 2: Category spending concerns
        # Find categories that take up too large a percentage of budget
        max_category_pct = self.user_preferences.get('max_category_percent', 0.3)
        for category, amount in sorted(category_spending.items(), key=lambda x: x[1], reverse=True):
            category_pct = amount / sum(category_spending.values()) if sum(category_spending.values()) > 0 else 0
            
            if category_pct > max_category_pct:
                insights.append({
                    'type': "category_concentration",
                    'category': category,
                    'title': f"High spending concentration in {category}",
                    'description': f"{category} represents {category_pct*100:.1f}% of your total expenses.",
                    'recommendation': f"Look for ways to reduce {category} expenses to below {max_category_pct*100:.0f}% of your budget.",
                    'impact': "high" if category_pct > 0.4 else "medium"
                })
                
        # Insight 3: Savings rate assessment
        savings_rate = ((sum(tx['amount'] for tx in transactions if tx.get('is_income', False)) - 
                        sum(tx['amount'] for tx in transactions if not tx.get('is_income', False))) / 
                        sum(tx['amount'] for tx in transactions if tx.get('is_income', False)) * 100
                      ) if sum(tx['amount'] for tx in transactions if tx.get('is_income', False)) > 0 else 0
                      
        if savings_rate < 10:
            insights.append({
                'type': "low_savings",
                'title': "Low savings rate",
                'description': f"Your current savings rate is {savings_rate:.1f}%.",
                'recommendation': "Aim for a savings rate of at least 15-20% for long-term financial health.",
                'impact': "high" if savings_rate < 5 else "medium"
            })
        elif savings_rate > 40:
            insights.append({
                'type': "high_savings_positive",
                'title': "Excellent savings rate",
                'description': f"Your current savings rate is {savings_rate:.1f}%.",
                'recommendation': "You're saving a significant portion of your income. Consider investing excess savings for better returns.",
                'impact': "low"
            })
            
        # Add more sophisticated ML-based insights if available
        if self.ml_enabled and 'monthly_patterns' in self.training_data:
            # Insight 4: Spending trend analysis
            try:
                recent_months = sorted(self.training_data['monthly_patterns'].items(), key=lambda x: x[0], reverse=True)[:6]
                if len(recent_months) >= 3:
                    month_values = [amount for _, amount in recent_months]
                    
                    # Check for consistent increase or decrease
                    increasing = all(month_values[i] <= month_values[i+1] for i in range(len(month_values)-1))
                    decreasing = all(month_values[i] >= month_values[i+1] for i in range(len(month_values)-1))
                    
                    if increasing:
                        # Calculate average monthly increase
                        avg_increase = (month_values[0] - month_values[-1]) / (len(month_values) - 1)
                        
                        insights.append({
                            'type': "spending_increasing_alert",
                            'title': "Spending consistently increasing",
                            'description': f"Your monthly spending has increased by an average of {avg_increase:.2f} € each month for the last {len(month_values)} months.",
                            'recommendation': "Review your expenses to identify and reduce non-essential spending.",
                            'impact': "high"
                        })
                    elif decreasing:
                        # Calculate average monthly decrease
                        avg_decrease = (month_values[-1] - month_values[0]) / (len(month_values) - 1)
                        
                        insights.append({
                            'type': "spending_decreasing_positive",
                            'title': "Spending consistently decreasing",
                            'description': f"Your monthly spending has decreased by an average of {abs(avg_decrease):.2f} € each month for the last {len(month_values)} months.",
                            'recommendation': "Great job reducing expenses. Consider allocating these savings toward your financial goals.",
                            'impact': "low"
                        })
            except Exception as e:
                self.logger.error("Error generating spending trend insight: %s", str(e))
                
        # Sort insights by impact
        impact_values = {"high": 3, "medium": 2, "low": 1}
        insights.sort(key=lambda x: impact_values.get(x.get('impact', 'low'), 0), reverse=True)
        
        return insights
        
    def _detect_spending_anomalies(self, transactions: List[Dict[str, Any]], 
                                  category_spending: Dict[str, float]) -> List[Dict[str, Any]]:
        """Detect anomalies in spending patterns using ML techniques."""
        anomalies = []
        
        if not self.ml_enabled or not self.training_data.get('spending_history'):
            return anomalies
            
        try:
            # Group transactions by category
            category_transactions = defaultdict(list)
            category_field = None
            
            # Determine category field
            for tx in transactions:
                if 'category' in tx:
                    category_field = 'category'
                    break
                elif 'category_name' in tx:
                    category_field = 'category_name'
                    break
                    
            if not category_field:
                return anomalies
                
            # Group transactions by category
            for tx in transactions:
                if tx.get('is_income', False):
                    continue
                category = tx.get(category_field, 'Uncategorized')
                category_transactions[category].append(tx)
                
            # Get historical category averages
            category_history = defaultdict(list)
            for tx in self.training_data.get('spending_history', []):
                category_history[tx['category']].append(tx['amount'])
                
            # Check for anomalies in each category
            threshold = self.MODEL_PARAMS['spending_anomaly_threshold']
            
            for category, current_amount in category_spending.items():
                if category in category_history and len(category_history[category]) >= 3:
                    # Calculate historical statistics
                    historical_amounts = category_history[category]
                    mean = np.mean(historical_amounts)
                    std_dev = np.std(historical_amounts)
                    
                    # Check if current amount is an anomaly
                    if std_dev > 0:
                        z_score = abs(current_amount - mean) / std_dev
                        
                        if z_score > threshold:
                            direction = "higher" if current_amount > mean else "lower"
                            percent_diff = abs(current_amount - mean) / mean * 100 if mean > 0 else 0
                            
                            anomalies.append({
                                'type': f"spending_anomaly_{direction}",
                                'category': category,
                                'title': f"Unusual {category} spending",
                                'description': f"Your {category} spending is {percent_diff:.1f}% {direction} than usual.",
                                'current_amount': current_amount,
                                'typical_amount': mean,
                                'z_score': z_score,
                                'severity': "high" if z_score > threshold * 1.5 else "medium"
                            })
                            
            # Check for unusual transaction volumes
            # Implementation would go here
            
        except Exception as e:
            self.logger.error("Error detecting spending anomalies: %s", str(e))
            
        return anomalies
        
    def _generate_spending_forecast(self, transactions: List[Dict[str, Any]], 
                                   category_spending: Dict[str, float]) -> Dict[str, Any]:
        """Generate ML-based forecast of future spending patterns."""
        forecast = {
            'categories': {},
            'total': {},
            'confidence': 'low'
        }
        
        if not self.ml_enabled or not self.training_data.get('monthly_patterns'):
            return forecast
            
        try:
            # Get historical monthly spending patterns
            monthly_patterns = self.training_data.get('monthly_patterns', {})
            
            if len(monthly_patterns) < 3:
                return forecast
                
            # Sort patterns by month
            sorted_patterns = sorted(monthly_patterns.items())
            months = list(range(len(sorted_patterns)))
            spending = [amount for _, amount in sorted_patterns]
            
            # Create and train linear regression model
            model = LinearRegression()
            model.fit(np.array(months).reshape(-1, 1), spending)
            
            # Generate predictions for next 3 months
            prediction_horizon = self.MODEL_PARAMS['prediction_horizon']
            future_months = list(range(len(months), len(months) + prediction_horizon))
            predictions = model.predict(np.array(future_months).reshape(-1, 1))
            
            # Store predictions
            last_date = sorted_patterns[-1][0]
            year, month = map(int, last_date.split('-'))
            
            for i, amount in enumerate(predictions):
                future_month = month + i + 1
                future_year = year
                
                while future_month > 12:
                    future_month -= 12
                    future_year += 1
                    
                # Format as YYYY-MM
                month_key = f"{future_year}-{future_month:02d}"
                forecast['total'][month_key] = max(0, float(amount))  # Ensure no negative predictions
                
            # Generate category-specific forecasts
            if len(self.training_data.get('spending_history', [])) >= 10:
                category_history = defaultdict(lambda: defaultdict(float))
                
                # Group historical spending by category and month
                for tx in self.training_data['spending_history']:
                    if 'year' in tx and 'month' in tx and tx['year'] and tx['month']:
                        month_key = f"{tx['year']}-{tx['month']:02d}"
                        category_history[tx['category']][month_key] += tx['amount']
                
                # Forecast each category with enough data points
                for category, monthly_data in category_history.items():
                    if len(monthly_data) >= 3:
                        # Prepare data for regression
                        sorted_data = sorted(monthly_data.items())
                        cat_months = list(range(len(sorted_data)))
                        cat_spending = [amount for _, amount in sorted_data]
                        
                        # Create category-specific model
                        cat_model = LinearRegression()
                        cat_model.fit(np.array(cat_months).reshape(-1, 1), cat_spending)
                        
                        # Generate predictions
                        cat_future_months = list(range(len(cat_months), len(cat_months) + prediction_horizon))
                        cat_predictions = cat_model.predict(np.array(cat_future_months).reshape(-1, 1))
                        
                        # Store predictions
                        forecast['categories'][category] = {}
                        
                        for i, amount in enumerate(cat_predictions):
                            future_month = month + i + 1
                            future_year = year
                            
                            while future_month > 12:
                                future_month -= 12
                                future_year += 1
                                
                            month_key = f"{future_year}-{future_month:02d}"
                            forecast['categories'][category][month_key] = max(0, float(amount))
            
            # Set confidence level based on data quality
            history_length = len(monthly_patterns)
            if history_length >= 12:
                forecast['confidence'] = 'high'
            elif history_length >= 6:
                forecast['confidence'] = 'medium'
            else:
                forecast['confidence'] = 'low'
                
        except Exception as e:
            self.logger.error("Error generating spending forecast: %s", str(e))
            
        return forecast
        
    def _record_spending_patterns(self, transactions_df, category_spending):
        """Record spending patterns for future reference and learning."""
        if not transactions_df.empty and 'date' in transactions_df.columns:
            # Extract current month/year
            try:
                recent_date = max(transactions_df['date'])
                date_parts = recent_date.split('-')
                
                if len(date_parts) >= 2:
                    current_year = int(date_parts[0])
                    current_month = int(date_parts[1])
                    
                    # Add to monthly patterns if not already present
                    month_key = f"{current_year}-{current_month:02d}"
                    
                    if month_key not in self.training_data.get('monthly_patterns', {}):
                        total_spending = sum(category_spending.values())
                        self.training_data.setdefault('monthly_patterns', {})[month_key] = total_spending
                        
                        # Limit history to last 24 months
                        if len(self.training_data.get('monthly_patterns', {})) > 24:
                            oldest_key = min(self.training_data['monthly_patterns'].keys())
                            del self.training_data['monthly_patterns'][oldest_key]
            except (ValueError, IndexError, AttributeError) as e:
                self.logger.error("Error recording spending pattern: %s", str(e))
                
    def _train_models(self):
        """Train machine learning models on accumulated data."""
        if not self.ml_enabled or not self.training_data.get('spending_history'):
            return
            
        # Only train periodically to avoid unnecessary computation
        if len(self.models) > 0 and len(self.training_data.get('spending_history', [])) % 10 != 0:
            return
            
        try:
            self.logger.info("Training budget optimization models...")
            
            # 1. Train category classifier using TF-IDF vectorizer
            if self.training_data.get('category_mappings'):
                # Prepare text and labels for category classification
                categories = []
                labels = []
                
                for category, budget_type in self.training_data['category_mappings'].items():
                    categories.append(category)
                    labels.append(budget_type)
                    
                # Add known categories from CATEGORY_TYPES
                for budget_type, category_list in self.CATEGORY_TYPES.items():
                    for category in category_list:
                        categories.append(category.lower())
                        labels.append(budget_type)
                
                if len(categories) >= 5:  # Need enough samples for meaningful training
                    # Create and train TF-IDF vectorizer
                    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=100)
                    X = vectorizer.fit_transform(categories)
                    
                    # Train KMeans for simple classification
                    kmeans = KMeans(n_clusters=3, random_state=42)
                    kmeans.fit(X)
                    
                    # Map clusters to budget types based on majority class
                    cluster_labels = {}
                    for cluster_id in range(kmeans.n_clusters):
                        # Get indices of samples in this cluster
                        cluster_indices = np.where(kmeans.labels_ == cluster_id)[0]
                        
                        # Get the budget types for these samples
                        cluster_types = [labels[i] for i in cluster_indices]
                        
                        # Find the most common budget type
                        if cluster_types:
                            # Count occurrences of each type
                            type_counts = {}
                            for t in cluster_types:
                                type_counts[t] = type_counts.get(t, 0) + 1
                                
                            # Find the most common
                            majority_type = max(type_counts.items(), key=lambda x: x[1])[0]
                            cluster_labels[cluster_id] = majority_type
                    
                    # Create a mapping function for prediction
                    def predict_budget_type(X):
                        clusters = kmeans.predict(X)
                        return np.array([cluster_labels.get(c, 'Wants') for c in clusters])
                    
                    # Attach the prediction function to kmeans
                    kmeans.predict = predict_budget_type
                    
                    # Store the trained models
                    self.models['tfidf_model'] = vectorizer
                    self.models['category_classifier'] = kmeans
                    
                    self.logger.info("Trained category classifier with %d samples", len(categories))
                    
            # 2. Train spending forecasting models for categories
            history = self.training_data.get('spending_history', [])
            if len(history) >= 10:
                # Group spending by category and month
                category_monthly = defaultdict(lambda: defaultdict(float))
                
                for tx in history:
                    if 'category' in tx and 'year' in tx and 'month' in tx:
                        month_key = f"{tx['year']}-{tx['month']:02d}"
                        category_monthly[tx['category']][month_key] += tx['amount']
                
                # Train separate models for each category with enough data
                for category, monthly_data in category_monthly.items():
                    if len(monthly_data) >= 6:  # Need enough months for meaningful prediction
                        # Prepare training data
                        sorted_data = sorted(monthly_data.items())
                        X = np.array(range(len(sorted_data))).reshape(-1, 1)
                        y = np.array([amount for _, amount in sorted_data])
                        
                        # Train linear regression model
                        model = LinearRegression()
                        model.fit(X, y)
                        
                        # Store model
                        self.category_predictors[category] = model
                        
                        self.logger.info("Trained forecast model for category: %s", category)
                
            # Save the trained models
            self._save_state()
            
        except Exception as e:
            self.logger.error("Error training budget models: %s", str(e))

    def optimize_budget(self, transactions: List[Dict[str, Any]],
                        target_income: Optional[float] = None,
                        personalize: bool = True) -> Dict[str, Any]:
        """
        Generate an optimized budget based on spending history, ML insights, and budget rules.

        Args:
            transactions: List of transaction dictionaries
            target_income: Target monthly income, or None to use average income from transactions
            personalize: Whether to apply personalized optimizations based on ML

        Returns:
            Dictionary with optimized budget recommendations
        """
        # Analyze current spending with ML-enhanced insights
        spending_analysis = self.analyze_spending(transactions)

        # Convert to DataFrame
        df = pd.DataFrame(transactions)

        # Get monthly income
        if target_income is not None:
            monthly_income = target_income
        else:
            # Calculate average monthly income from transactions
            if 'is_income' in df.columns and 'date' in df.columns:
                income_df = df[df['is_income']]

                if not income_df.empty:
                    income_df['date'] = pd.to_datetime(income_df['date'])
                    income_df['month'] = income_df['date'].dt.strftime('%Y-%m')

                    monthly_income_avg = income_df.groupby('month')['amount'].sum().mean()
                    monthly_income = monthly_income_avg
                else:
                    # Fallback if no income data
                    monthly_income = spending_analysis['total_expense'] * 1.2  # Assume 20% savings
            else:
                # Fallback if missing needed columns
                monthly_income = spending_analysis['total_expense'] * 1.2

        # Determine which allocation rule to use
        allocation_rule = self.user_preferences.get('allocation_rule', '50/30/20')
        
        if allocation_rule == '50/30/20':
            allocation = self.DEFAULT_ALLOCATION
        elif allocation_rule == 'custom' and self.user_preferences.get('custom_allocation'):
            allocation = self.user_preferences['custom_allocation']
        else:
            allocation = self.DEFAULT_ALLOCATION

        # Calculate optimal allocation based on selected rule
        optimal_allocation = {
            'Needs': monthly_income * allocation['Needs'],
            'Wants': monthly_income * allocation['Wants'],
            'Savings': monthly_income * allocation['Savings']
        }

        # Get current category spending
        category_spending = spending_analysis['category_spending']

        # Use ML-based category classifications when available
        if 'category_classifications' in spending_analysis and spending_analysis['category_classifications']:
            category_types = spending_analysis['category_classifications']
        else:
            # Classify categories into needs, wants, savings using rule-based method
            category_types = {}
            for category in category_spending.keys():
                for budget_type, categories in self.CATEGORY_TYPES.items():
                    if category in categories:
                        category_types[category] = budget_type
                        break

                # Default to Wants if not classified
                if category not in category_types:
                    category_types[category] = 'Wants'

        # Calculate current vs. optimal spending for each category type
        current_vs_optimal = {}
        for budget_type in allocation.keys():
            current = spending_analysis['needs_wants_savings'].get(budget_type, 0)
            optimal = optimal_allocation[budget_type]

            if current > optimal:
                # Reduce spending in this category
                reduction_needed = current - optimal
                reduction_factor = optimal / current if current > 0 else 0
            else:
                # Can increase spending or maintain
                reduction_needed = 0
                reduction_factor = 1

            current_vs_optimal[budget_type] = {
                'current': current,
                'optimal': optimal,
                'difference': optimal - current,
                'reduction_needed': reduction_needed,
                'reduction_factor': reduction_factor,
                'status': 'over_budget' if current > optimal * 1.05 else
                         'on_target' if current <= optimal * 1.05 and current >= optimal * 0.95 else
                         'under_budget'
            }

        # Generate recommended budget for each category with ML-based personalization
        recommended_budget = {}
        adjustment_factors = {}
        
        # Apply ML-based personalization if enabled
        if personalize and self.ml_enabled:
            # Use anomaly detection to make specific adjustments
            for anomaly in spending_analysis.get('anomalies', []):
                category = anomaly.get('category')
                if not category:
                    continue
                    
                severity = anomaly.get('severity', 'medium')
                anomaly_type = anomaly.get('type', '')
                
                # Calculate adjustment factor based on anomaly
                if 'higher' in anomaly_type:
                    # Reduce budget for categories with unusually high spending
                    adjustment_factors[category] = 0.8 if severity == 'high' else 0.9
                elif 'lower' in anomaly_type:
                    # Maintain reduced spending for categories already lower than usual
                    budget_type = category_types.get(category, 'Wants')
                    if current_vs_optimal[budget_type]['status'] != 'over_budget':
                        adjustment_factors[category] = 0.95  # Slight reduction from current lower level
            
            # Use insights for category-specific adjustments
            for insight in spending_analysis.get('insights', []):
                insight_type = insight.get('type', '')
                category = insight.get('category')
                
                if category and 'allocation_high' in insight_type:
                    # Ensure this category type is reduced more aggressively
                    for cat, cat_type in category_types.items():
                        if cat_type == category and cat not in adjustment_factors:
                            adjustment_factors[cat] = 0.85
                            
            # Apply forecast-based adjustments
            forecast = spending_analysis.get('forecast', {})
            if forecast.get('categories') and forecast.get('confidence') != 'low':
                for category, future_values in forecast['categories'].items():
                    # Check if category spending is trending up
                    if future_values:
                        # Get latest forecast month
                        last_month = max(future_values.keys())
                        forecast_amount = future_values[last_month]
                        current_amount = category_spending.get(category, 0)
                        
                        # If forecast is significantly higher than current, add adjustment
                        if forecast_amount > current_amount * 1.2 and category not in adjustment_factors:
                            adjustment_factors[category] = 0.9  # Preemptive reduction
        
        # Apply adjustments to each category
        for category, amount in category_spending.items():
            budget_type = category_types.get(category, 'Wants')
            
            # Get base reduction factor from category type
            base_factor = 1.0
            if current_vs_optimal[budget_type]['reduction_needed'] > 0:
                base_factor = current_vs_optimal[budget_type]['reduction_factor']
                
            # Apply personalized adjustment if available
            ml_factor = adjustment_factors.get(category, 1.0)
            
            # Final factor is the stricter of the two
            final_factor = min(base_factor, ml_factor)
            
            # Apply factor to get recommended budget
            recommended_budget[category] = amount * final_factor
            
            # Ensure minimum budget amounts
            if recommended_budget[category] < 5.0 and amount >= 5.0:
                recommended_budget[category] = 5.0  # Minimum practical budget amount
        
        # Handle priority categories (keep closer to current spending)
        priority_categories = self.user_preferences.get('priority_categories', [])
        for category in priority_categories:
            if category in recommended_budget and category in category_spending:
                current = category_spending[category]
                recommended = recommended_budget[category]
                
                # Adjust to be at most 10% less than current for priority categories
                min_recommended = current * 0.9
                if recommended < min_recommended:
                    recommended_budget[category] = min_recommended

        # Add savings category if not present
        if 'Savings' not in recommended_budget and 'Savings' not in category_spending:
            recommended_budget['Savings'] = optimal_allocation['Savings']

        # Calculate surplus/deficit
        total_recommended = sum(recommended_budget.values())
        budget_balance = monthly_income - total_recommended
        
        # Generate insights on budget adjustments
        budget_insights = []
        
        # Calculate the overall reduction percentages by type
        reduction_by_type = {}
        for budget_type in allocation.keys():
            original = current_vs_optimal[budget_type]['current']
            
            # Calculate the new totals by type
            new_total = sum(recommended_budget.get(cat, 0) 
                          for cat, type_name in category_types.items() 
                          if type_name == budget_type)
                          
            if original > 0:
                reduction_pct = (original - new_total) / original * 100
                reduction_by_type[budget_type] = reduction_pct
                
                if reduction_pct > 5:
                    budget_insights.append({
                        'type': 'budget_reduction',
                        'category_type': budget_type,
                        'description': f"Your {budget_type.lower()} budget is reduced by {reduction_pct:.1f}%",
                        'amount': original - new_total,
                        'percent': reduction_pct
                    })
        
        # Identify top categories with largest reductions
        category_reductions = []
        for category, amount in category_spending.items():
            if category in recommended_budget:
                reduction = amount - recommended_budget[category]
                if reduction > 0:
                    category_reductions.append({
                        'category': category,
                        'reduction': reduction,
                        'percent': (reduction / amount * 100) if amount > 0 else 0
                    })
                    
        # Get top 3 categories with highest absolute reductions
        top_reductions = sorted(category_reductions, key=lambda x: x['reduction'], reverse=True)[:3]
        for reduction in top_reductions:
            if reduction['reduction'] > 10:  # Only show significant reductions
                budget_insights.append({
                    'type': 'category_reduction',
                    'category': reduction['category'],
                    'description': f"Reduce {reduction['category']} by {reduction['reduction']:.2f} € ({reduction['percent']:.1f}%)",
                    'amount': reduction['reduction'],
                    'percent': reduction['percent']
                })

        return {
            'monthly_income': monthly_income,
            'optimal_allocation': optimal_allocation,
            'current_vs_optimal': current_vs_optimal,
            'recommended_budget': recommended_budget,
            'total_recommended': total_recommended,
            'budget_balance': budget_balance,
            'budget_insights': budget_insights,
            'allocation_rule': allocation_rule,
            'adjustment_factors': adjustment_factors,
            'category_types': category_types
        }

    def identify_savings_opportunities(self, transactions: List[Dict[str, Any]],
                                      use_ml: bool = True) -> List[Dict[str, Any]]:
        """
        Identify specific savings opportunities based on transaction data using AI/ML techniques.

        Args:
            transactions: List of transaction dictionaries
            use_ml: Whether to use ML-enhanced opportunity detection

        Returns:
            List of savings opportunities with intelligent recommendations
        """
        opportunities = []

        if not transactions:
            return opportunities

        # Convert to DataFrame
        df = pd.DataFrame(transactions)

        # Determine category field
        category_field = 'category' if 'category' in df.columns else 'category_name' if 'category_name' in df.columns else None
        if not category_field or 'amount' not in df.columns:
            return opportunities

        # Filter expenses - ensure 'is_income' exists in the columns before filtering
        try:
            expenses = df[~df['is_income']] if 'is_income' in df.columns else df
        except Exception as e:
            self.logger.error(f"Error filtering expenses: {e}")
            # Fall back to using the entire dataframe
            expenses = df

        if expenses.empty:
            return opportunities

        # Get ML-enhanced insights if available
        ml_insights = {}
        if use_ml and self.ml_enabled:
            # First run full analysis to get ML insights
            analysis = self.analyze_spending(transactions)
            ml_insights = {
                'anomalies': analysis.get('anomalies', []),
                'insights': analysis.get('insights', []),
                'forecast': analysis.get('forecast', {})
            }

        # 1. Check for duplicate subscriptions and recurring payments
        subscription_opportunities = self._find_subscription_opportunities(expenses, category_field, ml_insights)
        opportunities.extend(subscription_opportunities)

        # 2. Identify high-spending categories compared to dynamic benchmarks
        high_spending_opportunities = self._find_high_spending_opportunities(expenses, category_field, ml_insights)
        opportunities.extend(high_spending_opportunities)

        # 3. Identify frequent small transactions that add up
        small_transaction_opportunities = self._find_small_transaction_opportunities(expenses, category_field)
        opportunities.extend(small_transaction_opportunities)

        # 4. Identify savings from large discretionary purchases
        large_purchase_opportunities = self._find_large_purchase_opportunities(expenses, category_field, ml_insights)
        opportunities.extend(large_purchase_opportunities)
        
        # 5. Add subscription analyzer opportunities if available (integration with subscription_analyzer.py)
        if 'subscription_analyzer' in self.user_preferences.get('integrations', {}):
            subscription_analyzer = self.user_preferences['integrations']['subscription_analyzer']
            try:
                sub_recommendations = subscription_analyzer.generate_subscription_recommendations(transactions)
                # Filter out duplicate recommendations
                existing_types = {op['type'] for op in opportunities}
                for rec in sub_recommendations:
                    if rec['type'] not in existing_types:
                        opportunities.append(rec)
            except Exception as e:
                self.logger.error(f"Error getting subscription analyzer recommendations: {e}")
                
        # 6. Add ML-specific opportunities
        if use_ml and self.ml_enabled:
            ml_opportunities = self._generate_ml_savings_opportunities(transactions, ml_insights)
            opportunities.extend(ml_opportunities)

        # Deduplicate opportunities based on type and category
        deduplicated = {}
        for op in opportunities:
            key = f"{op['type']}_{op.get('category', '')}_{op.get('merchant', '')}"
            if key not in deduplicated or deduplicated[key]['potential_savings'] < op['potential_savings']:
                deduplicated[key] = op
                
        # Sort by potential savings (highest first)
        sorted_opportunities = sorted(deduplicated.values(), key=lambda x: x.get('potential_savings', 0), reverse=True)
        
        # Return top 10 opportunities to avoid overwhelming the user
        return sorted_opportunities[:10]
        
    def _find_subscription_opportunities(self, expenses_df, category_field, ml_insights):
        """Find subscription-related savings opportunities."""
        opportunities = []
        
        # Check for subscription category
        subscription_category = 'Subscriptions'
        subscription_keywords = ['subscription', 'monthly', 'membership', 'recurring']
        
        # Find transactions that are subscriptions
        subscription_expenses = expenses_df[expenses_df[category_field] == subscription_category]
        
        # If no explicit subscription category, look for keywords in description
        if 'description' in expenses_df.columns and subscription_expenses.empty:
            for keyword in subscription_keywords:
                keyword_mask = expenses_df['description'].str.contains(keyword, case=False, na=False)
                subscription_expenses = pd.concat([subscription_expenses, expenses_df[keyword_mask]])
        
        # Find duplicate subscriptions if we have merchant information
        if 'merchant' in subscription_expenses.columns and not subscription_expenses.empty:
            # Group by merchant and count
            merchant_counts = subscription_expenses.groupby('merchant').size()
            
            # Get merchants with multiple entries
            duplicate_merchants = merchant_counts[merchant_counts > 1].index.tolist()
            
            for merchant in duplicate_merchants:
                merchant_df = subscription_expenses[subscription_expenses['merchant'] == merchant]
                
                # Calculate stats
                total_spent = merchant_df['amount'].sum()
                avg_amount = merchant_df['amount'].mean()
                
                # Check for different amounts which might indicate multiple subscriptions
                amount_variation = merchant_df['amount'].nunique() > 1
                
                if amount_variation:
                    # Multiple different subscriptions to same service
                    opportunities.append({
                        'type': 'duplicate_subscription',
                        'category': 'Subscriptions',
                        'merchant': merchant,
                        'count': len(merchant_df),
                        'total_spent': total_spent,
                        'recommendation': f"You have multiple different subscriptions to {merchant}. Consider consolidating to a single plan.",
                        'potential_savings': total_spent * 0.5,  # Estimate 50% savings
                        'confidence': 'high'
                    })
                else:
                    # Possible duplicate payment for same subscription
                    opportunities.append({
                        'type': 'duplicate_payment',
                        'category': 'Subscriptions',
                        'merchant': merchant,
                        'count': len(merchant_df),
                        'total_spent': total_spent,
                        'recommendation': f"You may be paying {merchant} multiple times for the same service (€{avg_amount:.2f} x {len(merchant_df)}). Verify you're not double-paying.",
                        'potential_savings': total_spent - avg_amount,  # All but one payment could be saved
                        'confidence': 'medium'
                    })
        
        # Use ML anomalies to identify unusual subscription payments
        for anomaly in ml_insights.get('anomalies', []):
            if (anomaly.get('category') == 'Subscriptions' or 
                'subscription' in anomaly.get('category', '').lower()) and 'higher' in anomaly.get('type', ''):
                
                opportunities.append({
                    'type': 'subscription_anomaly',
                    'category': anomaly.get('category'),
                    'total_spent': anomaly.get('current_amount', 0),
                    'typical_amount': anomaly.get('typical_amount', 0),
                    'recommendation': f"Your {anomaly.get('category')} spending is unusually high. Review your subscriptions for services you may not be using.",
                    'potential_savings': (anomaly.get('current_amount', 0) - anomaly.get('typical_amount', 0)) * 0.8,
                    'confidence': 'high' if anomaly.get('severity') == 'high' else 'medium'
                })
                
        # Annual vs. monthly subscription opportunities
        # Common services that offer annual discounts
        annual_discount_services = {
            'netflix': 0.15,       # 15% discount for annual
            'spotify': 0.20,       # 20% discount for annual 
            'amazon prime': 0.25,  # 25% discount for annual
            'disney+': 0.20,       # 20% discount for annual
            'hbo': 0.15,           # 15% discount for annual
            'apple': 0.15,         # 15% discount for annual plans
            'microsoft': 0.15,     # 15% discount for annual
            'adobe': 0.20,         # 20% discount for annual
            'gym': 0.25,           # 25% discount for annual gym memberships
        }
        
        if 'description' in expenses_df.columns:
            for service, discount in annual_discount_services.items():
                # Find expenses matching this service
                service_expenses = expenses_df[expenses_df['description'].str.contains(service, case=False, na=False)]
                
                if not service_expenses.empty:
                    # Check if payments are monthly (more than 1 payment to same service)
                    if len(service_expenses) > 1:
                        total_annual = service_expenses['amount'].sum() * (12 / len(service_expenses))  # Extrapolate to annual
                        discounted_annual = total_annual * (1 - discount)
                        potential_savings = total_annual - discounted_annual
                        
                        if potential_savings > 10:  # Only suggest if savings are meaningful
                            opportunities.append({
                                'type': 'annual_subscription_opportunity',
                                'service': service.title(),
                                'monthly_amount': service_expenses['amount'].mean(),
                                'annual_equivalent': total_annual,
                                'discounted_annual': discounted_annual,
                                'recommendation': f"Switch to an annual plan for {service.title()} to save approximately €{potential_savings:.2f} per year.",
                                'potential_savings': potential_savings,
                                'confidence': 'medium'
                            })
        
        return opportunities
        
    def _find_high_spending_opportunities(self, expenses_df, category_field, ml_insights):
        """Identify high-spending categories compared to benchmarks."""
        opportunities = []
        
        # Calculate category spending
        if expenses_df.empty or category_field not in expenses_df.columns:
            return opportunities
            
        category_spending = expenses_df.groupby(category_field)['amount'].sum()
        
        # Calculate total spending
        total_spending = category_spending.sum()
        if total_spending == 0:
            return opportunities
            
        # Find adaptive benchmarks based on user preferences or defaults
        benchmarks = self.user_preferences.get('category_benchmarks', {
            'Food': 0.15,           # Should be under 15% of total
            'Entertainment': 0.10,  # Should be under 10% of total
            'Shopping': 0.10,       # Should be under 10% of total
            'Subscriptions': 0.05,  # Should be under 5% of total
            'Transportation': 0.15, # Should be under 15% of total
            'Housing': 0.35,        # Should be under 35% of total
            'Utilities': 0.10,      # Should be under 10% of total
        })
        
        # Add dynamic benchmarks from ML insights
        if ml_insights:
            # Implementation would go here to adjust benchmarks based on ML
            pass
            
        # Check each category against its benchmark
        for category, amount in category_spending.items():
            benchmark = benchmarks.get(category)
            if not benchmark:
                continue
                
            actual_percentage = amount / total_spending
            if actual_percentage > benchmark:
                excess_spending = amount - (total_spending * benchmark)
                percent_over = ((actual_percentage / benchmark) - 1) * 100
                
                # Determine severity based on how much over benchmark
                severity = "significantly" if percent_over > 50 else "moderately" if percent_over > 20 else "slightly"
                
                opportunities.append({
                    'type': 'high_category_spending',
                    'category': category,
                    'actual_percentage': actual_percentage * 100,
                    'benchmark_percentage': benchmark * 100,
                    'excess_spending': excess_spending,
                    'recommendation': f"Your {category} spending is {severity} high at {actual_percentage * 100:.1f}% of your budget (recommended: {benchmark * 100:.1f}%).",
                    'potential_savings': excess_spending,
                    'confidence': 'high' if percent_over > 50 else 'medium'
                })
        
        # Add ML-derived insights about high spending
        for insight in ml_insights.get('insights', []):
            if insight.get('type') == 'category_concentration':
                category = insight.get('category')
                # Check if we already have this category in opportunities
                if category and not any(op['category'] == category and op['type'] == 'high_category_spending' 
                                      for op in opportunities):
                    
                    opportunities.append({
                        'type': 'ml_high_spending',
                        'category': category,
                        'recommendation': insight.get('recommendation', f"Reduce spending in {category}."),
                        'potential_savings': total_spending * 0.05,  # Conservative estimate of 5% savings
                        'confidence': 'medium'
                    })
        
        return opportunities
        
    def _find_small_transaction_opportunities(self, expenses_df, category_field):
        """Identify patterns of frequent small transactions that add up."""
        opportunities = []
        
        if expenses_df.empty or 'amount' not in expenses_df.columns:
            return opportunities
            
        # Define what constitutes a "small transaction"
        small_threshold = 10  # Consider transactions under 10 euros as "small"
        
        # Find small transactions
        small_expenses = expenses_df[expenses_df['amount'] < small_threshold]
        if small_expenses.empty:
            return opportunities
            
        # Group by merchant if available, otherwise by category
        groupby_field = 'merchant' if 'merchant' in small_expenses.columns else category_field
        if not groupby_field or groupby_field not in small_expenses.columns:
            return opportunities
            
        # Group transactions
        merchant_groups = small_expenses.groupby(groupby_field)
        
        for name, group in merchant_groups:
            # Look for patterns of frequent small purchases
            if len(group) >= 5:  # At least 5 transactions
                total_spent = group['amount'].sum()
                avg_amount = group['amount'].mean()
                
                # Calculate frequency if date is available
                frequency_text = ""
                if 'date' in group.columns:
                    try:
                        group['date'] = pd.to_datetime(group['date'])
                        date_range = (group['date'].max() - group['date'].min()).days
                        
                        if date_range > 0:
                            transactions_per_week = len(group) / (date_range / 7)
                            frequency_text = f", averaging {transactions_per_week:.1f} times per week"
                    except:
                        pass
                
                # Calculate potential savings (higher for very frequent small purchases)
                savings_factor = 0.8 if len(group) >= 10 else 0.7 if len(group) >= 7 else 0.6
                
                opportunities.append({
                    'type': 'frequent_small_purchases',
                    'merchant_or_category': name,
                    'count': len(group),
                    'total_spent': total_spent,
                    'average_amount': avg_amount,
                    'recommendation': f"You made {len(group)} small purchases at {name}{frequency_text}, totaling €{total_spent:.2f}. Consider consolidating these purchases or finding alternatives.",
                    'potential_savings': total_spent * savings_factor,
                    'confidence': 'high' if len(group) >= 10 else 'medium'
                })
                
        return opportunities
        
    def _find_large_purchase_opportunities(self, expenses_df, category_field, ml_insights):
        """Identify large discretionary purchases that could be optimized."""
        opportunities = []
        
        if expenses_df.empty or 'amount' not in expenses_df.columns or not category_field or category_field not in expenses_df.columns:
            return opportunities
            
        # Define what constitutes a "large transaction"
        large_threshold = 100  # Transactions over 100 euros
        
        # Define discretionary categories
        discretionary_categories = ['Shopping', 'Entertainment', 'Personal Care', 'Travel', 'Dining Out', 'Hobbies']
        
        # Find large discretionary expenses
        large_expenses = expenses_df[(expenses_df['amount'] > large_threshold) & 
                                    (expenses_df[category_field].isin(discretionary_categories))]
        
        if large_expenses.empty:
            return opportunities
            
        # Analyze each large expense
        for _, expense in large_expenses.iterrows():
            category = expense[category_field]
            amount = expense['amount']
            
            # Calculate average spending in this category if possible
            category_expenses = expenses_df[expenses_df[category_field] == category]
            category_avg = category_expenses['amount'].mean() if not category_expenses.empty else 0
            
            # Only suggest for expenses significantly larger than average
            if amount > category_avg * 1.5:
                description = expense.get('description', f"{category} expense")
                date = expense.get('date', '')
                
                # Format date if available
                date_text = ""
                if date:
                    try:
                        if isinstance(date, str):
                            date_obj = pd.to_datetime(date)
                            date_text = f" on {date_obj.strftime('%Y-%m-%d')}"
                        else:
                            date_text = f" on {date}"
                    except:
                        pass
                        
                # Calculate potential savings
                savings_factor = 0.3  # Estimate 30% could be saved
                
                # Check if this is a repeated large expense
                repeated = len(category_expenses[category_expenses['amount'] > amount * 0.9]) > 1
                confidence = 'high' if repeated else 'medium'
                
                opportunities.append({
                    'type': 'large_discretionary_purchase',
                    'category': category,
                    'date': date,
                    'description': description,
                    'amount': amount,
                    'recommendation': f"Your large {category} expense of €{amount:.2f}{date_text} could be reduced. Consider researching cheaper alternatives or waiting for sales.",
                    'potential_savings': amount * savings_factor,
                    'confidence': confidence
                })
                
        # Use ML anomalies to identify unusual large expenses
        for anomaly in ml_insights.get('anomalies', []):
            category = anomaly.get('category', '')
            if (category in discretionary_categories and 
                'higher' in anomaly.get('type', '') and 
                not any(op['category'] == category and op['type'] == 'large_discretionary_purchase' 
                       for op in opportunities)):
                
                current_amount = anomaly.get('current_amount', 0)
                typical_amount = anomaly.get('typical_amount', 0)
                
                if current_amount > typical_amount * 1.3:  # At least 30% higher than typical
                    opportunities.append({
                        'type': 'unusual_large_expense',
                        'category': category,
                        'amount': current_amount,
                        'typical_amount': typical_amount,
                        'recommendation': f"Your {category} spending is unusually high (€{current_amount:.2f} vs. typically €{typical_amount:.2f}). Look for ways to reduce these expenses.",
                        'potential_savings': (current_amount - typical_amount) * 0.7,
                        'confidence': 'high' if anomaly.get('severity') == 'high' else 'medium'
                    })
                    
        return opportunities
        
    def _generate_ml_savings_opportunities(self, transactions, ml_insights):
        """Generate ML-specific savings opportunities not covered by other methods."""
        opportunities = []
        
        if not self.ml_enabled or not ml_insights:
            return opportunities
            
        # Use forecast data to predict and prevent future overspending
        forecast = ml_insights.get('forecast', {})
        if forecast and forecast.get('categories') and forecast.get('confidence') != 'low':
            # Identify categories with forecast increases
            for category, future_values in forecast['categories'].items():
                if not future_values:
                    continue
                    
                # Get the trend from forecast months
                months = sorted(future_values.keys())
                if len(months) >= 2:
                    first_month = months[0]
                    last_month = months[-1]
                    
                    first_value = future_values[first_month]
                    last_value = future_values[last_month]
                    
                    # Check if significant increase is predicted
                    if last_value > first_value * 1.2:  # 20% increase
                        increase_amount = last_value - first_value
                        
                        opportunities.append({
                            'type': 'forecast_prevention',
                            'category': category,
                            'predicted_increase': increase_amount,
                            'current_amount': first_value,
                            'forecast_amount': last_value,
                            'recommendation': f"Your {category} spending is predicted to increase by €{increase_amount:.2f} in the coming months. Take proactive steps to control these costs.",
                            'potential_savings': increase_amount * 0.8,  # Prevent 80% of the forecasted increase
                            'confidence': forecast.get('confidence', 'medium')
                        })
        
        # Use insights to generate additional opportunities
        for insight in ml_insights.get('insights', []):
            insight_type = insight.get('type', '')
            
            # Find savings opportunities in spending patterns
            if 'spending_increasing_alert' in insight_type:
                # Already have other opportunities that cover this
                pass
                
            # Find opportunities based on allocation insights
            elif 'allocation_high' in insight_type:
                category = insight.get('category')
                if category and category == 'Wants':
                    # Look for opportunities to reduce discretionary spending
                    opportunities.append({
                        'type': 'reduce_discretionary',
                        'category': 'Wants',
                        'recommendation': "Your discretionary spending is higher than recommended. Review your 'wants' categories and identify areas to cut back.",
                        'potential_savings': 100,  # Generic estimate
                        'confidence': 'medium'
                    })
            
            # Look for low savings opportunities
            elif insight_type == 'low_savings':
                opportunities.append({
                    'type': 'increase_savings_rate',
                    'recommendation': "Your savings rate is below recommended levels. Aim to gradually increase your monthly savings by reducing non-essential expenses.",
                    'potential_savings': 50,  # Generic estimate
                    'confidence': 'medium'
                })
                    
        return opportunities

    def generate_savings_plan(self, transactions: List[Dict[str, Any]],
                              savings_goal: float,
                              months_to_save: int) -> Dict[str, Any]:
        """
        Generate a savings plan to reach a financial goal.

        Args:
            transactions: List of transaction dictionaries
            savings_goal: Target amount to save
            months_to_save: Number of months to achieve the goal

        Returns:
            Dictionary with savings plan details
        """
        # Analyze current spending
        spending_analysis = self.analyze_spending(transactions)

        # Identify savings opportunities
        opportunities = self.identify_savings_opportunities(transactions)

        # Calculate required monthly savings
        monthly_savings_required = savings_goal / months_to_save

        # Current monthly savings (if any)
        current_savings = spending_analysis['needs_wants_savings'].get('Savings', 0)

        # Additional savings needed per month
        additional_savings_needed = max(0, monthly_savings_required - current_savings)

        # Calculate total potential savings from opportunities
        total_potential_savings = sum(op.get('potential_savings', 0) for op in opportunities)

        # Check if identified opportunities are enough
        if total_potential_savings >= additional_savings_needed:
            savings_feasible = True

            # Sort opportunities by potential savings (highest first)
            sorted_opportunities = sorted(opportunities, key=lambda x: x.get('potential_savings', 0), reverse=True)

            # Select opportunities until we reach the goal
            selected_opportunities = []
            cumulative_savings = 0

            for op in sorted_opportunities:
                if cumulative_savings < additional_savings_needed:
                    selected_opportunities.append(op)
                    cumulative_savings += op.get('potential_savings', 0)
                else:
                    break
        else:
            savings_feasible = False
            selected_opportunities = opportunities

            # Additional amount that needs to be cut from budget
            additional_cuts_needed = additional_savings_needed - total_potential_savings

        # Generate monthly budget with the savings goal incorporated
        current_income = spending_analysis['total_income']
        target_budget = {}

        # Start with current category spending
        for category, amount in spending_analysis['category_spending'].items():
            target_budget[category] = amount

        # Add or update Savings category
        target_budget['Savings'] = monthly_savings_required

        # If needed, reduce other categories proportionally
        if not savings_feasible:
            total_discretionary = 0

            # Calculate total in discretionary categories
            for category in target_budget.keys():
                if category in self.CATEGORY_TYPES['Wants']:
                    total_discretionary += target_budget[category]

            # Calculate reduction factor
            if total_discretionary > 0:
                reduction_factor = max(0, 1 - (additional_cuts_needed / total_discretionary))

                # Apply reduction to discretionary categories
                for category in list(target_budget.keys()):
                    if category in self.CATEGORY_TYPES['Wants']:
                        target_budget[category] *= reduction_factor

        return {
            'savings_goal': savings_goal,
            'months_to_save': months_to_save,
            'monthly_savings_required': monthly_savings_required,
            'current_savings': current_savings,
            'additional_savings_needed': additional_savings_needed,
            'identified_opportunities': opportunities,
            'total_potential_savings': total_potential_savings,
            'savings_feasible': savings_feasible,
            'selected_opportunities': selected_opportunities,
            'target_budget': target_budget,
            'additional_cuts_needed': 0 if savings_feasible else (additional_savings_needed - total_potential_savings)
        }

    def optimize_student_budget(self, transactions: List[Dict[str, Any]],
                                student_income: float) -> Dict[str, Any]:
        """
        Generate an optimized budget specifically for international students in Germany.

        Args:
            transactions: List of transaction dictionaries
            student_income: Monthly student income

        Returns:
            Dictionary with student budget recommendations
        """
        # Student-specific allocation (modified from standard 50/30/20)
        STUDENT_ALLOCATION = {
            'Needs': 0.65,  # 65% for needs (higher than standard due to fixed costs)
            'Wants': 0.2,  # 20% for wants (lower than standard)
            'Savings': 0.15  # 15% for savings (slightly lower than standard)
        }

        # Student-specific category classification
        STUDENT_CATEGORIES = {
            'Needs': [
                'Housing',
                'Utilities',
                'Groceries',
                'Transportation',
                'Telecommunications',
                'Healthcare',
                'Education',
                'Insurance'
            ],
            'Wants': [
                'Food',  # Restaurants, dining out
                'Entertainment',
                'Shopping',
                'Subscriptions',
                'Personal Care'
            ],
            'Savings': [
                'Savings',
                'Emergency Fund'
            ]
        }

        # Student-specific benchmarks (percentage of income)
        STUDENT_BENCHMARKS = {
            'Housing': 0.40,  # Up to 40% for rent
            'Groceries': 0.15,  # Up to 15% for groceries
            'Transportation': 0.10,  # Up to 10% for transportation
            'Telecommunications': 0.05,  # Up to 5% for phone/internet
            'Education': 0.05,  # Up to 5% for study materials
            'Entertainment': 0.08,  # Up to 8% for entertainment
            'Food': 0.10,  # Up to 10% for eating out
            'Shopping': 0.07,  # Up to 7% for shopping
            'Healthcare': 0.08,  # Up to 8% for health insurance
            'Savings': 0.15  # At least 15% for savings
        }

        # Analyze current spending
        spending_analysis = self.analyze_spending(transactions)

        # Generate recommended budget based on student benchmarks
        recommended_budget = {}

        for category, benchmark in STUDENT_BENCHMARKS.items():
            # Calculate recommended amount
            recommended_amount = student_income * benchmark

            # Get current spending if available
            current_spending = spending_analysis['category_spending'].get(category, 0)

            # Use the lower of current spending or recommended amount
            if current_spending > 0 and current_spending < recommended_amount:
                recommended_budget[category] = current_spending
            else:
                recommended_budget[category] = recommended_amount

        # Check for categories with current spending but no benchmark
        for category, amount in spending_analysis['category_spending'].items():
            if category not in recommended_budget:
                # Assign to appropriate type
                category_type = None
                for type_, categories in STUDENT_CATEGORIES.items():
                    if category in categories:
                        category_type = type_
                        break

                if category_type == 'Needs':
                    # For essential needs, keep current spending
                    recommended_budget[category] = amount
                else:
                    # For non-essential categories, reduce if needed
                    recommended_budget[category] = amount * 0.8  # 20% reduction

        # Calculate totals
        total_recommended = sum(recommended_budget.values())
        budget_balance = student_income - total_recommended

        # Adjust if over budget
        if budget_balance < 0:
            # Calculate reduction factor
            reduction_factor = student_income / total_recommended

            # Apply reduction to non-essential categories first
            total_reduction_needed = -budget_balance
            reduction_applied = 0

            # First try reducing Wants categories
            for category in list(recommended_budget.keys()):
                if any(category in categories for _, categories in [('Wants', STUDENT_CATEGORIES['Wants'])]):
                    original_value = recommended_budget[category]
                    reduced_value = original_value * 0.7  # 30% reduction
                    reduction = original_value - reduced_value

                    if reduction_applied + reduction <= total_reduction_needed:
                        recommended_budget[category] = reduced_value
                        reduction_applied += reduction
                    else:
                        # Apply partial reduction to reach target
                        remaining_reduction = total_reduction_needed - reduction_applied
                        recommended_budget[category] = original_value - remaining_reduction
                        reduction_applied = total_reduction_needed
                        break

            # If still over budget, reduce Needs categories (except Housing)
            if reduction_applied < total_reduction_needed:
                for category in list(recommended_budget.keys()):
                    if (category in STUDENT_CATEGORIES['Needs'] and
                            category != 'Housing' and
                            category != 'Healthcare'):

                        original_value = recommended_budget[category]
                        reduced_value = original_value * 0.9  # 10% reduction
                        reduction = original_value - reduced_value

                        if reduction_applied + reduction <= total_reduction_needed:
                            recommended_budget[category] = reduced_value
                            reduction_applied += reduction
                        else:
                            # Apply partial reduction to reach target
                            remaining_reduction = total_reduction_needed - reduction_applied
                            recommended_budget[category] = original_value - remaining_reduction
                            reduction_applied = total_reduction_needed
                            break

        # Recalculate totals after adjustments
        total_recommended = sum(recommended_budget.values())
        budget_balance = student_income - total_recommended

        # Generate student-specific tips
        student_tips = [
            "Look for student discounts on public transportation (semester ticket).",
            "Use university facilities for internet access and printing when possible.",
            "Cook at home and shop at discount supermarkets like Aldi and Lidl.",
            "Get a student job at the university for additional income.",
            "Check if you qualify for housing assistance or scholarships.",
            "Use the university's gym instead of a private fitness center.",
            "Buy used textbooks or borrow from the library.",
            "Take advantage of free or discounted cultural events with student ID.",
            "Get a student bank account with no fees.",
            "Consider shared housing to reduce rent costs."
        ]

        return {
            'student_income': student_income,
            'recommended_budget': recommended_budget,
            'total_recommended': total_recommended,
            'budget_balance': budget_balance,
            'student_tips': student_tips,
            'allocation': STUDENT_ALLOCATION,
            'benchmarks': STUDENT_BENCHMARKS
        }


# Example usage
if __name__ == "__main__":
    optimizer = BudgetOptimizer()

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

    # Analyze spending
    analysis = optimizer.analyze_spending(transactions)
    print(f"Total Income: {analysis['total_income']:.2f} €")
    print(f"Total Expense: {analysis['total_expense']:.2f} €")
    print(f"Savings Rate: {analysis['savings_rate']:.2f}%")

    # Generate student budget
    student_budget = optimizer.optimize_student_budget(transactions, 900)

    # Print recommended budget
    print("\nRecommended Student Budget:")
    for category, amount in student_budget['recommended_budget'].items():
        print(f"{category}: {amount:.2f} €")

    print(f"\nTotal Recommended: {student_budget['total_recommended']:.2f} €")
    print(f"Budget Balance: {student_budget['budget_balance']:.2f} €")