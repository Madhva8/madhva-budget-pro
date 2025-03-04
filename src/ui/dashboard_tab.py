#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modern Dashboard Tab Module

This module provides a Power BI-inspired dashboard view with detailed financial summaries,
interactive visualizations, and analytics for the Financial Planner application.
"""

import logging
import datetime
import calendar
import random
from typing import List, Dict, Any, Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QGridLayout,
    QTableWidget, QTableWidgetItem, QHeaderView, QSizePolicy, QPushButton,
    QGraphicsDropShadowEffect, QScrollArea, QComboBox, QSpacerItem,
    QProgressBar, QTabWidget, QToolButton, QGroupBox, QScroller
)
from PyQt5.QtCore import Qt, QSize, QDate, QPropertyAnimation, QRect, QEasingCurve, QEvent, QPoint
from PyQt5.QtGui import QColor, QPainter, QBrush, QPen, QFont, QIcon, QLinearGradient, QPainterPath, QPixmap, QWheelEvent
from PyQt5.QtChart import (
    QChart, QChartView, QPieSeries, QLineSeries, QBarSet, QBarSeries, 
    QValueAxis, QBarCategoryAxis, QSplineSeries, QPieSlice, QPercentBarSeries,
    QScatterSeries, QAreaSeries, QDateTimeAxis
)


class EnhancedScrollArea(QScrollArea):
    """Enhanced scroll area with improved wheel event handling."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # Enable kinetic scrolling for touch devices
        QScroller.grabGesture(self.viewport(), QScroller.TouchGesture)
        # Adjust scroll properties for smoother behavior
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # Track current scroll position
        self._last_wheel_event_pos = None
        self._scroll_speed_multiplier = 1.5  # Adjust scroll speed
    
    def wheelEvent(self, event: QWheelEvent):
        """Enhanced wheel event handling for more reliable scrolling."""
        # Get the number of degrees rotated
        delta = event.angleDelta().y()
        
        # Calculate scroll distance with speed adjustment
        scroll_amount = delta * self._scroll_speed_multiplier
        
        # Vertical scrolling
        current_pos = self.verticalScrollBar().value()
        target_pos = current_pos - scroll_amount
        
        # Enforce bounds
        target_pos = max(0, min(target_pos, self.verticalScrollBar().maximum()))
        
        # Apply scroll
        self.verticalScrollBar().setValue(int(target_pos))
        
        # Debug logging if scrolling issues persist
        logging.debug(f"Wheel event: delta={delta}, current={current_pos}, target={target_pos}, max={self.verticalScrollBar().maximum()}")
        
        # Accept the event to prevent it from being passed to the parent
        event.accept()

class DashboardTab(QWidget):
    """Power BI-inspired dashboard tab widget."""

    def __init__(self, db_manager):
        """
        Initialize the modern dashboard tab.

        Args:
            db_manager: Database manager instance
        """
        super().__init__()
        
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        
        # Dashboard time range - default to current month
        self.current_date = QDate.currentDate()
        self.current_month = self.current_date.month()
        self.current_year = self.current_date.year()
        self.time_range = "month"  # Options: month, quarter, year, custom
        
        # Initialize AI components dictionary (will be set from outside)
        self.ai_components = {}
        
        # Initialize UI
        self.init_ui()
        
        # Load initial data
        self.refresh_dashboard()
        
    def init_ui(self):
        """Initialize the modern Power BI-inspired user interface."""
        # Set properties for the widget
        self.setObjectName("modernDashboard")
        
        # Main layout with margins for clean look
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Header with dashboard controls
        header_layout = QHBoxLayout()
        
        # Dashboard title with modern styling
        title_label = QLabel("Financial Dashboard")
        title_label.setObjectName("dashboardTitle")
        title_label.setStyleSheet("""
            font-size: 24px; 
            font-weight: bold; 
            color: #2c3e50;
        """)
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Time range selector
        time_range_layout = QHBoxLayout()
        time_range_layout.setSpacing(5)
        
        time_range_label = QLabel("Time Range:")
        time_range_label.setStyleSheet("font-weight: bold;")
        time_range_layout.addWidget(time_range_label)
        
        self.time_range_combo = QComboBox()
        self.time_range_combo.addItems(["Current Month", "Last 3 Months", "Year to Date", "Last 12 Months"])
        self.time_range_combo.setCurrentIndex(0)
        self.time_range_combo.setFixedWidth(150)
        self.time_range_combo.currentIndexChanged.connect(self.on_time_range_changed)
        time_range_layout.addWidget(self.time_range_combo)
        
        header_layout.addLayout(time_range_layout)
        
        # Refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setIcon(QIcon.fromTheme("view-refresh"))
        refresh_btn.setFixedWidth(100)
        refresh_btn.clicked.connect(self.refresh_dashboard)
        header_layout.addWidget(refresh_btn)
        
        main_layout.addLayout(header_layout)
        
        # Create a scroll area to handle overflow
        scroll_area = EnhancedScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        
        # Container for all dashboard content
        dashboard_container = QWidget()
        dashboard_layout = QVBoxLayout(dashboard_container)
        dashboard_layout.setContentsMargins(0, 0, 0, 0)
        dashboard_layout.setSpacing(20)
        
        # Key metrics cards row
        metrics_label = QLabel("Key Metrics")
        metrics_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50; margin-top: 10px;")
        dashboard_layout.addWidget(metrics_label)
        
        # First row: Summary cards
        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(15)
        
        # Income card
        self.income_card = MetricCard(
            "Total Income", 
            "0.00 €", 
            "green",
            "↑ 0% vs last period",
            "Income received during the selected period"
        )
        metrics_row.addWidget(self.income_card)
        
        # Expense card
        self.expense_card = MetricCard(
            "Total Expenses", 
            "0.00 €", 
            "red",
            "↑ 0% vs last period",
            "Expenses during the selected period"
        )
        metrics_row.addWidget(self.expense_card)
        
        # Balance card
        self.balance_card = MetricCard(
            "Net Savings", 
            "0.00 €", 
            "blue",
            "↑ 0% vs last period",
            "Income minus expenses (net cash flow)"
        )
        metrics_row.addWidget(self.balance_card)
        
        # Savings rate card
        self.savings_rate_card = MetricCard(
            "Savings Rate", 
            "0%", 
            "purple",
            "Target: 20%",
            "Percentage of income saved"
        )
        metrics_row.addWidget(self.savings_rate_card)
        
        # Budget Status Card
        self.budget_status_card = BudgetProgressCard("Budget Status", 0)
        metrics_row.addWidget(self.budget_status_card)
        
        dashboard_layout.addLayout(metrics_row)

        # Dashboard Tabs for different views
        dashboard_tabs = QTabWidget()
        dashboard_tabs.setDocumentMode(True)
        dashboard_tabs.setStyleSheet("""
            QTabWidget::pane { 
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 10px;
            }
            QTabBar::tab {
                background: #f5f5f5;
                border: 1px solid #ddd;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: white;
                border-bottom: 1px solid white;
            }
        """)
        
        # Overview panel with charts
        overview_panel = QWidget()
        overview_layout = QVBoxLayout(overview_panel)
        
        # Dual overview charts
        overview_charts = QGridLayout()
        overview_charts.setSpacing(20)
        
        # Income/Expense Trend Chart with advanced features
        self.income_expense_trend = EnhancedTrendChart(
            "Income & Expenses", 
            ["Income", "Expenses", "Net Savings"],
            ["#2ecc71", "#e74c3c", "#3498db"]
        )
        overview_charts.addWidget(self.income_expense_trend, 0, 0, 1, 2)
        
        # Category spending chart with enhanced visuals
        self.category_chart = EnhancedPieChart("Spending by Category")
        overview_charts.addWidget(self.category_chart, 1, 0)
        
        # Spending insights widget
        self.spending_insights = SpendingInsightsWidget(self.db_manager)
        overview_charts.addWidget(self.spending_insights, 1, 1)
        
        overview_layout.addLayout(overview_charts)
        
        # Details and Analysis panel with more charts and insights
        analysis_panel = QWidget()
        analysis_layout = QVBoxLayout(analysis_panel)
        
        # Monthly comparison charts
        monthly_comparison = QHBoxLayout()
        
        # Monthly Income vs Expense Bars
        self.monthly_comparison_chart = EnhancedComparisonChart("Monthly Comparison")
        monthly_comparison.addWidget(self.monthly_comparison_chart)
        
        # Spending Breakdown by Category
        self.category_breakdown = CategoryBreakdownWidget("Category Details")
        monthly_comparison.addWidget(self.category_breakdown)
        
        analysis_layout.addLayout(monthly_comparison)
        
        # Advanced Analysis Section
        advanced_analysis = QHBoxLayout()
        
        # Savings Target Tracking
        self.savings_tracker = SavingsTargetWidget("Savings Goals")
        advanced_analysis.addWidget(self.savings_tracker)
        
        # Budget vs Actual
        self.budget_comparison = BudgetComparisonWidget("Budget Tracking")
        advanced_analysis.addWidget(self.budget_comparison)
        
        analysis_layout.addLayout(advanced_analysis)
        
        # Transactions panel with detailed records
        transactions_panel = QWidget()
        transactions_layout = QVBoxLayout(transactions_panel)
        
        # Transactions table with enhanced features
        self.transactions_widget = EnhancedTransactionsWidget(self.db_manager)
        transactions_layout.addWidget(self.transactions_widget)
        
        # Add the panels to the tabs
        dashboard_tabs.addTab(overview_panel, "Overview")
        dashboard_tabs.addTab(analysis_panel, "Analysis")
        dashboard_tabs.addTab(transactions_panel, "Transactions")
        
        dashboard_layout.addWidget(dashboard_tabs)

        # AI Insights section
        ai_insights_label = QLabel("AI Insights & Recommendations")
        ai_insights_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50; margin-top: 10px;")
        dashboard_layout.addWidget(ai_insights_label)

        # AI Insights Cards
        ai_insights_layout = QHBoxLayout()
        ai_insights_layout.setSpacing(15)
        
        # Subscription Analysis Card
        self.subscription_insights = InsightCard(
            "Subscription Analysis",
            "Analyze your recurring subscriptions",
            "subscription",
            "See ways to optimize your subscription spending"
        )
        ai_insights_layout.addWidget(self.subscription_insights)
        
        # Spending Pattern Card
        self.spending_patterns = InsightCard(
            "Spending Patterns",
            "Identify your spending habits",
            "chart",
            "View personalized spending insights"
        )
        ai_insights_layout.addWidget(self.spending_patterns)
        
        # Saving Opportunities Card
        self.saving_opportunities = InsightCard(
            "Saving Opportunities",
            "AI-detected saving opportunities",
            "lightbulb",
            "Discover ways to save more money"
        )
        ai_insights_layout.addWidget(self.saving_opportunities)
        
        dashboard_layout.addLayout(ai_insights_layout)
        
        # Set the dashboard container as the scroll area's widget
        scroll_area.setWidget(dashboard_container)
        main_layout.addWidget(scroll_area)
        
    def on_time_range_changed(self, index):
        """Handle time range selection change."""
        range_options = ["month", "quarter", "ytd", "year"]
        if index < len(range_options):
            self.time_range = range_options[index]
            self.refresh_dashboard()
    
    def refresh_dashboard(self):
        """Refresh all dashboard components with current data."""
        self.logger.info(f"Refreshing dashboard data for time range: {self.time_range}")
        
        try:
            # Determine date range based on selected time range
            current_date = QDate.currentDate()
            month = current_date.month()
            year = current_date.year()
            
            # Get data based on time range
            if self.time_range == "month":
                # Current month data
                monthly_summary = self.db_manager.get_monthly_summary(month, year)
                comparison_period = "Previous Month"
            elif self.time_range == "quarter":
                # Last 3 months data
                monthly_summary = self.db_manager.get_quarterly_summary(year, (month-1)//3 + 1)
                comparison_period = "Previous Quarter"
            elif self.time_range == "ytd":
                # Year to date
                monthly_summary = self.db_manager.get_ytd_summary(year)
                comparison_period = "Same Period Last Year"
            else:  # year
                # Last 12 months
                monthly_summary = self.db_manager.get_yearly_summary(year)
                comparison_period = "Previous Year"
            
            # Update metric cards with comparison data
            total_income = monthly_summary.get('total_income', 0)
            total_expenses = monthly_summary.get('total_expenses', 0)
            balance = monthly_summary.get('savings', 0)
            savings_rate = monthly_summary.get('savings_rate', 0)
            
            # Get comparison data (if available)
            income_change = monthly_summary.get('income_change_pct', 0)
            expense_change = monthly_summary.get('expense_change_pct', 0)
            balance_change = monthly_summary.get('savings_change_pct', 0)
            
            # Format comparison text with arrows
            income_comparison = f"{'↑' if income_change >= 0 else '↓'} {abs(income_change):.1f}% vs {comparison_period}"
            expense_comparison = f"{'↑' if expense_change >= 0 else '↓'} {abs(expense_change):.1f}% vs {comparison_period}"
            balance_comparison = f"{'↑' if balance_change >= 0 else '↓'} {abs(balance_change):.1f}% vs {comparison_period}"
            savings_target = f"Target: 20%" if savings_rate < 20 else "Target achieved!"
            
            # Update cards
            self.income_card.update_value(f"{total_income:.2f} €")
            self.income_card.update_comparison(income_comparison)
            
            self.expense_card.update_value(f"{total_expenses:.2f} €")
            self.expense_card.update_comparison(expense_comparison)
            
            self.balance_card.update_value(f"{balance:.2f} €")
            self.balance_card.update_comparison(balance_comparison)
            
            self.savings_rate_card.update_value(f"{savings_rate:.1f}%")
            self.savings_rate_card.update_comparison(savings_target)
            
            # Set colors for comparative indicators
            if income_change > 0:
                self.income_card.set_comparison_color("green")
            else:
                self.income_card.set_comparison_color("red")
                
            if expense_change > 0:
                self.expense_card.set_comparison_color("red") # Higher expenses is bad
            else:
                self.expense_card.set_comparison_color("green")
                
            if balance_change > 0:
                self.balance_card.set_comparison_color("green")
            else:
                self.balance_card.set_comparison_color("red")
                
            # Set colors based on target achievements
            if savings_rate >= 20:
                self.savings_rate_card.set_comparison_color("green")
            else:
                self.savings_rate_card.set_comparison_color("gray")
            
            # Update budget progress card
            budget_data = monthly_summary.get('budget_status', {})
            budget_used_pct = budget_data.get('used_percentage', 0)
            self.budget_status_card.update_progress(budget_used_pct)
            
            # Update trend charts
            trend_data = self.db_manager.get_trend_data(self.time_range, year, month)
            self.income_expense_trend.update_data(trend_data)
            
            # Update category breakdown
            category_data = monthly_summary.get('category_breakdown', [])
            self.category_chart.update_data(category_data)
            self.category_breakdown.update_data(category_data)
            
            # Update comparison chart
            comparison_data = self.db_manager.get_comparison_data(self.time_range, year, month)
            self.monthly_comparison_chart.update_data(comparison_data)
            
            # Update insights widgets
            self.spending_insights.update_insights(monthly_summary.get('insights', []))
            
            # Update savings tracker and budget comparison
            savings_goals = self.db_manager.get_savings_goals()
            self.savings_tracker.update_data(savings_goals, balance)
            
            budget_comparison = self.db_manager.get_budget_comparison()
            self.budget_comparison.update_data(budget_comparison)
            
            # Update transactions
            self.transactions_widget.refresh_transactions(self.time_range)
            
            # Update AI Insights if available
            if hasattr(self, 'ai_components') and self.ai_components:
                self.update_ai_insights()
            
            self.logger.info("Dashboard data refreshed successfully")
            
        except Exception as e:
            self.logger.error(f"Error refreshing dashboard: {e}")
            
    def update_ai_insights(self):
        """Update AI-powered insight cards with recommendations."""
        try:
            # Get all transactions for AI analysis
            transactions = self.db_manager.get_all_transactions()
            
            # Subscription insights
            if 'subscription_analyzer' in self.ai_components:
                subscription_analyzer = self.ai_components['subscription_analyzer']
                recommendations = subscription_analyzer.generate_subscription_recommendations(transactions)
                
                if recommendations:
                    savings = sum(rec.get('potential_savings', 0) for rec in recommendations)
                    self.subscription_insights.update_content(
                        f"Save up to {savings:.2f} € on subscriptions",
                        f"Found {len(recommendations)} ways to optimize your subscriptions",
                        f"{len(recommendations)} recommendations available"
                    )
                else:
                    self.subscription_insights.update_content(
                        "No subscription optimizations found",
                        "Your subscription spending appears optimized",
                        "No recommendations"
                    )
            
            # Spending pattern insights
            if 'financial_ai' in self.ai_components:
                financial_ai = self.ai_components['financial_ai']
                analysis = financial_ai.analyze_spending_patterns(transactions)
                
                if analysis and 'insights' in analysis:
                    insights = analysis['insights']
                    if insights:
                        alert_count = sum(1 for insight in insights if insight.get('type', '').endswith('_alert'))
                        self.spending_patterns.update_content(
                            f"{alert_count} potential issues detected",
                            "AI has analyzed your spending patterns",
                            f"{len(insights)} insights available"
                        )
                    else:
                        self.spending_patterns.update_content(
                            "No unusual spending patterns detected",
                            "Your spending appears normal",
                            "0 insights"
                        )
            
            # Saving opportunities
            if 'budget_optimizer' in self.ai_components:
                budget_optimizer = self.ai_components['budget_optimizer']
                opportunities = budget_optimizer.identify_savings_opportunities(transactions)
                
                if opportunities:
                    savings = sum(op.get('potential_savings', 0) for op in opportunities)
                    self.saving_opportunities.update_content(
                        f"Save up to {savings:.2f} € monthly",
                        f"Found {len(opportunities)} saving opportunities",
                        f"{len(opportunities)} opportunities available"
                    )
                else:
                    self.saving_opportunities.update_content(
                        "No saving opportunities found",
                        "Your budget appears optimized",
                        "No recommendations"
                    )
        
        except Exception as e:
            self.logger.error(f"Error updating AI insights: {e}")
    
    def set_ai_components(self, ai_components):
        """Set AI components from outside."""
        self.ai_components = ai_components
        self.update_ai_insights()


class MetricCard(QFrame):
    """Modern metric card with value, comparison, and trend indicator."""
    
    def __init__(self, title, value, color="blue", comparison="", tooltip=""):
        """
        Initialize a modern metric card.
        
        Args:
            title: Card title
            value: Initial value
            color: Color scheme (blue, green, red, purple)
            comparison: Comparative text (e.g. "↑ 10% vs last month")
            tooltip: Tooltip text explaining the metric
        """
        super().__init__()
        
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        # Apply modern card styling
        self.setObjectName("metricCard")
        self.setStyleSheet("""
            #metricCard {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
            }
        """)
        
        # Add shadow effect for depth
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 20))
        shadow.setOffset(0, 3)
        self.setGraphicsEffect(shadow)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(5)
        
        # Title label
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-size: 14px; color: #555; font-weight: 500;")
        if tooltip:
            self.title_label.setToolTip(tooltip)
        layout.addWidget(self.title_label)
        
        # Value label
        self.value_label = QLabel(value)
        self.set_color(color)
        layout.addWidget(self.value_label)
        
        # Comparison label
        self.comparison_label = QLabel(comparison)
        self.comparison_label.setStyleSheet("font-size: 12px; color: #777;")
        layout.addWidget(self.comparison_label)
        
    def update_value(self, value):
        """Update the displayed value."""
        self.value_label.setText(value)
        
    def update_comparison(self, comparison_text):
        """Update the comparison text."""
        self.comparison_label.setText(comparison_text)
        
    def set_color(self, color):
        """Set the color scheme of the card."""
        if color == "green":
            self.value_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #2ecc71;")
        elif color == "red":
            self.value_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #e74c3c;")
        elif color == "purple":
            self.value_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #9b59b6;")
        elif color == "orange":
            self.value_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #f39c12;")
        else:  # blue is default
            self.value_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #3498db;")
            
    def set_comparison_color(self, color):
        """Set the color of the comparison text."""
        if color == "green":
            self.comparison_label.setStyleSheet("font-size: 12px; font-weight: 500; color: #2ecc71;")
        elif color == "red":
            self.comparison_label.setStyleSheet("font-size: 12px; font-weight: 500; color: #e74c3c;")
        elif color == "gray":
            self.comparison_label.setStyleSheet("font-size: 12px; font-weight: 500; color: #777;")
        else:  # blue is default
            self.comparison_label.setStyleSheet("font-size: 12px; font-weight: 500; color: #3498db;")


class BudgetProgressCard(QFrame):
    """Budget progress card with visual indicator."""
    
    def __init__(self, title, progress_value=0):
        """
        Initialize the budget progress card.
        
        Args:
            title: Card title
            progress_value: Initial progress percentage (0-100)
        """
        super().__init__()
        
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        # Apply modern card styling
        self.setObjectName("metricCard")
        self.setStyleSheet("""
            #metricCard {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
            }
            QProgressBar {
                border: 1px solid #e0e0e0;
                border-radius: 5px;
                background-color: #f5f5f5;
                text-align: center;
                height: 12px;
            }
            QProgressBar::chunk {
                border-radius: 5px;
            }
        """)
        
        # Add shadow effect for depth
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 20))
        shadow.setOffset(0, 3)
        self.setGraphicsEffect(shadow)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(5)
        
        # Title label
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-size: 14px; color: #555; font-weight: 500;")
        self.title_label.setToolTip("Shows how much of your monthly budget has been spent")
        layout.addWidget(self.title_label)
        
        # Progress value
        self.value_label = QLabel(f"{progress_value}% Used")
        self.value_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #3498db;")
        layout.addWidget(self.value_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(progress_value)
        self.update_progress_color(progress_value)
        layout.addWidget(self.progress_bar)
        
        # Advice label
        self.advice_label = QLabel(self.get_advice_text(progress_value))
        self.advice_label.setStyleSheet("font-size: 12px; color: #777;")
        layout.addWidget(self.advice_label)
        
    def update_progress(self, value):
        """Update the progress value and related elements."""
        clamped_value = max(0, min(100, value))
        self.progress_bar.setValue(clamped_value)
        self.value_label.setText(f"{clamped_value}% Used")
        self.advice_label.setText(self.get_advice_text(clamped_value))
        self.update_progress_color(clamped_value)
        
    def update_progress_color(self, value):
        """Update the progress bar color based on value."""
        if value < 70:
            # Green - Good
            style = "QProgressBar::chunk { background-color: #2ecc71; }"
            self.value_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #2ecc71;")
        elif value < 90:
            # Orange - Warning
            style = "QProgressBar::chunk { background-color: #f39c12; }"
            self.value_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #f39c12;")
        else:
            # Red - Critical
            style = "QProgressBar::chunk { background-color: #e74c3c; }"
            self.value_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #e74c3c;")
            
        self.progress_bar.setStyleSheet(style)
        
    def get_advice_text(self, value):
        """Get advice text based on the progress value."""
        if value < 50:
            return "Budget on track"
        elif value < 70:
            return "Budget on target"
        elif value < 90:
            return "Approaching budget limit"
        else:
            return "Budget limit exceeded"


class EnhancedTrendChart(QFrame):
    """Advanced chart for visualizing financial trends."""
    
    def __init__(self, title, series_names=None, series_colors=None):
        """Initialize the enhanced trend chart."""
        super().__init__()
        
        if series_names is None:
            series_names = ["Income", "Expenses", "Savings"]
            
        if series_colors is None:
            series_colors = ["#2ecc71", "#e74c3c", "#3498db"]
            
        self.series_names = series_names
        self.series_colors = series_colors
        
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Apply modern card styling
        self.setObjectName("chartCard")
        self.setStyleSheet("""
            #chartCard {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
            }
        """)
        
        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 20))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Header layout with title and controls
        header_layout = QHBoxLayout()
        
        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Chart type selector
        chart_type_layout = QHBoxLayout()
        chart_type_layout.setSpacing(2)
        
        # Line chart button
        self.line_button = QToolButton()
        self.line_button.setText("Line")
        self.line_button.setCheckable(True)
        self.line_button.setChecked(True)
        self.line_button.clicked.connect(lambda: self.change_chart_type("line"))
        chart_type_layout.addWidget(self.line_button)
        
        # Bar chart button
        self.bar_button = QToolButton()
        self.bar_button.setText("Bar")
        self.bar_button.setCheckable(True)
        self.bar_button.clicked.connect(lambda: self.change_chart_type("bar"))
        chart_type_layout.addWidget(self.bar_button)
        
        # Area chart button
        self.area_button = QToolButton()
        self.area_button.setText("Area")
        self.area_button.setCheckable(True)
        self.area_button.clicked.connect(lambda: self.change_chart_type("area"))
        chart_type_layout.addWidget(self.area_button)
        
        header_layout.addLayout(chart_type_layout)
        
        layout.addLayout(header_layout)
        
        # Create the chart
        self.chart = QChart()
        self.chart.setTitle("")
        self.chart.setAnimationOptions(QChart.SeriesAnimations)
        self.chart.legend().setVisible(True)
        self.chart.legend().setAlignment(Qt.AlignBottom)
        
        # Create chart view
        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.Antialiasing)
        layout.addWidget(self.chart_view)
        
        # Current chart type
        self.chart_type = "line"
        
        # Initial dummy data
        self.data = []
        self.update_data([])
        
    def change_chart_type(self, chart_type):
        """Change the chart visualization type."""
        if chart_type == self.chart_type:
            return
            
        self.chart_type = chart_type
        
        # Update button states
        self.line_button.setChecked(chart_type == "line")
        self.bar_button.setChecked(chart_type == "bar")
        self.area_button.setChecked(chart_type == "area")
        
        # Redraw chart with current data
        self.update_data(self.data)
        
    def update_data(self, data):
        """
        Update the chart with new data.
        
        Args:
            data: List of dictionaries with monthly data
        """
        # Store data for redrawing if chart type changes
        self.data = data
        
        # Clear existing series
        self.chart.removeAllSeries()
        
        # No data case
        if not data:
            self.chart.setTitle("No data available")
            return
            
        # Sort data by month
        sorted_data = sorted(data, key=lambda x: x.get('month', 0))
        
        # Prepare axes
        categories = []
        
        for month_data in sorted_data:
            month_num = month_data.get('month', 0)
            if 1 <= month_num <= 12:
                month_name = calendar.month_abbr[month_num]
                categories.append(month_name)
        
        # Draw based on chart type
        if self.chart_type == "line":
            self._create_line_chart(sorted_data, categories)
        elif self.chart_type == "bar":
            self._create_bar_chart(sorted_data, categories)
        elif self.chart_type == "area":
            self._create_area_chart(sorted_data, categories)
            
    def _create_line_chart(self, data, categories):
        """Create line chart visualization."""
        # Create series
        series_list = []
        
        for i, name in enumerate(self.series_names):
            series = QSplineSeries()
            series.setName(name)
            
            # Set color if available
            if i < len(self.series_colors):
                pen = QPen(QColor(self.series_colors[i]))
                pen.setWidth(3)
                series.setPen(pen)
                
            series_list.append(series)
            
        # Fill with data
        max_value = 0
        min_value = 0
        
        for i, month_data in enumerate(data):
            values = [
                month_data.get('income', 0),
                month_data.get('expenses', 0),
                month_data.get('savings', 0)
            ]
            
            # Update min/max
            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))
            
            # Add data points
            for j, series in enumerate(series_list):
                if j < len(values):
                    series.append(i, values[j])
        
        # Add series to chart
        for series in series_list:
            self.chart.addSeries(series)
            
        # Create axes
        axis_x = QBarCategoryAxis()
        axis_x.append(categories)
        
        axis_y = QValueAxis()
        axis_y.setLabelFormat("%.2f €")
        axis_y.setTitleText("Amount")
        
        # Set range with padding
        padding = (max_value - min_value) * 0.1
        axis_y.setRange(min_value - padding if min_value < 0 else 0, max_value + padding)
        
        self.chart.addAxis(axis_x, Qt.AlignBottom)
        self.chart.addAxis(axis_y, Qt.AlignLeft)
        
        # Attach axes to all series
        for series in series_list:
            series.attachAxis(axis_x)
            series.attachAxis(axis_y)
            
    def _create_bar_chart(self, data, categories):
        """Create bar chart visualization."""
        # Create bar sets
        bar_sets = []
        
        for i, name in enumerate(self.series_names):
            bar_set = QBarSet(name)
            
            # Set color if available
            if i < len(self.series_colors):
                bar_set.setColor(QColor(self.series_colors[i]))
                
            bar_sets.append(bar_set)
            
        # Fill with data
        max_value = 0
        
        for month_data in data:
            values = [
                month_data.get('income', 0),
                month_data.get('expenses', 0),
                month_data.get('savings', 0)
            ]
            
            # Update max
            max_value = max(max_value, max(values))
            
            # Add values to bar sets
            for i, bar_set in enumerate(bar_sets):
                if i < len(values):
                    bar_set.append(values[i])
        
        # Create bar series
        series = QBarSeries()
        for bar_set in bar_sets:
            series.append(bar_set)
            
        self.chart.addSeries(series)
        
        # Create axes
        axis_x = QBarCategoryAxis()
        axis_x.append(categories)
        
        axis_y = QValueAxis()
        axis_y.setLabelFormat("%.2f €")
        axis_y.setTitleText("Amount")
        
        # Set range with padding
        axis_y.setRange(0, max_value * 1.1)
        
        self.chart.addAxis(axis_x, Qt.AlignBottom)
        self.chart.addAxis(axis_y, Qt.AlignLeft)
        
        series.attachAxis(axis_x)
        series.attachAxis(axis_y)
        
    def _create_area_chart(self, data, categories):
        """Create area chart visualization."""
        # Create base line series
        line_series_list = []
        area_series_list = []
        
        for i, name in enumerate(self.series_names):
            upper_series = QLineSeries()
            
            # Skip savings for area chart - it's derived
            if name == "Savings" or i >= 2:
                continue
                
            line_series_list.append(upper_series)
            
        # Fill with data
        max_value = 0
        
        for i, month_data in enumerate(data):
            values = [
                month_data.get('income', 0),
                month_data.get('expenses', 0)
            ]
            
            # Update max
            max_value = max(max_value, max(values))
            
            # Add data points
            for j, series in enumerate(line_series_list):
                if j < len(values):
                    series.append(i, values[j])
        
        # Create area series
        # Income area
        income_area = QAreaSeries(line_series_list[0])
        income_area.setName(self.series_names[0])
        color = QColor(self.series_colors[0])
        color.setAlpha(150)
        income_area.setBrush(QBrush(color))
        income_area.setPen(QPen(Qt.NoPen))
        
        # Expenses area
        if len(line_series_list) > 1:
            expenses_area = QAreaSeries(line_series_list[1])
            expenses_area.setName(self.series_names[1])
            color = QColor(self.series_colors[1])
            color.setAlpha(150)
            expenses_area.setBrush(QBrush(color))
            expenses_area.setPen(QPen(Qt.NoPen))
            area_series_list = [income_area, expenses_area]
        else:
            area_series_list = [income_area]
        
        # Add series to chart
        for series in area_series_list:
            self.chart.addSeries(series)
            
        # Create axes
        axis_x = QBarCategoryAxis()
        axis_x.append(categories)
        
        axis_y = QValueAxis()
        axis_y.setLabelFormat("%.2f €")
        axis_y.setTitleText("Amount")
        
        # Set range with padding
        axis_y.setRange(0, max_value * 1.1)
        
        self.chart.addAxis(axis_x, Qt.AlignBottom)
        self.chart.addAxis(axis_y, Qt.AlignLeft)
        
        # Attach axes to all series
        for series in area_series_list:
            series.attachAxis(axis_x)
            series.attachAxis(axis_y)


class EnhancedPieChart(QFrame):
    """Enhanced pie chart with interactive features for category visualization."""
    
    def __init__(self, title="Spending by Category"):
        """
        Initialize the enhanced pie chart.
        
        Args:
            title: Chart title
        """
        super().__init__()
        
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Apply modern card styling
        self.setObjectName("chartCard")
        self.setStyleSheet("""
            #chartCard {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
            }
        """)
        
        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 20))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Title with info button
        header_layout = QHBoxLayout()
        
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Info label for selected category
        self.info_label = QLabel("Click on a category for details")
        self.info_label.setStyleSheet("font-size: 12px; color: #7f8c8d; font-style: italic;")
        header_layout.addWidget(self.info_label)
        
        layout.addLayout(header_layout)
        
        # Create the chart
        self.chart = QChart()
        self.chart.setTitle("")
        self.chart.setAnimationOptions(QChart.SeriesAnimations)
        self.chart.legend().setVisible(True)
        self.chart.legend().setAlignment(Qt.AlignRight)
        
        # Create chart view with rounded corners
        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.Antialiasing)
        layout.addWidget(self.chart_view)
        
        # Category breakdown label for details
        self.category_detail = QLabel()
        self.category_detail.setAlignment(Qt.AlignCenter)
        self.category_detail.setStyleSheet("font-size: 13px; color: #34495e; margin-top: 5px;")
        layout.addWidget(self.category_detail)
        
        # Track selected category
        self.selected_category = None
        
        # Category colors (modern palette)
        self.category_colors = [
            "#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6", 
            "#1abc9c", "#d35400", "#34495e", "#27ae60", "#e67e22",
            "#16a085", "#c0392b", "#8e44ad", "#f1c40f", "#2980b9"
        ]
        
        # Initial dummy data
        self.update_data([])
        
    def update_data(self, category_data):
        """
        Update the chart with new data.
        
        Args:
            category_data: List of category breakdowns
        """
        # Clear existing series
        self.chart.removeAllSeries()
        self.category_detail.setText("")
        
        # Filter for expense categories only and sort by amount
        expense_categories = [cat for cat in category_data 
                             if cat.get('type', '') == 'expense' and cat.get('total', 0) > 0]
        
        # No data case
        if not expense_categories:
            self.chart.setTitle("No expense data available")
            self.info_label.setText("No data available")
            return
            
        # Sort by amount
        expense_categories.sort(key=lambda x: x.get('total', 0), reverse=True)
        
        # Take top 6 categories, group others
        top_categories = expense_categories[:6]
        other_categories = expense_categories[6:]
        
        # Create pie series
        series = QPieSeries()
        
        # Total expenses for percentage calculation
        total_expenses = sum(cat.get('total', 0) for cat in expense_categories)
        
        # Add top categories
        for i, category in enumerate(top_categories):
            name = category.get('name', 'Unknown')
            amount = category.get('total', 0)
            percentage = (amount / total_expenses) * 100 if total_expenses > 0 else 0
            
            # Create label with percentage
            label = f"{name}: {percentage:.1f}%"
            
            # Add slice
            slice = series.append(label, amount)
            
            # Set color
            color_index = i % len(self.category_colors)
            slice.setBrush(QColor(self.category_colors[color_index]))
            
            # Store category data in slice
            slice.setProperty("category_name", name)
            slice.setProperty("category_amount", amount)
            slice.setProperty("category_percentage", percentage)
            
            # Connect signals
            slice.hovered.connect(self.handle_slice_hovered)
            slice.clicked.connect(self.handle_slice_clicked)
            
            # Explode the largest slice slightly
            if i == 0:
                slice.setExploded(True)
                slice.setLabelVisible(True)
        
        # Add "Other" category if needed
        if other_categories:
            other_total = sum(cat.get('total', 0) for cat in other_categories)
            other_percentage = (other_total / total_expenses) * 100 if total_expenses > 0 else 0
            
            # Add slice
            other_slice = series.append(f"Other: {other_percentage:.1f}%", other_total)
            other_slice.setBrush(QColor("#95a5a6"))  # Gray for "Other"
            
            # Store other categories in slice
            other_slice.setProperty("category_name", "Other")
            other_slice.setProperty("category_amount", other_total)
            other_slice.setProperty("category_percentage", other_percentage)
            other_slice.setProperty("other_categories", other_categories)
            
            # Connect signals
            other_slice.hovered.connect(self.handle_slice_hovered)
            other_slice.clicked.connect(self.handle_slice_clicked)
        
        # Add to chart
        self.chart.addSeries(series)
        
        # Update info label
        self.info_label.setText("Click on a category for details")
        
    def handle_slice_hovered(self, state, slice):
        """Handle mouse hover over pie slice."""
        if state:
            # Hover in - make slice stand out
            slice.setExploded(True)
            
            # Show tooltip-like info
            name = slice.property("category_name")
            amount = slice.property("category_amount")
            percentage = slice.property("category_percentage")
            
            self.info_label.setText(f"{name}: {amount:.2f} € ({percentage:.1f}%)")
        else:
            # Hover out - reset to normal unless selected
            if self.selected_category != slice.property("category_name"):
                slice.setExploded(False)
                
            self.info_label.setText("Click on a category for details")
            
    def handle_slice_clicked(self, slice):
        """Handle click on pie slice for detailed view."""
        # Get category data
        name = slice.property("category_name")
        amount = slice.property("category_amount")
        percentage = slice.property("category_percentage")
        
        # Update detail text
        if name == "Other":
            # Show breakdown of "Other" categories
            other_categories = slice.property("other_categories")
            
            if other_categories:
                detail_text = "<b>Other Categories:</b><br>"
                
                for cat in other_categories[:5]:  # Show top 5 other categories
                    cat_name = cat.get('name', 'Unknown')
                    cat_amount = cat.get('total', 0)
                    cat_percentage = (cat_amount / amount) * 100 if amount > 0 else 0
                    
                    detail_text += f"• {cat_name}: {cat_amount:.2f} € ({cat_percentage:.1f}%)<br>"
                    
                if len(other_categories) > 5:
                    detail_text += f"• ... and {len(other_categories) - 5} more categories"
                    
                self.category_detail.setText(detail_text)
        else:
            # Show details for single category
            self.category_detail.setText(
                f"<b>{name}</b><br>"
                f"Amount: {amount:.2f} €<br>"
                f"Percentage of total: {percentage:.1f}%"
            )
            
        # Update selected state
        self.selected_category = name


class InsightCard(QFrame):
    """Interactive card showing AI insights."""
    
    def __init__(self, title, subtitle, icon_type="lightbulb", action_text="View details"):
        """
        Initialize an insight card.
        
        Args:
            title: Card title
            subtitle: Card subtitle or description
            icon_type: Type of icon to display (lightbulb, chart, money, subscription)
            action_text: Text for the action button
        """
        super().__init__()
        
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumHeight(160)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        # Set hover functionality
        self.setMouseTracking(True)
        self.is_hovered = False
        
        # Apply modern card styling
        self.setObjectName("insightCard")
        self.setStyleSheet("""
            #insightCard {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
            }
            #insightCard:hover {
                background-color: #f8f9fa;
                border: 1px solid #d0d0d0;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        
        # Add shadow effect for depth
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 20))
        shadow.setOffset(0, 3)
        self.setGraphicsEffect(shadow)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(8)
        
        # Header layout with icon
        header_layout = QHBoxLayout()
        
        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Icon
        if icon_type == "lightbulb":
            icon_color = "#f39c12"  # Yellow
        elif icon_type == "chart":
            icon_color = "#3498db"  # Blue
        elif icon_type == "money":
            icon_color = "#2ecc71"  # Green
        elif icon_type == "subscription":
            icon_color = "#9b59b6"  # Purple
        else:
            icon_color = "#95a5a6"  # Gray
            
        icon_label = QLabel("💡")  # Default to lightbulb emoji
        
        if icon_type == "chart":
            icon_label.setText("📊")
        elif icon_type == "money":
            icon_label.setText("💰")
        elif icon_type == "subscription":
            icon_label.setText("🔄")
            
        icon_label.setStyleSheet(f"font-size: 20px; color: {icon_color};")
        header_layout.addWidget(icon_label)
        
        layout.addLayout(header_layout)
        
        # Subtitle
        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setStyleSheet("font-size: 14px; color: #7f8c8d;")
        self.subtitle_label.setWordWrap(True)
        layout.addWidget(self.subtitle_label)
        
        # Spacer
        layout.addStretch()
        
        # Action button
        action_layout = QHBoxLayout()
        action_layout.addStretch()
        
        self.action_button = QPushButton(action_text)
        self.action_button.setCursor(Qt.PointingHandCursor)
        action_layout.addWidget(self.action_button)
        
        layout.addLayout(action_layout)
        
        # Status label
        self.status_label = QLabel("No recommendations")
        self.status_label.setStyleSheet("font-size: 12px; color: #7f8c8d; font-style: italic;")
        layout.addWidget(self.status_label)
        
    def update_content(self, title, subtitle, status_text):
        """Update the card content."""
        # Find and update title
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i)
            if isinstance(item, QHBoxLayout):
                for j in range(item.count()):
                    widget = item.itemAt(j).widget()
                    if isinstance(widget, QLabel) and not widget.text() in ["💡", "📊", "💰", "🔄"]:
                        widget.setText(title)
                        break
        
        # Update subtitle and status
        self.subtitle_label.setText(subtitle)
        self.status_label.setText(status_text)
        
    def enterEvent(self, event):
        """Handle mouse enter event."""
        self.is_hovered = True
        # Could add animation here
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        """Handle mouse leave event."""
        self.is_hovered = False
        # Could add animation here
        super().leaveEvent(event)
        
        
class SpendingInsightsWidget(QFrame):
    """Widget showing AI-generated insights about spending patterns."""
    
    def __init__(self, db_manager):
        """
        Initialize the spending insights widget.
        
        Args:
            db_manager: Database manager instance
        """
        super().__init__()
        
        self.db_manager = db_manager
        
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Apply modern card styling
        self.setObjectName("chartCard")
        self.setStyleSheet("""
            #chartCard {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
            }
        """)
        
        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 20))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        title_label = QLabel("Spending Insights")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(title_label)
        
        # Insights group
        insights_group = QGroupBox()
        insights_group.setStyleSheet("""
            QGroupBox {
                border: none;
                padding-top: 10px;
            }
        """)
        insights_layout = QVBoxLayout(insights_group)
        insights_layout.setSpacing(10)
        
        # Placeholder for insights
        self.insights_layout = insights_layout
        
        # Add group to main layout with scroll
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setWidget(insights_group)
        layout.addWidget(scroll_area)
        
        # Initial insights
        self.update_insights([])
        
    def update_insights(self, insights):
        """
        Update the displayed insights.
        
        Args:
            insights: List of insight dictionaries
        """
        # Clear existing insights
        while self.insights_layout.count():
            child = self.insights_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
        # No insights case
        if not insights:
            no_insights_label = QLabel("No insights available yet. Add more transactions to generate insights.")
            no_insights_label.setStyleSheet("color: #7f8c8d; font-style: italic;")
            no_insights_label.setAlignment(Qt.AlignCenter)
            no_insights_label.setWordWrap(True)
            self.insights_layout.addWidget(no_insights_label)
            return
            
        # Add insights
        for insight in insights[:5]:  # Show top 5 insights
            insight_card = self._create_insight_card(insight)
            self.insights_layout.addWidget(insight_card)
            
        # Add spacer at the end
        self.insights_layout.addStretch()
        
    def _create_insight_card(self, insight):
        """Create a card for a single insight."""
        card = QFrame()
        card.setObjectName("insightItemCard")
        card.setStyleSheet("""
            #insightItemCard {
                background-color: #f8f9fa;
                border-radius: 6px;
                border-left: 4px solid #3498db;
                padding: 5px;
            }
        """)
        
        # Set border color based on insight type
        insight_type = insight.get('type', '')
        if insight_type.endswith('alert'):
            card.setStyleSheet("""
                #insightItemCard {
                    background-color: #f8f9fa;
                    border-radius: 6px;
                    border-left: 4px solid #e74c3c;
                    padding: 5px;
                }
            """)
        elif insight_type.endswith('positive'):
            card.setStyleSheet("""
                #insightItemCard {
                    background-color: #f8f9fa;
                    border-radius: 6px;
                    border-left: 4px solid #2ecc71;
                    padding: 5px;
                }
            """)
        
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(10, 10, 10, 10)
        card_layout.setSpacing(5)
        
        # Title
        title_label = QLabel(insight.get('title', 'Insight'))
        title_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        card_layout.addWidget(title_label)
        
        # Description
        description = insight.get('description', '')
        if description:
            desc_label = QLabel(description)
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("color: #34495e;")
            card_layout.addWidget(desc_label)
            
        # Recommendation
        recommendation = insight.get('recommendation', '')
        if recommendation:
            rec_label = QLabel(recommendation)
            rec_label.setWordWrap(True)
            rec_label.setStyleSheet("color: #16a085; font-style: italic;")
            card_layout.addWidget(rec_label)
            
        return card
        
        
class EnhancedComparisonChart(QFrame):
    """Chart showing monthly comparison of income and expenses."""
    
    def __init__(self, title):
        """Initialize the comparison chart."""
        super().__init__()
        
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Apply modern card styling
        self.setObjectName("chartCard")
        self.setStyleSheet("""
            #chartCard {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
            }
        """)
        
        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 20))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(title_label)
        
        # Create the chart
        self.chart = QChart()
        self.chart.setTitle("")
        self.chart.setAnimationOptions(QChart.SeriesAnimations)
        self.chart.legend().setVisible(True)
        self.chart.legend().setAlignment(Qt.AlignBottom)
        
        # Create chart view
        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.Antialiasing)
        layout.addWidget(self.chart_view)
        
        # Initial dummy data
        self.update_data([])
        
    def update_data(self, data):
        """Update chart with comparison data."""
        # Clear existing series
        self.chart.removeAllSeries()
        
        # No data case
        if not data:
            self.chart.setTitle("No data available")
            return
        
        # Create percent bar series
        series = QPercentBarSeries()
        
        # Create bar sets
        income_set = QBarSet("Income")
        income_set.setColor(QColor("#2ecc71"))  # Green
        
        expense_set = QBarSet("Expenses")
        expense_set.setColor(QColor("#e74c3c"))  # Red
        
        savings_set = QBarSet("Savings")
        savings_set.setColor(QColor("#3498db"))  # Blue
        
        # Categories for months
        categories = []
        
        # Add data
        for month_data in data:
            month_num = month_data.get('month', 0)
            if 1 <= month_num <= 12:
                month_name = calendar.month_abbr[month_num]
                categories.append(month_name)
                
                income = month_data.get('income', 0)
                expenses = month_data.get('expenses', 0)
                savings = month_data.get('savings', 0)
                
                total = income
                
                # Calculate percentages manually
                if total > 0:
                    income_set.append(100)  # 100%
                    expense_percentage = (expenses / income) * 100
                    expense_set.append(expense_percentage)
                    
                    # Savings can be negative
                    savings_percentage = (savings / income) * 100 if savings > 0 else 0
                    savings_set.append(savings_percentage)
                else:
                    income_set.append(0)
                    expense_set.append(0)
                    savings_set.append(0)
        
        # Add sets to series
        series.append(income_set)
        series.append(expense_set)
        series.append(savings_set)
        
        # Add series to chart
        self.chart.addSeries(series)
        
        # Create axes
        axis_x = QBarCategoryAxis()
        axis_x.append(categories)
        
        axis_y = QValueAxis()
        axis_y.setRange(0, 100)
        axis_y.setLabelFormat("%.0f%%")
        
        self.chart.addAxis(axis_x, Qt.AlignBottom)
        self.chart.addAxis(axis_y, Qt.AlignLeft)
        
        series.attachAxis(axis_x)
        series.attachAxis(axis_y)
        
        
class CategoryBreakdownWidget(QFrame):
    """Widget showing detailed category breakdown."""
    
    def __init__(self, title):
        """Initialize the category breakdown widget."""
        super().__init__()
        
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Apply modern card styling
        self.setObjectName("chartCard")
        self.setStyleSheet("""
            #chartCard {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
            }
        """)
        
        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 20))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(title_label)
        
        # Table for category breakdown
        self.category_table = QTableWidget()
        self.category_table.setColumnCount(3)
        self.category_table.setHorizontalHeaderLabels(["Category", "Amount", "% of Total"])
        
        # Set column stretch
        header = self.category_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        
        # Style the table
        self.category_table.setAlternatingRowColors(True)
        self.category_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.category_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.category_table.verticalHeader().setVisible(False)
        self.category_table.setShowGrid(False)
        
        layout.addWidget(self.category_table)
        
        # Initial dummy data
        self.update_data([])
        
    def update_data(self, category_data):
        """Update the table with category data."""
        # Clear table
        self.category_table.setRowCount(0)
        
        # Filter for expense categories only and sort by amount
        expense_categories = [cat for cat in category_data 
                             if cat.get('type', '') == 'expense' and cat.get('total', 0) > 0]
        
        # No data case
        if not expense_categories:
            return
            
        # Sort by amount
        expense_categories.sort(key=lambda x: x.get('total', 0), reverse=True)
        
        # Calculate total for percentages
        total_expenses = sum(cat.get('total', 0) for cat in expense_categories)
        
        # Add rows
        for row, category in enumerate(expense_categories):
            self.category_table.insertRow(row)
            
            # Category name
            name_item = QTableWidgetItem(category.get('name', 'Unknown'))
            self.category_table.setItem(row, 0, name_item)
            
            # Amount
            amount = category.get('total', 0)
            amount_item = QTableWidgetItem(f"{amount:.2f} €")
            amount_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.category_table.setItem(row, 1, amount_item)
            
            # Percentage
            percentage = (amount / total_expenses) * 100 if total_expenses > 0 else 0
            percentage_item = QTableWidgetItem(f"{percentage:.1f}%")
            percentage_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.category_table.setItem(row, 2, percentage_item)
            
            
class SavingsTargetWidget(QFrame):
    """Widget showing savings goals and progress."""
    
    def __init__(self, title):
        """Initialize the savings target widget."""
        super().__init__()
        
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumSize(400, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Apply modern card styling
        self.setObjectName("chartCard")
        self.setStyleSheet("""
            #chartCard {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
            }
            QProgressBar {
                border: 1px solid #e0e0e0;
                border-radius: 5px;
                background-color: #f5f5f5;
                text-align: center;
                height: 12px;
            }
            QProgressBar::chunk {
                border-radius: 5px;
                background-color: #3498db;
            }
        """)
        
        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 20))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(title_label)
        
        # Progress bar container
        self.progress_container = QVBoxLayout()
        self.progress_container.setSpacing(15)
        
        layout.addLayout(self.progress_container)
        
        # Message for no goals
        self.no_goals_label = QLabel("No savings goals set up yet.")
        self.no_goals_label.setAlignment(Qt.AlignCenter)
        self.no_goals_label.setStyleSheet("color: #7f8c8d; font-style: italic;")
        layout.addWidget(self.no_goals_label)
        
        # Initial dummy data
        self.update_data([], 0)
        
    def update_data(self, goals, current_savings):
        """Update the widget with savings goals data."""
        # Clear progress container
        while self.progress_container.count():
            child = self.progress_container.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
        # Show or hide no goals message
        self.no_goals_label.setVisible(not goals)
        
        # No goals case
        if not goals:
            return
            
        # Add progress bars for each goal
        for goal in goals:
            goal_name = goal.get('name', 'Unnamed Goal')
            goal_target = goal.get('target_amount', 0)
            goal_current = min(current_savings, goal_target)  # Cap at target
            
            # Calculate percentage
            goal_percentage = (goal_current / goal_target) * 100 if goal_target > 0 else 0
            
            # Create goal layout
            goal_layout = QVBoxLayout()
            goal_layout.setSpacing(5)
            
            # Goal header with name and amounts
            header_layout = QHBoxLayout()
            
            name_label = QLabel(goal_name)
            name_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
            header_layout.addWidget(name_label)
            
            header_layout.addStretch()
            
            amount_label = QLabel(f"{goal_current:.2f} € / {goal_target:.2f} €")
            amount_label.setStyleSheet("color: #34495e;")
            header_layout.addWidget(amount_label)
            
            goal_layout.addLayout(header_layout)
            
            # Progress bar
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 100)
            progress_bar.setValue(int(goal_percentage))
            
            # Color based on progress
            if goal_percentage < 33:
                progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #e74c3c; }")
            elif goal_percentage < 66:
                progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #f39c12; }")
            else:
                progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #2ecc71; }")
                
            goal_layout.addWidget(progress_bar)
            
            # Add to container
            self.progress_container.addLayout(goal_layout)
            
        # Add spacer at the end
        self.progress_container.addStretch()
        
        
class BudgetComparisonWidget(QFrame):
    """Widget comparing budget vs actual spending."""
    
    def __init__(self, title):
        """Initialize the budget comparison widget."""
        super().__init__()
        
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumSize(400, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Apply modern card styling
        self.setObjectName("chartCard")
        self.setStyleSheet("""
            #chartCard {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
            }
        """)
        
        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 20))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(title_label)
        
        # Create the chart
        self.chart = QChart()
        self.chart.setTitle("")
        self.chart.setAnimationOptions(QChart.SeriesAnimations)
        self.chart.legend().setVisible(True)
        self.chart.legend().setAlignment(Qt.AlignBottom)
        
        # Create chart view
        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.Antialiasing)
        layout.addWidget(self.chart_view)
        
        # Initial dummy data
        self.update_data([])
        
    def update_data(self, budget_data):
        """Update the chart with budget comparison data."""
        # Clear existing series
        self.chart.removeAllSeries()
        
        # No data case
        if not budget_data:
            self.chart.setTitle("No budget data available")
            return
            
        # Create bar sets
        budget_set = QBarSet("Budget")
        budget_set.setColor(QColor("#3498db"))  # Blue
        
        actual_set = QBarSet("Actual")
        actual_set.setColor(QColor("#e74c3c"))  # Red
        
        # Categories for chart
        categories = []
        
        # Add data
        for category_data in budget_data:
            category = category_data.get('category', 'Other')
            budget = category_data.get('budget', 0)
            actual = category_data.get('actual', 0)
            
            categories.append(category)
            budget_set.append(budget)
            actual_set.append(actual)
            
        # Create bar series
        series = QBarSeries()
        series.append(budget_set)
        series.append(actual_set)
        
        # Add series to chart
        self.chart.addSeries(series)
        
        # Create axes
        axis_x = QBarCategoryAxis()
        axis_x.append(categories)
        
        axis_y = QValueAxis()
        axis_y.setLabelFormat("%.2f €")
        
        # Find maximum value for y-axis scaling
        max_budget = max(budget_set) if len(budget_set) > 0 else 0
        max_actual = max(actual_set) if len(actual_set) > 0 else 0
        max_value = max(max_budget, max_actual)
        
        # Add some headroom above the maximum value
        axis_y.setRange(0, max_value * 1.1)
        
        self.chart.addAxis(axis_x, Qt.AlignBottom)
        self.chart.addAxis(axis_y, Qt.AlignLeft)
        
        series.attachAxis(axis_x)
        series.attachAxis(axis_y)
        
        
class EnhancedTransactionsWidget(QFrame):
    """Enhanced transactions table with filtering and sorting."""
    
    def __init__(self, db_manager):
        """Initialize the transactions widget."""
        super().__init__()
        
        self.db_manager = db_manager
        
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Apply modern card styling
        self.setObjectName("chartCard")
        self.setStyleSheet("""
            #chartCard {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
            }
        """)
        
        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 20))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Header with title and controls
        header_layout = QHBoxLayout()
        
        # Title
        title_label = QLabel("Recent Transactions")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Filter dropdown
        filter_layout = QHBoxLayout()
        filter_layout.setSpacing(5)
        
        filter_label = QLabel("Filter:")
        filter_label.setStyleSheet("font-weight: bold;")
        filter_layout.addWidget(filter_label)
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All", "Income", "Expenses"])
        self.filter_combo.currentIndexChanged.connect(self.apply_filter)
        filter_layout.addWidget(self.filter_combo)
        
        header_layout.addLayout(filter_layout)
        
        layout.addLayout(header_layout)
        
        # Transactions table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Date", "Description", "Category", "Amount", "Type"])
        
        # Set stretch for columns
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Date
        header.setSectionResizeMode(1, QHeaderView.Stretch)  # Description
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Category
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Amount
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Type
        
        # Style the table
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        
        layout.addWidget(self.table)
        
        # Initial load
        self.refresh_transactions()
        
    def refresh_transactions(self, time_range=None):
        """
        Load recent transactions from the database with optional time range filter.
        
        Args:
            time_range: Optional time range filter (month, quarter, ytd, year)
        """
        try:
            # Get transactions with time range
            if time_range:
                transactions = self.db_manager.get_transactions_by_range(time_range)
            else:
                transactions = self.db_manager.get_transactions(limit=50)
                
            # Store transactions for filtering
            self.transactions = transactions
            
            # Apply current filter
            self.apply_filter(self.filter_combo.currentIndex())
            
        except Exception as e:
            logging.error(f"Error loading transactions: {e}")
            
    def apply_filter(self, filter_index):
        """Apply filter to transactions."""
        # Get filtered transactions
        if filter_index == 0:  # All
            filtered_transactions = self.transactions
        elif filter_index == 1:  # Income
            filtered_transactions = [tx for tx in self.transactions if tx.get('is_income', False)]
        else:  # Expenses
            filtered_transactions = [tx for tx in self.transactions if not tx.get('is_income', False)]
            
        # Update table
        self.update_table(filtered_transactions)
        
    def update_table(self, transactions):
        """Update the table with transactions."""
        # Clear table
        self.table.setRowCount(0)
        
        # Add transactions
        for row, tx in enumerate(transactions):
            self.table.insertRow(row)
            
            # Date
            date_str = tx.get('date', '')
            try:
                # Format date for display
                date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                formatted_date = date_obj.strftime("%d.%m.%Y")
            except ValueError:
                formatted_date = date_str
                
            date_item = QTableWidgetItem(formatted_date)
            self.table.setItem(row, 0, date_item)
            
            # Description
            desc_item = QTableWidgetItem(tx.get('description', ''))
            self.table.setItem(row, 1, desc_item)
            
            # Category
            cat_item = QTableWidgetItem(tx.get('category_name', ''))
            self.table.setItem(row, 2, cat_item)
            
            # Amount
            amount = tx.get('amount', 0)
            amount_item = QTableWidgetItem(f"{amount:.2f} €")
            amount_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            
            # Set color based on transaction type
            if tx.get('is_income'):
                amount_item.setForeground(QColor('#2ecc71'))  # Green
            else:
                amount_item.setForeground(QColor('#e74c3c'))  # Red
                
            self.table.setItem(row, 3, amount_item)
            
            # Type
            type_item = QTableWidgetItem("Income" if tx.get('is_income') else "Expense")
            type_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 4, type_item)
            
            
def update_data(self, category_data):
        """
        Update the chart with new data.
        
        Args:
            category_data: List of category breakdowns
        """
        # Clear existing series
        self.chart.removeAllSeries()
        
        # Filter for expense categories only and sort by amount
        expense_categories = [cat for cat in category_data 
                             if cat.get('type') == 'expense' and cat.get('total', 0) > 0]
        
        # No data case
        if not expense_categories:
            self.chart.setTitle("No expense data available")
            return
            
        # Sort by amount
        expense_categories.sort(key=lambda x: x.get('total', 0), reverse=True)
        
        # Take top 7 categories, group others
        top_categories = expense_categories[:7]
        other_categories = expense_categories[7:]
        
        # Create pie series
        series = QPieSeries()
        
        # Add top categories
        for category in top_categories:
            name = category.get('name', 'Unknown')
            amount = category.get('total', 0)
            
            # Add slice
            slice = series.append(name, amount)
            
            # Set color if available
            if 'color' in category and category['color']:
                slice.setBrush(QColor(category['color']))
            
            # Explode the largest slice
            if category == top_categories[0]:
                slice.setExploded(True)
                slice.setLabelVisible(True)
        
        # Add "Other" category if needed
        if other_categories:
            other_total = sum(cat.get('total', 0) for cat in other_categories)
            series.append("Other", other_total)
        
        # Add to chart
        self.chart.addSeries(series)


class MonthlyTrendChart(QFrame):
    """Line chart showing income, expenses, and savings trends over time."""
    
    def __init__(self):
        """Initialize the monthly trend chart."""
        super().__init__()
        
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Apply modern card styling
        self.setObjectName("chartCard")
        
        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 30))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        title_label = QLabel("Monthly Trends")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title_label)
        
        # Create the chart
        self.chart = QChart()
        self.chart.setTitle("")
        self.chart.setAnimationOptions(QChart.SeriesAnimations)
        self.chart.legend().setVisible(True)
        self.chart.legend().setAlignment(Qt.AlignBottom)
        
        # Create chart view
        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.Antialiasing)
        layout.addWidget(self.chart_view)
        
        # Initial dummy data
        self.update_data([])
        
    def update_data(self, yearly_data):
        """
        Update the chart with new data.
        
        Args:
            yearly_data: List of dictionaries with monthly data
        """
        # Clear existing series
        self.chart.removeAllSeries()
        
        # No data case
        if not yearly_data:
            self.chart.setTitle("No data available")
            return
            
        # Create line series for income, expenses, and savings
        income_series = QLineSeries()
        income_series.setName("Income")
        income_series.setPen(QPen(QColor("#2ecc71"), 2))  # Green
        
        expense_series = QLineSeries()
        expense_series.setName("Expenses")
        expense_series.setPen(QPen(QColor("#e74c3c"), 2))  # Red
        
        savings_series = QLineSeries()
        savings_series.setName("Savings")
        savings_series.setPen(QPen(QColor("#3498db"), 2))  # Blue
        
        # Fill with data and collect categories
        categories = []
        
        # Sort data by month
        sorted_data = sorted(yearly_data, key=lambda x: x.get('month', 0))
        
        # Track min/max values for axis scaling
        max_value = 0
        min_value = 0
        
        for i, month_data in enumerate(sorted_data):
            month_num = month_data.get('month', 0)
            if 1 <= month_num <= 12:
                month_name = calendar.month_abbr[month_num]
                categories.append(month_name)
                
                income = month_data.get('income', 0)
                expenses = month_data.get('expenses', 0)
                savings = month_data.get('savings', 0)
                
                # Update min/max
                max_value = max(max_value, income, expenses, savings)
                min_value = min(min_value, savings)  # Savings can be negative
                
                # Add points (x-coordinate is month index)
                income_series.append(i, income)
                expense_series.append(i, expenses)
                savings_series.append(i, savings)
        
        # Add series to chart
        self.chart.addSeries(income_series)
        self.chart.addSeries(expense_series)
        self.chart.addSeries(savings_series)
        
        # Create axes
        axis_x = QBarCategoryAxis()
        axis_x.append(categories)
        
        axis_y = QValueAxis()
        axis_y.setLabelFormat("%.2f €")
        axis_y.setTitleText("Amount")
        
        # Add some padding above the maximum value and below minimum value
        axis_y.setRange(min_value * 1.1 if min_value < 0 else 0, max_value * 1.1)
        
        self.chart.addAxis(axis_x, Qt.AlignBottom)
        self.chart.addAxis(axis_y, Qt.AlignLeft)
        
        income_series.attachAxis(axis_x)
        income_series.attachAxis(axis_y)
        expense_series.attachAxis(axis_x)
        expense_series.attachAxis(axis_y)
        savings_series.attachAxis(axis_x)
        savings_series.attachAxis(axis_y)


class RecentTransactionsWidget(QFrame):
    """Widget showing recent transactions."""
    
    def __init__(self, db_manager):
        """
        Initialize the recent transactions widget.
        
        Args:
            db_manager: Database manager instance
        """
        super().__init__()
        
        self.db_manager = db_manager
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Apply modern card styling
        self.setObjectName("chartCard")
        
        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 30))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        title_label = QLabel("Recent Transactions")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title_label)
        
        # Transactions table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Date", "Description", "Category", "Amount"])
        
        # Set stretch for columns
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Date
        header.setSectionResizeMode(1, QHeaderView.Stretch)  # Description
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Category
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Amount
        
        # Style the table
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.verticalHeader().setVisible(False)
        
        layout.addWidget(self.table)
        
        # Initial load
        self.refresh_transactions()
        
    def refresh_transactions(self):
        """Load recent transactions from the database."""
        try:
            # Get last 10 transactions
            transactions = self.db_manager.get_transactions(limit=10)
            
            # Clear table
            self.table.setRowCount(0)
            
            # Add transactions
            for row, tx in enumerate(transactions):
                self.table.insertRow(row)
                
                # Date
                date_str = tx.get('date', '')
                try:
                    # Format date for display
                    date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                    formatted_date = date_obj.strftime("%d.%m.%Y")
                except ValueError:
                    formatted_date = date_str
                    
                date_item = QTableWidgetItem(formatted_date)
                self.table.setItem(row, 0, date_item)
                
                # Description
                desc_item = QTableWidgetItem(tx.get('description', ''))
                self.table.setItem(row, 1, desc_item)
                
                # Category
                cat_item = QTableWidgetItem(tx.get('category_name', ''))
                self.table.setItem(row, 2, cat_item)
                
                # Amount
                amount = tx.get('amount', 0)
                amount_item = QTableWidgetItem(f"{amount:.2f} €")
                amount_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                
                # Set color based on transaction type
                if tx.get('is_income'):
                    amount_item.setForeground(QColor('#2ecc71'))  # Green
                else:
                    amount_item.setForeground(QColor('#e74c3c'))  # Red
                    
                self.table.setItem(row, 3, amount_item)
                
        except Exception as e:
            logging.error(f"Error loading recent transactions: {e}")