#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modern Dashboard Tab Module

This module provides a dashboard view with financial summaries and visualizations
for the Financial Planner application using PySide6.
"""

import logging
import datetime
import calendar
from typing import List, Dict, Any, Optional
import matplotlib
matplotlib.use('QtAgg')  # Use generic QtAgg backend which works with both PySide6 and PyQt5
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

# Set up matplotlib for dark mode compatibility
def configure_matplotlib_for_theme(is_dark_mode=False):
    """Configure matplotlib colors based on light or dark theme."""
    if is_dark_mode:
        # Dark mode settings
        rcParams['text.color'] = 'white'
        rcParams['axes.facecolor'] = 'none'
        rcParams['axes.edgecolor'] = '#666666'
        rcParams['axes.labelcolor'] = 'white'
        rcParams['xtick.color'] = 'white'
        rcParams['ytick.color'] = 'white'
        rcParams['grid.color'] = '#444444'
        rcParams['figure.facecolor'] = 'none'
        rcParams['savefig.facecolor'] = 'none'
        rcParams['axes.grid'] = True
        rcParams['grid.alpha'] = 0.3
    else:
        # Light mode settings
        rcParams['text.color'] = 'black'
        rcParams['axes.facecolor'] = 'none'
        rcParams['axes.edgecolor'] = '#E5E5E5'
        rcParams['axes.labelcolor'] = 'black'
        rcParams['xtick.color'] = 'black'
        rcParams['ytick.color'] = 'black'
        rcParams['grid.color'] = '#E5E5E5'
        rcParams['figure.facecolor'] = 'none'
        rcParams['savefig.facecolor'] = 'none'
        rcParams['axes.grid'] = True
        rcParams['grid.alpha'] = 0.7

# Default to light mode initially
configure_matplotlib_for_theme(False)
import datetime
from matplotlib.patches import FancyBboxPatch
# Optional imports with fallbacks for better compatibility
try:
    from sklearn.linear_model import LinearRegression
except ImportError:
    # Create a simple fallback LinearRegression
    class LinearRegression:
        def __init__(self):
            self.coef_ = None
            self.intercept_ = 0
            
        def fit(self, X, y):
            X = np.array(X).flatten()
            y = np.array(y)
            n = len(X)
            if n <= 1:
                self.coef_ = [0]
                return self
                
            x_mean = np.mean(X)
            y_mean = np.mean(y)
            
            # Calculate slope (coef_)
            numerator = sum((X[i] - x_mean) * (y[i] - y_mean) for i in range(n))
            denominator = sum((X[i] - x_mean) ** 2 for i in range(n))
            
            if denominator == 0:
                self.coef_ = [0]
            else:
                self.coef_ = [numerator / denominator]
                
            # Calculate intercept
            self.intercept_ = y_mean - self.coef_[0] * x_mean
            return self
            
        def predict(self, X):
            X = np.array(X).flatten()
            return self.intercept_ + self.coef_[0] * X

import calendar
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
import re
import matplotlib.dates as mdates

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QGridLayout,
    QTableWidget, QTableWidgetItem, QHeaderView, QSizePolicy,
    QScrollArea, QGraphicsDropShadowEffect, QComboBox, QPushButton,
    QDialog, QTextBrowser, QToolTip, QTabWidget
)
from PySide6.QtCore import Qt, QSize, QDate, Signal, Slot, Property, QPoint, QEvent
from PySide6.QtGui import QColor, QFont, QBrush, QCursor


class InteractiveCanvas(FigureCanvas):
    """An interactive matplotlib canvas that can emit signals on hover/click."""
    
    def __init__(self, figure, parent=None):
        super().__init__(figure)
        self.setParent(parent)
        self.figure = figure
        self.active_element = None
        self.hovered_category = None
        self.hovered_index = None
        self.category_data = {}
        
    def set_category_data(self, data):
        """Store the category data for tooltip display."""
        self.category_data = data
        
    def mouseMoveEvent(self, event):
        """Handle mouse hover events to show tooltips on chart elements."""
        super().mouseMoveEvent(event)
        if not hasattr(self.figure, 'axes') or not self.figure.axes:
            return
            
        # Get the axes and mouse position
        ax = self.figure.axes[0]
        x, y = event.position().x(), event.position().y()
        
        # Convert to data coordinates
        inv = ax.transData.inverted()
        data_x, data_y = inv.transform((x, y))
        
        # Check if we're hovering over a pie chart wedge
        if hasattr(ax, 'patches') and ax.patches:
            for i, wedge in enumerate(ax.patches):
                if isinstance(wedge, matplotlib.patches.Wedge) and wedge.contains_point((data_x, data_y)):
                    if self.hovered_index != i:
                        self.hovered_index = i
                        if i < len(self.category_data):
                            category, amount = list(self.category_data.items())[i]
                            QToolTip.showText(
                                QCursor.pos(),
                                f"<b>{category}</b><br>Amount: {amount:.2f} €<br>Click for details",
                                self
                            )
                    return
                    
        # Check if we're hovering over a bar chart bar
        if hasattr(ax, 'containers') and ax.containers:
            for container in ax.containers:
                for i, rectangle in enumerate(container):
                    if isinstance(rectangle, matplotlib.patches.Rectangle) and rectangle.contains_point((data_x, data_y)):
                        height = rectangle.get_height()
                        QToolTip.showText(
                            QCursor.pos(),
                            f"<b>{ax.get_xticklabels()[i].get_text()}</b><br>Amount: {height:.2f} €<br>Click for details",
                            self
                        )
                        return
        
        # Hide tooltip if not hovering over anything            
        QToolTip.hideText()
        self.hovered_index = None
        
    def mouseReleaseEvent(self, event):
        """Handle mouse click events on chart elements."""
        super().mouseReleaseEvent(event)
        if not hasattr(self.figure, 'axes') or not self.figure.axes:
            return
            
        # Get the axes and mouse position
        ax = self.figure.axes[0]
        x, y = event.position().x(), event.position().y()
        
        # Convert to data coordinates
        inv = ax.transData.inverted()
        data_x, data_y = inv.transform((x, y))
        
        # Check if we clicked on a pie chart wedge
        if hasattr(ax, 'patches') and ax.patches:
            for i, wedge in enumerate(ax.patches):
                if isinstance(wedge, matplotlib.patches.Wedge) and wedge.contains_point((data_x, data_y)):
                    if i < len(self.category_data):
                        # Signal to parent that category was clicked
                        category, amount = list(self.category_data.items())[i]
                        self.parent().show_category_details(category, amount)
                    return
                    
        # Check if we clicked on a bar chart bar  
        if hasattr(ax, 'containers') and ax.containers:
            for container in ax.containers:
                for i, rectangle in enumerate(container):
                    if isinstance(rectangle, matplotlib.patches.Rectangle) and rectangle.contains_point((data_x, data_y)):
                        month = ax.get_xticklabels()[i].get_text()
                        height = rectangle.get_height()
                        self.parent().show_month_details(month, height, container.get_label())
                        return


class FinancialInsightDialog(QDialog):
    """Dialog for displaying AI-generated financial insights."""
    
    def __init__(self, parent=None, title="Financial Insights"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(600, 400)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Create text browser for rich text display
        self.text_browser = QTextBrowser()
        self.text_browser.setOpenExternalLinks(True)
        
        # Add to layout
        layout.addWidget(self.text_browser)
        
        # Add close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)
        
    def set_content(self, html_content):
        """Set the HTML content of the insight dialog."""
        self.text_browser.setHtml(html_content)


class ModernDashboardTab(QWidget):
    """Widget representing the modern dashboard tab in the application."""

    def __init__(self, db_manager, ai_components=None, dark_mode=False):
        """
        Initialize the dashboard tab.

        Args:
            db_manager: Database manager instance
            ai_components: Dictionary of AI components including subscription_analyzer, etc.
            dark_mode: Whether to use dark mode for charts (default: False)
        """
        super().__init__()
        
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        self.ai_components = ai_components or {}
        self.dark_mode = dark_mode
        
        # Configure matplotlib theme based on dark mode
        configure_matplotlib_for_theme(self.dark_mode)
        
        # Log available AI components
        if self.ai_components:
            self.logger.info(f"Dashboard initialized with AI components: {list(self.ai_components.keys())}")
        else:
            self.logger.warning("No AI components provided to dashboard")
        
        # Store transactions and analysis data for interactive features
        self.current_transactions = []
        self.category_data = {}
        self.yearly_data = []
        self.forecast_data = {}
        
        # Date range for filtering
        self.current_date_range = "1M"  # Default to 1 month for better pie chart visibility
        
        # Initialize UI
        self.init_ui()
        
        # Load initial data
        self.refresh_dashboard()
        
    def init_ui(self):
        """Initialize the user interface with macOS-style design."""
        # Main layout with margins for modern look
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Dashboard header
        header_layout = QHBoxLayout()
        
        # Dashboard title with modern styling
        title_label = QLabel("Financial Dashboard")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: var(--text-color, black);")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch(1)  # Flexible spacer
        
        # Time range filter
        time_range_label = QLabel("Time Range:")
        time_range_label.setStyleSheet("color: var(--text-color, black);")
        header_layout.addWidget(time_range_label)
        
        self.time_range_combo = QComboBox()
        self.time_range_combo.addItems(["1M", "3M", "6M", "1Y", "All"])
        self.time_range_combo.setCurrentText("1M")  # Default to 1 month for better pie chart view
        self.time_range_combo.currentTextChanged.connect(self.on_time_range_changed)
        header_layout.addWidget(self.time_range_combo)
        
        # Analysis button
        self.analysis_button = QPushButton("AI Insights")
        self.analysis_button.clicked.connect(self.show_ai_insights)
        header_layout.addWidget(self.analysis_button)
        
        main_layout.addLayout(header_layout)
        
        # Create a tab widget for different dashboard views
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid var(--border-color, #E5E5E5);
                border-radius: 5px;
                margin-top: -1px;
            }
            QTabBar::tab {
                background-color: transparent;
                border: 1px solid var(--border-color, #E5E5E5);
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 8px 16px;
                margin-right: 4px;
            }
            QTabBar::tab:selected {
                background-color: var(--card-bg-color, white);
                border-bottom: 1px solid var(--card-bg-color, white);
            }
        """)
        
        # Create Overview tab
        overview_tab = QWidget()
        overview_layout = QVBoxLayout(overview_tab)
        
        # Create a scroll area to handle overflow
        overview_scroll = QScrollArea()
        overview_scroll.setWidgetResizable(True)
        overview_scroll.setFrameShape(QFrame.NoFrame)
        
        # Container for overview content
        overview_container = QWidget()
        overview_container_layout = QVBoxLayout(overview_container)
        overview_container_layout.setContentsMargins(0, 0, 0, 0)
        overview_container_layout.setSpacing(20)
        
        # First row: Summary cards
        summary_row = QHBoxLayout()
        summary_row.setSpacing(15)
        
        # Income card
        self.income_card = SummaryCard("Income", "0.00 €", "green")
        summary_row.addWidget(self.income_card)
        
        # Expense card
        self.expense_card = SummaryCard("Expenses", "0.00 €", "red")
        summary_row.addWidget(self.expense_card)
        
        # Savings transfers card - for transfers from previous months
        self.savings_transfers_card = SummaryCard("Savings Transfers", "0.00 €", "purple")
        summary_row.addWidget(self.savings_transfers_card)
        
        # Balance card - net result of income minus expenses
        self.balance_card = SummaryCard("Balance", "0.00 €", "blue")
        summary_row.addWidget(self.balance_card)
        
        overview_container_layout.addLayout(summary_row)
        
        # Second row: Charts grid (2x2)
        charts_row = QGridLayout()
        charts_row.setSpacing(15)
        
        # Spending by Category chart
        self.category_chart_frame = QFrame()
        self.category_chart_frame.setFrameShape(QFrame.StyledPanel)
        self.category_chart_frame.setObjectName("chartCard")
        self.category_chart_frame.setStyleSheet("""
            #chartCard {
                background-color: var(--card-bg-color, white);
                border: 1px solid var(--border-color, #E5E5E5);
                border-radius: 10px;
            }
        """)
        
        category_chart_layout = QVBoxLayout(self.category_chart_frame)
        category_chart_layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        category_title = QLabel("Spending by Category")
        category_title.setStyleSheet("font-size: 16px; font-weight: bold; color: var(--text-color, black);")
        category_chart_layout.addWidget(category_title)
        
        # Chart container
        self.category_chart_container = QWidget()
        self.category_chart_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.category_chart_container.setMinimumHeight(250)
        
        category_container_layout = QVBoxLayout(self.category_chart_container)
        category_container_layout.setContentsMargins(0, 0, 0, 0)
        
        category_chart_layout.addWidget(self.category_chart_container)
        
        charts_row.addWidget(self.category_chart_frame, 0, 0)
        
        # Income vs Expenses chart
        self.income_expense_frame = QFrame()
        self.income_expense_frame.setFrameShape(QFrame.StyledPanel)
        self.income_expense_frame.setObjectName("chartCard")
        self.income_expense_frame.setStyleSheet("""
            #chartCard {
                background-color: var(--card-bg-color, white);
                border: 1px solid var(--border-color, #E5E5E5);
                border-radius: 10px;
            }
        """)
        
        income_expense_layout = QVBoxLayout(self.income_expense_frame)
        income_expense_layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        income_expense_title = QLabel("Income vs Expenses")
        income_expense_title.setStyleSheet("font-size: 16px; font-weight: bold; color: var(--text-color, black);")
        income_expense_layout.addWidget(income_expense_title)
        
        # Chart container
        self.income_expense_container = QWidget()
        self.income_expense_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.income_expense_container.setMinimumHeight(250)
        
        income_expense_container_layout = QVBoxLayout(self.income_expense_container)
        income_expense_container_layout.setContentsMargins(0, 0, 0, 0)
        
        income_expense_layout.addWidget(self.income_expense_container)
        
        charts_row.addWidget(self.income_expense_frame, 0, 1)
        
        # Monthly trend chart
        self.trend_frame = QFrame()
        self.trend_frame.setFrameShape(QFrame.StyledPanel)
        self.trend_frame.setObjectName("chartCard")
        self.trend_frame.setStyleSheet("""
            #chartCard {
                background-color: var(--card-bg-color, white);
                border: 1px solid var(--border-color, #E5E5E5);
                border-radius: 10px;
            }
        """)
        
        trend_layout = QVBoxLayout(self.trend_frame)
        trend_layout.setContentsMargins(15, 15, 15, 15)
        
        # Title with forecast indicator
        trend_title = QLabel("Monthly Trends & Forecast")
        trend_title.setStyleSheet("font-size: 16px; font-weight: bold; color: var(--text-color, black);")
        trend_layout.addWidget(trend_title)
        
        # Chart container
        self.trend_container = QWidget()
        self.trend_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.trend_container.setMinimumHeight(250)
        
        trend_container_layout = QVBoxLayout(self.trend_container)
        trend_container_layout.setContentsMargins(0, 0, 0, 0)
        
        trend_layout.addWidget(self.trend_container)
        
        charts_row.addWidget(self.trend_frame, 1, 0)
        
        # Recent transactions table
        self.transactions_frame = QFrame()
        self.transactions_frame.setFrameShape(QFrame.StyledPanel)
        self.transactions_frame.setObjectName("chartCard")
        self.transactions_frame.setStyleSheet("""
            #chartCard {
                background-color: var(--card-bg-color, white);
                border: 1px solid var(--border-color, #E5E5E5);
                border-radius: 10px;
            }
        """)
        
        transactions_layout = QVBoxLayout(self.transactions_frame)
        transactions_layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        transactions_title = QLabel("Recent Transactions")
        transactions_title.setStyleSheet("font-size: 16px; font-weight: bold; color: var(--text-color, black);")
        transactions_layout.addWidget(transactions_title)
        
        # Transactions table with modern styling
        self.transactions_table = QTableWidget()
        self.transactions_table.setColumnCount(4)
        self.transactions_table.setHorizontalHeaderLabels(["Date", "Description", "Category", "Amount"])
        
        # Style the table to match macOS aesthetics
        self.transactions_table.setStyleSheet("""
            QTableWidget {
                border: none;
                background-color: transparent;
                gridline-color: var(--separator-color, #F0F0F0);
            }
            QHeaderView::section {
                background-color: transparent;
                border: none;
                border-bottom: 1px solid var(--separator-color, #F0F0F0);
                padding: 4px;
                font-weight: bold;
                color: var(--header-text-color, #666666);
            }
            QTableWidget::item {
                border-bottom: 1px solid var(--separator-color, #F0F0F0);
                padding: 4px;
            }
        """)
        
        # Configure columns
        header = self.transactions_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Date
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Amount
        
        self.transactions_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.transactions_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.transactions_table.setAlternatingRowColors(True)
        self.transactions_table.verticalHeader().setVisible(False)
        
        transactions_layout.addWidget(self.transactions_table)
        
        charts_row.addWidget(self.transactions_frame, 1, 1)
        
        overview_container_layout.addLayout(charts_row)
        
        # Set the overview container as the scroll area's widget
        overview_scroll.setWidget(overview_container)
        overview_layout.addWidget(overview_scroll)
        
        # Add the overview tab to the tab widget
        self.tab_widget.addTab(overview_tab, "Overview")
        
        # Create Forecasting tab
        forecast_tab = QWidget()
        forecast_layout = QVBoxLayout(forecast_tab)

        # Create a scroll area for the forecast tab
        forecast_scroll = QScrollArea()
        forecast_scroll.setWidgetResizable(True)
        forecast_scroll.setFrameShape(QFrame.NoFrame)

        # Container for forecast content
        forecast_container = QWidget()
        forecast_container_layout = QVBoxLayout(forecast_container)
        forecast_container_layout.setContentsMargins(0, 0, 0, 0)
        forecast_container_layout.setSpacing(20)

        # Forecast Header
        forecast_header = QLabel("Financial Forecast")
        forecast_header.setStyleSheet("font-size: 20px; font-weight: bold; color: var(--text-color, black);")
        forecast_container_layout.addWidget(forecast_header)

        # Forecast description
        forecast_description = QLabel("This tab shows your financial forecast based on historical data and trends.")
        forecast_description.setStyleSheet("font-size: 14px; color: var(--text-color, black);")
        forecast_description.setWordWrap(True)
        forecast_container_layout.addWidget(forecast_description)

        # Add forecast chart
        self.forecast_chart_frame = QFrame()
        self.forecast_chart_frame.setFrameShape(QFrame.StyledPanel)
        self.forecast_chart_frame.setObjectName("chartCard")
        self.forecast_chart_frame.setStyleSheet("""
            #chartCard {
                background-color: var(--card-bg-color, white);
                border: 1px solid var(--border-color, #E5E5E5);
                border-radius: 10px;
            }
        """)

        forecast_chart_layout = QVBoxLayout(self.forecast_chart_frame)
        forecast_chart_layout.setContentsMargins(15, 15, 15, 15)

        # Chart title
        forecast_chart_title = QLabel("3-Month Financial Forecast")
        forecast_chart_title.setStyleSheet("font-size: 16px; font-weight: bold; color: var(--text-color, black);")
        forecast_chart_layout.addWidget(forecast_chart_title)

        # Chart container
        self.forecast_chart_container = QWidget()
        self.forecast_chart_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.forecast_chart_container.setMinimumHeight(300)

        forecast_chart_container_layout = QVBoxLayout(self.forecast_chart_container)
        forecast_chart_container_layout.setContentsMargins(0, 0, 0, 0)

        forecast_chart_layout.addWidget(self.forecast_chart_container)
        forecast_container_layout.addWidget(self.forecast_chart_frame)

        # Add forecast details section
        self.forecast_details_frame = QFrame()
        self.forecast_details_frame.setFrameShape(QFrame.StyledPanel)
        self.forecast_details_frame.setObjectName("chartCard")
        self.forecast_details_frame.setStyleSheet("""
            #chartCard {
                background-color: var(--card-bg-color, white);
                border: 1px solid var(--border-color, #E5E5E5);
                border-radius: 10px;
            }
        """)

        forecast_details_layout = QVBoxLayout(self.forecast_details_frame)
        forecast_details_layout.setContentsMargins(15, 15, 15, 15)

        # Details title
        forecast_details_title = QLabel("Forecast Details")
        forecast_details_title.setStyleSheet("font-size: 16px; font-weight: bold; color: var(--text-color, black);")
        forecast_details_layout.addWidget(forecast_details_title)

        # Create forecast details table
        self.forecast_table = QTableWidget()
        self.forecast_table.setColumnCount(4)
        self.forecast_table.setHorizontalHeaderLabels(["Month", "Income", "Expenses", "Balance"])

        # Style the table
        self.forecast_table.setStyleSheet("""
            QTableWidget {
                border: none;
                background-color: transparent;
                gridline-color: var(--separator-color, #F0F0F0);
            }
            QHeaderView::section {
                background-color: transparent;
                border: none;
                border-bottom: 1px solid var(--separator-color, #F0F0F0);
                padding: 4px;
                font-weight: bold;
                color: var(--header-text-color, #666666);
            }
            QTableWidget::item {
                border-bottom: 1px solid var(--separator-color, #F0F0F0);
                padding: 4px;
            }
        """)

        # Configure columns
        header = self.forecast_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.forecast_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.forecast_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.forecast_table.setAlternatingRowColors(True)
        self.forecast_table.verticalHeader().setVisible(False)

        forecast_details_layout.addWidget(self.forecast_table)
        forecast_container_layout.addWidget(self.forecast_details_frame)

        # Set the forecast container as the scroll area's widget
        forecast_scroll.setWidget(forecast_container)
        forecast_layout.addWidget(forecast_scroll)

        self.tab_widget.addTab(forecast_tab, "Forecasting")
        
        # Create Budget Optimization tab with AI insights
        budget_tab = QWidget()
        budget_layout = QVBoxLayout(budget_tab)
        
        # Create a scroll area for the budget tab
        budget_scroll = QScrollArea()
        budget_scroll.setWidgetResizable(True)
        budget_scroll.setFrameShape(QFrame.NoFrame)
        
        # Container for budget content
        budget_container = QWidget()
        budget_container_layout = QVBoxLayout(budget_container)
        budget_container_layout.setContentsMargins(0, 0, 0, 0)
        budget_container_layout.setSpacing(20)
        
        # Budget tab header
        budget_header = QLabel("AI Budget Optimization")
        budget_header.setStyleSheet("font-size: 20px; font-weight: bold; color: var(--text-color, black);")
        budget_container_layout.addWidget(budget_header)
        
        # Budget tab description
        budget_description = QLabel(
            "This tab uses machine learning to analyze your spending patterns and provide personalized budget recommendations."
        )
        budget_description.setStyleSheet("font-size: 14px; color: var(--text-color, black);")
        budget_description.setWordWrap(True)
        budget_container_layout.addWidget(budget_description)
        
        # Budget allocation visualization
        self.budget_allocation_frame = QFrame()
        self.budget_allocation_frame.setFrameShape(QFrame.StyledPanel)
        self.budget_allocation_frame.setObjectName("chartCard")
        self.budget_allocation_frame.setStyleSheet("""
            #chartCard {
                background-color: var(--card-bg-color, white);
                border: 1px solid var(--border-color, #E5E5E5);
                border-radius: 10px;
            }
        """)
        
        budget_allocation_layout = QVBoxLayout(self.budget_allocation_frame)
        budget_allocation_layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        budget_allocation_title = QLabel("Budget Allocation")
        budget_allocation_title.setStyleSheet("font-size: 16px; font-weight: bold; color: var(--text-color, black);")
        budget_allocation_layout.addWidget(budget_allocation_title)
        
        # Chart container
        self.budget_allocation_container = QWidget()
        self.budget_allocation_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.budget_allocation_container.setMinimumHeight(250)
        
        budget_allocation_container_layout = QVBoxLayout(self.budget_allocation_container)
        budget_allocation_container_layout.setContentsMargins(0, 0, 0, 0)
        
        budget_allocation_layout.addWidget(self.budget_allocation_container)
        budget_container_layout.addWidget(self.budget_allocation_frame)
        
        # Budget allocation controls - 50/30/20 vs custom
        budget_controls_frame = QFrame()
        budget_controls_frame.setFrameShape(QFrame.StyledPanel)
        budget_controls_frame.setObjectName("chartCard")
        budget_controls_frame.setStyleSheet("""
            #chartCard {
                background-color: var(--card-bg-color, white);
                border: 1px solid var(--border-color, #E5E5E5);
                border-radius: 10px;
            }
        """)
        
        budget_controls_layout = QVBoxLayout(budget_controls_frame)
        budget_controls_layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        budget_controls_title = QLabel("Budget Optimization Controls")
        budget_controls_title.setStyleSheet("font-size: 16px; font-weight: bold; color: var(--text-color, black);")
        budget_controls_layout.addWidget(budget_controls_title)
        
        # Controls container
        controls_container = QWidget()
        controls_layout = QHBoxLayout(controls_container)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        
        # Budget rule selector
        rule_label = QLabel("Budget Rule:")
        rule_label.setStyleSheet("font-weight: bold;")
        controls_layout.addWidget(rule_label)
        
        self.budget_rule_combo = QComboBox()
        self.budget_rule_combo.addItems(["50/30/20 Rule", "Custom Allocation"])
        self.budget_rule_combo.setMinimumWidth(150)
        controls_layout.addWidget(self.budget_rule_combo)
        
        controls_layout.addStretch()
        
        # Optimize button
        self.optimize_button = QPushButton("Optimize Budget")
        self.optimize_button.setMinimumWidth(150)
        controls_layout.addWidget(self.optimize_button)
        
        budget_controls_layout.addWidget(controls_container)
        budget_container_layout.addWidget(budget_controls_frame)
        
        # Budget recommendations
        self.budget_recommendations_frame = QFrame()
        self.budget_recommendations_frame.setFrameShape(QFrame.StyledPanel)
        self.budget_recommendations_frame.setObjectName("chartCard")
        self.budget_recommendations_frame.setStyleSheet("""
            #chartCard {
                background-color: var(--card-bg-color, white);
                border: 1px solid var(--border-color, #E5E5E5);
                border-radius: 10px;
            }
        """)
        
        budget_recommendations_layout = QVBoxLayout(self.budget_recommendations_frame)
        budget_recommendations_layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        budget_recommendations_title = QLabel("Personalized Budget Recommendations")
        budget_recommendations_title.setStyleSheet("font-size: 16px; font-weight: bold; color: var(--text-color, black);")
        budget_recommendations_layout.addWidget(budget_recommendations_title)
        
        # Recommendations text
        self.budget_recommendations_text = QTextBrowser()
        self.budget_recommendations_text.setStyleSheet("""
            QTextBrowser {
                border: none;
                background-color: transparent;
            }
        """)
        self.budget_recommendations_text.setMinimumHeight(150)
        budget_recommendations_layout.addWidget(self.budget_recommendations_text)
        
        budget_container_layout.addWidget(self.budget_recommendations_frame)
        
        # Savings opportunities
        self.savings_opportunities_frame = QFrame()
        self.savings_opportunities_frame.setFrameShape(QFrame.StyledPanel)
        self.savings_opportunities_frame.setObjectName("chartCard")
        self.savings_opportunities_frame.setStyleSheet("""
            #chartCard {
                background-color: var(--card-bg-color, white);
                border: 1px solid var(--border-color, #E5E5E5);
                border-radius: 10px;
            }
        """)
        
        savings_opportunities_layout = QVBoxLayout(self.savings_opportunities_frame)
        savings_opportunities_layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        savings_opportunities_title = QLabel("AI-Detected Savings Opportunities")
        savings_opportunities_title.setStyleSheet("font-size: 16px; font-weight: bold; color: var(--text-color, black);")
        savings_opportunities_layout.addWidget(savings_opportunities_title)
        
        # Opportunities table
        self.savings_opportunities_table = QTableWidget()
        self.savings_opportunities_table.setColumnCount(3)
        self.savings_opportunities_table.setHorizontalHeaderLabels(["Opportunity", "Recommendation", "Potential Savings"])
        
        # Style the table
        self.savings_opportunities_table.setStyleSheet("""
            QTableWidget {
                border: none;
                background-color: transparent;
                gridline-color: var(--separator-color, #F0F0F0);
            }
            QHeaderView::section {
                background-color: transparent;
                border: none;
                border-bottom: 1px solid var(--separator-color, #F0F0F0);
                padding: 4px;
                font-weight: bold;
                color: var(--header-text-color, #666666);
            }
            QTableWidget::item {
                border-bottom: 1px solid var(--separator-color, #F0F0F0);
                padding: 4px;
            }
        """)
        
        # Configure columns
        header = self.savings_opportunities_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Savings
        self.savings_opportunities_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.savings_opportunities_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.savings_opportunities_table.setAlternatingRowColors(True)
        self.savings_opportunities_table.verticalHeader().setVisible(False)
        
        savings_opportunities_layout.addWidget(self.savings_opportunities_table)
        budget_container_layout.addWidget(self.savings_opportunities_frame)
        
        # Set the budget container as the scroll area's widget
        budget_scroll.setWidget(budget_container)
        budget_layout.addWidget(budget_scroll)
        
        # Add budget tab to the tab widget
        self.tab_widget.addTab(budget_tab, "Budget Optimization")
        
        # Connect signals for budget optimization
        self.budget_rule_combo.currentIndexChanged.connect(self.on_budget_rule_changed)
        self.optimize_button.clicked.connect(self.optimize_budget)
        self.savings_opportunities_table.itemClicked.connect(self.on_opportunity_clicked)

        # Create Subscriptions tab
        subscriptions_tab = QWidget()
        subscriptions_layout = QVBoxLayout(subscriptions_tab)

        # Create a scroll area for the subscriptions tab
        subscriptions_scroll = QScrollArea()
        subscriptions_scroll.setWidgetResizable(True)
        subscriptions_scroll.setFrameShape(QFrame.NoFrame)

        # Container for subscriptions content
        subscriptions_container = QWidget()
        subscriptions_container_layout = QVBoxLayout(subscriptions_container)
        subscriptions_container_layout.setContentsMargins(0, 0, 0, 0)
        subscriptions_container_layout.setSpacing(20)

        # Subscriptions Header
        subscriptions_header = QLabel("Subscription Management")
        subscriptions_header.setStyleSheet("font-size: 20px; font-weight: bold; color: var(--text-color, black);")
        subscriptions_container_layout.addWidget(subscriptions_header)

        # Subscriptions description
        subscriptions_description = QLabel("Track and optimize your recurring subscriptions to save money.")
        subscriptions_description.setStyleSheet("font-size: 14px; color: var(--text-color, black);")
        subscriptions_description.setWordWrap(True)
        subscriptions_container_layout.addWidget(subscriptions_description)

        # Add subscriptions summary card
        self.subscriptions_summary_frame = QFrame()
        self.subscriptions_summary_frame.setFrameShape(QFrame.StyledPanel)
        self.subscriptions_summary_frame.setObjectName("chartCard")
        self.subscriptions_summary_frame.setStyleSheet("""
            #chartCard {
                background-color: var(--card-bg-color, white);
                border: 1px solid var(--border-color, #E5E5E5);
                border-radius: 10px;
            }
        """)

        subscriptions_summary_layout = QVBoxLayout(self.subscriptions_summary_frame)
        subscriptions_summary_layout.setContentsMargins(15, 15, 15, 15)

        # Summary title
        subscriptions_summary_title = QLabel("Subscriptions Summary")
        subscriptions_summary_title.setStyleSheet("font-size: 16px; font-weight: bold; color: var(--text-color, black);")
        subscriptions_summary_layout.addWidget(subscriptions_summary_title)

        # Total subscriptions amount
        self.subscriptions_total_label = QLabel("Total Monthly: €0.00")
        self.subscriptions_total_label.setStyleSheet("font-size: 20px; font-weight: bold; color: var(--text-color, black);")
        subscriptions_summary_layout.addWidget(self.subscriptions_total_label)

        # Monthly percentage
        self.subscriptions_percentage_label = QLabel("0% of monthly expenses")
        self.subscriptions_percentage_label.setStyleSheet("font-size: 14px; color: var(--text-color, black);")
        subscriptions_summary_layout.addWidget(self.subscriptions_percentage_label)

        subscriptions_container_layout.addWidget(self.subscriptions_summary_frame)

        # Add subscriptions list
        self.subscriptions_list_frame = QFrame()
        self.subscriptions_list_frame.setFrameShape(QFrame.StyledPanel)
        self.subscriptions_list_frame.setObjectName("chartCard")
        self.subscriptions_list_frame.setStyleSheet("""
            #chartCard {
                background-color: var(--card-bg-color, white);
                border: 1px solid var(--border-color, #E5E5E5);
                border-radius: 10px;
            }
        """)

        subscriptions_list_layout = QVBoxLayout(self.subscriptions_list_frame)
        subscriptions_list_layout.setContentsMargins(15, 15, 15, 15)

        # List title
        subscriptions_list_title = QLabel("Your Subscriptions")
        subscriptions_list_title.setStyleSheet("font-size: 16px; font-weight: bold; color: var(--text-color, black);")
        subscriptions_list_layout.addWidget(subscriptions_list_title)

        # Create subscriptions table
        self.subscriptions_table = QTableWidget()
        self.subscriptions_table.setColumnCount(4)
        self.subscriptions_table.setHorizontalHeaderLabels(["Service", "Frequency", "Amount", "Category"])

        # Style the table
        self.subscriptions_table.setStyleSheet("""
            QTableWidget {
                border: none;
                background-color: transparent;
                gridline-color: var(--separator-color, #F0F0F0);
            }
            QHeaderView::section {
                background-color: transparent;
                border: none;
                border-bottom: 1px solid var(--separator-color, #F0F0F0);
                padding: 4px;
                font-weight: bold;
                color: var(--header-text-color, #666666);
            }
            QTableWidget::item {
                border-bottom: 1px solid var(--separator-color, #F0F0F0);
                padding: 4px;
            }
        """)

        # Configure columns
        header = self.subscriptions_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Amount
        self.subscriptions_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.subscriptions_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.subscriptions_table.setAlternatingRowColors(True)
        self.subscriptions_table.verticalHeader().setVisible(False)

        subscriptions_list_layout.addWidget(self.subscriptions_table)
        subscriptions_container_layout.addWidget(self.subscriptions_list_frame)

        # Add recommendations section
        self.recommendations_frame = QFrame()
        self.recommendations_frame.setFrameShape(QFrame.StyledPanel)
        self.recommendations_frame.setObjectName("chartCard")
        self.recommendations_frame.setStyleSheet("""
            #chartCard {
                background-color: var(--card-bg-color, white);
                border: 1px solid var(--border-color, #E5E5E5);
                border-radius: 10px;
            }
        """)

        recommendations_layout = QVBoxLayout(self.recommendations_frame)
        recommendations_layout.setContentsMargins(15, 15, 15, 15)

        # Recommendations title
        recommendations_title = QLabel("Savings Recommendations")
        recommendations_title.setStyleSheet("font-size: 16px; font-weight: bold; color: var(--text-color, black);")
        recommendations_layout.addWidget(recommendations_title)

        # Recommendations list
        self.recommendations_list = QTextBrowser()
        self.recommendations_list.setStyleSheet("""
            QTextBrowser {
                border: none;
                background-color: transparent;
            }
        """)
        self.recommendations_list.setMinimumHeight(150)
        recommendations_layout.addWidget(self.recommendations_list)

        subscriptions_container_layout.addWidget(self.recommendations_frame)

        # Set the subscriptions container as the scroll area's widget
        subscriptions_scroll.setWidget(subscriptions_container)
        subscriptions_layout.addWidget(subscriptions_scroll)

        self.tab_widget.addTab(subscriptions_tab, "Subscriptions")
        
        # Add tab widget to main layout
        main_layout.addWidget(self.tab_widget)
        
    def refresh_dashboard(self):
        """Refresh all dashboard components with current data."""
        self.logger.info("Refreshing dashboard data")
        
        # Force a sync with the transactions tab first
        try:
            if hasattr(self, 'parentWidget') and self.parent() and hasattr(self.parent(), 'transactions_tab'):
                self.logger.info("Syncing with transactions tab first")
                # Call refresh on the transactions tab to ensure data is updated
                self.parent().transactions_tab.load_transactions()
                # Add a small delay to ensure database updates are complete
                from PySide6.QtCore import QTimer
                QTimer.singleShot(100, self.update_dashboard_data)
                return
        except Exception as e:
            self.logger.error(f"Error syncing with transactions tab: {e}")
        
        # If we couldn't sync with transactions tab, continue with normal refresh
        self.update_dashboard_data()
        
    def update_dashboard_data(self):
        """Update all dashboard components with current data."""
        self.logger.info("Updating dashboard data after sync")
        
        # Clear any previous forecast data to force regeneration
        self.forecast_data = {}
        
        # Add debugging to see the connection with transaction tab
        if hasattr(self, 'parentWidget') and self.parent() and hasattr(self.parent(), 'transactions_tab'):
            self.logger.info("Connected to transactions tab with data")
        else:
            self.logger.warning("No connection to transactions tab detected")
        
        # Initialize budget tab if first run
        if hasattr(self, 'budget_allocation_container'):
            self.update_budget_allocation_chart()
        
        try:
            # Import datetime up here so we can use it multiple times
            import datetime
            
            # Ensure database connection is active
            if not hasattr(self.db_manager, 'conn') or self.db_manager.conn is None:
                self.logger.warning("Database connection not available, reconnecting...")
                try:
                    self.db_manager._connect()
                    self.logger.info("Successfully reconnected to database")
                except Exception as db_err:
                    self.logger.error(f"Failed to reconnect to database: {db_err}")
            
            # Get all transactions to find the last month with data
            transactions = self.db_manager.get_transactions()
            self.current_transactions = transactions  # Store for AI insights
            
            # Current date for fallback
            now = datetime.datetime.now()
            
            # Find the most recent date with transactions
            if transactions:
                # Sort by date in descending order
                transactions.sort(key=lambda x: x['date'], reverse=True)
                # Get the most recent date
                last_date = transactions[0]['date']
                # Parse to get year and month
                try:
                    last_date_parts = last_date.split('-')
                    year = int(last_date_parts[0])
                    month = int(last_date_parts[1])
                    
                    # Create a datetime object for further calculations
                    current_date = datetime.datetime(year, month, 1)
                    
                    self.logger.info(f"Found most recent transactions in {month}/{year}")
                except (ValueError, IndexError) as e:
                    # If parsing fails, use current date
                    current_date = now
                    month = now.month
                    year = now.year
                    self.logger.error(f"Error parsing transaction date: {e}, using current date instead")
            else:
                # If no transactions, use current date
                current_date = now
                month = now.month
                year = now.year
                self.logger.info(f"No transactions found, using current date {month}/{year}")
            
            # Log the date we're using for fetch
            self.logger.info(f"Fetching monthly summary for {month}/{year}")
            
            # Get monthly summary data
            monthly_summary = self.db_manager.get_monthly_summary(month, year)
            
            # Update summary cards with all values
            total_income = monthly_summary.get('total_income', 0)
            total_expenses = monthly_summary.get('total_expenses', 0)
            savings_transfers = monthly_summary.get('savings_transfers', 0)
            balance = monthly_summary.get('savings', 0)
            
            # Add debugging to see the actual numbers
            self.logger.info(f"Dashboard summary - Income: {total_income:.2f}, Expenses: {total_expenses:.2f}, " +
                             f"Savings Transfers: {savings_transfers:.2f}, Balance: {balance:.2f}")
            
            # Debug category data
            self.logger.info("Debugging category data for pie chart")
            try:
                if hasattr(self.db_manager, 'connection') and self.db_manager.connection:
                    cursor = self.db_manager.connection.cursor()
                    cursor.execute("""
                        SELECT c.name as category, COALESCE(SUM(t.amount), 0) as total
                        FROM transactions t
                        LEFT JOIN categories c ON t.category_id = c.id
                        WHERE t.is_income = 0
                        GROUP BY t.category_id
                        ORDER BY total DESC
                    """)
                    
                    rows = cursor.fetchall()
                    self.logger.info(f"Found {len(rows)} expense categories with data:")
                    for row in rows:
                        category_name = row[0] if row[0] else "Other"
                        amount = float(row[1])
                        if amount > 0:
                            self.logger.info(f"  Category: {category_name}, Amount: {amount:.2f}")
            except Exception as e:
                self.logger.error(f"Error debugging category data: {e}")
            
            # Update cards with proper formatted values - make sure to force conversion to float
            self.income_card.update_value(f"{float(total_income):.2f} €")
            self.expense_card.update_value(f"{float(total_expenses):.2f} €")
            self.savings_transfers_card.update_value(f"{float(savings_transfers):.2f} €")
            self.balance_card.update_value(f"{float(balance):.2f} €")
            
            # Add comparison with previous month if available
            previous_month = month - 1 if month > 1 else 12
            previous_year = year if month > 1 else year - 1
            previous_summary = self.db_manager.get_monthly_summary(previous_month, previous_year)
            
            if previous_summary:
                prev_income = previous_summary.get('total_income', 0)
                prev_expenses = previous_summary.get('total_expenses', 0)
                prev_balance = previous_summary.get('savings', 0)
                
                # Calculate percentage changes
                income_change = ((total_income - prev_income) / prev_income * 100) if prev_income else 0
                expense_change = ((total_expenses - prev_expenses) / prev_expenses * 100) if prev_expenses else 0
                balance_change = ((balance - prev_balance) / prev_balance * 100) if prev_balance else 0
                
                # Update cards with comparison data
                self.income_card.update_comparison(income_change)
                self.expense_card.update_comparison(expense_change)
                self.balance_card.update_comparison(balance_change)
            
            # Log the actual values that are being set
            self.logger.info(f"Dashboard cards updated - Income: {float(total_income):.2f} €, " +
                           f"Expenses: {float(total_expenses):.2f} €, " +
                           f"Savings Transfers: {float(savings_transfers):.2f} €, " +
                           f"Balance: {float(balance):.2f} €")
            
            # Set color for balance based on value
            if balance >= 0:
                self.balance_card.set_color("green")
            else:
                self.balance_card.set_color("red")
            
            # Get transactions over time based on selected range
            start_date = None
            if self.current_date_range == "1M":
                # Just the current month
                start_date = current_date.replace(day=1).strftime("%Y-%m-%d")
            elif self.current_date_range == "3M":
                start_date = (current_date.replace(day=1) - datetime.timedelta(days=90)).strftime("%Y-%m-%d")
            elif self.current_date_range == "6M":
                start_date = (current_date.replace(day=1) - datetime.timedelta(days=180)).strftime("%Y-%m-%d")
            elif self.current_date_range == "1Y":
                start_date = (current_date.replace(day=1) - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
            else:
                # Default to all-time data
                start_date = None
            
            # Set end date to current or future month end
            end_date = (now + datetime.timedelta(days=31)).replace(day=1).strftime("%Y-%m-%d")
            
            # Log the date range we're using for fetch
            self.logger.info(f"Fetching transactions from {start_date or 'beginning'} to {end_date}")
            
            # Get transactions for this period
            if start_date:
                filtered_transactions = self.db_manager.get_transactions(
                    start_date=start_date,
                    end_date=end_date,
                    exclude_categories=['Initial Balance', 'Previous Month', 'Balance Forward']
                )
            else:
                filtered_transactions = transactions
            
            # FORCE a direct database query to get the most current spending by category data
            try:
                self.logger.info("*** CRITICAL: Performing direct database query for category data")
                # Only do this if we have DB access - guaranteed to be the freshest data
                if hasattr(self.db_manager, 'connection') and self.db_manager.connection:
                    # Build the date filter for the SQL query - for pie chart, show all expense transactions
                    date_filter = "WHERE t.is_income = 0"
                    params = []
                    
                    self.logger.info(f"SQL Query using all expense transactions for pie chart")
                    
                    # Create SQL query for spending by category
                    query = f"""
                    SELECT c.name as category, COALESCE(SUM(t.amount), 0) as total
                    FROM transactions t
                    LEFT JOIN categories c ON t.category_id = c.id
                    {date_filter}
                    GROUP BY t.category_id
                    ORDER BY total DESC
                    """
                    
                    # Execute the query directly
                    cursor = self.db_manager.connection.cursor()
                    cursor.execute(query, params)
                    
                    # Get the category data directly
                    direct_category_data = {}
                    
                    for row in cursor.fetchall():
                        category_name = row[0] if row[0] else "Other"
                        amount = float(row[1])
                        # Only include categories with non-zero positive amounts
                        if amount > 0:
                            direct_category_data[category_name] = amount
                    
                    self.logger.info(f"Direct category query returned {len(direct_category_data)} categories with data")
                    
                    # Add some logging about the categories and values
                    if direct_category_data:
                        self.logger.info(f"Categories found: {list(direct_category_data.keys())}")
                        self.logger.info(f"Category values: {list(direct_category_data.values())}")
                    
                    # Use the directly queried data to update the chart
                    self.update_category_chart_with_data(direct_category_data)
                else:
                    # Fall back to the original method
                    self.update_category_chart(filtered_transactions)
            except Exception as e:
                self.logger.error(f"Error during direct category update: {e}")
                # Fall back to the original method
                self.update_category_chart(filtered_transactions)
            
            # Get yearly data for charts
            yearly_data = []
            
            # Get data from the last 3 years
            for y in range(year-2, year+1):
                yearly_summary = self.db_manager.get_yearly_summary(y)
                
                # Add year to data items for better tracking
                for item in yearly_summary:
                    item['year'] = y
                    
                yearly_data.extend(yearly_summary)
            
            # Update the income vs expenses chart
            self.update_income_expense_chart(yearly_data)
            
            # Update the monthly trend chart 
            # This will also compute forecast data that we'll use in the forecast tab
            self.update_trend_chart(yearly_data)
            
            # Update recent transactions table
            self.update_recent_transactions(filtered_transactions)

            # Generate forecast data directly here to ensure it's available
            self.generate_forecast_data(yearly_data)
            
            # Update forecast tab with the forecast data
            self.update_forecast_tab()
            
            # Update subscriptions tab
            self.update_subscriptions_tab(transactions)
            
            # Update budget optimization tab if available
            if hasattr(self, 'savings_opportunities_table'):
                # Detect savings opportunities
                if 'budget_optimizer' in self.ai_components:
                    try:
                        self.logger.info("Finding savings opportunities for budget tab")
                        # Create a sanitized copy of transactions with proper structure for budget_optimizer
                        sanitized_transactions = []
                        for tx in transactions:
                            sanitized_tx = {
                                'date': tx.get('date', ''),
                                'amount': tx.get('amount', 0),
                                'description': tx.get('description', ''),
                                'category_name': tx.get('category_name', 'Uncategorized'),
                                'is_income': bool(tx.get('is_income', False))
                            }
                            sanitized_transactions.append(sanitized_tx)
                            
                        savings_opportunities = self.ai_components['budget_optimizer'].identify_savings_opportunities(sanitized_transactions)
                        self.update_savings_opportunities(savings_opportunities)
                    except Exception as e:
                        self.logger.error(f"Error detecting savings opportunities: {e}")
            
            self.logger.info("Dashboard data refreshed successfully")
            
        except Exception as e:
            self.logger.error(f"Error refreshing dashboard: {e}", exc_info=True)
    
    def update_category_chart(self, transactions):
        """Update the spending by category pie chart with interactive features."""
        try:
            # Use the same year/month as determined earlier
            import datetime
            
            # Get the latest month with data if available
            transactions_copy = list(transactions)  # Create a copy to avoid modifying original
            if transactions_copy:
                # Sort by date in descending order
                transactions_copy.sort(key=lambda x: x['date'], reverse=True)
                # Get the most recent date
                last_date = transactions_copy[0]['date']
                # Parse to get year and month
                try:
                    last_date_parts = last_date.split('-')
                    chart_year = int(last_date_parts[0])
                    chart_month = int(last_date_parts[1])
                except (ValueError, IndexError):
                    # If parsing fails, use current date
                    now = datetime.datetime.now()
                    chart_month = now.month
                    chart_year = now.year
            else:
                # If no transactions, use current date
                now = datetime.datetime.now()
                chart_month = now.month
                chart_year = now.year
                
            # Use all available expense transactions regardless of date
            self.logger.info(f"Using all expense transactions for pie chart, time range: {self.current_date_range}")
            
            # Filter for all expenses, with no date filtering
            expenses = [tx for tx in transactions 
                       if not tx.get('is_income') 
                       and tx.get('category_name') not in ['Initial Balance', 'Previous Month', 'Balance Forward', 'Savings']
                       and "um " not in (tx.get('description', '') or '')
                       and tx.get('category_id') != 13]  # 13 is the Savings category ID
                       
            self.logger.info(f"Found {len(expenses)} expense transactions for pie chart")
            
            # Store expenses for details dialog
            self.expenses_by_category = {}
            for tx in expenses:
                category = tx.get('category_name', 'Other')
                if category not in self.expenses_by_category:
                    self.expenses_by_category[category] = []
                self.expenses_by_category[category].append(tx)
            
            # Clear any existing chart
            if hasattr(self, 'category_canvas'):
                self.category_chart_container.layout().removeWidget(self.category_canvas)
                self.category_canvas.deleteLater()
            
            # Process expenses to create category data
            category_totals = {}
            for tx in expenses:
                category = tx.get('category_name', 'Other')
                amount = tx.get('amount', 0)
                if category not in category_totals:
                    category_totals[category] = 0
                category_totals[category] += amount
            
            # Log the data
            self.logger.info(f"Category data: {category_totals}")
            
            # Call update_category_chart_with_data with the generated data
            self.update_category_chart_with_data(category_totals)
                
        except Exception as e:
            self.logger.error(f"Error updating category chart: {e}", exc_info=True)
            
    def update_category_chart_with_data(self, category_data):
        """Update the spending by category pie chart using pre-calculated category data.
        
        Args:
            category_data: Dictionary mapping category names to total amounts
        """
        try:
            # If no category data was provided or it's empty, try to retrieve it directly from the database
            if not category_data and hasattr(self.db_manager, 'connection') and self.db_manager.connection:
                self.logger.info("No category data provided, performing emergency direct database query")
                
                # Execute a direct query to get category data
                cursor = self.db_manager.connection.cursor()
                cursor.execute("""
                    SELECT c.name as category, COALESCE(SUM(t.amount), 0) as total
                    FROM transactions t
                    LEFT JOIN categories c ON t.category_id = c.id
                    WHERE t.is_income = 0
                    GROUP BY t.category_id
                    ORDER BY total DESC
                """)
                
                # Get the results
                category_data = {}
                for row in cursor.fetchall():
                    category_name = row[0] if row[0] else "Other"
                    amount = float(row[1])
                    # Only include categories with non-zero amounts
                    if amount > 0:
                        category_data[category_name] = amount
                
            self.logger.info(f"Updating category chart with direct data: {category_data}")
            
            # Store the data for later reference
            self.category_totals = category_data
            
            # Clear any existing chart
            if hasattr(self, 'category_canvas'):
                self.category_chart_container.layout().removeWidget(self.category_canvas)
                self.category_canvas.deleteLater()
                
            if not category_data:
                # Create empty chart with message
                fig = Figure(facecolor='none')
                fig.set_tight_layout(True)
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, "No expense data found.\nAdd transactions via bank statements or manually in the Transactions tab.", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=12, color='gray')
                ax.axis('off')
                
                self.category_canvas = InteractiveCanvas(fig, self)
                self.category_canvas.setStyleSheet("background-color: transparent;")
                self.category_chart_container.layout().addWidget(self.category_canvas)
                return
                
            # Use the category data directly
            category_totals = category_data
                
            # Store category data for hover/click interactions
            self.category_data = category_totals
                
            # Sort categories by amount (descending)
            categories = sorted(category_totals.keys(), 
                               key=lambda x: category_totals[x], 
                               reverse=True)
            amounts = [category_totals[cat] for cat in categories]
            
            # Define Mac-like colors
            colors = [
                '#FF3B30',  # Red
                '#FF9500',  # Orange
                '#FFCC00',  # Yellow
                '#34C759',  # Green
                '#007AFF',  # Blue
                '#5856D6',  # Purple
                '#AF52DE',  # Magenta
                '#FF2D55',  # Pink
                '#E0A800',  # Gold
                '#28C7B7',  # Teal
                '#59A9FF',  # Light Blue
                '#8E8E93',  # Gray
            ]
            
            # Ensure we have enough colors
            while len(colors) < len(categories):
                colors.extend(colors)
            colors = colors[:len(categories)]
            
            # Create pie chart
            fig = Figure(facecolor='none')
            fig.set_tight_layout(True)
            ax = fig.add_subplot(111)
            
            # Set figure background to be transparent
            ax.set_facecolor('none')
            
            # Create pie chart with more visible outline and exploded wedges for better interactivity
            explode = [0.02] * len(amounts)  # Slightly explode all wedges
            wedges, texts, autotexts = ax.pie(
                amounts, 
                labels=None,
                autopct='%1.1f%%',
                startangle=90,
                colors=colors,
                explode=explode,
                wedgeprops={'edgecolor': 'white', 'linewidth': 1.5, 'antialiased': True},
                shadow=True,
                radius=0.9  # Make pie slightly smaller to fit better
            )
            
            # Equal aspect ratio ensures that pie is drawn as a circle
            ax.axis('equal')
            
            # Add a circular border around the pie chart for better visibility in dark mode
            from matplotlib.patches import Circle
            circle = Circle((0, 0), 0.95, fill=False, 
                           edgecolor='#E5E5E5',
                           linewidth=1.5)
            ax.add_artist(circle)
            
            # Customize the percentage text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(10)
                autotext.set_weight('bold')
            
            # Add legend with category names and amounts with better styling for dark mode
            legend = ax.legend(
                wedges, 
                [f"{cat}: €{category_totals[cat]:.2f}" for cat in categories],
                title="Categories (click for details)",
                loc="center left",
                bbox_to_anchor=(1.0, 0.5),
                fontsize=9,
                frameon=True
            )
            
            # Style the legend for dark mode compatibility 
            frame = legend.get_frame()
            frame.set_facecolor('white')
            frame.set_edgecolor('#E5E5E5')
            frame.set_boxstyle("round,pad=0.5")  # Rounded corners
            
            # Set legend title color
            legend.get_title().set_fontsize(10)
            legend.get_title().set_fontweight('bold')
            legend.get_title().set_color('black')
            
            # Set legend text color for visibility in dark mode
            for text in legend.get_texts():
                text.set_color('black')
            
            # Add interactive canvas to the container
            self.category_canvas = InteractiveCanvas(fig, self)
            self.category_canvas.set_category_data(category_totals)
            self.category_canvas.setStyleSheet("background-color: transparent;")
            self.category_chart_container.layout().addWidget(self.category_canvas)
            
        except Exception as e:
            self.logger.error(f"Error updating category chart: {e}", exc_info=True)
    
    def show_category_details(self, category, amount):
        """Show detailed transactions for a selected category."""
        try:
            # Create a dialog for displaying category details
            dialog = FinancialInsightDialog(self, f"{category} Details")
            
            # Build HTML content for the dialog
            html_content = f"""
            <h2 style="color: #007AFF;">{category} Spending Details</h2>
            <p>Total amount: <b>€{amount:.2f}</b></p>
            <h3>Transactions:</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="background-color: #F5F5F7;">
                    <th style="text-align: left; padding: 8px; border-bottom: 1px solid #E5E5E5;">Date</th>
                    <th style="text-align: left; padding: 8px; border-bottom: 1px solid #E5E5E5;">Description</th>
                    <th style="text-align: right; padding: 8px; border-bottom: 1px solid #E5E5E5;">Amount</th>
                </tr>
            """
            
            # Add transactions for this category
            if category in self.expenses_by_category:
                transactions = sorted(self.expenses_by_category[category], 
                                    key=lambda x: x.get('date', ''), 
                                    reverse=True)
                
                for i, tx in enumerate(transactions):
                    date_str = tx.get('date', '')
                    try:
                        # Format date for better readability
                        date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                        formatted_date = date_obj.strftime("%d %b %Y")
                    except (ValueError, TypeError):
                        formatted_date = date_str
                        
                    row_style = 'background-color: #F9F9F9;' if i % 2 == 0 else ''
                    html_content += f"""
                    <tr style="{row_style}">
                        <td style="text-align: left; padding: 8px; border-bottom: 1px solid #E5E5E5;">{formatted_date}</td>
                        <td style="text-align: left; padding: 8px; border-bottom: 1px solid #E5E5E5;">{tx.get('description', '')}</td>
                        <td style="text-align: right; padding: 8px; border-bottom: 1px solid #E5E5E5; color: #FF3B30;">€{tx.get('amount', 0):.2f}</td>
                    </tr>
                    """
            
            html_content += """
            </table>
            <h3>Spending Pattern Analysis:</h3>
            """
            
            # Add spending pattern analysis
            if category in self.expenses_by_category and len(self.expenses_by_category[category]) > 1:
                transactions = self.expenses_by_category[category]
                
                # Calculate average and frequency
                avg_amount = sum(tx.get('amount', 0) for tx in transactions) / len(transactions)
                
                # Extract dates and sort chronologically
                dates = [datetime.datetime.strptime(tx.get('date', ''), "%Y-%m-%d").date() 
                        for tx in transactions 
                        if tx.get('date')]
                dates.sort()
                
                # Calculate average interval if we have multiple dates
                if len(dates) > 1:
                    intervals = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
                    avg_interval = sum(intervals) / len(intervals)
                    
                    if avg_interval < 15:
                        frequency = "multiple times per month"
                    elif 15 <= avg_interval <= 40:
                        frequency = "monthly"
                    elif 40 < avg_interval <= 100:
                        frequency = "every few months"
                    else:
                        frequency = "infrequently"
                        
                    html_content += f"""
                    <p>You spend <b>€{avg_amount:.2f}</b> on average for {category}, {frequency}.</p>
                    """
                    
                    # Check for trends
                    if len(transactions) >= 3:
                        amounts = [tx.get('amount', 0) for tx in transactions]
                        is_increasing = all(amounts[i] >= amounts[i-1] for i in range(1, len(amounts)))
                        is_decreasing = all(amounts[i] <= amounts[i-1] for i in range(1, len(amounts)))
                        
                        if is_increasing:
                            html_content += f"""
                            <p style="color: #FF3B30;"><b>Warning:</b> Your spending in this category has been consistently increasing.</p>
                            """
                        elif is_decreasing:
                            html_content += f"""
                            <p style="color: #34C759;"><b>Good news:</b> Your spending in this category has been decreasing.</p>
                            """
                        else:
                            # Calculate variance
                            variance = sum((x - avg_amount)**2 for x in amounts) / len(amounts)
                            std_dev = variance**0.5
                            
                            if std_dev / avg_amount > 0.25:
                                html_content += f"""
                                <p style="color: #FF9500;"><b>Note:</b> Your spending in this category varies significantly (±{std_dev:.2f}€).</p>
                                """
                            else:
                                html_content += f"""
                                <p style="color: #007AFF;"><b>Insight:</b> Your spending in this category is fairly consistent.</p>
                                """
            else:
                html_content += f"""
                <p>Not enough data to analyze spending patterns in this category.</p>
                """
            
            # Set the HTML content and show the dialog
            dialog.set_content(html_content)
            dialog.exec()
            
        except Exception as e:
            self.logger.error(f"Error showing category details: {e}", exc_info=True)
    
    def update_income_expense_chart(self, yearly_data):
        """Update the income vs expenses interactive bar chart."""
        try:
            # Clear any existing chart
            if hasattr(self, 'income_expense_canvas'):
                self.income_expense_container.layout().removeWidget(self.income_expense_canvas)
                self.income_expense_canvas.deleteLater()
                
            if not yearly_data:
                # Create empty chart with message
                fig = Figure(facecolor='none')
                fig.set_tight_layout(True)
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, "No data available", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=12, color='gray')
                ax.axis('off')
                
                self.income_expense_canvas = InteractiveCanvas(fig, self)
                self.income_expense_canvas.setStyleSheet("background-color: transparent;")
                self.income_expense_container.layout().addWidget(self.income_expense_canvas)
                return
                
            # Store yearly data for detail displays
            self.yearly_data = yearly_data
            
            # Filter based on selected time range
            filtered_data = self.filter_data_by_time_range(yearly_data)
            
            # Sort data by month
            sorted_data = sorted(filtered_data, key=lambda x: (x.get('year', 0), x.get('month', 0)))
            
            # Prepare data for plotting
            months = []
            for data in sorted_data:
                month = data.get('month', 0)
                year = data.get('year', datetime.datetime.now().year)
                if month > 0 and month <= 12:
                    months.append(f"{calendar.month_abbr[month]} {str(year)[2:]}")
                else:
                    months.append(f"M{month} {str(year)[2:]}")
                    
            income = [data.get('income', 0) for data in sorted_data]
            expenses = [data.get('expenses', 0) for data in sorted_data]
            
            # Create the figure
            fig = Figure(facecolor='none')
            fig.set_tight_layout(True)
            ax = fig.add_subplot(111)
            
            # Set figure background and border
            ax.set_facecolor('none')  # Transparent background
            
            # Add a frame outline - important for visibility in dark mode
            for spine in ax.spines.values():
                spine.set_edgecolor('#E5E5E5')
                spine.set_linewidth(1.5)
            
            # Set the positions for bars
            x = np.arange(len(months))
            bar_width = 0.35
            
            # Create bars with custom styling for better interactivity
            income_bars = ax.bar(
                x - bar_width/2, income, bar_width, 
                color='#34C759', label='Income',
                alpha=0.8, edgecolor='#2A9747', linewidth=1.5,
                zorder=3  # Put bars above grid
            )
            
            expense_bars = ax.bar(
                x + bar_width/2, expenses, bar_width, 
                color='#FF3B30', label='Expenses',
                alpha=0.8, edgecolor='#CC2F29', linewidth=1.5,
                zorder=3  # Put bars above grid
            )
            
            # Add bar values on top
            for i, v in enumerate(income):
                ax.text(i - bar_width/2, v + 50, f"{v:.0f}", 
                       ha='center', va='bottom', fontsize=8,
                       color='#2A9747', fontweight='bold')
                
            for i, v in enumerate(expenses):
                ax.text(i + bar_width/2, v + 50, f"{v:.0f}", 
                       ha='center', va='bottom', fontsize=8,
                       color='#CC2F29', fontweight='bold')
            
            # Add net income/expense line
            net_values = [inc - exp for inc, exp in zip(income, expenses)]
            ax.plot(x, net_values, 'o-', color='#007AFF', linewidth=2, 
                   label='Net Balance', zorder=4, markersize=6)
            
            # Add fill between line and x-axis to show profit/loss areas
            for i in range(len(net_values)-1):
                if net_values[i] >= 0 and net_values[i+1] >= 0:
                    # Both positive - green fill
                    ax.fill_between([x[i], x[i+1]], [net_values[i], net_values[i+1]], 
                                   color='#34C75933', zorder=2)
                elif net_values[i] < 0 and net_values[i+1] < 0:
                    # Both negative - red fill
                    ax.fill_between([x[i], x[i+1]], [net_values[i], net_values[i+1]], 
                                   color='#FF3B3033', zorder=2)
                else:
                    # Crossing from positive to negative or vice versa
                    # Find the crossing point
                    if net_values[i] >= 0 and net_values[i+1] < 0:
                        # Crossing from positive to negative
                        ratio = abs(net_values[i]) / (abs(net_values[i]) + abs(net_values[i+1]))
                        cross_x = x[i] + ratio * (x[i+1] - x[i])
                        
                        # Fill positive area
                        ax.fill_between([x[i], cross_x], [net_values[i], 0], 
                                       color='#34C75933', zorder=2)
                        
                        # Fill negative area
                        ax.fill_between([cross_x, x[i+1]], [0, net_values[i+1]], 
                                       color='#FF3B3033', zorder=2)
                    else:
                        # Crossing from negative to positive
                        ratio = abs(net_values[i]) / (abs(net_values[i]) + abs(net_values[i+1]))
                        cross_x = x[i] + ratio * (x[i+1] - x[i])
                        
                        # Fill negative area
                        ax.fill_between([x[i], cross_x], [net_values[i], 0], 
                                       color='#FF3B3033', zorder=2)
                        
                        # Fill positive area
                        ax.fill_between([cross_x, x[i+1]], [0, net_values[i+1]], 
                                       color='#34C75933', zorder=2)
            
            # Add labels and title
            ax.set_xlabel('Month', color='black')
            ax.set_ylabel('Amount (€)', color='black')
            ax.set_xticks(x)
            ax.set_xticklabels(months, rotation=45, ha='right')
            
            # Set tick colors for visibility in both light and dark modes
            ax.tick_params(axis='x', colors='black')
            ax.tick_params(axis='y', colors='black')
            
            # Add legend with better visibility in dark mode
            legend = ax.legend(frameon=True, loc='upper center', bbox_to_anchor=(0.5, 1.1), 
                             ncol=3, fancybox=True, shadow=True)
            frame = legend.get_frame()
            frame.set_facecolor('white')
            frame.set_edgecolor('#E5E5E5')
            
            # Set legend text color for visibility in dark mode
            for text in legend.get_texts():
                text.set_color('black')
            
            # Add grid lines for better readability, but respecting theme
            ax.grid(axis='y', linestyle='--', color='#E5E5E5', alpha=0.7, zorder=1)
            
            # Add padding to y-axis to make room for the bar value labels
            y_max = max(max(income), max(expenses)) if income and expenses else 1000
            ax.set_ylim(bottom=min(min(net_values, default=0), 0), top=y_max * 1.15)
            
            # Add the chart to the container
            self.income_expense_canvas = InteractiveCanvas(fig, self)
            self.income_expense_canvas.setStyleSheet("background-color: transparent;")
            self.income_expense_container.layout().addWidget(self.income_expense_canvas)
            
        except Exception as e:
            self.logger.error(f"Error updating income expense chart: {e}", exc_info=True)
            
    def show_month_details(self, month, height, data_type):
        """Show detail dialog for a specific month's financial data."""
        try:
            # Find the month and year from label
            month_parts = month.split()
            if len(month_parts) >= 2:
                month_abbr = month_parts[0]
                year_short = month_parts[1]
                
                # Convert month abbr to number
                month_num = 0
                for i, abbr in enumerate(calendar.month_abbr):
                    if abbr == month_abbr:
                        month_num = i
                        break
                
                # Full year
                year = int("20" + year_short) if year_short.isdigit() and len(year_short) == 2 else datetime.datetime.now().year
                
                # Find the matching data
                month_data = None
                for data in self.yearly_data:
                    if data.get('month') == month_num and data.get('year', datetime.datetime.now().year) == year:
                        month_data = data
                        break
                
                if month_data:
                    # Create dialog
                    dialog = FinancialInsightDialog(self, f"{month} Financial Details")
                    
                    # Format the month name
                    month_name = calendar.month_name[month_num] if 1 <= month_num <= 12 else f"Month {month_num}"
                    
                    # Determine if clicked on income or expenses
                    is_income = data_type == "Income"
                    metric_name = "Income" if is_income else "Expenses"
                    metric_value = month_data.get('income' if is_income else 'expenses', 0)
                    metric_color = "#34C759" if is_income else "#FF3B30"
                    
                    # Get transactions for this month
                    month_str = f"{year}-{month_num:02d}"
                    transactions = self.db_manager.get_transactions(
                        start_date=f"{month_str}-01",
                        end_date=f"{month_str}-{calendar.monthrange(year, month_num)[1]}"
                    )
                    
                    # Filter to income or expenses
                    filtered_txs = [tx for tx in transactions if tx.get('is_income', False) == is_income]
                    
                    # Build HTML content
                    html_content = f"""
                    <h2 style="color: {metric_color};">{month_name} {year} {metric_name}</h2>
                    <p>Total {metric_name.lower()}: <b>€{metric_value:.2f}</b></p>
                    """
                    
                    # Add comparison to previous month
                    prev_month = month_num - 1 if month_num > 1 else 12
                    prev_year = year if month_num > 1 else year - 1
                    
                    prev_data = None
                    for data in self.yearly_data:
                        if data.get('month') == prev_month and data.get('year', year) == prev_year:
                            prev_data = data
                            break
                    
                    if prev_data:
                        prev_value = prev_data.get('income' if is_income else 'expenses', 0)
                        difference = metric_value - prev_value
                        percent_change = (difference / prev_value * 100) if prev_value else 0
                        
                        change_direction = "up" if difference > 0 else "down"
                        change_color = "#34C759" if (is_income and difference > 0) or (not is_income and difference < 0) else "#FF3B30"
                        
                        html_content += f"""
                        <p>Compared to {calendar.month_name[prev_month]} {prev_year}: 
                           <span style="color: {change_color}; font-weight: bold;">
                             {change_direction} {abs(percent_change):.1f}% (€{abs(difference):.2f})
                           </span>
                        </p>
                        """
                    
                    # Show transactions if available
                    if filtered_txs:
                        # Sort by amount
                        sorted_txs = sorted(filtered_txs, key=lambda x: x.get('amount', 0), reverse=True)
                        
                        html_content += f"""
                        <h3>Transactions:</h3>
                        <table style="width: 100%; border-collapse: collapse;">
                            <tr style="background-color: #F5F5F7;">
                                <th style="text-align: left; padding: 8px; border-bottom: 1px solid #E5E5E5;">Date</th>
                                <th style="text-align: left; padding: 8px; border-bottom: 1px solid #E5E5E5;">Category</th>
                                <th style="text-align: left; padding: 8px; border-bottom: 1px solid #E5E5E5;">Description</th>
                                <th style="text-align: right; padding: 8px; border-bottom: 1px solid #E5E5E5;">Amount</th>
                            </tr>
                        """
                        
                        for i, tx in enumerate(sorted_txs):
                            date_str = tx.get('date', '')
                            try:
                                # Format date for better readability
                                date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                                formatted_date = date_obj.strftime("%d %b")
                            except (ValueError, TypeError):
                                formatted_date = date_str
                                
                            category = tx.get('category_name', 'Uncategorized')
                            description = tx.get('description', '')
                            amount = tx.get('amount', 0)
                            
                            row_style = 'background-color: #F9F9F9;' if i % 2 == 0 else ''
                            amount_color = metric_color
                            
                            html_content += f"""
                            <tr style="{row_style}">
                                <td style="text-align: left; padding: 8px; border-bottom: 1px solid #E5E5E5;">{formatted_date}</td>
                                <td style="text-align: left; padding: 8px; border-bottom: 1px solid #E5E5E5;">{category}</td>
                                <td style="text-align: left; padding: 8px; border-bottom: 1px solid #E5E5E5;">{description}</td>
                                <td style="text-align: right; padding: 8px; border-bottom: 1px solid #E5E5E5; color: {amount_color};">€{amount:.2f}</td>
                            </tr>
                            """
                            
                        html_content += "</table>"
                        
                        # For expenses, add category breakdown
                        if not is_income and filtered_txs:
                            category_totals = {}
                            for tx in filtered_txs:
                                category = tx.get('category_name', 'Other')
                                if category not in category_totals:
                                    category_totals[category] = 0
                                category_totals[category] += tx.get('amount', 0)
                                
                            # Sort categories by amount
                            sorted_categories = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)
                            
                            html_content += f"""
                            <h3>Expense Categories:</h3>
                            <table style="width: 100%; border-collapse: collapse;">
                                <tr style="background-color: #F5F5F7;">
                                    <th style="text-align: left; padding: 8px; border-bottom: 1px solid #E5E5E5;">Category</th>
                                    <th style="text-align: right; padding: 8px; border-bottom: 1px solid #E5E5E5;">Amount</th>
                                    <th style="text-align: right; padding: 8px; border-bottom: 1px solid #E5E5E5;">% of Total</th>
                                </tr>
                            """
                            
                            total_expenses = sum(category_totals.values())
                            
                            for i, (category, amount) in enumerate(sorted_categories):
                                row_style = 'background-color: #F9F9F9;' if i % 2 == 0 else ''
                                percentage = (amount / total_expenses * 100) if total_expenses else 0
                                
                                html_content += f"""
                                <tr style="{row_style}">
                                    <td style="text-align: left; padding: 8px; border-bottom: 1px solid #E5E5E5;">{category}</td>
                                    <td style="text-align: right; padding: 8px; border-bottom: 1px solid #E5E5E5;">€{amount:.2f}</td>
                                    <td style="text-align: right; padding: 8px; border-bottom: 1px solid #E5E5E5;">{percentage:.1f}%</td>
                                </tr>
                                """
                                
                            html_content += "</table>"
                    
                    # Set content and show dialog
                    dialog.set_content(html_content)
                    dialog.exec()
                    
        except Exception as e:
            self.logger.error(f"Error showing month details: {e}", exc_info=True)
    
    def on_time_range_changed(self, range_text):
        """Handle time range combo box changes."""
        self.current_date_range = range_text
        self.refresh_dashboard()
        
    def filter_data_by_time_range(self, data):
        """Filter data based on selected time range."""
        if not data or self.current_date_range == "All":
            return data
            
        # Get current date
        now = datetime.datetime.now()
        
        # Parse range value
        range_value = self.current_date_range
        months_back = 0
        
        if range_value == "1M":
            months_back = 1
        elif range_value == "3M":
            months_back = 3
        elif range_value == "6M":
            months_back = 6
        elif range_value == "1Y":
            months_back = 12
            
        # Calculate cutoff date
        cutoff_year = now.year
        cutoff_month = now.month - months_back
        
        while cutoff_month <= 0:
            cutoff_month += 12
            cutoff_year -= 1
            
        # Filter data
        filtered_data = []
        for item in data:
            year = item.get('year', now.year)
            month = item.get('month', 0)
            
            if year > cutoff_year or (year == cutoff_year and month >= cutoff_month):
                filtered_data.append(item)
                
        return filtered_data
        
    def show_ai_insights(self):
        """Show AI-generated insights dialog."""
        try:
            # Create dialog
            dialog = FinancialInsightDialog(self, "AI Financial Insights")
            
            # Start building HTML content
            html_content = """
            <h2 style="color: #007AFF;">AI Financial Insights</h2>
            """
            
            # Add income and expense trends analysis
            if self.yearly_data:
                # Get income and expense data
                sorted_data = sorted(self.yearly_data, key=lambda x: (x.get('year', 0), x.get('month', 0)))
                income = [data.get('income', 0) for data in sorted_data]
                expenses = [data.get('expenses', 0) for data in sorted_data]
                net_values = [inc - exp for inc, exp in zip(income, expenses)]
                
                # Analyze trends
                if len(income) >= 3:
                    # Income trend
                    income_trend = "increasing" if income[-1] > income[-2] > income[-3] else \
                                  "decreasing" if income[-1] < income[-2] < income[-3] else \
                                  "stable"
                    
                    income_trend_color = "#34C759" if income_trend == "increasing" else \
                                        "#FF3B30" if income_trend == "decreasing" else \
                                        "#007AFF"
                    
                    # Expense trend
                    expense_trend = "increasing" if expenses[-1] > expenses[-2] > expenses[-3] else \
                                   "decreasing" if expenses[-1] < expenses[-2] < expenses[-3] else \
                                   "stable"
                    
                    expense_trend_color = "#FF3B30" if expense_trend == "increasing" else \
                                         "#34C759" if expense_trend == "decreasing" else \
                                         "#007AFF"
                    
                    # Net trend
                    net_trend = "improving" if net_values[-1] > net_values[-2] > net_values[-3] else \
                               "worsening" if net_values[-1] < net_values[-2] < net_values[-3] else \
                               "stable"
                    
                    net_trend_color = "#34C759" if net_trend == "improving" else \
                                     "#FF3B30" if net_trend == "worsening" else \
                                     "#007AFF"
                    
                    # Add trend analysis
                    html_content += f"""
                    <h3>Financial Trends</h3>
                    <p>Based on your last 3 months of financial data:</p>
                    <ul>
                        <li>Your income is <span style="color: {income_trend_color}; font-weight: bold;">{income_trend}</span></li>
                        <li>Your expenses are <span style="color: {expense_trend_color}; font-weight: bold;">{expense_trend}</span></li>
                        <li>Your financial situation is <span style="color: {net_trend_color}; font-weight: bold;">{net_trend}</span></li>
                    </ul>
                    """
                    
                    # Add forecast
                    if len(income) >= 6:
                        try:
                            # Simple linear regression for forecasting
                            X = np.array(range(len(income))).reshape(-1, 1)
                            
                            # Income forecast
                            model_income = LinearRegression()
                            model_income.fit(X, np.array(income))
                            next_month_income = float(model_income.predict([[len(income)]])[0])
                            
                            # Expense forecast
                            model_expenses = LinearRegression()
                            model_expenses.fit(X, np.array(expenses))
                            next_month_expenses = float(model_expenses.predict([[len(expenses)]])[0])
                            
                            # Net forecast
                            next_month_net = next_month_income - next_month_expenses
                            
                            # Format next month name
                            last_month = sorted_data[-1].get('month', 0)
                            last_year = sorted_data[-1].get('year', datetime.datetime.now().year)
                            
                            next_month = last_month + 1
                            next_year = last_year
                            
                            if next_month > 12:
                                next_month = 1
                                next_year += 1
                                
                            next_month_name = calendar.month_name[next_month] if 1 <= next_month <= 12 else f"Month {next_month}"
                            
                            html_content += f"""
                            <h3>Next Month Forecast</h3>
                            <p>For {next_month_name} {next_year}, we predict:</p>
                            <ul>
                                <li>Income: <b>€{next_month_income:.2f}</b> ({"+" if next_month_income > income[-1] else ""}{next_month_income - income[-1]:.2f}€)</li>
                                <li>Expenses: <b>€{next_month_expenses:.2f}</b> ({"+" if next_month_expenses > expenses[-1] else ""}{next_month_expenses - expenses[-1]:.2f}€)</li>
                                <li>Net Balance: <b>€{next_month_net:.2f}</b></li>
                            </ul>
                            """
                        except Exception as e:
                            self.logger.error(f"Error generating forecast: {e}", exc_info=True)
            
            # Add subscription insights if available
            if 'subscription_analyzer' in self.ai_components and self.current_transactions:
                try:
                    analyzer = self.ai_components['subscription_analyzer']
                    recommendations = analyzer.generate_subscription_recommendations(self.current_transactions)
                    
                    if recommendations:
                        html_content += """
                        <h3>Subscription Recommendations</h3>
                        <p>Based on your subscription services:</p>
                        <ul>
                        """
                        
                        for i, rec in enumerate(recommendations[:3]):  # Show top 3 recommendations
                            html_content += f"""
                            <li>{rec['recommendation']}</li>
                            """
                            
                        html_content += "</ul>"
                except Exception as e:
                    self.logger.error(f"Error generating subscription recommendations: {e}", exc_info=True)
            
            # Add spending analysis
            if self.category_data:
                try:
                    # Sort categories by amount
                    sorted_categories = sorted(self.category_data.items(), key=lambda x: x[1], reverse=True)
                    top_categories = sorted_categories[:3]
                    
                    html_content += """
                    <h3>Top Spending Categories</h3>
                    <p>Your top spending categories this month:</p>
                    <ul>
                    """
                    
                    for category, amount in top_categories:
                        html_content += f"""
                        <li><b>{category}:</b> €{amount:.2f}</li>
                        """
                        
                    html_content += "</ul>"
                    
                    # Add some actionable advice
                    if len(sorted_categories) >= 2:
                        top_category, top_amount = sorted_categories[0]
                        second_category, second_amount = sorted_categories[1]
                        
                        total_expenses = sum(self.category_data.values())
                        top_percentage = (top_amount / total_expenses * 100) if total_expenses > 0 else 0
                        
                        if top_percentage > 40:
                            html_content += f"""
                            <p style="color: #FF9500;"><b>Spending Tip:</b> {top_percentage:.1f}% of your expenses are in '{top_category}'. 
                            Consider reviewing to see if you can reduce spending in this category.</p>
                            """
                except Exception as e:
                    self.logger.error(f"Error generating category analysis: {e}", exc_info=True)
            
            # Set content and show dialog
            dialog.set_content(html_content)
            dialog.exec()
            
        except Exception as e:
            self.logger.error(f"Error showing AI insights: {e}", exc_info=True)
    
    def update_trend_chart(self, yearly_data):
        """Update the monthly trends line chart with forecast."""
        try:
            # Clear any existing chart
            if hasattr(self, 'trend_canvas'):
                self.trend_container.layout().removeWidget(self.trend_canvas)
                self.trend_canvas.deleteLater()
                
            if not yearly_data:
                # Create empty chart with message
                fig = Figure(facecolor='none')
                fig.set_tight_layout(True)
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, "No data available", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=12, color='gray')
                ax.axis('off')
                
                self.trend_canvas = InteractiveCanvas(fig, self)
                self.trend_canvas.setStyleSheet("background-color: transparent;")
                self.trend_container.layout().addWidget(self.trend_canvas)
                return
                
            # Filter based on selected time range
            filtered_data = self.filter_data_by_time_range(yearly_data)
            
            # Sort data by month chronologically
            sorted_data = sorted(filtered_data, key=lambda x: (x.get('year', 0), x.get('month', 0)))
            
            # Prepare data for plotting - Include year in month labels for clarity
            months = []
            for data in sorted_data:
                month = data.get('month', 0)
                year = data.get('year', datetime.datetime.now().year)
                if month > 0 and month <= 12:
                    months.append(f"{calendar.month_abbr[month]} {str(year)[2:]}")
                else:
                    months.append(f"M{month} {str(year)[2:]}")
                    
            income = [data.get('income', 0) for data in sorted_data]
            expenses = [data.get('expenses', 0) for data in sorted_data]
            savings = [data.get('savings', 0) for data in sorted_data]
            
            # Also include savings transfers if present
            has_transfers = any(data.get('savings_transfers', 0) > 0 for data in sorted_data)
            if has_transfers:
                savings_transfers = [data.get('savings_transfers', 0) for data in sorted_data]
            
            # Create the figure with higher resolution for better appearance
            fig = Figure(facecolor='none', figsize=(8, 5), dpi=100)
            fig.set_tight_layout(True)
            ax = fig.add_subplot(111)
            
            # Set figure background and border
            ax.set_facecolor('none')  # Transparent background
            
            # Add a frame outline - important for visibility in dark mode
            for spine in ax.spines.values():
                spine.set_edgecolor('#E5E5E5')
                spine.set_linewidth(1.5)
            
            # Set the x positions
            x = np.arange(len(months))
            
            # Generate forecast data if we have enough history
            forecast_months = []
            income_forecast = []
            expenses_forecast = []
            savings_forecast = []
            
            if len(income) >= 4:
                self.logger.info(f"Generating forecast data from {len(income)} months of history")
                try:
                    # Number of months to forecast
                    forecast_period = 3
                    
                    # Get most recent date to continue from
                    last_month = sorted_data[-1].get('month', 0)
                    last_year = sorted_data[-1].get('year', datetime.datetime.now().year)
                    
                    # Generate forecast months
                    for i in range(1, forecast_period + 1):
                        forecast_month = last_month + i
                        forecast_year = last_year
                        
                        if forecast_month > 12:
                            forecast_month -= 12
                            forecast_year += 1
                            
                        if forecast_month > 0 and forecast_month <= 12:
                            forecast_months.append(f"{calendar.month_abbr[forecast_month]} {str(forecast_year)[2:]}")
                        else:
                            forecast_months.append(f"M{forecast_month} {str(forecast_year)[2:]}")
                    
                    # Income forecast using Linear Regression 
                    X = np.array(range(len(income))).reshape(-1, 1)
                    y = np.array(income)
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    # Predict future income
                    for i in range(1, forecast_period + 1):
                        prediction = float(model.predict([[len(income) + i - 1]])[0])
                        # Ensure we don't predict negative income
                        income_forecast.append(max(0, prediction))
                    
                    # Expense forecast
                    X = np.array(range(len(expenses))).reshape(-1, 1)
                    y = np.array(expenses)
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    # Predict future expenses
                    for i in range(1, forecast_period + 1):
                        prediction = float(model.predict([[len(expenses) + i - 1]])[0])
                        # Ensure we don't predict negative expenses
                        expenses_forecast.append(max(0, prediction))
                        
                    # Calculate forecast savings
                    for i in range(forecast_period):
                        savings_forecast.append(income_forecast[i] - expenses_forecast[i])
                        
                    # Store forecast data for other uses
                    self.forecast_data = {
                        'months': forecast_months,
                        'income': income_forecast,
                        'expenses': expenses_forecast,
                        'savings': savings_forecast
                    }
                    self.logger.info(f"Successfully generated forecast data with {len(forecast_months)} months")
                    
                except Exception as e:
                    self.logger.error(f"Error generating forecast data: {e}", exc_info=True)
                    forecast_months = []
                    income_forecast = []
                    expenses_forecast = []
                    savings_forecast = []
            
            # Create lines for actual data with enhanced styling
            income_line, = ax.plot(
                x, income, 'o-', 
                color='#34C759', label='Income', 
                linewidth=2.5, markersize=7,
                markerfacecolor='white', markeredgewidth=2,
                markeredgecolor='#34C759'
            )
            
            expenses_line, = ax.plot(
                x, expenses, 'o-', 
                color='#FF3B30', label='Expenses', 
                linewidth=2.5, markersize=7,
                markerfacecolor='white', markeredgewidth=2,
                markeredgecolor='#FF3B30'
            )
            
            # Include savings transfers if they exist
            if has_transfers:
                savings_transfers_line, = ax.plot(
                    x, savings_transfers, 'o-', 
                    color='#AF52DE', label='Savings Transfers', 
                    linewidth=2.5, markersize=7,
                    markerfacecolor='white', markeredgewidth=2,
                    markeredgecolor='#AF52DE'
                )
                
            # Always include the savings/balance line
            balance_line, = ax.plot(
                x, savings, 'o-', 
                color='#007AFF', label='Balance', 
                linewidth=2.5, markersize=7,
                markerfacecolor='white', markeredgewidth=2, 
                markeredgecolor='#007AFF'
            )
            
            # Add forecast data if available
            if forecast_months and income_forecast and expenses_forecast and savings_forecast:
                # Calculate x positions for forecast
                forecast_x = np.arange(len(months), len(months) + len(forecast_months))
                
                # Add vertical line to separate actual from forecast
                ax.axvline(x=len(months)-0.5, color='#8E8E93', linestyle='--', linewidth=1.5, alpha=0.7)
                
                # Plot forecast lines with dashed style
                ax.plot(
                    forecast_x, income_forecast, 'o--', 
                    color='#34C759', label='Income (Forecast)',
                    linewidth=2, markersize=6, alpha=0.8,
                    markerfacecolor='#34C759', markeredgewidth=1,
                    markeredgecolor='#34C759'
                )
                
                ax.plot(
                    forecast_x, expenses_forecast, 'o--', 
                    color='#FF3B30', label='Expenses (Forecast)',
                    linewidth=2, markersize=6, alpha=0.8,
                    markerfacecolor='#FF3B30', markeredgewidth=1,
                    markeredgecolor='#FF3B30'
                )
                
                ax.plot(
                    forecast_x, savings_forecast, 'o--', 
                    color='#007AFF', label='Balance (Forecast)',
                    linewidth=2, markersize=6, alpha=0.8,
                    markerfacecolor='#007AFF', markeredgewidth=1,
                    markeredgecolor='#007AFF'
                )
                
                # Add forecast label
                ax.text(
                    len(months) + len(forecast_months)/2 - 0.5, 
                    ax.get_ylim()[1] * 0.95,
                    "FORECAST", 
                    ha='center', va='top',
                    fontsize=10, color='#8E8E93',
                    bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='#E5E5E5', alpha=0.8)
                )
                
                # Extend x-axis to include forecast months
                all_months = months + forecast_months
                all_x = np.arange(len(all_months))
                ax.set_xticks(all_x)
                ax.set_xticklabels(all_months, rotation=45, ha='right')
            else:
                # No forecast data, just use actual data
                ax.set_xticks(x)
                ax.set_xticklabels(months, rotation=45, ha='right')
            
            # Add labels and title
            ax.set_xlabel('Month', color='black', fontsize=10)
            ax.set_ylabel('Amount (€)', color='black', fontsize=10)
            
            # Set tick colors for visibility in both light and dark modes
            ax.tick_params(axis='x', colors='black')
            ax.tick_params(axis='y', colors='black')
            
            # Add legend with better visibility in dark mode
            legend = ax.legend(
                frameon=True, 
                loc='upper center', 
                bbox_to_anchor=(0.5, 1.15), 
                ncol=2, 
                fontsize=9,
                fancybox=True, 
                shadow=True
            )
            frame = legend.get_frame()
            frame.set_facecolor('white')
            frame.set_edgecolor('#E5E5E5')
            
            # Set legend text color for visibility in dark mode
            for text in legend.get_texts():
                text.set_color('black')
            
            # Add grid lines for better readability, but respecting theme
            ax.grid(True, linestyle='--', color='#E5E5E5', alpha=0.7)
            
            # Add zero line for reference
            ax.axhline(y=0, color='#8E8E93', linestyle='-', linewidth=1, alpha=0.5)
            
            # Add trend channel (envelope) for balance line if we have enough data points
            if len(savings) >= 4:
                # Calculate moving average for smoother trend
                window_size = min(3, len(savings))
                if window_size > 1:
                    savings_ma = np.convolve(savings, np.ones(window_size)/window_size, mode='valid')
                    # Pad beginning to maintain length
                    padding = [savings_ma[0]] * (len(savings) - len(savings_ma))
                    savings_ma = np.concatenate((padding, savings_ma))
                    
                    # Calculate upper and lower bounds (simple deviation-based)
                    deviations = [abs(savings[i] - savings_ma[i]) for i in range(len(savings))]
                    avg_deviation = sum(deviations) / len(deviations)
                    
                    upper_bound = [savings_ma[i] + avg_deviation * 1.5 for i in range(len(savings_ma))]
                    lower_bound = [savings_ma[i] - avg_deviation * 1.5 for i in range(len(savings_ma))]
                    
                    # Plot trend envelope
                    ax.fill_between(
                        x, upper_bound, lower_bound, 
                        color='#007AFF', alpha=0.1,
                        label='_Balance Trend'  # Hide from legend
                    )
            
            # Add the chart to the container using interactive canvas
            self.trend_canvas = InteractiveCanvas(fig, self)
            self.trend_canvas.setStyleSheet("background-color: transparent;")
            self.trend_container.layout().addWidget(self.trend_canvas)
            
        except Exception as e:
            self.logger.error(f"Error updating trend chart: {e}", exc_info=True)
    
    def update_recent_transactions(self, transactions):
        """Update the recent transactions table."""
        try:
            # Clear the table
            self.transactions_table.setRowCount(0)
            
            if not transactions:
                return
                
            # Sort transactions by date (newest first)
            sorted_transactions = sorted(
                transactions, 
                key=lambda tx: tx.get('date', ''), 
                reverse=True
            )
            
            # Take the 5 most recent transactions
            recent_transactions = sorted_transactions[:5]
            
            # Add transactions to table
            for row, tx in enumerate(recent_transactions):
                self.transactions_table.insertRow(row)
                
                # Date column
                date_str = tx.get('date', '')
                try:
                    # Format date for better readability
                    date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                    formatted_date = date_obj.strftime("%d %b %Y")
                except (ValueError, TypeError):
                    formatted_date = date_str
                    
                date_item = QTableWidgetItem(formatted_date)
                self.transactions_table.setItem(row, 0, date_item)
                
                # Description column - truncate if too long
                description = tx.get('description', '')
                if len(description) > 30:
                    description = description[:27] + "..."
                desc_item = QTableWidgetItem(description)
                self.transactions_table.setItem(row, 1, desc_item)
                
                # Category column
                category_item = QTableWidgetItem(tx.get('category_name', 'Other'))
                self.transactions_table.setItem(row, 2, category_item)
                
                # Amount column
                amount = tx.get('amount', 0)
                amount_item = QTableWidgetItem(f"{amount:.2f} €")
                amount_item.setTextAlignment(int(Qt.AlignRight | Qt.AlignVCenter))
                
                # Color code income vs expense
                if tx.get('is_income'):
                    amount_item.setForeground(QBrush(QColor('#34C759')))  # Green
                else:
                    amount_item.setForeground(QBrush(QColor('#FF3B30')))  # Red
                
                self.transactions_table.setItem(row, 3, amount_item)
            
        except Exception as e:
            self.logger.error(f"Error updating recent transactions: {e}", exc_info=True)
    
    def update_forecast_tab(self):
        """Update the forecast tab with forecast data."""
        try:
            self.logger.info("Updating forecast tab with forecast data")
            # Clear any existing chart
            if hasattr(self, 'forecast_canvas'):
                self.forecast_chart_container.layout().removeWidget(self.forecast_canvas)
                self.forecast_canvas.deleteLater()
                
            # Check if we have forecast data
            if not self.forecast_data or not self.forecast_data.get('months'):
                # Create empty chart with message
                fig = Figure(facecolor='none')
                fig.set_tight_layout(True)
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, "No forecast data available. Add more transactions to generate a forecast.", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=12, color='gray')
                ax.axis('off')
                
                self.forecast_canvas = InteractiveCanvas(fig, self)
                self.forecast_canvas.setStyleSheet("background-color: transparent;")
                self.forecast_chart_container.layout().addWidget(self.forecast_canvas)
                
                # Clear the table
                self.forecast_table.setRowCount(0)
                self.logger.warning("No forecast data available to display")
                return
            
            # Get forecast data
            forecast_months = self.forecast_data.get('months', [])
            forecast_income = self.forecast_data.get('income', [])
            forecast_expenses = self.forecast_data.get('expenses', [])
            forecast_savings = self.forecast_data.get('savings', [])
            
            self.logger.info(f"Displaying forecast data: {len(forecast_months)} months of forecast: {forecast_months}")
            
            if not forecast_months or not forecast_income or not forecast_expenses:
                self.logger.warning("Missing forecast data components")
                return
                
            # Create the figure
            fig = Figure(facecolor='none', figsize=(8, 6), dpi=100)
            fig.set_tight_layout(True)
            ax = fig.add_subplot(111)
            
            # Set figure background and border
            ax.set_facecolor('none')  # Transparent background
            
            # Add a frame outline - important for visibility in dark mode
            for spine in ax.spines.values():
                spine.set_edgecolor('#E5E5E5')
                spine.set_linewidth(1.5)
                
            # Set the x positions
            x = np.arange(len(forecast_months))
            
            # Bar width
            bar_width = 0.35
            
            # Create bars with custom styling
            income_bars = ax.bar(
                x - bar_width/2, forecast_income, bar_width, 
                color='#34C759', label='Income',
                alpha=0.8, edgecolor='#2A9747', linewidth=1.5,
                zorder=3  # Put bars above grid
            )
            
            expense_bars = ax.bar(
                x + bar_width/2, forecast_expenses, bar_width, 
                color='#FF3B30', label='Expenses',
                alpha=0.8, edgecolor='#CC2F29', linewidth=1.5,
                zorder=3  # Put bars above grid
            )
            
            # Add net income/expense line
            ax.plot(x, forecast_savings, 'o-', color='#007AFF', linewidth=2.5, 
                   label='Net Balance', zorder=4, markersize=8,
                   markerfacecolor='white', markeredgewidth=2, 
                   markeredgecolor='#007AFF')
            
            # Add bar values on top
            for i, v in enumerate(forecast_income):
                ax.text(i - bar_width/2, v + 50, f"{v:.0f}", 
                       ha='center', va='bottom', fontsize=9,
                       color='#2A9747', fontweight='bold')
                
            for i, v in enumerate(forecast_expenses):
                ax.text(i + bar_width/2, v + 50, f"{v:.0f}", 
                       ha='center', va='bottom', fontsize=9,
                       color='#CC2F29', fontweight='bold')
            
            # Add labels and title
            ax.set_xlabel('Month', color='black', fontsize=10)
            ax.set_ylabel('Amount (€)', color='black', fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(forecast_months, rotation=45, ha='right')
            
            # Set tick colors for visibility in both light and dark modes
            ax.tick_params(axis='x', colors='black')
            ax.tick_params(axis='y', colors='black')
            
            # Add legend with better visibility in dark mode
            legend = ax.legend(
                frameon=True, 
                loc='upper center', 
                bbox_to_anchor=(0.5, 1.15), 
                ncol=3, 
                fontsize=9,
                fancybox=True, 
                shadow=True
            )
            frame = legend.get_frame()
            frame.set_facecolor('white')
            frame.set_edgecolor('#E5E5E5')
            
            # Set legend text color for visibility in dark mode
            for text in legend.get_texts():
                text.set_color('black')
            
            # Add grid lines for better readability, but respecting theme
            ax.grid(True, linestyle='--', color='#E5E5E5', alpha=0.7)
            
            # Add the chart to the container using interactive canvas
            self.forecast_canvas = InteractiveCanvas(fig, self)
            self.forecast_canvas.setStyleSheet("background-color: transparent;")
            self.forecast_chart_container.layout().addWidget(self.forecast_canvas)
            
            # Update the forecast table
            self.forecast_table.setRowCount(len(forecast_months))
            
            for i, month in enumerate(forecast_months):
                # Month name
                month_item = QTableWidgetItem(month)
                self.forecast_table.setItem(i, 0, month_item)
                
                # Income
                income_item = QTableWidgetItem(f"{forecast_income[i]:.2f} €")
                income_item.setTextAlignment(int(Qt.AlignRight | Qt.AlignVCenter))
                income_item.setForeground(QBrush(QColor('#34C759')))
                self.forecast_table.setItem(i, 1, income_item)
                
                # Expenses
                expense_item = QTableWidgetItem(f"{forecast_expenses[i]:.2f} €")
                expense_item.setTextAlignment(int(Qt.AlignRight | Qt.AlignVCenter))
                expense_item.setForeground(QBrush(QColor('#FF3B30')))
                self.forecast_table.setItem(i, 2, expense_item)
                
                # Balance
                balance_item = QTableWidgetItem(f"{forecast_savings[i]:.2f} €")
                balance_item.setTextAlignment(int(Qt.AlignRight | Qt.AlignVCenter))
                
                # Color balance based on value
                if forecast_savings[i] >= 0:
                    balance_item.setForeground(QBrush(QColor('#34C759')))
                else:
                    balance_item.setForeground(QBrush(QColor('#FF3B30')))
                    
                self.forecast_table.setItem(i, 3, balance_item)
                
        except Exception as e:
            self.logger.error(f"Error updating forecast tab: {e}", exc_info=True)
    
    def on_budget_rule_changed(self, index):
        """Handle budget rule selection change."""
        # Update UI based on selected rule
        if index == 0:  # 50/30/20 Rule
            # Set standard allocation rule
            if 'budget_optimizer' in self.ai_components:
                optimizer = self.ai_components['budget_optimizer']
                optimizer.user_preferences['allocation_rule'] = '50/30/20'
                optimizer.user_preferences['custom_allocation'] = None
            
            # Update allocation chart
            self.update_budget_allocation_chart()
            
        else:  # Custom Allocation
            # TO DO: Implement custom allocation UI
            # For now, just use a slightly different allocation
            if 'budget_optimizer' in self.ai_components:
                optimizer = self.ai_components['budget_optimizer']
                optimizer.user_preferences['allocation_rule'] = 'custom'
                optimizer.user_preferences['custom_allocation'] = {
                    'Needs': 0.55,  # 55% for needs
                    'Wants': 0.25,  # 25% for wants
                    'Savings': 0.20  # 20% for savings
                }
            
            # Update allocation chart
            self.update_budget_allocation_chart()
    
    def optimize_budget(self):
        """Run budget optimization and update UI with results."""
        try:
            # Check if we have the budget optimizer
            if not self.ai_components or 'budget_optimizer' not in self.ai_components:
                self.logger.warning("Budget optimizer not found in AI components")
                self.budget_recommendations_text.setHtml(
                    "<p>Budget optimization is not available. Please check your AI components.</p>"
                )
                return
                
            # Get transactions
            transactions = self.db_manager.get_transactions()
            if not transactions:
                self.budget_recommendations_text.setHtml(
                    "<p>No transaction data available for budget optimization.</p>"
                )
                return
                
            # Run budget optimization
            budget_optimizer = self.ai_components['budget_optimizer']
            budget_results = budget_optimizer.optimize_budget(
                transactions, 
                target_income=None,  # Use income from transactions
                personalize=True     # Use AI personalization
            )
            
            # Update budget allocation chart
            self.update_budget_allocation_chart(budget_results)
            
            # Update budget recommendations
            self.update_budget_recommendations(budget_results)
            
            # Update savings opportunities
            savings_opportunities = budget_optimizer.identify_savings_opportunities(transactions)
            self.update_savings_opportunities(savings_opportunities)
            
        except Exception as e:
            self.logger.error(f"Error optimizing budget: {e}", exc_info=True)
            self.budget_recommendations_text.setHtml(
                f"<p>Error during budget optimization: {str(e)}</p>"
            )
    
    def update_budget_allocation_chart(self, budget_data=None):
        """Update the budget allocation chart."""
        try:
            # Clear any existing chart
            if hasattr(self, 'budget_allocation_canvas'):
                self.budget_allocation_container.layout().removeWidget(self.budget_allocation_canvas)
                self.budget_allocation_canvas.deleteLater()
            
            # Create figure
            fig = Figure(facecolor='none')
            fig.set_tight_layout(True)
            ax = fig.add_subplot(111)
            
            # Set figure background and border
            ax.set_facecolor('none')
            
            # Get allocation percentages
            if budget_data:
                # Use data from optimization
                current_allocation = budget_data['current_vs_optimal']
                optimal_allocation = budget_data['optimal_allocation']
                
                # Calculate percentages
                total_current = sum(data['current'] for data in current_allocation.values())
                total_optimal = sum(optimal_allocation.values())
                
                current_percentages = {
                    budget_type: data['current'] / total_current * 100 if total_current > 0 else 0
                    for budget_type, data in current_allocation.items()
                }
                
                optimal_percentages = {
                    budget_type: amount / total_optimal * 100 if total_optimal > 0 else 0
                    for budget_type, amount in optimal_allocation.items()
                }
                
                # Get recommended budget
                recommended_budget = budget_data['recommended_budget']
                recommended_totals = {budget_type: 0 for budget_type in current_allocation}
                
                # Sum up category amounts by budget type
                for category, amount in recommended_budget.items():
                    budget_type = budget_data['category_types'].get(category, 'Wants')
                    recommended_totals[budget_type] += amount
                
                # Calculate percentages
                total_recommended = sum(recommended_totals.values())
                recommended_percentages = {
                    budget_type: amount / total_recommended * 100 if total_recommended > 0 else 0
                    for budget_type, amount in recommended_totals.items()
                }
                
                # Values to plot
                labels = list(current_allocation.keys())
                current_values = [current_percentages[label] for label in labels]
                optimal_values = [optimal_percentages[label] for label in labels]
                recommended_values = [recommended_percentages[label] for label in labels]
                
                # Colors for each budget type
                colors = {
                    'Needs': '#3498db',     # Blue
                    'Wants': '#e74c3c',     # Red
                    'Savings': '#2ecc71'    # Green
                }
                
                bar_colors = [colors.get(label, '#95a5a6') for label in labels]
                
                # Set position for bars
                x = np.arange(len(labels))
                width = 0.25
                
                # Create bars
                ax.bar(x - width, current_values, width, label='Current', color=bar_colors, alpha=0.6)
                ax.bar(x, optimal_values, width, label='Target', color=bar_colors)
                ax.bar(x + width, recommended_values, width, label='Recommended', color=bar_colors, alpha=0.8)
                
                # Add value labels on bars
                for i, v in enumerate(current_values):
                    ax.text(i - width, v + 1, f"{v:.1f}%", ha='center', fontsize=8)
                
                for i, v in enumerate(optimal_values):
                    ax.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=8)
                    
                for i, v in enumerate(recommended_values):
                    ax.text(i + width, v + 1, f"{v:.1f}%", ha='center', fontsize=8)
                
            else:
                # Use default allocation for visualization
                if 'budget_optimizer' in self.ai_components:
                    optimizer = self.ai_components['budget_optimizer']
                    rule = optimizer.user_preferences.get('allocation_rule', '50/30/20')
                    
                    if rule == '50/30/20':
                        allocation = optimizer.DEFAULT_ALLOCATION
                    elif rule == 'custom' and optimizer.user_preferences.get('custom_allocation'):
                        allocation = optimizer.user_preferences['custom_allocation']
                    else:
                        allocation = optimizer.DEFAULT_ALLOCATION
                else:
                    # Default 50/30/20 rule
                    allocation = {
                        'Needs': 0.5,
                        'Wants': 0.3,
                        'Savings': 0.2
                    }
                
                # Values to plot (just the ideal values)
                labels = list(allocation.keys())
                values = [allocation[label] * 100 for label in labels]
                
                # Colors for each budget type
                colors = {
                    'Needs': '#3498db',     # Blue
                    'Wants': '#e74c3c',     # Red
                    'Savings': '#2ecc71'    # Green
                }
                
                bar_colors = [colors.get(label, '#95a5a6') for label in labels]
                
                # Create pie chart for simple allocation
                wedges, texts, autotexts = ax.pie(
                    values, 
                    labels=labels,
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=bar_colors,
                    explode=[0.02] * len(values),
                    wedgeprops={'edgecolor': 'white', 'linewidth': 1},
                    textprops={'fontsize': 12, 'weight': 'bold'}
                )
                
                # Customize the percentage text
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontsize(10)
                    autotext.set_weight('bold')
                
                # Equal aspect ratio ensures that pie is drawn as a circle
                ax.axis('equal')
                
                # Set title
                ax.set_title("Budget Allocation Rule", fontsize=14, pad=20)
            
            # Create canvas and add to layout
            self.budget_allocation_canvas = FigureCanvas(fig)
            self.budget_allocation_canvas.setStyleSheet("background-color: transparent;")
            self.budget_allocation_container.layout().addWidget(self.budget_allocation_canvas)
            
        except Exception as e:
            self.logger.error(f"Error updating budget allocation chart: {e}", exc_info=True)
    
    def update_budget_recommendations(self, budget_results):
        """Update the budget recommendations display."""
        try:
            if not budget_results:
                self.budget_recommendations_text.setHtml(
                    "<p>No budget recommendations available. Please run budget optimization first.</p>"
                )
                return
                
            # Create HTML content for recommendations
            html_content = f"""
            <h3>Optimized Budget Summary</h3>
            <p><b>Monthly Income:</b> €{budget_results['monthly_income']:.2f}</p>
            <p><b>Recommended Monthly Budget:</b> €{budget_results['total_recommended']:.2f}</p>
            <p><b>Remaining Balance:</b> €{budget_results['budget_balance']:.2f}</p>
            
            <h3>Budget Insights</h3>
            <ul>
            """
            
            # Add budget insights
            if 'budget_insights' in budget_results and budget_results['budget_insights']:
                for insight in budget_results['budget_insights']:
                    html_content += f"<li>{insight['description']}</li>"
            else:
                html_content += "<li>Your budget is well balanced according to the selected rule.</li>"
                
            html_content += """
            </ul>
            
            <h3>Category Recommendations</h3>
            <table width="100%" cellspacing="0" cellpadding="4" style="border-collapse: collapse;">
            <tr style="background-color: #f5f5f7;">
                <th style="text-align: left; border-bottom: 1px solid #e0e0e0;">Category</th>
                <th style="text-align: right; border-bottom: 1px solid #e0e0e0;">Current</th>
                <th style="text-align: right; border-bottom: 1px solid #e0e0e0;">Recommended</th>
                <th style="text-align: right; border-bottom: 1px solid #e0e0e0;">Change</th>
            </tr>
            """
            
            # Add top 5 categories by spending amount
            category_spending = {cat: amt for cat, amt in budget_results['recommended_budget'].items()}
            top_categories = sorted(category_spending.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for i, (category, recommended) in enumerate(top_categories):
                # Get current spending if available
                current = 0
                if 'category_spending' in budget_results:
                    current = budget_results['category_spending'].get(category, 0)
                    
                # Calculate change
                change = recommended - current
                change_pct = (change / current * 100) if current > 0 else 0
                change_text = f"{change:.2f} € ({change_pct:.1f}%)" if current > 0 else f"{change:.2f} €"
                
                # Row style
                row_style = 'background-color: #f9f9f9;' if i % 2 == 0 else ''
                
                # Change color based on direction
                change_color = 'green' if change >= 0 else 'red'
                
                html_content += f"""
                <tr style="{row_style}">
                    <td style="border-bottom: 1px solid #e0e0e0;">{category}</td>
                    <td style="text-align: right; border-bottom: 1px solid #e0e0e0;">€{current:.2f}</td>
                    <td style="text-align: right; border-bottom: 1px solid #e0e0e0;">€{recommended:.2f}</td>
                    <td style="text-align: right; border-bottom: 1px solid #e0e0e0; color: {change_color};">{change_text}</td>
                </tr>
                """
            
            html_content += """
            </table>
            """
            
            # Set the HTML content
            self.budget_recommendations_text.setHtml(html_content)
            
        except Exception as e:
            self.logger.error(f"Error updating budget recommendations: {e}", exc_info=True)
            self.budget_recommendations_text.setHtml(
                f"<p>Error displaying budget recommendations: {str(e)}</p>"
            )
    
    def update_savings_opportunities(self, opportunities):
        """Update the savings opportunities table."""
        try:
            # Clear the table
            self.savings_opportunities_table.setRowCount(0)
            
            if not opportunities:
                return
                
            # Add opportunities to table
            for i, opportunity in enumerate(opportunities):
                self.savings_opportunities_table.insertRow(i)
                
                # Opportunity type
                opportunity_type = opportunity.get('type', 'Unknown')
                type_label = self._format_opportunity_type(opportunity_type)
                type_item = QTableWidgetItem(type_label)
                self.savings_opportunities_table.setItem(i, 0, type_item)
                
                # Recommendation
                recommendation = opportunity.get('recommendation', '')
                rec_item = QTableWidgetItem(recommendation)
                rec_item.setToolTip(recommendation)  # Show full text on hover
                self.savings_opportunities_table.setItem(i, 1, rec_item)
                
                # Potential savings
                savings = opportunity.get('potential_savings', 0)
                savings_item = QTableWidgetItem(f"€{savings:.2f}")
                savings_item.setTextAlignment(int(Qt.AlignRight | Qt.AlignVCenter))
                savings_item.setForeground(QBrush(QColor('#2ecc71')))  # Green
                self.savings_opportunities_table.setItem(i, 2, savings_item)
                
                # Store opportunity data in first column item
                type_item.setData(Qt.UserRole, opportunity)
                
        except Exception as e:
            self.logger.error(f"Error updating savings opportunities: {e}", exc_info=True)
    
    def _format_opportunity_type(self, opportunity_type):
        """Format opportunity type for display."""
        # Clean up type name and capitalize
        type_name = opportunity_type.replace('_', ' ').title()
        
        # Special cases for better readability
        if 'Subscription' in type_name:
            return "Subscription Optimization"
        elif 'High Category' in type_name:
            return "Category Spending"
        elif 'Small' in type_name:
            return "Small Purchases"
        elif 'Discretionary' in type_name:
            return "Discretionary Spending"
        elif 'Forecast' in type_name:
            return "Future Prevention"
        
        return type_name
    
    def on_opportunity_clicked(self, item):
        """Show detailed information about the clicked opportunity."""
        row = item.row()
        data_item = self.savings_opportunities_table.item(row, 0)
        if not data_item:
            return
            
        opportunity = data_item.data(Qt.UserRole)
        if not opportunity:
            return
            
        # Create dialog
        dialog = FinancialInsightDialog(self, "Savings Opportunity Details")
        
        # Format html content
        opportunity_type = self._format_opportunity_type(opportunity.get('type', 'Unknown'))
        recommendation = opportunity.get('recommendation', '')
        savings = opportunity.get('potential_savings', 0)
        confidence = opportunity.get('confidence', 'medium').title()
        
        html_content = f"""
        <h2 style="color: #2ecc71;">{opportunity_type}</h2>
        
        <h3>Recommendation</h3>
        <p>{recommendation}</p>
        
        <h3>Potential Savings</h3>
        <p style="font-size: 18px; font-weight: bold;">€{savings:.2f}</p>
        
        <p><b>Confidence Level:</b> {confidence}</p>
        """
        
        # Add specific details based on opportunity type
        if 'category' in opportunity:
            html_content += f"<p><b>Category:</b> {opportunity['category']}</p>"
            
        if 'merchant' in opportunity:
            html_content += f"<p><b>Merchant:</b> {opportunity['merchant']}</p>"
            
        if 'count' in opportunity:
            html_content += f"<p><b>Transaction Count:</b> {opportunity['count']}</p>"
            
        if 'total_spent' in opportunity:
            html_content += f"<p><b>Total Spent:</b> €{opportunity['total_spent']:.2f}</p>"
            
        # Set content and show dialog
        dialog.set_content(html_content)
        dialog.exec()
    
    def update_subscriptions_tab(self, transactions):
        """Update the subscriptions tab with subscription data."""
        try:
            self.logger.info("Updating subscriptions tab with subscription data")
            
            # Check if we have the subscription analyzer
            if not self.ai_components or 'subscription_analyzer' not in self.ai_components:
                self.logger.warning(f"Subscription analyzer not found in AI components: {list(self.ai_components.keys()) if self.ai_components else 'None'}")
                
                # Display a message in all subscription tab elements
                self.subscriptions_total_label.setText("Total Monthly: €0.00")
                self.subscriptions_percentage_label.setText("0% of monthly expenses")
                self.subscriptions_table.setRowCount(0)
                self.recommendations_list.setHtml(
                    "<p>Subscription analysis is not available. Please check your AI components.</p>" +
                    "<p>The subscription analyzer component was not found or is not properly initialized.</p>"
                )
                return
                
            # Check if we have transactions
            if not transactions or len(transactions) < 5:
                self.logger.warning(f"Not enough transactions for subscription analysis: {len(transactions) if transactions else 0}")
                self.recommendations_list.setHtml(
                    "<p>Not enough transaction data to analyze subscriptions. Please add more transactions.</p>"
                )
                return
                
            analyzer = self.ai_components['subscription_analyzer']
            self.logger.info(f"Using subscription analyzer with {len(transactions)} transactions")
            
            # Analyze subscriptions
            try:
                analysis = analyzer.analyze_subscription_spending(transactions)
                self.logger.info("Subscription analysis completed successfully")
            except Exception as e:
                self.logger.error(f"Error during subscription analysis: {e}", exc_info=True)
                self.recommendations_list.setHtml(
                    f"<p>Error analyzing subscriptions: {str(e)}</p>"
                )
                return
                
            subscriptions = analysis.get('subscriptions', [])
            self.logger.info(f"Found {len(subscriptions)} subscription transactions")
            
            # Update summary labels
            monthly_estimate = analysis.get('monthly_subscription_estimate', 0)
            percentage = analysis.get('subscription_percentage', 0)
            
            self.logger.info(f"Monthly subscription estimate: €{monthly_estimate:.2f} ({percentage:.1f}%)")
            self.subscriptions_total_label.setText(f"Total Monthly: €{monthly_estimate:.2f}")
            self.subscriptions_percentage_label.setText(f"{percentage:.1f}% of monthly expenses")
            
            # Update the subscription table
            self.subscriptions_table.setRowCount(0)
            
            # Group by service to avoid duplicates
            service_data = {}
            for sub in subscriptions:
                service_name = sub.get('service_name', 'Unknown')
                if service_name not in service_data:
                    service_data[service_name] = {
                        'amount': 0,
                        'frequency': sub.get('frequency', 'monthly'),
                        'service_type': sub.get('service_type', 'Other')
                    }
                service_data[service_name]['amount'] += sub.get('amount', 0)
            
            # Add services to table
            for i, (service, data) in enumerate(sorted(service_data.items(), key=lambda x: x[1]['amount'], reverse=True)):
                self.subscriptions_table.insertRow(i)
                
                # Service name
                service_item = QTableWidgetItem(service.capitalize())
                self.subscriptions_table.setItem(i, 0, service_item)
                
                # Frequency
                frequency = data.get('frequency', 'monthly').capitalize()
                frequency_item = QTableWidgetItem(frequency)
                self.subscriptions_table.setItem(i, 1, frequency_item)
                
                # Amount
                amount = data.get('amount', 0)
                amount_item = QTableWidgetItem(f"{amount:.2f} €")
                amount_item.setTextAlignment(int(Qt.AlignRight | Qt.AlignVCenter))
                amount_item.setForeground(QBrush(QColor('#FF3B30')))
                self.subscriptions_table.setItem(i, 2, amount_item)
                
                # Category
                category = data.get('service_type', 'Other').capitalize()
                category_item = QTableWidgetItem(category)
                self.subscriptions_table.setItem(i, 3, category_item)
            
            # Generate recommendations
            try:
                recommendations = analyzer.generate_subscription_recommendations(transactions)
                self.logger.info(f"Generated {len(recommendations)} subscription recommendations")
            except Exception as e:
                self.logger.error(f"Error generating subscription recommendations: {e}", exc_info=True)
                self.recommendations_list.setHtml(
                    f"<p>Error generating subscription recommendations: {str(e)}</p>"
                )
                return
            
            # Update recommendations text
            html_content = "<h3>Savings Opportunities</h3>"
            
            if recommendations:
                html_content += "<ul>"
                for rec in recommendations:
                    savings = rec.get('potential_savings', 0)
                    recommendation = rec.get('recommendation', '')
                    html_content += f"<li><b>Save €{savings:.2f}:</b> {recommendation}</li>"
                html_content += "</ul>"
            else:
                html_content += "<p>No recommendations available. Your subscription spending appears to be optimized.</p>"
            
            self.logger.info("Subscription tab updated successfully")    
            self.recommendations_list.setHtml(html_content)
            
        except Exception as e:
            self.logger.error(f"Error updating subscriptions tab: {e}", exc_info=True)
    
    def generate_forecast_data(self, yearly_data):
        """Generate forecast data from yearly data."""
        try:
            self.logger.info("Explicitly generating forecast data...")
            
            if not yearly_data or len(yearly_data) < 4:
                self.logger.warning(f"Not enough data to generate forecast: {len(yearly_data) if yearly_data else 0} months")
                self.forecast_data = {}
                return
                
            # Get the data needed for forecast
            sorted_data = sorted(yearly_data, key=lambda x: (x.get('year', 0), x.get('month', 0)))
            income = [data.get('income', 0) for data in sorted_data]
            expenses = [data.get('expenses', 0) for data in sorted_data]
            
            # Number of months to forecast
            forecast_period = 3
            
            # Get most recent date to continue from
            last_month = sorted_data[-1].get('month', 1)
            last_year = sorted_data[-1].get('year', datetime.datetime.now().year)
            
            # Generate forecast months
            forecast_months = []
            for i in range(1, forecast_period + 1):
                forecast_month = last_month + i
                forecast_year = last_year
                
                if forecast_month > 12:
                    forecast_month -= 12
                    forecast_year += 1
                    
                if forecast_month > 0 and forecast_month <= 12:
                    forecast_months.append(f"{calendar.month_abbr[forecast_month]} {str(forecast_year)[2:]}")
                else:
                    forecast_months.append(f"M{forecast_month} {str(forecast_year)[2:]}")
            
            # Simple linear regression for forecasting
            X = np.array(range(len(income))).reshape(-1, 1)
            
            # Income forecast
            model_income = LinearRegression()
            model_income.fit(X, np.array(income))
            income_forecast = []
            for i in range(1, forecast_period + 1):
                prediction = float(model_income.predict([[len(income) + i - 1]])[0])
                # Ensure we don't predict negative income
                income_forecast.append(max(0, prediction))
            
            # Expense forecast
            model_expenses = LinearRegression()
            model_expenses.fit(X, np.array(expenses))
            expenses_forecast = []
            for i in range(1, forecast_period + 1):
                prediction = float(model_expenses.predict([[len(expenses) + i - 1]])[0])
                # Ensure we don't predict negative expenses
                expenses_forecast.append(max(0, prediction))
                
            # Calculate forecast savings
            savings_forecast = []
            for i in range(forecast_period):
                savings_forecast.append(income_forecast[i] - expenses_forecast[i])
                
            # Store forecast data
            self.forecast_data = {
                'months': forecast_months,
                'income': income_forecast,
                'expenses': expenses_forecast,
                'savings': savings_forecast
            }
            
            self.logger.info(f"Successfully generated forecast data with {len(forecast_months)} months: months={forecast_months}, income={income_forecast}")
            
        except Exception as e:
            self.logger.error(f"Error generating forecast data: {e}", exc_info=True)
            self.forecast_data = {}

    def parent_resized(self, width):
        """Handle parent window resize events."""
        # Adjust layout based on window width
        if width < 800:
            # Compact mode
            pass
        else:
            # Normal mode
            pass
            
    def set_dark_mode(self, dark_mode):
        """
        Update the dark mode setting and refresh charts.
        
        Args:
            dark_mode: Boolean indicating whether dark mode is enabled
        """
        if self.dark_mode == dark_mode:
            return  # No change needed
            
        self.dark_mode = dark_mode
        self.logger.info(f"Dashboard dark mode set to {dark_mode}")
        
        # Update matplotlib configuration
        configure_matplotlib_for_theme(self.dark_mode)
        
        # Refresh all charts with new theme
        if hasattr(self, 'category_data') and self.category_data:
            self.update_category_chart(self.current_transactions)
            
        if hasattr(self, 'yearly_data') and self.yearly_data:
            self.update_income_expense_chart(self.yearly_data)
            self.update_trend_chart(self.yearly_data)
            
        if hasattr(self, 'forecast_data') and self.forecast_data:
            self.update_forecast_tab()
            
        if hasattr(self, 'budget_allocation_container'):
            self.update_budget_allocation_chart()


class SummaryCard(QFrame):
    """Widget for displaying summary information with modern metrics card design."""
    
    def __init__(self, title, value, color="blue"):
        """
        Initialize a summary card.
        
        Args:
            title: Card title
            value: Initial value
            color: Color scheme (blue, green, red, purple)
        """
        super().__init__()
        
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumHeight(130)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        # Apply macOS-style card styling
        self.setObjectName("summaryCard")
        self.setStyleSheet("""
            #summaryCard {
                background-color: var(--card-bg-color, white);
                border: 1px solid var(--border-color, #E5E5E5);
                border-radius: 12px;
            }
        """)
        
        # Add shadow effect for depth
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 30))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(8)
        
        # Title label
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-size: 14px; font-weight: bold; color: var(--header-text-color, #666666);")
        layout.addWidget(self.title_label)
        
        # Value label
        self.value_label = QLabel(value)
        layout.addWidget(self.value_label)
        
        # Comparison label for showing change vs previous period
        self.comparison_label = QLabel("")
        self.comparison_label.setStyleSheet("font-size: 12px;")
        layout.addWidget(self.comparison_label)
        
        # Set initial color
        self.set_color(color)
        
    def update_value(self, value):
        """Update the displayed value."""
        self.value_label.setText(value)
        
    def update_comparison(self, percentage_change):
        """
        Update the comparison indicator showing change vs previous period.
        
        Args:
            percentage_change: Percentage change (positive or negative)
        """
        if abs(percentage_change) < 0.1:
            # Too small to show
            self.comparison_label.setText("No change from last month")
            self.comparison_label.setStyleSheet("font-size: 12px; color: #8E8E93;")
            return
            
        if percentage_change > 0:
            direction = "▲"  # Up arrow
            color = "#34C759" if self.title_label.text() in ["Income", "Balance"] else "#FF3B30"
        else:
            direction = "▼"  # Down arrow
            color = "#FF3B30" if self.title_label.text() in ["Income", "Balance"] else "#34C759"
            
        # Format text
        self.comparison_label.setText(f"{direction} {abs(percentage_change):.1f}% from last month")
        self.comparison_label.setStyleSheet(f"font-size: 12px; color: {color}; font-weight: bold;")
        
    def set_color(self, color):
        """Set the color scheme of the card using macOS system colors."""
        if color == "green":
            self.value_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #34C759;")
        elif color == "red":
            self.value_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #FF3B30;")
        elif color == "purple":
            self.value_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #AF52DE;")
        else:  # blue is default
            self.value_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #007AFF;")