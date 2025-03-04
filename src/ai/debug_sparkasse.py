#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sparkasse Bank Statement Parser Debug Utility

This script helps debug and test the Sparkasse bank statement parser
with sample statements. It visualizes parsed transactions and shows the
categorization results.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import seaborn as sns
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QTabWidget, QComboBox, QCheckBox, QGridLayout, QSplitter, QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont

# Add parent directory to path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the necessary modules
try:
    from ai.sparkasse_parser import SparkasseParser
    from ai.statement_translator import StatementTranslator
    from ai.transaction_categories import TransactionCategorizer
except ImportError:
    print("Error importing required modules. Make sure you have the following modules:")
    print("- sparkasse_parser.py")
    print("- statement_translator.py")
    print("- transaction_categories.py")
    sys.exit(1)


class TransactionTableWidget(QTableWidget):
    """A table widget for displaying transactions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_table()

    def setup_table(self):
        """Set up the table columns and properties."""
        headers = ["Date", "Description", "Amount", "Type", "Category", "Merchant"]
        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(headers)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setEditTriggers(QTableWidget.NoEditTriggers)

    def set_transactions(self, transactions):
        """
        Populate the table with transaction data.

        Args:
            transactions: List of transaction dictionaries
        """
        self.setRowCount(0)  # Clear table

        if not transactions:
            return

        # Get category colors for visual differentiation
        category_colors = self._get_category_colors(transactions)

        self.setRowCount(len(transactions))

        for i, tx in enumerate(transactions):
            # Date
            date_item = QTableWidgetItem(tx.get('date', ''))
            self.setItem(i, 0, date_item)

            # Description
            desc_item = QTableWidgetItem(tx.get('description', ''))
            self.setItem(i, 1, desc_item)

            # Amount
            amount = tx.get('amount', 0)
            amount_item = QTableWidgetItem(f"{amount:.2f} €")
            amount_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

            # Set text color based on transaction type
            if tx.get('is_income', False):
                amount_item.setForeground(QColor('#2ecc71'))  # Green for income
            else:
                amount_item.setForeground(QColor('#e74c3c'))  # Red for expenses

            self.setItem(i, 2, amount_item)

            # Type
            type_item = QTableWidgetItem(tx.get('type', ''))
            self.setItem(i, 3, type_item)

            # Category
            category = tx.get('category', '')
            category_item = QTableWidgetItem(category)

            # Set background color based on category
            if category in category_colors:
                category_item.setBackground(QColor(category_colors[category]))

            self.setItem(i, 4, category_item)

            # Merchant
            merchant_item = QTableWidgetItem(tx.get('merchant', ''))
            self.setItem(i, 5, merchant_item)

    def _get_category_colors(self, transactions):
        """
        Generate colors for each unique category.

        Args:
            transactions: List of transaction dictionaries

        Returns:
            Dictionary mapping categories to colors
        """
        categories = set()
        for tx in transactions:
            category = tx.get('category')
            if category:
                categories.add(category)

        # Default colors for common categories
        default_colors = {
            'Housing': '#3498db',  # Blue
            'Food': '#e67e22',  # Orange
            'Transportation': '#9b59b6',  # Purple
            'Subscriptions': '#f1c40f',  # Yellow
            'Entertainment': '#e74c3c',  # Red
            'Shopping': '#2ecc71',  # Green
            'Health': '#1abc9c',  # Turquoise
            'Education': '#34495e',  # Dark Blue
            'Income': '#27ae60',  # Dark Green
            'Other': '#7f8c8d'  # Gray
        }

        # Use default colors for known categories and generate for others
        colors = {}
        cmap = plt.cm.get_cmap('tab20')

        for i, category in enumerate(categories):
            if category in default_colors:
                colors[category] = default_colors[category]
            else:
                # Generate a color from the colormap
                rgb = cmap(i % 20)[:3]  # Get RGB from colormap (ignore alpha)
                hex_color = '#%02x%02x%02x' % tuple(int(c * 255) for c in rgb)
                colors[category] = hex_color

        return colors


class VisualizationWidget(QWidget):
    """Widget for visualizing transaction data."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Main layout
        self.layout = QVBoxLayout(self)

        # Visualization type selector
        self.viz_type_combo = QComboBox()
        self.viz_type_combo.addItems([
            "Category Breakdown (Pie Chart)",
            "Monthly Spending (Bar Chart)",
            "Income vs Expenses (Bar Chart)",
            "Daily Spending (Line Chart)",
            "Category Trends (Line Chart)"
        ])
        self.viz_type_combo.currentIndexChanged.connect(self.update_visualization)

        # Figure for matplotlib plots
        self.figure = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)

        # Add widgets to layout
        self.layout.addWidget(QLabel("Visualization Type:"))
        self.layout.addWidget(self.viz_type_combo)
        self.layout.addWidget(self.canvas)

        # Store transactions
        self.transactions = []

    def set_transactions(self, transactions):
        """
        Set the transactions to visualize.

        Args:
            transactions: List of transaction dictionaries
        """
        self.transactions = transactions
        self.update_visualization()

    def update_visualization(self):
        """Update the visualization based on the selected type."""
        if not self.transactions:
            return

        # Clear the figure
        self.figure.clear()

        # Get the selected visualization type
        viz_type = self.viz_type_combo.currentText()

        # Create DataFrame from transactions
        df = pd.DataFrame(self.transactions)

        # Convert date to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        # Create visualization based on selected type
        if viz_type.startswith("Category Breakdown"):
            self._create_category_pie_chart(df)
        elif viz_type.startswith("Monthly Spending"):
            self._create_monthly_bar_chart(df)
        elif viz_type.startswith("Income vs Expenses"):
            self._create_income_expense_chart(df)
        elif viz_type.startswith("Daily Spending"):
            self._create_daily_line_chart(df)
        elif viz_type.startswith("Category Trends"):
            self._create_category_trend_chart(df)

        # Refresh the canvas
        self.canvas.draw()

    def _create_category_pie_chart(self, df):
        """Create a pie chart of spending by category."""
        # Filter expenses
        expenses = df[~df['is_income']] if 'is_income' in df.columns else df

        if 'category' in expenses.columns and not expenses.empty:
            # Group by category
            category_totals = expenses.groupby('category')['amount'].sum()

            # Create pie chart
            ax = self.figure.add_subplot(111)
            category_totals.plot.pie(
                ax=ax,
                autopct='%1.1f%%',
                startangle=90,
                shadow=False,
                labels=None,  # Hide labels on the pie
                legend=True,
                fontsize=10,
                explode=[0.05] * len(category_totals)  # Explode all slices
            )

            # Add title and legend
            ax.set_title('Spending by Category', fontsize=12)
            ax.set_ylabel('')  # Remove y-label
            ax.legend(category_totals.index, loc='center left', bbox_to_anchor=(1, 0.5))

    def _create_monthly_bar_chart(self, df):
        """Create a bar chart of monthly spending."""
        # Filter expenses
        expenses = df[~df['is_income']] if 'is_income' in df.columns else df

        if 'date' in expenses.columns and not expenses.empty:
            # Create month column
            expenses['month'] = expenses['date'].dt.strftime('%Y-%m')

            # Group by month
            monthly_totals = expenses.groupby('month')['amount'].sum()

            # Create bar chart
            ax = self.figure.add_subplot(111)
            monthly_totals.plot.bar(ax=ax, color='skyblue')

            # Add title and labels
            ax.set_title('Monthly Spending', fontsize=12)
            ax.set_xlabel('Month')
            ax.set_ylabel('Amount (€)')

            # Add amount labels on top of bars
            for i, v in enumerate(monthly_totals):
                ax.text(i, v + (monthly_totals.max() * 0.02), f'{v:.0f} €',
                        ha='center', fontsize=8)

            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)

    def _create_income_expense_chart(self, df):
        """Create a bar chart comparing income and expenses."""
        if 'is_income' in df.columns and 'date' in df.columns and not df.empty:
            # Create month column
            df['month'] = df['date'].dt.strftime('%Y-%m')

            # Filter income and expenses
            income = df[df['is_income']]
            expenses = df[~df['is_income']]

            # Group by month
            monthly_income = income.groupby('month')['amount'].sum()
            monthly_expenses = expenses.groupby('month')['amount'].sum()

            # Combine into a single DataFrame
            monthly_data = pd.DataFrame({
                'Income': monthly_income,
                'Expenses': monthly_expenses
            }).fillna(0)

            # Create bar chart
            ax = self.figure.add_subplot(111)
            monthly_data.plot.bar(ax=ax, color=['#2ecc71', '#e74c3c'])

            # Add title and labels
            ax.set_title('Monthly Income vs Expenses', fontsize=12)
            ax.set_xlabel('Month')
            ax.set_ylabel('Amount (€)')

            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)

            # Add legend
            ax.legend()

    def _create_daily_line_chart(self, df):
        """Create a line chart of daily spending."""
        # Filter expenses
        expenses = df[~df['is_income']] if 'is_income' in df.columns else df

        if 'date' in expenses.columns and not expenses.empty:
            # Group by date
            daily_totals = expenses.groupby('date')['amount'].sum()

            # Create line chart
            ax = self.figure.add_subplot(111)
            daily_totals.plot.line(ax=ax, marker='o', linestyle='-', color='#3498db')

            # Add title and labels
            ax.set_title('Daily Spending', fontsize=12)
            ax.set_xlabel('Date')
            ax.set_ylabel('Amount (€)')

            # Format x-axis dates
            plt.xticks(rotation=45)
            plt.tight_layout()

    def _create_category_trend_chart(self, df):
        """Create a line chart showing spending trends by category."""
        # Filter expenses
        expenses = df[~df['is_income']] if 'is_income' in df.columns else df

        if 'date' in expenses.columns and 'category' in expenses.columns and not expenses.empty:
            # Create month column
            expenses['month'] = expenses['date'].dt.strftime('%Y-%m')

            # Group by month and category
            category_monthly = expenses.pivot_table(
                index='month',
                columns='category',
                values='amount',
                aggfunc='sum',
                fill_value=0
            )

            # Select top 5 categories for readability
            top_categories = expenses.groupby('category')['amount'].sum().nlargest(5).index
            category_monthly = category_monthly[top_categories]

            # Create line chart
            ax = self.figure.add_subplot(111)
            category_monthly.plot.line(ax=ax, marker='o')

            # Add title and labels
            ax.set_title('Monthly Spending by Category (Top 5)', fontsize=12)
            ax.set_xlabel('Month')
            ax.set_ylabel('Amount (€)')

            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)

            # Add legend
            ax.legend()

            # Adjust layout
            plt.tight_layout()


class SparkasseDebugger(QMainWindow):
    """Main window for debugging Sparkasse statement parsing."""

    def __init__(self):
        super().__init__()

        # Initialize parser components
        self.parser = SparkasseParser()
        self.translator = StatementTranslator()
        self.categorizer = TransactionCategorizer()

        # Initialize UI
        self.init_ui()

        # Store parsed transactions
        self.transactions = []

    def init_ui(self):
        """Initialize the user interface."""
        # Set window properties
        self.setWindowTitle("Sparkasse Parser Debugger")
        self.setMinimumSize(1000, 800)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)

        # Top section - File loading controls
        top_layout = QHBoxLayout()

        # File selection button
        self.file_button = QPushButton("Select Sparkasse Statement PDF")
        self.file_button.clicked.connect(self.load_file)
        top_layout.addWidget(self.file_button)

        # File path label
        self.file_label = QLabel("No file selected")
        top_layout.addWidget(self.file_label, 1)

        # Parse button
        self.parse_button = QPushButton("Parse Statement")
        self.parse_button.clicked.connect(self.parse_statement)
        self.parse_button.setEnabled(False)
        top_layout.addWidget(self.parse_button)

        main_layout.addLayout(top_layout)

        # Middle section - Control panel
        control_layout = QHBoxLayout()

        # Translation checkbox
        self.translate_check = QCheckBox("Translate Descriptions")
        self.translate_check.setChecked(True)
        control_layout.addWidget(self.translate_check)

        # Categorization checkbox
        self.categorize_check = QCheckBox("Categorize Transactions")
        self.categorize_check.setChecked(True)
        control_layout.addWidget(self.categorize_check)

        # Add stretch
        control_layout.addStretch()

        # Export buttons
        self.export_csv_button = QPushButton("Export to CSV")
        self.export_csv_button.clicked.connect(self.export_to_csv)
        self.export_csv_button.setEnabled(False)
        control_layout.addWidget(self.export_csv_button)

        main_layout.addLayout(control_layout)

        # Bottom section - Results tabs
        self.tab_widget = QTabWidget()

        # Transactions tab
        self.transactions_table = TransactionTableWidget()
        self.tab_widget.addTab(self.transactions_table, "Transactions")

        # Visualization tab
        self.visualization_widget = VisualizationWidget()
        self.tab_widget.addTab(self.visualization_widget, "Visualizations")

        # Statistics tab
        self.stats_widget = QWidget()
        self.stats_layout = QVBoxLayout(self.stats_widget)
        self.stats_table = QTableWidget()
        self.stats_layout.addWidget(self.stats_table)
        self.tab_widget.addTab(self.stats_widget, "Statistics")

        main_layout.addWidget(self.tab_widget, 1)

        # Status bar
        self.statusBar().showMessage("Ready")

    def load_file(self):
        """Open file dialog to select a PDF file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Sparkasse Statement PDF", "",
            "PDF Files (*.pdf);;All Files (*)", options=options
        )

        if file_path:
            self.file_label.setText(file_path)
            self.parse_button.setEnabled(True)
            self.statusBar().showMessage(f"File selected: {file_path}")

    def parse_statement(self):
        """Parse the selected statement file."""
        file_path = self.file_label.text()

        if file_path == "No file selected":
            self.statusBar().showMessage("Please select a file first")
            return

        try:
            # Parse the PDF file
            self.statusBar().showMessage("Parsing PDF...")
            transactions = self.parser.parse_pdf(file_path)

            if not transactions:
                self.statusBar().showMessage("No transactions found in the file")
                return

            # Translate if needed
            if self.translate_check.isChecked():
                self.statusBar().showMessage("Translating transactions...")
                transactions = self.translator.translate_transactions(transactions)

            # Categorize if needed
            if self.categorize_check.isChecked():
                self.statusBar().showMessage("Categorizing transactions...")
                transactions = self.categorizer.categorize_transactions(transactions)

            # Store the transactions
            self.transactions = transactions

            # Update the UI
            self.transactions_table.set_transactions(transactions)
            self.visualization_widget.set_transactions(transactions)
            self.update_statistics(transactions)

            # Enable export button
            self.export_csv_button.setEnabled(True)

            self.statusBar().showMessage(f"Successfully parsed {len(transactions)} transactions")

        except Exception as e:
            self.statusBar().showMessage(f"Error parsing file: {str(e)}")
            import traceback
            traceback.print_exc()

    def update_statistics(self, transactions):
        """Update the statistics tab with transaction data."""
        if not transactions:
            return

        # Convert to DataFrame
        df = pd.DataFrame(transactions)

        # Calculate basic statistics
        stats = []

        # Total transactions
        stats.append(("Total Transactions", len(transactions)))

        # Total income and expenses
        if 'is_income' in df.columns:
            income_total = df[df['is_income']]['amount'].sum() if not df[df['is_income']].empty else 0
            expense_total = df[~df['is_income']]['amount'].sum() if not df[~df['is_income']].empty else 0
            stats.append(("Total Income", f"{income_total:.2f} €"))
            stats.append(("Total Expenses", f"{expense_total:.2f} €"))
            stats.append(("Net Cash Flow", f"{income_total - expense_total:.2f} €"))

        # Date range
        if 'date' in df.columns:
            date_min = df['date'].min()
            date_max = df['date'].max()
            stats.append(("Date Range", f"{date_min} to {date_max}"))

        # Category statistics
        if 'category' in df.columns:
            categories = df['category'].value_counts()
            stats.append(("Number of Categories", len(categories)))
            stats.append(("Most Common Category", f"{categories.index[0]} ({categories.iloc[0]} transactions)"))

            # Category amounts
            expense_df = df[~df['is_income']] if 'is_income' in df.columns else df
            if not expense_df.empty:
                category_amounts = expense_df.groupby('category')['amount'].sum().sort_values(ascending=False)
                for category, amount in category_amounts.items():
                    stats.append((f"Category: {category}", f"{amount:.2f} €"))

        # Update the statistics table
        self.stats_table.setRowCount(len(stats))
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["Statistic", "Value"])

        for i, (stat, value) in enumerate(stats):
            self.stats_table.setItem(i, 0, QTableWidgetItem(stat))
            self.stats_table.setItem(i, 1, QTableWidgetItem(str(value)))

        # Resize columns to contents
        self.stats_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.stats_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)

    def export_to_csv(self):
        """Export the parsed transactions to a CSV file."""
        if not self.transactions:
            self.statusBar().showMessage("No transactions to export")
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Transactions as CSV", "",
            "CSV Files (*.csv);;All Files (*)", options=options
        )

        if file_path:
            try:
                # Create DataFrame and export to CSV
                df = pd.DataFrame(self.transactions)
                df.to_csv(file_path, index=False)
                self.statusBar().showMessage(f"Transactions exported to {file_path}")
            except Exception as e:
                self.statusBar().showMessage(f"Error exporting transactions: {str(e)}")


def main():
    """Run the Sparkasse debugger application."""
    app = QApplication(sys.argv)
    window = SparkasseDebugger()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()