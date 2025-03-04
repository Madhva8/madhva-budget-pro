"""
AI Package

This package contains AI components for financial analysis,
bank statement parsing, and transaction categorization.
"""

try:
    from ai.sparkasse_parser import SparkasseParser
    from ai.statement_translator import StatementTranslator
    from ai.transaction_categories import TransactionCategorizer
    from ai.financial_ai import FinancialAI
    from ai.budget_optimizer import BudgetOptimizer
    from ai.investment_advisor import InvestmentAdvisor
    from ai.pdf_diagnostic import PDFDiagnostic
    from ai.banking_app import BankingAppConnector
    __all__ = [
        'SparkasseParser',
        'StatementTranslator',
        'TransactionCategorizer',
        'FinancialAI',
        'BudgetOptimizer',
        'InvestmentAdvisor',
        'PDFDiagnostic',
        'BankingAppConnector'
    ]
except ImportError:
    # Some modules might be missing dependencies
    pass