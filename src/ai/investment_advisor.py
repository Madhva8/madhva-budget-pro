#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Investment Advisor Module

This module provides investment advice and recommendations for
international students based on financial data and goals.
"""

import numpy as np
import pandas as pd
import datetime
from typing import List, Dict, Any, Optional, Tuple, Union


class InvestmentAdvisor:
    """
    Investment advisor for international students with focus on Germany.
    """

    # Investment options available in Germany
    INVESTMENT_OPTIONS = {
        'Savings': {
            'min_investment': 0,
            'liquidity': 'High',
            'risk': 'Low',
            'expected_return': 0.5,  # 0.5% annual return
            'costs': 'None',
            'tax_implications': 'Interest is taxable',
            'description': 'Regular savings accounts at German banks',
            'suitable_for': 'Emergency funds, short-term goals',
            'min_timeframe': 0  # months
        },
        'Festgeld': {  # Fixed-term deposit
            'min_investment': 1000,
            'liquidity': 'Medium',
            'risk': 'Low',
            'expected_return': 2.5,  # 2.5% annual return
            'costs': 'None',
            'tax_implications': 'Interest is taxable',
            'description': 'Fixed-term deposit accounts with higher interest rates',
            'suitable_for': 'Medium-term savings with fixed timeframe',
            'min_timeframe': 12  # months
        },
        'ETF_Savings_Plan': {
            'min_investment': 25,
            'liquidity': 'Medium',
            'risk': 'Medium',
            'expected_return': 6.0,  # 6% annual return (long-term average)
            'costs': '0.2-0.5% annual fee, may have purchase fees',
            'tax_implications': 'Dividends and capital gains are taxable',
            'description': 'Regular investments in diversified index ETFs',
            'suitable_for': 'Long-term wealth building, retirement',
            'min_timeframe': 36  # months
        },
        'Robo_Advisor': {
            'min_investment': 500,
            'liquidity': 'Medium',
            'risk': 'Medium',
            'expected_return': 5.0,  # 5% annual return
            'costs': '0.5-1.0% annual fee',
            'tax_implications': 'Dividends and capital gains are taxable',
            'description': 'Automated investment services (e.g., Scalable Capital, Oskar)',
            'suitable_for': 'Hands-off investing with some customization',
            'min_timeframe': 24  # months
        },
        'Individual_Stocks': {
            'min_investment': 50,
            'liquidity': 'High',
            'risk': 'High',
            'expected_return': 8.0,  # 8% annual return (but highly variable)
            'costs': 'Trading fees, may have account fees',
            'tax_implications': 'Dividends and capital gains are taxable',
            'description': 'Direct investment in company stocks',
            'suitable_for': 'Experienced investors, high risk tolerance',
            'min_timeframe': 60  # months
        },
        'Crypto': {
            'min_investment': 10,
            'liquidity': 'High',
            'risk': 'Very High',
            'expected_return': 0.0,  # Too volatile to estimate
            'costs': 'Trading fees, spread',
            'tax_implications': 'Capital gains taxable, complex rules',
            'description': 'Cryptocurrency investments (Bitcoin, Ethereum, etc.)',
            'suitable_for': 'Speculative portion of portfolio only',
            'min_timeframe': 48  # months
        }
    }

    # Risk profiles
    RISK_PROFILES = {
        'Conservative': {
            'description': 'Focus on capital preservation with minimal risk',
            'suitable_investments': ['Savings', 'Festgeld'],
            'max_allocation_percent': {
                'Savings': 80,
                'Festgeld': 20,
                'ETF_Savings_Plan': 0,
                'Robo_Advisor': 0,
                'Individual_Stocks': 0,
                'Crypto': 0
            }
        },
        'Moderately Conservative': {
            'description': 'Primarily focused on safety with small growth component',
            'suitable_investments': ['Savings', 'Festgeld', 'ETF_Savings_Plan'],
            'max_allocation_percent': {
                'Savings': 60,
                'Festgeld': 25,
                'ETF_Savings_Plan': 15,
                'Robo_Advisor': 0,
                'Individual_Stocks': 0,
                'Crypto': 0
            }
        },
        'Balanced': {
            'description': 'Balance between capital preservation and growth',
            'suitable_investments': ['Savings', 'Festgeld', 'ETF_Savings_Plan', 'Robo_Advisor'],
            'max_allocation_percent': {
                'Savings': 35,
                'Festgeld': 25,
                'ETF_Savings_Plan': 30,
                'Robo_Advisor': 10,
                'Individual_Stocks': 0,
                'Crypto': 0
            }
        },
        'Growth': {
            'description': 'Focus on long-term growth with moderate risk',
            'suitable_investments': ['Savings', 'ETF_Savings_Plan', 'Robo_Advisor', 'Individual_Stocks'],
            'max_allocation_percent': {
                'Savings': 20,
                'Festgeld': 10,
                'ETF_Savings_Plan': 50,
                'Robo_Advisor': 15,
                'Individual_Stocks': 5,
                'Crypto': 0
            }
        },
        'Aggressive': {
            'description': 'Maximum growth potential with higher risk',
            'suitable_investments': ['Savings', 'ETF_Savings_Plan', 'Robo_Advisor', 'Individual_Stocks', 'Crypto'],
            'max_allocation_percent': {
                'Savings': 10,
                'Festgeld': 0,
                'ETF_Savings_Plan': 55,
                'Robo_Advisor': 20,
                'Individual_Stocks': 10,
                'Crypto': 5
            }
        }
    }

    # Student-specific investment providers in Germany
    RECOMMENDED_PROVIDERS = {
        'Savings': ['DKB', 'ING', 'Consorsbank', 'N26', 'Commerzbank'],
        'Festgeld': ['DKB', 'ING', 'Consorsbank', 'Volkswagen Bank'],
        'ETF_Savings_Plan': ['Trade Republic', 'Scalable Capital', 'DKB', 'ING', 'comdirect'],
        'Robo_Advisor': ['Scalable Capital', 'Oskar', 'Quirion', 'Whitebox', 'growney'],
        'Individual_Stocks': ['Trade Republic', 'Scalable Capital', 'justTRADE', 'comdirect', 'ING'],
        'Crypto': ['Bison', 'Coinbase', 'Binance', 'Kraken']
    }

    def __init__(self):
        """Initialize the investment advisor."""
        pass

    def assess_investment_readiness(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess if a student is ready to invest based on their financial situation.

        Args:
            financial_data: Dictionary with financial information
                - monthly_income: Monthly income
                - monthly_expenses: Monthly expenses
                - emergency_fund: Current emergency fund amount
                - savings: Current savings
                - debt: Current debt (optional)
                - risk_tolerance: Self-assessed risk tolerance (1-10, optional)

        Returns:
            Dictionary with investment readiness assessment
        """
        # Default response for insufficient data
        if not all(key in financial_data for key in ['monthly_income', 'monthly_expenses']):
            return {
                'ready_to_invest': False,
                'reasons': ['Insufficient financial data provided'],
                'recommendations': ['Track your income and expenses for at least 3 months'],
                'next_steps': ['Build an emergency fund', 'Create a budget', 'Pay off high-interest debt']
            }

        # Extract financial data
        monthly_income = financial_data.get('monthly_income', 0)
        monthly_expenses = financial_data.get('monthly_expenses', 0)
        emergency_fund = financial_data.get('emergency_fund', 0)
        savings = financial_data.get('savings', 0)
        debt = financial_data.get('debt', 0)
        risk_tolerance = financial_data.get('risk_tolerance', 5)  # Default to middle

        # Calculate key metrics
        monthly_surplus = monthly_income - monthly_expenses
        months_of_expenses_covered = emergency_fund / monthly_expenses if monthly_expenses > 0 else 0
        debt_to_income_ratio = debt / monthly_income if monthly_income > 0 else float('inf')

        # Define thresholds
        MIN_SURPLUS = 50  # Minimum monthly surplus in EUR
        MIN_EMERGENCY_MONTHS = 3  # Minimum months of expenses in emergency fund
        MAX_DEBT_RATIO = 0.4  # Maximum acceptable debt-to-income ratio

        # Check if ready to invest
        ready_to_invest = True
        reasons = []
        recommendations = []
        next_steps = []

        # Check monthly surplus
        if monthly_surplus < MIN_SURPLUS:
            ready_to_invest = False
            reasons.append('Insufficient monthly surplus')
            recommendations.append('Increase income or reduce expenses to have at least €50 monthly surplus')
            next_steps.append('Review budget to find potential savings')

        # Check emergency fund
        if months_of_expenses_covered < MIN_EMERGENCY_MONTHS:
            ready_to_invest = False
            reasons.append('Insufficient emergency fund')
            recommendations.append(f'Build emergency fund to cover at least {MIN_EMERGENCY_MONTHS} months of expenses')
            target_fund = monthly_expenses * MIN_EMERGENCY_MONTHS
            next_steps.append(f'Save €{target_fund - emergency_fund:.2f} more in emergency fund')

        # Check debt ratio
        if debt > 0 and debt_to_income_ratio > MAX_DEBT_RATIO:
            ready_to_invest = False
            reasons.append('High debt-to-income ratio')
            recommendations.append('Focus on paying down high-interest debt before investing')
            next_steps.append('Create a debt repayment plan')

        # Determine potential monthly investment amount
        if ready_to_invest:
            # If ready to invest, suggest investing 50% of monthly surplus
            suggested_monthly_investment = monthly_surplus * 0.5

            # Determine risk profile based on input risk tolerance and age
            risk_profile = self._determine_risk_profile(risk_tolerance)

            # Suggest allocation based on risk profile
            allocation = self._suggest_allocation(
                risk_profile,
                suggested_monthly_investment,
                investment_timeframe=financial_data.get('investment_timeframe', 60)  # Default to 5 years
            )

            return {
                'ready_to_invest': True,
                'reasons': ['Sufficient emergency fund', 'Healthy financial situation'],
                'suggested_monthly_investment': suggested_monthly_investment,
                'risk_profile': risk_profile,
                'allocation': allocation,
                'next_steps': [
                    'Open investment accounts with recommended providers',
                    'Set up automatic monthly transfers',
                    'Review investment performance quarterly'
                ]
            }
        else:
            return {
                'ready_to_invest': False,
                'reasons': reasons,
                'recommendations': recommendations,
                'next_steps': next_steps,
                'focus': 'financial_foundation'
            }

    def _determine_risk_profile(self, risk_tolerance: int) -> str:
        """
        Determine the appropriate risk profile based on risk tolerance.

        Args:
            risk_tolerance: Self-assessed risk tolerance (1-10)

        Returns:
            Risk profile name
        """
        if risk_tolerance <= 2:
            return 'Conservative'
        elif risk_tolerance <= 4:
            return 'Moderately Conservative'
        elif risk_tolerance <= 6:
            return 'Balanced'
        elif risk_tolerance <= 8:
            return 'Growth'
        else:
            return 'Aggressive'

    def _suggest_allocation(self, risk_profile: str, monthly_amount: float,
                            investment_timeframe: int) -> Dict[str, Any]:
        """
        Suggest investment allocation based on risk profile and amount.

        Args:
            risk_profile: Risk profile name
            monthly_amount: Monthly investment amount
            investment_timeframe: Investment timeframe in months

        Returns:
            Dictionary with suggested allocation
        """
        if risk_profile not in self.RISK_PROFILES:
            risk_profile = 'Balanced'  # Default to balanced

        profile = self.RISK_PROFILES[risk_profile]
        max_allocations = profile['max_allocation_percent']

        # Filter investments based on timeframe
        suitable_investments = [
            inv for inv in profile['suitable_investments']
            if self.INVESTMENT_OPTIONS[inv]['min_timeframe'] <= investment_timeframe
        ]

        if not suitable_investments:
            suitable_investments = ['Savings']  # Default to savings if no suitable options

        # Calculate initial allocation based on max percentages for suitable investments
        total_allowed_percent = sum(max_allocations[inv] for inv in suitable_investments)

        allocation = {}
        for investment in suitable_investments:
            if total_allowed_percent > 0:
                allocation[investment] = (max_allocations[investment] / total_allowed_percent) * 100
            else:
                allocation[investment] = 0

        # Calculate EUR amounts
        allocation_amounts = {}
        for investment, percent in allocation.items():
            amount = monthly_amount * (percent / 100)
            min_required = self.INVESTMENT_OPTIONS[investment]['min_investment']

            # Check if amount meets minimum investment requirement
            if amount < min_required and min_required > 0:
                # If not, adjust allocation to meet minimums
                if monthly_amount >= min_required:
                    allocation_amounts[investment] = min_required
                else:
                    # If total amount is less than minimum, can't invest in this option
                    allocation_amounts[investment] = 0
            else:
                allocation_amounts[investment] = amount

        # Adjust percentages based on actual amounts
        total_allocated = sum(allocation_amounts.values())
        if total_allocated > 0:
            adjusted_percentages = {
                investment: (amount / total_allocated) * 100
                for investment, amount in allocation_amounts.items()
                if amount > 0
            }
        else:
            adjusted_percentages = {investment: 0 for investment in allocation_amounts}

        # Recommend providers for each investment type
        recommended_providers = {}
        for investment in allocation_amounts:
            if investment in self.RECOMMENDED_PROVIDERS and allocation_amounts[investment] > 0:
                recommended_providers[investment] = self.RECOMMENDED_PROVIDERS[investment]

        return {
            'profile': risk_profile,
            'monthly_amount': monthly_amount,
            'percentages': adjusted_percentages,
            'amounts': allocation_amounts,
            'providers': recommended_providers
        }

    def get_investment_options(self, min_amount: float = 0,
                               max_risk: str = 'Very High') -> Dict[str, Dict[str, Any]]:
        """
        Get available investment options based on criteria.

        Args:
            min_amount: Minimum investment amount available
            max_risk: Maximum risk level acceptable

        Returns:
            Dictionary of suitable investment options
        """
        risk_levels = ['Low', 'Medium', 'High', 'Very High']
        max_risk_index = risk_levels.index(max_risk) if max_risk in risk_levels else len(risk_levels) - 1

        # Filter options based on criteria
        suitable_options = {}
        for name, option in self.INVESTMENT_OPTIONS.items():
            if option['min_investment'] <= min_amount:
                option_risk_index = risk_levels.index(option['risk']) if option['risk'] in risk_levels else 0
                if option_risk_index <= max_risk_index:
                    suitable_options[name] = option

        return suitable_options

    def calculate_investment_projection(self, monthly_amount: float,
                                        allocation: Dict[str, float],
                                        years: int = 10) -> Dict[str, Any]:
        """
        Calculate projected investment growth over time.

        Args:
            monthly_amount: Monthly investment amount
            allocation: Dictionary mapping investment types to percentages
            years: Number of years to project

        Returns:
            Dictionary with projection results
        """
        # Calculate monthly allocation amounts
        monthly_allocation = {}
        for investment, percentage in allocation.items():
            monthly_allocation[investment] = monthly_amount * (percentage / 100)

        # Calculate expected returns
        expected_returns = {}
        for investment, amount in monthly_allocation.items():
            if investment in self.INVESTMENT_OPTIONS:
                annual_return = self.INVESTMENT_OPTIONS[investment]['expected_return'] / 100
                expected_returns[investment] = annual_return
            else:
                expected_returns[investment] = 0.01  # Default to 1% if unknown

        # Calculate projections
        months = years * 12
        projection = {
            'monthly_contribution': monthly_amount,
            'total_contribution': monthly_amount * months,
            'years': years,
            'final_balance': 0,
            'by_investment': {},
            'yearly_projections': []
        }

        # Initialize balance for each investment type
        balances = {investment: 0 for investment in monthly_allocation}

        # Calculate compound growth for each month
        for month in range(1, months + 1):
            for investment, amount in monthly_allocation.items():
                # Add monthly contribution
                balances[investment] += amount

                # Apply monthly growth rate (annual rate divided by 12)
                monthly_return = expected_returns[investment] / 12
                balances[investment] *= (1 + monthly_return)

            # Record yearly projections
            if month % 12 == 0:
                year = month // 12
                yearly_total = sum(balances.values())
                yearly_contribution = monthly_amount * month

                projection['yearly_projections'].append({
                    'year': year,
                    'balance': yearly_total,
                    'contribution': yearly_contribution,
                    'growth': yearly_total - yearly_contribution
                })

        # Calculate final values
        for investment, balance in balances.items():
            projection['by_investment'][investment] = {
                'final_balance': balance,
                'total_contribution': monthly_allocation[investment] * months,
                'growth': balance - (monthly_allocation[investment] * months),
                'annual_return': expected_returns[investment] * 100
            }

        projection['final_balance'] = sum(balances.values())
        projection['total_growth'] = projection['final_balance'] - projection['total_contribution']
        projection['effective_annual_return'] = (
                                                        ((projection['final_balance'] / projection[
                                                            'total_contribution']) ** (1 / years)) - 1
                                                ) * 100 if projection['total_contribution'] > 0 else 0

        return projection

    def get_student_specific_advice(self) -> List[Dict[str, str]]:
        """
        Get investment advice specific to international students in Germany.

        Returns:
            List of advice dictionaries
        """
        return [
            {
                'title': 'Understand Tax Implications',
                'description': 'Germany has a 25% flat tax on investment income plus solidarity surcharge. The first €801 of investment income per year is tax-free (Sparerpauschbetrag).',
                'action': 'Use tax-efficient accounts and strategies to minimize tax burden.'
            },
            {
                'title': 'Student Banking Benefits',
                'description': 'Many German banks offer free accounts and special benefits for students under 30.',
                'action': 'Compare student banking offers at DKB, N26, and Commerzbank.'
            },
            {
                'title': 'Start Small with ETF Savings Plans',
                'description': 'ETF savings plans (Sparplan) allow investments starting from €25 per month with no additional fees at some brokers.',
                'action': 'Consider Trade Republic or Scalable Capital for fee-free ETF savings plans.'
            },
            {
                'title': 'Residence Permit Considerations',
                'description': 'For non-EU students, having sufficient funds in a blocked account (Sperrkonto) is required for residence permits.',
                'action': 'Ensure you maintain the required minimum in your blocked account before investing elsewhere.'
            },
            {
                'title': 'Currency Exchange Costs',
                'description': 'Converting between your home currency and EUR can incur significant fees.',
                'action': 'Use services like Wise (formerly TransferWise) for better exchange rates when transferring money from home.'
            },
            {
                'title': 'Long-term vs. Short-term Planning',
                'description': 'As an international student, consider whether you plan to stay in Germany long-term or return home after studies.',
                'action': 'If returning home, consider investment options that are easily transferable or accessible internationally.'
            },
            {
                'title': 'Emergency Fund Priority',
                'description': 'International students should maintain a larger emergency fund (4-6 months) due to potential visa issues and travel needs.',
                'action': 'Keep your emergency fund in a liquid account with no withdrawal restrictions.'
            },
            {
                'title': 'Investment Documentation',
                'description': 'Keep detailed records of all investments for tax purposes and visa applications.',
                'action': 'Create a system for organizing financial documentation, including annual tax statements.'
            }
        ]

    def generate_investment_plan(self, financial_data: Dict[str, Any],
                                 investment_goals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a comprehensive investment plan based on financial situation and goals.

        Args:
            financial_data: Dictionary with financial information
            investment_goals: List of investment goals with timeframes and target amounts

        Returns:
            Dictionary with comprehensive investment plan
        """
        # Check if ready to invest
        readiness = self.assess_investment_readiness(financial_data)

        if not readiness['ready_to_invest']:
            return {
                'status': 'not_ready',
                'readiness_assessment': readiness,
                'focus_areas': readiness['next_steps'],
                'general_advice': self.get_student_specific_advice()
            }

        # If ready to invest, create a plan
        monthly_amount = readiness['suggested_monthly_investment']
        risk_profile = readiness['risk_profile']

        # Organize goals by timeframe
        short_term_goals = []  # 0-2 years
        medium_term_goals = []  # 2-5 years
        long_term_goals = []  # 5+ years

        for goal in investment_goals:
            months = goal.get('timeframe_months', 60)  # Default to 5 years

            if months <= 24:
                short_term_goals.append(goal)
            elif months <= 60:
                medium_term_goals.append(goal)
            else:
                long_term_goals.append(goal)

        # Create allocations for different timeframes
        allocations = {}
        projections = {}

        # Short-term allocation (conservative)
        if short_term_goals:
            short_term_allocation = self._suggest_allocation(
                'Conservative',
                monthly_amount * 0.3,  # 30% of investment budget
                investment_timeframe=24
            )
            allocations['short_term'] = short_term_allocation

            projections['short_term'] = self.calculate_investment_projection(
                short_term_allocation['monthly_amount'],
                short_term_allocation['percentages'],
                years=2
            )

        # Medium-term allocation (balanced)
        if medium_term_goals:
            medium_term_allocation = self._suggest_allocation(
                'Balanced' if risk_profile in ['Balanced', 'Growth', 'Aggressive'] else 'Moderately Conservative',
                monthly_amount * 0.4,  # 40% of investment budget
                investment_timeframe=60
            )
            allocations['medium_term'] = medium_term_allocation

            projections['medium_term'] = self.calculate_investment_projection(
                medium_term_allocation['monthly_amount'],
                medium_term_allocation['percentages'],
                years=5
            )

        # Long-term allocation (based on risk profile)
        if long_term_goals:
            long_term_allocation = self._suggest_allocation(
                risk_profile,
                monthly_amount * 0.3,  # 30% of investment budget
                investment_timeframe=120
            )
            allocations['long_term'] = long_term_allocation

            projections['long_term'] = self.calculate_investment_projection(
                long_term_allocation['monthly_amount'],
                long_term_allocation['percentages'],
                years=10
            )

        # If no goals in a category, redistribute
        if not short_term_goals and not medium_term_goals:
            # All long-term
            long_term_allocation = self._suggest_allocation(
                risk_profile,
                monthly_amount,
                investment_timeframe=120
            )
            allocations = {'long_term': long_term_allocation}

            projections['long_term'] = self.calculate_investment_projection(
                long_term_allocation['monthly_amount'],
                long_term_allocation['percentages'],
                years=10
            )
        elif not short_term_goals and not long_term_goals:
            # All medium-term
            medium_term_allocation = self._suggest_allocation(
                'Balanced' if risk_profile in ['Balanced', 'Growth', 'Aggressive'] else 'Moderately Conservative',
                monthly_amount,
                investment_timeframe=60
            )
            allocations = {'medium_term': medium_term_allocation}

            projections['medium_term'] = self.calculate_investment_projection(
                medium_term_allocation['monthly_amount'],
                medium_term_allocation['percentages'],
                years=5
            )
        elif not medium_term_goals and not long_term_goals:
            # All short-term
            short_term_allocation = self._suggest_allocation(
                'Conservative',
                monthly_amount,
                investment_timeframe=24
            )
            allocations = {'short_term': short_term_allocation}

            projections['short_term'] = self.calculate_investment_projection(
                short_term_allocation['monthly_amount'],
                short_term_allocation['percentages'],
                years=2
            )

        # Generate action plan
        action_plan = self._generate_action_plan(allocations)

        # Generate student-specific recommendations
        student_advice = self.get_student_specific_advice()

        return {
            'status': 'ready',
            'readiness_assessment': readiness,
            'monthly_investment': monthly_amount,
            'risk_profile': risk_profile,
            'goals': {
                'short_term': short_term_goals,
                'medium_term': medium_term_goals,
                'long_term': long_term_goals
            },
            'allocations': allocations,
            'projections': projections,
            'action_plan': action_plan,
            'student_specific_advice': student_advice
        }

    def _generate_action_plan(self, allocations: Dict[str, Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Generate an action plan based on investment allocations.

        Args:
            allocations: Dictionary with investment allocations

        Returns:
            List of action steps
        """
        action_plan = []

        # Step 1: Basic setup
        action_plan.append({
            'step': 1,
            'title': 'Set up investment accounts',
            'description': 'Open accounts with the recommended providers based on your investment allocation.',
            'timeframe': '1-2 weeks'
        })

        # Step 2: First investments
        action_plan.append({
            'step': 2,
            'title': 'Make initial investments',
            'description': 'Fund your accounts and make your first investments according to your allocation plan.',
            'timeframe': '1 week after account setup'
        })

        # Step 3: Automation
        action_plan.append({
            'step': 3,
            'title': 'Set up automatic transfers',
            'description': 'Automate your monthly investments to maintain consistent investing habits.',
            'timeframe': 'Once accounts are funded'
        })

        # Step 4: Review providers
        providers_needed = set()
        for timeframe, allocation in allocations.items():
            for investment, providers in allocation.get('providers', {}).items():
                if providers:
                    providers_needed.update(providers[:1])  # Add first recommended provider

        provider_recommendations = ""
        for provider in providers_needed:
            provider_recommendations += f"• {provider}\n"

        action_plan.append({
            'step': 4,
            'title': 'Research recommended providers',
            'description': f"Compare features and fees of these recommended providers:\n{provider_recommendations}",
            'timeframe': 'Before opening accounts'
        })

        # Step 5: Regular review
        action_plan.append({
            'step': 5,
            'title': 'Schedule regular reviews',
            'description': 'Set calendar reminders to review your investment performance quarterly and rebalance annually.',
            'timeframe': 'Ongoing'
        })

        # Step 6: Tax planning
        action_plan.append({
            'step': 6,
            'title': 'Plan for taxes',
            'description': 'Understand the tax implications of your investments in Germany and keep necessary documentation.',
            'timeframe': 'Before first tax filing'
        })

        return action_plan


# Example usage
if __name__ == "__main__":
    advisor = InvestmentAdvisor()

    # Example financial data
    financial_data = {
        'monthly_income': 1000,
        'monthly_expenses': 800,
        'emergency_fund': 2400,  # 3 months of expenses
        'savings': 3000,
        'debt': 0,
        'risk_tolerance': 7  # Medium-high risk tolerance
    }

    # Check if ready to invest
    readiness = advisor.assess_investment_readiness(financial_data)

    if readiness['ready_to_invest']:
        print(f"Ready to invest! Suggested monthly amount: €{readiness['suggested_monthly_investment']:.2f}")
        print(f"Recommended risk profile: {readiness['risk_profile']}")

        # Print allocation
        print("\nRecommended Allocation:")
        for investment, percent in readiness['allocation']['percentages'].items():
            amount = readiness['allocation']['amounts'][investment]
            print(f"{investment}: {percent:.1f}% (€{amount:.2f} per month)")
    else:
        print("Not ready to invest yet.")
        print("Reasons:")
        for reason in readiness['reasons']:
            print(f"- {reason}")

        print("\nNext steps:")
        for step in readiness['next_steps']:
            print(f"- {step}")