"""Models for the expense tracker functionality."""

from dataclasses import dataclass
from datetime import date
from typing import List, Optional
from enum import Enum

class TransactionCategory(str, Enum):
    """Categories for transactions."""
    # Expense categories
    RENT = "Rent"
    FOOD = "Food"
    TRANSPORT = "Transport"
    SHOPPING = "Shopping"
    BILLS = "Bills"
    ENTERTAINMENT = "Entertainment"
    OTHER = "Other"
    
    # Income categories
    SALARY = "Salary"
    SCHOLARSHIP = "Scholarship"
    BONUS = "Bonus"
    GIFT = "Gift"
    INVESTMENT = "Investment Return"

@dataclass
class Transaction:
    """Represents a financial transaction (expense or income)."""
    date: date
    description: str
    amount: float
    category: TransactionCategory
    is_avoidable: Optional[bool] = None
    transaction_type: str = "expense"  # "expense" or "income"

@dataclass
class SavingsGoal:
    """Represents a savings goal with target amount and monthly target."""
    total_goal: float
    monthly_target: float
    current_savings: float = 0.0

    @property
    def months_required(self) -> int:
        """Calculate estimated months required to achieve the goal."""
        if self.monthly_target <= 0:
            return 0
        months = self.total_goal / self.monthly_target
        return int(months) + 1 if months % 1 > 0 else int(months)

    @property
    def progress(self) -> float:
        """Calculate progress towards the goal as a percentage."""
        if self.total_goal <= 0:
            return 0.0
        return min(self.current_savings / self.total_goal, 1.0)

@dataclass
class FinancialSummary:
    """Represents a summary of financial transactions."""
    total_earnings: float
    total_expenses: float
    avoidable_expenses: float
    current_savings: float

    @property
    def savings_rate(self) -> float:
        """Calculate the savings rate as a percentage of earnings."""
        if self.total_earnings <= 0:
            return 0.0
        return (self.current_savings / self.total_earnings) * 100 