"""Service layer for expense tracker functionality."""

from datetime import date
from typing import List, Optional

from app.models.expense_tracker import (
    Transaction,
    TransactionCategory,
    SavingsGoal,
    FinancialSummary
)

class ExpenseTrackerService:
    """Service class for managing expense tracking functionality."""

    def __init__(self):
        """Initialize the expense tracker service."""
        self._transactions: List[Transaction] = []
        self._savings_goal: Optional[SavingsGoal] = None

    def add_transaction(
        self,
        description: str,
        amount: float,
        category: TransactionCategory,
        transaction_type: str,
        is_avoidable: Optional[bool] = None
    ) -> Transaction:
        """Add a new transaction to the tracker."""
        transaction = Transaction(
            date=date.today(),
            description=description,
            amount=amount,
            category=category,
            is_avoidable=is_avoidable,
            transaction_type=transaction_type
        )
        self._transactions.append(transaction)
        return transaction

    def set_savings_goal(self, total_goal: float, monthly_target: float) -> SavingsGoal:
        """Set or update the savings goal."""
        self._savings_goal = SavingsGoal(
            total_goal=total_goal,
            monthly_target=monthly_target,
            current_savings=self._calculate_current_savings()
        )
        return self._savings_goal

    def get_savings_goal(self) -> Optional[SavingsGoal]:
        """Get the current savings goal."""
        if self._savings_goal:
            self._savings_goal.current_savings = self._calculate_current_savings()
        return self._savings_goal

    def get_financial_summary(self) -> FinancialSummary:
        """Get a summary of all financial transactions."""
        total_earnings = sum(
            t.amount for t in self._transactions if t.transaction_type == "income"
        )
        total_expenses = sum(
            t.amount for t in self._transactions if t.transaction_type == "expense"
        )
        avoidable_expenses = sum(
            t.amount for t in self._transactions 
            if t.transaction_type == "expense" and t.is_avoidable
        )
        current_savings = total_earnings - total_expenses

        return FinancialSummary(
            total_earnings=total_earnings,
            total_expenses=total_expenses,
            avoidable_expenses=avoidable_expenses,
            current_savings=current_savings
        )

    def get_transactions(self) -> List[Transaction]:
        """Get all transactions sorted by date."""
        return sorted(self._transactions, key=lambda x: x.date, reverse=True)

    def _calculate_current_savings(self) -> float:
        """Calculate current savings from transactions."""
        summary = self.get_financial_summary()
        return summary.current_savings 