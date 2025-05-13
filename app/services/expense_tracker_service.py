"""Service layer for expense tracker functionality."""

from datetime import date
from typing import List, Optional, Dict, Any
import logging

from app.models.expense_tracker import (
    Transaction,
    TransactionCategory,
    SavingsGoal,
    FinancialSummary
)
from app.services.expense_database_service import (
    get_user_id_by_username,
    save_transaction,
    get_transactions,
    save_savings_goal,
    get_savings_goal
)

# Configure logging
logger = logging.getLogger(__name__)

class ExpenseTrackerService:
    """Service class for managing expense tracking functionality."""

    def __init__(self, username: Optional[str] = None):
        """
        Initialize the expense tracker service.
        
        Args:
            username: Username of the current user
        """
        self._transactions: List[Transaction] = []
        self._savings_goal: Optional[SavingsGoal] = None
        self._username = username
        self._user_id = None
        
        # If username is provided, load user data
        if username:
            self._user_id = get_user_id_by_username(username)
            if self._user_id:
                self._load_data_from_db()
            else:
                logger.warning(f"User ID not found for username: {username}")

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
        
        # Save to database if user_id is available
        if self._user_id:
            save_transaction(
                user_id=self._user_id,
                transaction_date=transaction.date,
                description=transaction.description,
                amount=transaction.amount,
                category=transaction.category.value,
                is_avoidable=transaction.is_avoidable,
                transaction_type=transaction.transaction_type
            )
        
        return transaction

    def set_savings_goal(self, total_goal: float, monthly_target: float) -> bool:
        """
        Set or update the savings goal.
        
        Args:
            total_goal: Total savings goal amount
            monthly_target: Monthly savings target
            
        Returns:
            True if the savings goal was successfully set, False otherwise
        """
        self._savings_goal = SavingsGoal(
            total_goal=total_goal,
            monthly_target=monthly_target,
            current_savings=self._calculate_current_savings()
        )
        
        # Save to database if user_id is available
        if self._user_id:
            try:
                success = save_savings_goal(
                    user_id=self._user_id,
                    total_goal=total_goal,
                    monthly_target=monthly_target
                )
                return success
            except Exception as e:
                logger.error(f"Error saving savings goal: {e}")
                return False
        
        return True

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
        
    def _load_data_from_db(self) -> None:
        """Load user's transaction data from the database."""
        if not self._user_id:
            return
            
        # Load transactions
        db_transactions = get_transactions(self._user_id)
        self._transactions = []
        
        for t in db_transactions:
            # Convert date string to date object
            transaction_date = date.fromisoformat(t['date'])
            
            self._transactions.append(Transaction(
                date=transaction_date,
                description=t['description'],
                amount=t['amount'],
                category=TransactionCategory(t['category']),
                is_avoidable=t['is_avoidable'],
                transaction_type=t['transaction_type']
            ))
            
        # Load savings goal
        goal = get_savings_goal(self._user_id)
        if goal:
            self._savings_goal = SavingsGoal(
                total_goal=goal['total_goal'],
                monthly_target=goal['monthly_target'],
                current_savings=self._calculate_current_savings()
            ) 