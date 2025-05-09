"""
Expense tracking service for the DhanKanya application.

This module provides functions for handling expense tracking,
budgeting, and financial goal management.
"""

import streamlit as st
import pandas as pd
import datetime
from typing import Dict, List, Tuple, Optional, Any

def initialize_session() -> None:
    """
    Initialize session state variables for expense tracking.
    
    This function sets up the initial state for expense tracking,
    including expenses, income, savings, and goals.
    """
    if "expenses" not in st.session_state:
        st.session_state.expenses = []
        
    if "earnings" not in st.session_state:
        st.session_state.earnings = []
        
    if "income" not in st.session_state:
        st.session_state.income = 4000
        
    if "savings_goal" not in st.session_state:
        st.session_state.savings_goal = 45000
        
    if "savings" not in st.session_state:
        st.session_state.savings = 0
        
    if "goals" not in st.session_state:
        st.session_state.goals = []
    
    # Ensure savings are calculated on initialization
    update_savings()

def calculate_total_expenses() -> float:
    """
    Calculate the total expenses recorded in the session.
    
    Returns:
        The sum of all expenses.
    """
    if not st.session_state.expenses:
        return 0
    
    return sum(expense.get("amount", 0) for expense in st.session_state.expenses)

def calculate_total_earnings() -> float:
    """
    Calculate the total earnings recorded in the session.
    
    Returns:
        The sum of all earnings.
    """
    if not st.session_state.get("earnings"):
        return 0
    
    return sum(earning.get("amount", 0) for earning in st.session_state.earnings)

def update_savings() -> None:
    """Update the savings amount based on income and expenses."""
    total_expenses = calculate_total_expenses()
    st.session_state.savings = st.session_state.income - total_expenses

def add_expense(date: datetime.date, category: str, description: str, amount: float, avoidable: bool = False) -> None:
    """
    Add a new expense to the tracking system.
    
    Args:
        date: The date of the expense.
        category: The expense category.
        description: A description of the expense.
        amount: The expense amount.
        avoidable: Whether the expense could be avoided.
    """
    st.session_state.expenses.append({
        "date": date,
        "category": category,
        "description": description,
        "amount": amount,
        "avoidable": avoidable
    })
    update_savings()

def add_earning(date: datetime.date, category: str, description: str, amount: float) -> None:
    """
    Add a new earning to the tracking system.
    
    Args:
        date: The date of the earning.
        category: The earning category.
        description: A description of the earning.
        amount: The earning amount.
    """
    if "earnings" not in st.session_state:
        st.session_state.earnings = []
        
    st.session_state.earnings.append({
        "date": date,
        "category": category,
        "description": description,
        "amount": amount
    })
    
    # Update income with the new earning
    st.session_state.income += amount
    update_savings()

def add_goal(name: str, target_amount: float, target_date: datetime.date) -> None:
    """
    Add a new financial goal.
    
    Args:
        name: The name of the goal.
        target_amount: The target amount to save.
        target_date: The target date to achieve the goal.
    """
    st.session_state.goals.append({
        "name": name,
        "target_amount": target_amount,
        "target_date": target_date,
        "current_amount": 0
    })

def update_goal(index: int, amount: float) -> None:
    """
    Update the progress of a goal.
    
    Args:
        index: The index of the goal to update.
        amount: The amount to add to the goal's current amount.
    """
    if 0 <= index < len(st.session_state.goals):
        st.session_state.goals[index]["current_amount"] += amount

def get_expense_summary() -> pd.DataFrame:
    """
    Get a summary of expenses by category.
    
    Returns:
        A DataFrame with expense totals by category.
    """
    if not st.session_state.expenses:
        return pd.DataFrame(columns=["category", "total"])
    
    # Group expenses by category and sum the amounts
    expenses_by_category = {}
    for expense in st.session_state.expenses:
        category = expense.get("category", "Other")
        amount = expense.get("amount", 0)
        expenses_by_category[category] = expenses_by_category.get(category, 0) + amount
    
    # Convert to DataFrame
    summary = pd.DataFrame({
        "category": list(expenses_by_category.keys()),
        "total": list(expenses_by_category.values())
    })
    
    return summary

def get_savings_progress() -> float:
    """
    Calculate the savings progress as a percentage of the goal.
    
    Returns:
        The percentage of savings goal achieved.
    """
    if st.session_state.savings_goal == 0:
        return 0
    return min(100, (st.session_state.savings / st.session_state.savings_goal) * 100)

def get_avoidable_expenses() -> float:
    """
    Calculate the total of avoidable expenses.
    
    Returns:
        The sum of all avoidable expenses.
    """
    if not st.session_state.expenses:
        return 0
    
    return sum(expense.get("amount", 0) for expense in st.session_state.expenses 
               if expense.get("avoidable", False)) 