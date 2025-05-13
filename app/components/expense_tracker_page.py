"""
Expense tracker page component for the DhanKanya application.

This module contains the UI components for the expense tracker page,
which allows users to track their expenses, income, and savings goals.
"""

import streamlit as st
import pandas as pd
import datetime
import altair as alt
import logging
from typing import List, Dict, Any, Optional

from app.services.expense_service import (
    initialize_session,
    update_savings,
    add_expense,
    add_earning,
    add_goal,
    update_goal,
    get_expense_summary,
    get_savings_progress,
    get_avoidable_expenses,
    calculate_total_earnings,
    calculate_total_expenses
)
from app.utils.helpers import format_inr
from app.models.expense_tracker import TransactionCategory
from app.services.expense_tracker_service import ExpenseTrackerService
from app.components.auth_pages import get_current_user

# Configure logging
logger = logging.getLogger(__name__)

def render() -> None:
    """Render the expense tracker page with a modern, reactive interface."""
    # Initialize session state
    initialize_session()
    
    # Get current user
    current_user = get_current_user()
    if not current_user:
        st.warning("You must be signed in to use the expense tracker.")
        return
        
    username = current_user.get("username")
    
    # Ensure the expense_tracker service is initialized with the username
    if "expense_tracker" not in st.session_state or "expense_tracker_username" not in st.session_state or st.session_state.expense_tracker_username != username:
        st.session_state.expense_tracker = ExpenseTrackerService(username=username)
        st.session_state.expense_tracker_username = username
        
        # Load savings goal from the service
        savings_goal = st.session_state.expense_tracker.get_savings_goal()
        if savings_goal:
            st.session_state.savings_goal = savings_goal.total_goal
            st.session_state.monthly_target = savings_goal.monthly_target
            st.session_state.savings = savings_goal.current_savings
        
        # Load transactions from the service to session state
        service_transactions = st.session_state.expense_tracker.get_transactions()
        
        # Convert to the format expected by the existing code
        st.session_state.transaction_history = []
        st.session_state.expenses = []
        st.session_state.earnings = []
        
        for transaction in service_transactions:
            # Add to transaction history
            st.session_state.transaction_history.append({
                "date": transaction.date,
                "category": transaction.category.value,
                "description": transaction.description,
                "amount": transaction.amount,
                "transaction_type": "Expense" if transaction.transaction_type == "expense" else "Earning",
                "avoidable": transaction.is_avoidable
            })
            
            # Add to appropriate list based on type
            if transaction.transaction_type == "expense":
                st.session_state.expenses.append({
                    "date": transaction.date,
                    "category": transaction.category.value,
                    "description": transaction.description,
                    "amount": transaction.amount,
                    "avoidable": transaction.is_avoidable
                })
            else:
                st.session_state.earnings.append({
                    "date": transaction.date,
                    "category": transaction.category.value,
                    "description": transaction.description,
                    "amount": transaction.amount
                })
    
    # Initialize transaction history in session state if not exists
    if "transaction_history" not in st.session_state:
        st.session_state.transaction_history = []
    
    # Page header
    st.title("💰 Smart Expense Tracker")
    st.markdown("Track your expenses, manage your savings, and achieve your financial goals!")
    
    # Main container
    with st.container():
        # Create a placeholder for metrics and progress that will be updated after transactions
        metrics_placeholder = st.empty()
        
        # Savings goal section
        with st.container():
            st.subheader("🎯 Savings Goal")
            
            # Two columns for goal inputs
            goal_col1, goal_col2 = st.columns(2)
            
            with goal_col1:
                savings_goal = st.number_input(
                    "Total Savings Goal (₹)",
                    min_value=0,
                    step=1000,
                    value=int(st.session_state.savings_goal),
                    key="savings_goal_input",
                    on_change=update_goals_and_progress
                )
            
            with goal_col2:
                monthly_target = st.number_input(
                    "Monthly Savings Target (₹)",
                    min_value=0,
                    step=500,
                    value=int(st.session_state.monthly_target),
                    key="monthly_target_input",
                    on_change=update_monthly_target
                )
            
            # Create a placeholder for progress and time estimate
            progress_placeholder = st.empty()
        
        # Transaction input section
        st.subheader("💸 Add Transaction")
        
        # Create tabs for expense and income
        expense_tabs = st.tabs(["Add Expense", "Add Income"])
        
        with expense_tabs[0]:
            with st.form("expense_form", clear_on_submit=True):
                # Two columns for better layout
                exp_col1, exp_col2 = st.columns(2)
                
                with exp_col1:
                    exp_amount = st.number_input(
                        "Amount (₹)",
                        min_value=0,
                        step=100,
                        help="Enter the expense amount"
                    )
                    exp_category = st.selectbox(
                        "Category",
                        [cat.value for cat in TransactionCategory if cat.value in ["Rent", "Food", "Transport", "Shopping", "Bills", "Entertainment", "Other"]],
                        help="Select the expense category"
                    )
                
                with exp_col2:
                    exp_desc = st.text_input(
                        "Description",
                        placeholder="What did you spend on?",
                        help="Add a description for this expense"
                    )
                    exp_avoidable = st.checkbox(
                        "Is this expense avoidable?",
                        help="Mark if this expense could have been avoided"
                    )
                
                exp_submit = st.form_submit_button("Add Expense", use_container_width=True)
                
                if exp_submit and exp_desc and exp_amount > 0:
                    # Add transaction to the tracker
                    transaction = st.session_state.expense_tracker.add_transaction(
                        description=exp_desc,
                        amount=exp_amount,
                        category=TransactionCategory(exp_category),
                        transaction_type="expense",
                        is_avoidable=exp_avoidable
                    )
                    
                    # Update session state
                    st.session_state.expenses.append({
                        "date": datetime.date.today(),
                        "category": exp_category,
                        "description": exp_desc,
                        "amount": exp_amount,
                        "avoidable": exp_avoidable
                    })
                    
                    # Update transaction history
                    st.session_state.transaction_history.append({
                        "date": datetime.date.today(),
                        "category": exp_category,
                        "description": exp_desc,
                        "amount": exp_amount,
                        "transaction_type": "Expense",
                        "avoidable": exp_avoidable
                    })
                    
                    # Update savings
                    update_savings()
                    
                    st.success("Expense added successfully!")
        
        with expense_tabs[1]:
            with st.form("income_form", clear_on_submit=True):
                # Two columns for better layout
                inc_col1, inc_col2 = st.columns(2)
                
                with inc_col1:
                    inc_amount = st.number_input(
                        "Amount (₹)",
                        min_value=0,
                        step=100,
                        help="Enter the income amount"
                    )
                    inc_category = st.selectbox(
                        "Category",
                        [cat.value for cat in TransactionCategory if cat.value in ["Salary", "Scholarship", "Bonus", "Gift", "Investment Return", "Other"]],
                        help="Select the income category"
                    )
                
                with inc_col2:
                    inc_desc = st.text_input(
                        "Description",
                        placeholder="What's the source of income?",
                        help="Add a description for this income"
                    )
                
                inc_submit = st.form_submit_button("Add Income", use_container_width=True)
                
                if inc_submit and inc_desc and inc_amount > 0:
                    # Add transaction to the tracker
                    transaction = st.session_state.expense_tracker.add_transaction(
                        description=inc_desc,
                        amount=inc_amount,
                        category=TransactionCategory(inc_category),
                        transaction_type="income"
                    )
                    
                    # Update session state
                    if "earnings" not in st.session_state:
                        st.session_state.earnings = []
                    st.session_state.earnings.append({
                        "date": datetime.date.today(),
                        "category": inc_category,
                        "description": inc_desc,
                        "amount": inc_amount
                    })
                    
                    # Update transaction history
                    st.session_state.transaction_history.append({
                        "date": datetime.date.today(),
                        "category": inc_category,
                        "description": inc_desc,
                        "amount": inc_amount,
                        "transaction_type": "Earning",
                        "avoidable": False
                    })
                    
                    # Update savings
                    update_savings()
                    
                    st.success("Income added successfully!")
        
        # Update metrics and progress after all transactions are processed
        with metrics_placeholder.container():
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Total Savings",
                    format_inr(st.session_state.savings),
                    delta=format_inr(st.session_state.savings - st.session_state.savings_goal) if st.session_state.savings > st.session_state.savings_goal else None,
                    delta_color="normal"
                )
            
            with col2:
                total_expenses = calculate_total_expenses()
                st.metric(
                    "Total Expenses",
                    format_inr(total_expenses),
                    delta=f"-{format_inr(total_expenses)}" if total_expenses > 0 else None,
                    delta_color="inverse"
                )
            
            with col3:
                total_earnings = calculate_total_earnings()
                st.metric(
                    "Total Earnings",
                    format_inr(total_earnings),
                    delta=f"+{format_inr(total_earnings)}" if total_earnings > 0 else None,
                    delta_color="normal"
                )
        
        # Update progress and time estimate
        with progress_placeholder.container():
            # Progress visualization
            progress = min(max(st.session_state.savings / savings_goal, 0.0), 1.0) if savings_goal > 0 else 0
            st.progress(progress, text=f"{progress:.1%} of goal achieved")
            
            # Calculate time estimate only if monthly target is set
            if st.session_state.monthly_target > 0 and savings_goal > st.session_state.savings:
                remaining = savings_goal - st.session_state.savings
                months_needed = int(remaining / st.session_state.monthly_target) + (1 if remaining % st.session_state.monthly_target > 0 else 0)
                st.write(f"🕒 At your current monthly target, you will reach your goal in approximately **{months_needed} months**.")
            elif savings_goal <= st.session_state.savings and savings_goal > 0:
                st.write("🎉 Congratulations! You've reached your savings goal!")
        
        # Transaction history
        st.subheader("📝 Recent Transactions")
        render_recent_transactions()
        
        # Analysis sections
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Spending by Category")
            render_category_analysis()
            
        with col2:
            st.subheader("📈 Time Analysis")
            render_time_analysis()

def render_recent_transactions() -> None:
    """Render the recent transactions list with filters and sorting."""
    if not st.session_state.transaction_history:
        st.info("No transactions recorded yet. Start by adding income or expenses.")
        return
    
    # Filter controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filter_type = st.selectbox(
            "Filter by",
            ["All", "Expenses", "Income"],
            key="filter_type"
        )
    
    with col2:
        categories = ["All Categories"]
        if filter_type == "Expenses":
            categories.extend([cat.value for cat in TransactionCategory if cat.value in ["Rent", "Food", "Transport", "Shopping", "Bills", "Entertainment", "Other"]])
        elif filter_type == "Income":
            categories.extend([cat.value for cat in TransactionCategory if cat.value in ["Salary", "Scholarship", "Bonus", "Gift", "Investment Return", "Other"]])
        else:
            categories.extend([cat.value for cat in TransactionCategory])
        
        filter_category = st.selectbox(
            "Category",
            categories,
            key="filter_category"
        )
    
    with col3:
        sort_by = st.selectbox(
            "Sort by",
            ["Date (newest)", "Date (oldest)", "Amount (highest)", "Amount (lowest)"],
            key="sort_by"
        )
    
    # Apply filters
    filtered_transactions = st.session_state.transaction_history.copy()
    
    if filter_type == "Expenses":
        filtered_transactions = [t for t in filtered_transactions if t["transaction_type"] == "Expense"]
    elif filter_type == "Income":
        filtered_transactions = [t for t in filtered_transactions if t["transaction_type"] == "Earning"]
    
    if filter_category != "All Categories":
        filtered_transactions = [t for t in filtered_transactions if t["category"] == filter_category]
    
    # Apply sorting
    if sort_by == "Date (newest)":
        filtered_transactions = sorted(filtered_transactions, key=lambda x: x["date"], reverse=True)
    elif sort_by == "Date (oldest)":
        filtered_transactions = sorted(filtered_transactions, key=lambda x: x["date"])
    elif sort_by == "Amount (highest)":
        filtered_transactions = sorted(filtered_transactions, key=lambda x: x["amount"], reverse=True)
    elif sort_by == "Amount (lowest)":
        filtered_transactions = sorted(filtered_transactions, key=lambda x: x["amount"])
    
    # Display transactions
    if not filtered_transactions:
        st.info("No transactions match your filters.")
        return
    
    for transaction in filtered_transactions[:10]:  # Show only the most recent 10 transactions
        with st.container():
            cols = st.columns([2, 3, 2, 2])
            with cols[0]:
                st.write(f"**{transaction['date'].strftime('%Y-%m-%d')}**")
            with cols[1]:
                st.write(transaction["description"])
            with cols[2]:
                st.write(transaction["category"])
            with cols[3]:
                amount_text = format_inr(transaction["amount"])
                if transaction["transaction_type"] == "Expense":
                    st.write(f"🔴 -{amount_text}")
                else:
                    st.write(f"🟢 +{amount_text}")
            
            st.markdown("---")
    
    # Show button to view all transactions
    if len(filtered_transactions) > 10:
        if st.button("Show All Transactions"):
            # Convert to DataFrame for better display
            df = pd.DataFrame(filtered_transactions)
            df["date"] = df["date"].astype(str)
            df["amount"] = df["amount"].map(lambda x: format_inr(x))
            st.dataframe(df[["date", "description", "category", "transaction_type", "amount"]])

def render_category_analysis() -> None:
    """Render the spending by category analysis."""
    if not st.session_state.expenses:
        st.info("No expenses recorded yet. Add some expenses to see the analysis.")
        return
    
    # Group expenses by category
    categories = {}
    for expense in st.session_state.expenses:
        category = expense["category"]
        amount = expense["amount"]
        categories[category] = categories.get(category, 0) + amount
    
    # Convert to DataFrame for chart
    df = pd.DataFrame({
        "category": list(categories.keys()),
        "amount": list(categories.values())
    })
    
    # Create pie chart
    chart = alt.Chart(df).mark_arc().encode(
        theta=alt.Theta(field="amount", type="quantitative"),
        color=alt.Color(field="category", type="nominal", legend=alt.Legend(title="Categories")),
        tooltip=["category", "amount"]
    ).properties(
        width=300,
        height=300
    )
    
    st.altair_chart(chart, use_container_width=True)
    
    # Display category breakdown
    st.write("Category Breakdown:")
    for index, row in df.iterrows():
        st.write(f"- {row['category']}: {format_inr(row['amount'])} ({row['amount']/sum(df['amount'])*100:.1f}%)")

def render_time_analysis() -> None:
    """Render the spending over time analysis."""
    if not st.session_state.transaction_history:
        st.info("No transactions recorded yet. Add some transactions to see the analysis.")
        return
    
    # Prepare data for the chart
    transactions_df = pd.DataFrame(st.session_state.transaction_history)
    
    if transactions_df.empty:
        return
    
    # Ensure date is datetime
    transactions_df["date"] = pd.to_datetime(transactions_df["date"])
    
    # Group by date and transaction type
    income_by_date = transactions_df[transactions_df["transaction_type"] == "Earning"].groupby(transactions_df["date"].dt.date)["amount"].sum().reset_index()
    expense_by_date = transactions_df[transactions_df["transaction_type"] == "Expense"].groupby(transactions_df["date"].dt.date)["amount"].sum().reset_index()
    
    income_by_date["type"] = "Income"
    expense_by_date["type"] = "Expense"
    
    combined_df = pd.concat([income_by_date, expense_by_date])
    
    # Create the chart
    chart = alt.Chart(combined_df).mark_bar().encode(
        x="date:T",
        y="amount:Q",
        color=alt.Color("type:N", scale=alt.Scale(domain=["Income", "Expense"], range=["green", "red"])),
        tooltip=["date", "amount", "type"]
    ).properties(
        width=400,
        height=300
    )
    
    st.altair_chart(chart, use_container_width=True)
    
    # Display additional stats
    
    if len(expense_by_date) > 1:
        avg_daily_expense = expense_by_date["amount"].mean()
        st.write(f"Average daily spending: {format_inr(avg_daily_expense)}")
    
    if len(income_by_date) > 0 and len(expense_by_date) > 0:
        avg_income = income_by_date["amount"].mean()
        avg_expense = expense_by_date["amount"].mean()
        st.write(f"Average income to expense ratio: {avg_income/avg_expense:.2f}")

def update_goals_and_progress() -> None:
    """Update savings goals and progress in both session state and database."""
    st.session_state.savings_goal = st.session_state.savings_goal_input
    
    if "expense_tracker" in st.session_state:
        # Get current monthly target
        monthly_target = st.session_state.monthly_target
        
        # Save to database through the expense tracker service
        success = st.session_state.expense_tracker.set_savings_goal(
            total_goal=st.session_state.savings_goal,
            monthly_target=monthly_target
        )
        
        if not success:
            logger.error("Failed to update savings goal in database")

def update_monthly_target() -> None:
    """Update monthly target in both session state and database."""
    st.session_state.monthly_target = st.session_state.monthly_target_input
    
    if "expense_tracker" in st.session_state:
        # Get current savings goal
        savings_goal = st.session_state.savings_goal
        
        # Save to database through the expense tracker service
        success = st.session_state.expense_tracker.set_savings_goal(
            total_goal=savings_goal,
            monthly_target=st.session_state.monthly_target
        )
        
        if not success:
            logger.error("Failed to update monthly target in database")

def update_income() -> None:
    """Update income values in session state."""
    # This function is kept for backward compatibility
    pass 