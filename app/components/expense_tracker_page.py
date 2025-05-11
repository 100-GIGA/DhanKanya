"""
Expense tracker page component for the DhanKanya application.

This module contains the UI components for the expense tracker page,
which allows users to track their expenses, income, and savings goals.
"""

import streamlit as st
import pandas as pd
import datetime
import altair as alt
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

def render() -> None:
    """Render the expense tracker page with a modern, reactive interface."""
    # Initialize session state
    initialize_session()
    
    # Ensure the expense_tracker service is initialized
    if "expense_tracker" not in st.session_state:
        st.session_state.expense_tracker = ExpenseTrackerService()
    
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
            
            # Estimated time calculation
            if savings_goal > 0 and monthly_target > 0:
                remaining_amount = max(0, savings_goal - st.session_state.savings)
                months_required = remaining_amount / monthly_target
                months_required = int(months_required) + 1 if months_required % 1 > 0 else int(months_required)
                st.caption(f"Estimated time to achieve goal: {months_required} months")
        
        # Transaction history section
        st.subheader("📊 Transaction History")
        
        # Create tabs for different views
        history_tabs = st.tabs(["Recent Transactions", "Category Analysis", "Time Analysis"])
        
        with history_tabs[0]:
            render_recent_transactions()
        
        with history_tabs[1]:
            render_category_analysis()
        
        with history_tabs[2]:
            render_time_analysis()

def render_recent_transactions() -> None:
    """Render the recent transactions view with filtering options."""
    if not st.session_state.transaction_history:
        st.info("No transactions recorded yet. Add some expenses or earnings to see them here.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(st.session_state.transaction_history)
    
    # Add filters
    with st.container(border=True):
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            # Date range filter
            min_date = df["date"].min() if "date" in df.columns else datetime.date.today() - datetime.timedelta(days=30)
            max_date = df["date"].max() if "date" in df.columns else datetime.date.today()
            
            date_range = st.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date - datetime.timedelta(days=365),
                max_value=datetime.date.today()
            )
        
        with filter_col2:
            # Transaction type filter
            transaction_types = ["All", "Expense", "Earning"]
            selected_type = st.selectbox("Transaction Type", transaction_types)
    
    # Apply filters
    filtered_df = df.copy()
    
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[(filtered_df["date"] >= start_date) & 
                                (filtered_df["date"] <= end_date)]
    
    if selected_type != "All":
        filtered_df = filtered_df[filtered_df["transaction_type"] == selected_type]
    
    # Sort by date (most recent first)
    filtered_df = filtered_df.sort_values("date", ascending=False)
    
    # Display transactions
    if filtered_df.empty:
        st.info("No transactions match the selected filters.")
    else:
        # Format for display
        display_df = filtered_df.copy()
        display_df["amount"] = display_df["amount"].apply(lambda x: format_inr(x))
        display_df["date"] = display_df["date"].apply(lambda x: x.strftime("%d %b %Y"))
        
        # Clean up column names
        column_rename = {
            "date": "Date",
            "description": "Description",
            "category": "Category",
            "amount": "Amount",
            "transaction_type": "Type"
        }
        display_df = display_df.rename(columns=column_rename)
        
        # Display the table
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

def render_category_analysis() -> None:
    """Render the category analysis view with charts."""
    # Get expense summary
    expense_summary = get_expense_summary()
    
    if expense_summary.empty:
        st.info("No expense data available for category analysis.")
        return
    
    # Create two columns for different visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart for expense categories
        st.markdown("#### Expense Distribution by Category")
        
        pie_chart = alt.Chart(expense_summary).mark_arc().encode(
            theta=alt.Theta(field="total", type="quantitative"),
            color=alt.Color(field="category", type="nominal"),
            tooltip=["category", "total"]
        ).properties(
            height=300
        )
        
        st.altair_chart(pie_chart, use_container_width=True)
    
    with col2:
        # Bar chart for expense categories
        st.markdown("#### Expense Amount by Category")
        
        bar_chart = alt.Chart(expense_summary).mark_bar().encode(
            x=alt.X("category", sort="-y"),
            y=alt.Y("total", title="Amount (₹)"),
            color=alt.Color("category"),
            tooltip=["category", "total"]
        ).properties(
            height=300
        )
        
        st.altair_chart(bar_chart, use_container_width=True)

def render_time_analysis() -> None:
    """Render the time analysis view with trend charts."""
    if not st.session_state.transaction_history:
        st.info("No transaction data available for time analysis.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(st.session_state.transaction_history)
    
    # Convert date column to datetime if it's not already
    df["date"] = pd.to_datetime(df["date"])
    
    # Add month column for grouping
    df["month"] = df["date"].dt.to_period("M")
    
    # Group by month and transaction type
    monthly_summary = df.groupby(["month", "transaction_type"])["amount"].sum().reset_index()
    
    # Create line chart for trends
    st.markdown("#### Monthly Transaction Trends")
    
    line_chart = alt.Chart(monthly_summary).mark_line().encode(
        x=alt.X("month:T", title="Month"),
        y=alt.Y("amount", title="Amount (₹)"),
        color="transaction_type",
        tooltip=["month", "transaction_type", "amount"]
    ).properties(
        height=400
    )
    
    st.altair_chart(line_chart, use_container_width=True)
    
    # Monthly summary table
    st.markdown("#### Monthly Summary")
    
    # Pivot the data for the table
    pivot_df = monthly_summary.pivot(
        index="month",
        columns="transaction_type",
        values="amount"
    ).fillna(0)
    
    # Ensure both Expense and Earning columns exist
    if "Expense" not in pivot_df.columns:
        pivot_df["Expense"] = 0
    if "Earning" not in pivot_df.columns:
        pivot_df["Earning"] = 0
    
    # Calculate net amount
    pivot_df["Net"] = pivot_df["Earning"] - pivot_df["Expense"]
    
    # Format for display
    display_pivot = pivot_df.copy()
    display_pivot["Expense"] = display_pivot["Expense"].apply(lambda x: format_inr(x))
    display_pivot["Earning"] = display_pivot["Earning"].apply(lambda x: format_inr(x))
    display_pivot["Net"] = display_pivot["Net"].apply(lambda x: format_inr(x))
    
    # Sort by month descending
    display_pivot = display_pivot.sort_values("month", ascending=False)
    
    # Display the table
    st.dataframe(display_pivot, use_container_width=True, hide_index=True)

def update_goals_and_progress() -> None:
    """Update goals and progress when savings goal changes."""
    st.session_state.savings_goal = st.session_state.savings_goal_input
    update_savings()

def update_monthly_target() -> None:
    """Update monthly target when it changes."""
    st.session_state.monthly_target = st.session_state.monthly_target_input

def update_income() -> None:
    """Update income when monthly target changes."""
    st.session_state.income = st.session_state.monthly_target_input
    update_savings() 