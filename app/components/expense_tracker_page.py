"""
Expense tracker page component for the DhanKanya application.

This module contains the UI components for the expense tracker page,
which allows users to track their expenses, income, and savings goals.
"""

import streamlit as st
import pandas as pd
import datetime
import altair as alt
from typing import List, Dict, Any

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

def render(use_reactive_interface: bool = True) -> None:
    """
    Render the expense tracker page with budgeting and goal tracking functionality.
    
    Args:
        use_reactive_interface: If True, displays the reactive single-page interface.
                                If False, displays the original tabbed interface.
    """
    if use_reactive_interface:
        expense_tracker_page()
    else:
        # Initialize session state
        initialize_session()
        
        st.title("💰 Savings and Budgeting")
        
        # Create tabs for different sections with more descriptive names
        tabs = st.tabs(["Dashboard", "Track Expenses", "Set Savings Goals", "Transaction History"])
        
        # Dashboard tab
        with tabs[0]:
            render_dashboard()
        
        # Track Expenses tab
        with tabs[1]:
            render_add_expense()
        
        # Set Savings Goals tab
        with tabs[2]:
            render_set_goals()
        
        # Transaction History tab
        with tabs[3]:
            render_history()

def expense_tracker_page() -> None:
    """
    Render the goal-oriented expense tracker page with reactive updates.
    """
    # Initialize session
    initialize_session()
    
    # Page title
    st.title("💰 Goal-Oriented Expense Tracker")

    # Savings Goal Section
    st.subheader("Your Savings Goal")
    
    # Make sure the session_state key exists before using it
    if "savings_goal_input" not in st.session_state:
        st.session_state.savings_goal_input = st.session_state.savings_goal
        
    # Add on_change callback to make savings goal input reactive
    savings_goal = st.number_input(
        "Total Savings Goal (₹)", 
        min_value=0, 
        step=1000,
        value=int(st.session_state.savings_goal),
        key="savings_goal_input",
        on_change=update_goals_and_progress
    )
    
    # Make sure the session_state key exists before using it
    if "monthly_target_input" not in st.session_state:
        st.session_state.monthly_target_input = st.session_state.income
    
    # Add on_change callback to make monthly target input reactive
    monthly_target = st.number_input(
        "Monthly Savings Target (₹)", 
        min_value=0, 
        step=500,
        value=int(st.session_state.income),
        key="monthly_target_input",
        on_change=update_income
    )

    # Calculate and display estimated time to achieve goal
    if savings_goal > 0 and monthly_target > 0:
        months_required = savings_goal / monthly_target
        months_required = int(months_required) + 1 if months_required % 1 > 0 else int(months_required)
        st.markdown(f"*Estimated time to achieve goal: **{months_required} months***")

    # Progress Calculation
    progress = min(st.session_state.savings / savings_goal, 1.0) if savings_goal > 0 else 0

    # Savings Progress
    st.subheader("Savings Progress")
    st.progress(progress)
    
    pm_col1, pm_col2, pm_col3 = st.columns(3)
    pm_col1.metric("Saved Amount", format_inr(st.session_state.savings))
    pm_col2.metric("More to Save", format_inr(max(savings_goal - st.session_state.savings, 0)))
    pm_col3.metric("Total Target", format_inr(savings_goal))

    # Earnings and Expenses Input Side by Side
    expns_col1, expns_col2 = st.columns(2)
    
    with expns_col1:
        st.subheader("Add Expenses")
        with st.form("expenses_form"):
            exp_amount = st.number_input("Expense Amount (₹)", min_value=0, step=100)
            exp_desc = st.text_input("Expense Description")
            exp_category = st.selectbox("Category", 
                ["Rent", "Food", "Transport", "Shopping", "Bills", "Entertainment", "Other"])
            exp_avoidable = st.checkbox("Is this expense avoidable?")
            exp_submit = st.form_submit_button("Add Expense")
            
            if exp_submit and exp_desc and exp_amount > 0:
                # Use the service function directly
                add_expense(
                    date=datetime.date.today(),
                    category=exp_category,
                    description=exp_desc,
                    amount=exp_amount,
                    avoidable=exp_avoidable
                )
                st.rerun()  # Force UI update

    with expns_col2:
        st.subheader("Add Earnings")
        with st.form("earnings_form"):
            e_amount = st.number_input("Earning Amount (₹)", min_value=0, step=100)
            e_desc = st.text_input("Earning Description")
            e_category = st.selectbox("Category", 
                ["Scholarship", "Bonus", "Gift", "Investment Return", "Other"])
            e_submit = st.form_submit_button("Add Earning")
            
            if e_submit and e_desc and e_amount > 0:
                # Use the service function directly
                add_earning(
                    date=datetime.date.today(),
                    category=e_category,
                    description=e_desc,
                    amount=e_amount
                )
                st.rerun()  # Force UI update

    # Summary Section
    st.subheader("Financial Summary")
    
    # Calculate the totals using service functions
    total_earnings = calculate_total_earnings()
    total_expenses = calculate_total_expenses()
    total_avoidable_expenses = get_avoidable_expenses()

    # Display the metrics
    fs_col1, fs_col2, fs_col3 = st.columns(3)
    fs_col1.metric("Total Earnings", format_inr(total_earnings))
    fs_col2.metric("Total Expenses", format_inr(total_expenses))
    fs_col3.metric("Avoidable Expenses", format_inr(total_avoidable_expenses))

    # Ledger Table
    st.subheader("📜 Earnings & Expenses Ledger")
    
    # Create a combined list of earnings and expenses
    ledger_data = []
    
    # Add earnings to the ledger data
    for e in st.session_state.get("earnings", []):
        ledger_entry = e.copy()
        ledger_entry["type"] = "Earning"
        ledger_entry["avoidable"] = "-"
        ledger_data.append(ledger_entry)
    
    # Add expenses to the ledger data
    for e in st.session_state.expenses:
        ledger_entry = e.copy()
        ledger_entry["type"] = "Expense"
        ledger_data.append(ledger_entry)

    # Create a DataFrame from the ledger data
    ledger_df = pd.DataFrame(ledger_data)
    
    # Display the ledger
    if not ledger_df.empty:
        ledger_df["avoidable"] = ledger_df.get("avoidable", "-")
        ledger_df = ledger_df.sort_values(by="date", ascending=False)
        st.dataframe(ledger_df, use_container_width=True, hide_index=True)
    else:
        st.info("No earnings or expenses recorded yet.")

def update_goals_and_progress() -> None:
    """Update savings goal and progress when the input changes."""
    # Use the key specified in the number_input widget
    st.session_state.savings_goal = st.session_state.savings_goal_input

def update_income() -> None:
    """Update income when the monthly target input changes."""
    # Use the key specified in the number_input widget
    st.session_state.income = st.session_state.monthly_target_input
    update_savings()

# Keep the existing rendering functions for other tabs
def render_dashboard() -> None:
    """Render the dashboard with financial overview."""
    st.header("Financial Dashboard")
    
    # Update savings amount
    update_savings()
    
    # Add a brief explanation
    st.markdown("""
    This dashboard provides a real-time overview of your finances, including your income, expenses, 
    savings progress, and financial goals. All metrics update automatically as you add expenses or earnings.
    """)
    
    # Create 3 columns for financial overview with improved metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Monthly Income", 
            format_inr(st.session_state.income),
            help="Your current monthly income"
        )
    
    with col2:
        total_expenses = calculate_total_expenses()
        st.metric(
            "Total Expenses", 
            format_inr(total_expenses),
            delta=f"-{format_inr(total_expenses)}" if total_expenses > 0 else None,
            delta_color="inverse",
            help="Total expenses recorded in this period"
        )
    
    with col3:
        st.metric(
            "Savings", 
            format_inr(st.session_state.savings),
            delta=f"+{format_inr(st.session_state.savings)}" if st.session_state.savings > 0 else f"{format_inr(st.session_state.savings)}",
            help="Current savings (Income - Expenses)"
        )
    
    # Add spacer
    st.markdown("---")
    
    # Progress bar for savings goal with improved visualization
    savings_progress = get_savings_progress()
    
    # Create 2 columns for savings goal progress
    prog_col1, prog_col2 = st.columns([2, 1])
    
    with prog_col1:
        st.subheader("Savings Goal Progress")
        st.progress(savings_progress / 100)
        st.caption(f"{savings_progress:.1f}% of goal - {format_inr(st.session_state.savings)} / {format_inr(st.session_state.savings_goal)}")
    
    with prog_col2:
        if st.session_state.savings_goal > 0 and st.session_state.income > 0:
            months_required = st.session_state.savings_goal / (st.session_state.income - calculate_total_expenses())
            if months_required > 0:
                months_required = int(months_required) + 1 if months_required % 1 > 0 else int(months_required)
                st.info(f"😊 Estimated time to achieve goal: **{months_required} months**")
            else:
                st.warning("⚠️ Your expenses exceed your income. Adjust your budget to achieve your savings goal.")
    
    # Add spacer
    st.markdown("---")
    
    # Expense breakdown chart with improved visualization
    st.subheader("Expense Breakdown")
    expense_summary = get_expense_summary()
    
    if not expense_summary.empty:
        # Create a more visually appealing chart
        chart = alt.Chart(expense_summary).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field="total", type="quantitative"),
            color=alt.Color(field="category", type="nominal", scale=alt.Scale(scheme='category10')),
            tooltip=["category", "total"]
        ).properties(width=400, height=300)
        
        st.altair_chart(chart, use_container_width=True)
        
        # Add a table with expense breakdown
        st.subheader("Expense Categories")
        expense_summary = expense_summary.sort_values("total", ascending=False)
        expense_summary["percentage"] = expense_summary["total"] / expense_summary["total"].sum() * 100
        expense_summary["total"] = expense_summary["total"].apply(lambda x: format_inr(x))
        expense_summary["percentage"] = expense_summary["percentage"].apply(lambda x: f"{x:.1f}%")
        st.dataframe(expense_summary, hide_index=True, use_container_width=True)
    else:
        st.info("No expenses recorded yet. Add some expenses to see a breakdown.")
    
    # Goals progress with improved visualization
    if st.session_state.goals:
        st.markdown("---")
        st.subheader("Financial Goals Progress")
        
        # Create columns for goals
        goals_cols = st.columns(min(3, len(st.session_state.goals)))
        
        for i, goal in enumerate(st.session_state.goals):
            col_index = i % 3
            with goals_cols[col_index]:
                progress = (goal["current_amount"] / goal["target_amount"]) * 100 if goal["target_amount"] > 0 else 0
                st.markdown(f"#### {goal['name']}")
                st.progress(min(100, progress) / 100)
                st.caption(f"{progress:.1f}% - {format_inr(goal['current_amount'])} / {format_inr(goal['target_amount'])}")
                
                time_left = (goal["target_date"] - datetime.date.today()).days
                if time_left > 0:
                    st.caption(f"⏱️ *{time_left} days remaining*")
                else:
                    st.caption("⚠️ *Goal deadline has passed*")

def render_add_expense() -> None:
    """Render the form for adding expenses and earnings with improved UI."""
    st.header("Track Your Money Flow")
    
    # Add helpful description
    st.markdown("""
    Keep track of where your money comes from and where it goes. This information will help you 
    identify spending patterns and opportunities to save more effectively.
    """)
    
    # Create tabs for expenses and earnings
    expense_tabs = st.tabs(["Add Expense", "Add Earning"])
    
    # Expense tab
    with expense_tabs[0]:
        st.subheader("Record New Expense")
        
        # Create expense form with better organization
        with st.form("expense_form", clear_on_submit=True):
            # Create two columns for date and category
            col1, col2 = st.columns(2)
            
            with col1:
                # Expense date
                expense_date = st.date_input(
                    "Date", 
                    datetime.date.today(),
                    help="When was this expense incurred?"
                )
            
            with col2:
                # Expense category with more options
                categories = [
                    "Food", "Transportation", "Education", "Entertainment", 
                    "Clothing", "Healthcare", "Utilities", "Rent", "Shopping", 
                    "Bills", "Other"
                ]
                category = st.selectbox(
                    "Category", 
                    categories,
                    help="Select the category that best describes this expense"
                )
            
            # Expense description
            description = st.text_input(
                "Description",
                help="Add details about this expense (e.g., 'Lunch at Restaurant')"
            )
            
            # Create two columns for amount and avoidable checkbox
            col3, col4 = st.columns([3, 2])
            
            with col3:
                # Expense amount
                amount = st.number_input(
                    "Amount (₹)", 
                    min_value=0.0, 
                    step=10.0,
                    help="How much did you spend?"
                )
            
            with col4:
                # Is this expense avoidable?
                avoidable = st.checkbox(
                    "Avoidable expense?",
                    help="Could this expense have been avoided or reduced?"
                )
            
            # Submit button
            submitted = st.form_submit_button("Add Expense", use_container_width=True)
            
            if submitted:
                if amount <= 0:
                    st.error("Please enter a valid amount.")
                elif not description:
                    st.warning("Adding a description helps you track expenses better.")
                    # Still add the expense but warn about missing description
                    add_expense(expense_date, category, "Unlabeled expense", amount, avoidable)
                    st.success("Expense added successfully!")
                    st.rerun()
                else:
                    add_expense(expense_date, category, description, amount, avoidable)
                    st.success("Expense added successfully!")
                    st.rerun()
        
        # Quick expense summary
        if st.session_state.expenses:
            st.subheader("Recent Expenses")
            recent_expenses = sorted(st.session_state.expenses, key=lambda x: x.get('date', datetime.date.today()), reverse=True)[:5]
            for expense in recent_expenses:
                st.markdown(f"""
                **{expense.get('description', 'Unnamed')}** - {format_inr(expense.get('amount', 0))}  
                *{expense.get('category', 'Uncategorized')} · {expense.get('date').strftime('%d %b %Y')}*
                """)
    
    # Earnings tab
    with expense_tabs[1]:
        st.subheader("Record New Earning")
        
        # Create earning form with better organization
        with st.form("earning_form", clear_on_submit=True):
            # Create two columns for date and category
            col1, col2 = st.columns(2)
            
            with col1:
                # Earning date
                earning_date = st.date_input(
                    "Date", 
                    datetime.date.today(),
                    help="When did you receive this income?"
                )
            
            with col2:
                # Earning category with relevant options
                categories = [
                    "Salary", "Bonus", "Gift", "Scholarship", "Investment Return", 
                    "Side Hustle", "Refund", "Other"
                ]
                category = st.selectbox(
                    "Source", 
                    categories,
                    help="Where did this money come from?"
                )
            
            # Earning description
            description = st.text_input(
                "Description",
                help="Add details about this earning (e.g., 'May Salary')"
            )
            
            # Earning amount
            amount = st.number_input(
                "Amount (₹)", 
                min_value=0.0, 
                step=100.0,
                help="How much did you earn?"
            )
            
            # Submit button
            submitted = st.form_submit_button("Add Earning", use_container_width=True)
            
            if submitted:
                if amount <= 0:
                    st.error("Please enter a valid amount.")
                elif not description:
                    st.warning("Adding a description helps you track earnings better.")
                    # Still add the earning but warn about missing description
                    add_earning(earning_date, category, "Unlabeled earning", amount)
                    st.success("Earning added successfully!")
                    st.rerun()
                else:
                    add_earning(earning_date, category, description, amount)
                    st.success("Earning added successfully!")
                    st.rerun()
        
        # Quick earnings summary
        if st.session_state.get("earnings", []):
            st.subheader("Recent Earnings")
            recent_earnings = sorted(st.session_state.earnings, key=lambda x: x.get('date', datetime.date.today()), reverse=True)[:5]
            for earning in recent_earnings:
                st.markdown(f"""
                **{earning.get('description', 'Unnamed')}** - {format_inr(earning.get('amount', 0))}  
                *{earning.get('category', 'Uncategorized')} · {earning.get('date').strftime('%d %b %Y')}*
                """)

def render_set_goals() -> None:
    """Render the form for setting financial goals with improved UI."""
    st.header("Set Financial Goals")
    
    # Add helpful guidance
    st.markdown("""
    Setting clear financial goals helps you stay motivated and track your progress. 
    Whether you're saving for education, a new gadget, or an emergency fund, 
    defining your goals is the first step to achieving them.
    """)
    
    # Create three sections: Income settings, savings goal, and specific goals
    st.subheader("Income & Basic Savings Goal")
    
    # Display current income and savings goal in a card-like layout
    with st.container(border=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Income with better labeling and help text
            new_income = st.number_input(
                "Monthly Income (₹)", 
                min_value=0.0, 
                value=float(st.session_state.income),
                step=1000.0,
                help="Your regular monthly income from all sources"
            )
            if new_income != st.session_state.income:
                st.session_state.income = new_income
                update_savings()
                st.success("Income updated! Your savings have been recalculated.")
        
        with col2:
            # Savings goal with better labeling and help text
            new_savings_goal = st.number_input(
                "Monthly Savings Target (₹)", 
                min_value=0.0, 
                value=float(st.session_state.savings_goal),
                step=500.0,
                help="How much you aim to save each month"
            )
            if new_savings_goal != st.session_state.savings_goal:
                st.session_state.savings_goal = new_savings_goal
                st.success("Savings goal updated!")
        
        # Monthly budget calculator
        st.divider()
        st.markdown("#### Monthly Budget Calculator")
        
        income = st.session_state.income
        savings_target = st.session_state.savings_goal
        available_for_expenses = max(0, income - savings_target)
        
        st.caption(f"Income: {format_inr(income)}")
        st.caption(f"Savings Target: {format_inr(savings_target)}")
        st.caption(f"**Available for Expenses: {format_inr(available_for_expenses)}**")
        
        # Show a warning if the savings goal is too ambitious
        if savings_target > income * 0.5:
            st.warning("⚠️ Your savings target is more than 50% of your income. Make sure this is realistic for your situation.")
    
    # Add space
    st.markdown("---")
    
    # Specific financial goals
    st.subheader("Specific Financial Goals")
    
    # Use tabs to separate adding new goals and managing existing goals
    goal_tabs = st.tabs(["Add New Goal", "Manage Existing Goals"])
    
    # Tab for adding new goal
    with goal_tabs[0]:
        with st.form("goal_form"):
            st.markdown("#### Create a New Goal")
            
            # Create columns for better layout
            form_col1, form_col2 = st.columns(2)
            
            with form_col1:
                goal_name = st.text_input(
                    "Goal Name",
                    help="Give your goal a meaningful name (e.g., 'Emergency Fund')"
                )
                target_amount = st.number_input(
                    "Target Amount (₹)", 
                    min_value=100.0, 
                    step=1000.0,
                    help="Total amount you need to save for this goal"
                )
            
            with form_col2:
                target_date = st.date_input(
                    "Target Date", 
                    min_value=datetime.date.today(),
                    help="When do you aim to achieve this goal?"
                )
                
                # Calculate months until target
                months_until_target = (target_date - datetime.date.today()).days / 30
                monthly_contribution = target_amount / max(1, months_until_target)
                
                if months_until_target > 0:
                    st.caption(f"Suggested monthly contribution: {format_inr(monthly_contribution)}")
            
            # Why is this goal important?
            goal_reason = st.text_area(
                "Why is this goal important to you?",
                help="Remembering your motivation can help you stay committed"
            )
            
            submitted = st.form_submit_button("Add Goal", use_container_width=True)
            
            if submitted:
                if not goal_name:
                    st.error("Please enter a goal name.")
                elif target_amount <= 0:
                    st.error("Please enter a valid target amount.")
                else:
                    # Create a more detailed goal with the reason
                    goal_data = {
                        "name": goal_name,
                        "target_amount": target_amount,
                        "target_date": target_date,
                        "reason": goal_reason,
                        "current_amount": 0,
                        "created_date": datetime.date.today()
                    }
                    
                    if "goals" not in st.session_state:
                        st.session_state.goals = []
                        
                    st.session_state.goals.append(goal_data)
                    st.success("Goal added successfully!")
                    st.rerun()
    
    # Tab for managing existing goals
    with goal_tabs[1]:
        if not st.session_state.goals:
            st.info("You haven't created any goals yet. Use the 'Add New Goal' tab to get started.")
        else:
            st.markdown("#### Your Financial Goals")
            
            # Iterate through goals with better visualization
            for i, goal in enumerate(st.session_state.goals):
                with st.container(border=True):
                    # Calculate progress percentage
                    progress_pct = (goal["current_amount"] / goal["target_amount"]) * 100 if goal["target_amount"] > 0 else 0
                    
                    # Create a layout with columns
                    g_col1, g_col2 = st.columns([3, 1])
                    
                    with g_col1:
                        st.markdown(f"##### {goal['name']}")
                        
                        # Progress bar
                        st.progress(min(1.0, progress_pct / 100))
                        st.caption(f"{progress_pct:.1f}% - {format_inr(goal['current_amount'])} / {format_inr(goal['target_amount'])}")
                        
                        # Time remaining calculation
                        days_remaining = (goal["target_date"] - datetime.date.today()).days
                        if days_remaining > 0:
                            months_remaining = days_remaining // 30
                            if goal["current_amount"] < goal["target_amount"]:
                                amount_needed = goal["target_amount"] - goal["current_amount"]
                                monthly_needed = amount_needed / max(1, months_remaining)
                                st.caption(f"⏱️ **{months_remaining} months remaining** - Save {format_inr(monthly_needed)}/month to reach goal")
                            else:
                                st.success("🎉 Goal achieved! Congratulations!")
                        else:
                            if goal["current_amount"] < goal["target_amount"]:
                                st.warning("⚠️ Goal deadline has passed")
                            else:
                                st.success("🎉 Goal achieved! Congratulations!")
                                
                        # Show the goal reason if it exists
                        if goal.get("reason"):
                            with st.expander("Why this goal matters"):
                                st.write(goal["reason"])
                    
                    with g_col2:
                        # Contribution input
                        contrib_amount = st.number_input(
                            f"Add to Goal",
                            min_value=0.0,
                            key=f"contrib_{i}",
                            step=100.0
                        )
                        
                        # Add button
                        if st.button(f"Add", key=f"add_to_goal_{i}", use_container_width=True):
                            if contrib_amount <= 0:
                                st.error("Please enter a valid amount.")
                            else:
                                update_goal(i, contrib_amount)
                                st.success(f"Added {format_inr(contrib_amount)}!")
                                st.rerun()

def render_history() -> None:
    """Render the transaction history with improved filtering and visualization."""
    st.header("Transaction History")
    
    # Add explanation
    st.markdown("""
    View and analyze your past transactions to identify spending patterns and track your financial progress.
    Use the filters below to focus on specific date ranges or transaction types.
    """)
    
    # Check if there are any transactions to display
    has_expenses = len(st.session_state.expenses) > 0
    has_earnings = len(st.session_state.get("earnings", [])) > 0
    
    if not has_expenses and not has_earnings:
        st.info("No transactions recorded yet. Add some expenses or earnings to see them here.")
        return
    
    # Combine expenses and earnings into a single dataframe for analysis
    transactions = []
    
    # Add expenses
    for expense in st.session_state.expenses:
        expense_copy = expense.copy()
        expense_copy["transaction_type"] = "Expense"
        transactions.append(expense_copy)
    
    # Add earnings
    for earning in st.session_state.get("earnings", []):
        earning_copy = earning.copy()
        earning_copy["transaction_type"] = "Earning"
        earning_copy["avoidable"] = False  # Earnings are never avoidable
        transactions.append(earning_copy)
    
    # Convert to DataFrame
    if transactions:
        df = pd.DataFrame(transactions)
        
        # Add Filters in a clean UI
        st.subheader("Filter Transactions")
        
        with st.container(border=True):
            # Create 3 columns for filters
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            with filter_col1:
                # Date range filter
                min_date = df["date"].min() if "date" in df.columns else datetime.date.today() - datetime.timedelta(days=30)
                max_date = df["date"].max() if "date" in df.columns else datetime.date.today()
                
                # Date range with a cleaner interface
                date_range = st.date_input(
                    "Date Range",
                    value=(min_date, max_date),
                    min_value=min_date - datetime.timedelta(days=365),
                    max_value=datetime.date.today(),
                    help="Filter transactions by date range"
                )
                
                # Handle single date selection
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    start_date, end_date = date_range
                else:
                    start_date = date_range if date_range else min_date
                    end_date = date_range if date_range else max_date
            
            with filter_col2:
                # Transaction type filter
                transaction_types = ["All", "Expense", "Earning"]
                selected_type = st.selectbox(
                    "Transaction Type",
                    transaction_types,
                    help="Filter by transaction type"
                )
            
            with filter_col3:
                # Category filter
                all_categories = list(df["category"].unique()) if "category" in df.columns else []
                categories = ["All"] + all_categories
                selected_category = st.selectbox(
                    "Category",
                    categories,
                    help="Filter by expense or earning category"
                )
        
        # Apply filters
        filtered_df = df.copy()
        
        # Date filter
        if "date" in filtered_df.columns:
            filtered_df = filtered_df[(filtered_df["date"] >= start_date) & 
                                    (filtered_df["date"] <= end_date)]
        
        # Transaction type filter
        if selected_type != "All" and "transaction_type" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["transaction_type"] == selected_type]
        
        # Category filter
        if selected_category != "All" and "category" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["category"] == selected_category]
        
        # Sort by date (most recent first)
        if "date" in filtered_df.columns:
            filtered_df = filtered_df.sort_values("date", ascending=False)
        
        # Use Tabs to separate different views of the transactions
        history_tabs = st.tabs(["Transactions List", "Summary Charts", "Time Analysis"])
        
        # Transactions list view
        with history_tabs[0]:
            if filtered_df.empty:
                st.info("No transactions match the selected filters.")
            else:
                # Show transaction count and totals
                st.subheader(f"Showing {len(filtered_df)} Transactions")
                
                # Calculate summary stats
                expenses_total = filtered_df[filtered_df["transaction_type"] == "Expense"]["amount"].sum() if "transaction_type" in filtered_df.columns else 0
                earnings_total = filtered_df[filtered_df["transaction_type"] == "Earning"]["amount"].sum() if "transaction_type" in filtered_df.columns else 0
                net_amount = earnings_total - expenses_total
                
                # Show summary in columns
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                summary_col1.metric("Total Expenses", format_inr(expenses_total))
                summary_col2.metric("Total Earnings", format_inr(earnings_total))
                summary_col3.metric("Net Amount", format_inr(net_amount), delta=format_inr(net_amount))
                
                # Prepare data for display
                display_df = filtered_df.copy()
                
                # Format amount for display
                if "amount" in display_df.columns:
                    display_df["amount"] = display_df["amount"].apply(lambda x: format_inr(x))
                
                # Format date for display
                if "date" in display_df.columns:
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
                
                # Select columns to display
                columns_to_show = ["Date", "Description", "Category", "Amount", "Type"]
                if "avoidable" in display_df.columns:
                    display_df["Avoidable"] = display_df["avoidable"].apply(lambda x: "Yes" if x else "No")
                    columns_to_show.append("Avoidable")
                
                # Show the filtered data
                st.dataframe(
                    display_df[columns_to_show], 
                    use_container_width=True,
                    hide_index=True
                )
                
                # Export option
                if st.button("Export Data to CSV"):
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"transactions_{datetime.date.today().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
        
        # Summary Charts
        with history_tabs[1]:
            if filtered_df.empty:
                st.info("No transactions match the selected filters.")
            else:
                # Create two columns for charts
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    # Category Breakdown Chart
                    st.subheader("Category Breakdown")
                    
                    if "category" in filtered_df.columns and "amount" in filtered_df.columns:
                        # Group by category and transaction type
                        category_summary = filtered_df.groupby(["category", "transaction_type"])["amount"].sum().reset_index()
                        
                        if not category_summary.empty:
                            chart = alt.Chart(category_summary).mark_bar().encode(
                                x=alt.X("category:N", title="Category"),
                                y=alt.Y("amount:Q", title="Amount (₹)"),
                                color=alt.Color("transaction_type:N", title="Type"),
                                tooltip=["category", "transaction_type", "amount"]
                            ).properties(height=300)
                            
                            st.altair_chart(chart, use_container_width=True)
                
                with chart_col2:
                    # Transaction Type Pie Chart
                    st.subheader("Expenses vs. Earnings")
                    
                    if "transaction_type" in filtered_df.columns and "amount" in filtered_df.columns:
                        # Group by transaction type
                        type_summary = filtered_df.groupby("transaction_type")["amount"].sum().reset_index()
                        
                        if not type_summary.empty and len(type_summary) > 1:
                            pie = alt.Chart(type_summary).mark_arc(innerRadius=50).encode(
                                theta=alt.Theta(field="amount", type="quantitative"),
                                color=alt.Color(field="transaction_type", type="nominal", 
                                                scale=alt.Scale(domain=["Expense", "Earning"], 
                                                                range=["#FF6B6B", "#4CAF50"])),
                                tooltip=["transaction_type", "amount"]
                            ).properties(height=300)
                            
                            st.altair_chart(pie, use_container_width=True)
                        else:
                            st.info("Need both expense and earning data for comparison chart.")
                
                # Additional Analysis: Avoidable Expenses
                if "avoidable" in filtered_df.columns:
                    avoidable_expenses = filtered_df[(filtered_df["transaction_type"] == "Expense") & 
                                                    (filtered_df["avoidable"] == True)]
                    
                    if not avoidable_expenses.empty:
                        st.subheader("Avoidable Expenses Analysis")
                        
                        # Calculate savings potential
                        avoidable_total = avoidable_expenses["amount"].sum()
                        expense_total = filtered_df[filtered_df["transaction_type"] == "Expense"]["amount"].sum()
                        avoidable_percentage = (avoidable_total / expense_total) * 100 if expense_total > 0 else 0
                        
                        st.markdown(f"""
                        **Savings Opportunity:** {format_inr(avoidable_total)} ({avoidable_percentage:.1f}% of expenses)
                        
                        These are expenses that you identified as potentially avoidable. Reducing these could 
                        help you reach your savings goals faster.
                        """)
                        
                        # Show top avoidable expense categories
                        if "category" in avoidable_expenses.columns:
                            top_avoidable = avoidable_expenses.groupby("category")["amount"].sum().sort_values(ascending=False).reset_index()
                            
                            if not top_avoidable.empty:
                                chart = alt.Chart(top_avoidable).mark_bar().encode(
                                    x=alt.X("amount:Q", title="Amount (₹)"),
                                    y=alt.Y("category:N", title="Category", sort="-x"),
                                    tooltip=["category", "amount"]
                                ).properties(height=200)
                                
                                st.altair_chart(chart, use_container_width=True)
        
        # Time Analysis
        with history_tabs[2]:
            if filtered_df.empty or "date" not in filtered_df.columns:
                st.info("No transactions with date information match the selected filters.")
            else:
                st.subheader("Transaction Trends Over Time")
                
                # Create a copy with month information
                time_df = filtered_df.copy()
                time_df["month"] = time_df["date"].apply(lambda x: x.strftime("%Y-%m"))
                
                # Group by month and transaction type
                monthly_summary = time_df.groupby(["month", "transaction_type"])["amount"].sum().reset_index()
                
                if not monthly_summary.empty:
                    # Line chart for trends
                    line_chart = alt.Chart(monthly_summary).mark_line(point=True).encode(
                        x=alt.X("month:T", title="Month"),
                        y=alt.Y("amount:Q", title="Amount (₹)"),
                        color=alt.Color("transaction_type:N", title="Type",
                                        scale=alt.Scale(domain=["Expense", "Earning"], 
                                                       range=["#FF6B6B", "#4CAF50"])),
                        tooltip=["month", "transaction_type", "amount"]
                    ).properties(height=300)
                    
                    st.altair_chart(line_chart, use_container_width=True)
                    
                    # Monthly summary table
                    st.subheader("Monthly Summary")
                    
                    # Pivot the data to create a table with months as rows
                    pivot_df = monthly_summary.pivot(index="month", columns="transaction_type", values="amount").reset_index()
                    
                    # Fill missing values
                    if "Expense" not in pivot_df.columns:
                        pivot_df["Expense"] = 0
                    if "Earning" not in pivot_df.columns:
                        pivot_df["Earning"] = 0
                    
                    # Calculate Net
                    pivot_df["Net"] = pivot_df["Earning"] - pivot_df["Expense"]
                    
                    # Format for display
                    display_pivot = pivot_df.copy()
                    display_pivot["Expense"] = display_pivot["Expense"].apply(lambda x: format_inr(x))
                    display_pivot["Earning"] = display_pivot["Earning"].apply(lambda x: format_inr(x))
                    display_pivot["Net"] = display_pivot["Net"].apply(lambda x: format_inr(x))
                    
                    # Sort by month descending (most recent first)
                    display_pivot = display_pivot.sort_values("month", ascending=False)
                    
                    # Display the table
                    st.dataframe(display_pivot, use_container_width=True, hide_index=True)
                else:
                    st.info("Not enough time-series data available for trend analysis.") 