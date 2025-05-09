"""
DhanKanya - Financial Empowerment for Girls in India.

Main application entry point for the DhanKanya financial literacy application.
This file sets up the Streamlit interface and coordinates the various components.
"""

import streamlit as st
import logging
import sys
import traceback

from app.services.ai_service import create_anthropic_client
from app.utils.helpers import log_system_info, check_env_file
from app.components import home_page, templates_page, expense_tracker_page
from config.settings import APP_TITLE, APP_ICON, APP_LAYOUT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main application entry point.
    
    Sets up the Streamlit interface, initializes services,
    and renders the appropriate page based on user selection.
    """
    # Configure page
    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout=APP_LAYOUT)

    # Log system information for diagnostics
    log_system_info()
    
    # Check .env file
    check_env_file()

    # Navigation menu items
    menu = ["Start with Voice", "Build your Wealth", "Savings and Budgeting"]
    
    # Initialize session state for navigation if not exists
    if "nav_selection" not in st.session_state:
        st.session_state.nav_selection = menu[0]
    
    # Create sidebar navigation with direct buttons
    st.sidebar.title("Navigation")
    
    # Create a button for each menu item
    for item in menu:
        if st.sidebar.button(item, key=f"nav_{item}", use_container_width=True):
            st.session_state.nav_selection = item
            st.rerun()

    # Create the Anthropic client with error handling
    try:
        logger.info("Attempting to create Anthropic client...")
        client = create_anthropic_client()
        st.sidebar.success("AI assistant initialized successfully!")
        
    except Exception as e:
        logger.error("=== Anthropic Client Error ===")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Error args: {e.args}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        st.sidebar.error(f"Failed to initialize the AI assistant. Error: {str(e)}")
        return

    # Display the selected page based on session state
    choice = st.session_state.nav_selection
    if choice == "Start with Voice":
        home_page.render(client)
    elif choice == "Build your Wealth":
        templates_page.render(client)
    elif choice == "Savings and Budgeting":
        expense_tracker_page.render()

if __name__ == "__main__":
    main() 