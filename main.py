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
    st.set_page_config(
        page_title="DhanKanya - Financial Empowerment",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Log system information for diagnostics
    log_system_info()
    
    # Check .env file
    check_env_file()

    # Add custom CSS for layout and spacing only (no hardcoded colors)
    st.markdown(
        """
        <style>
        .stApp {
            max-width: none;
            padding: 1rem 2rem;
        }
        .stMetric {
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        }
        .stForm {
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        }
        .stButton > button {
            width: 100%;
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }
        .stDataFrame {
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        }
        h1, h2, h3 {
            padding-top: 1rem;
            padding-bottom: 0.5rem;
        }
        .element-container {
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Navigation menu items with icons
    menu_items = [
        {"name": "Start with Voice", "icon": "🎙️"},
        {"name": "Build your Wealth", "icon": "💎"},
        {"name": "Savings and Budgeting", "icon": "💰"}
    ]
    
    # Initialize session state for navigation if not exists
    if "nav_selection" not in st.session_state:
        st.session_state.nav_selection = menu_items[0]["name"]
    
    # Create sidebar navigation with modern styling
    with st.sidebar:
        # Logo and title
        st.image("./assets/images/logo.png", width=100)
        st.title("DhanKanya")
        st.markdown("---")
        
        # Add navigation buttons with icons
        for item in menu_items:
            # Create a button with icon and name
            if st.button(
                f"{item['icon']} {item['name']}",
                key=f"nav_{item['name']}",
                use_container_width=True,
                type="primary" if st.session_state.nav_selection == item["name"] else "secondary"
            ):
                st.session_state.nav_selection = item["name"]
                st.rerun()

    # Create the Anthropic client with error handling
    try:
        logger.info("Attempting to create Anthropic client...")
        client = create_anthropic_client()
    except Exception as e:
        logger.error("=== Anthropic Client Error ===")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Error args: {e.args}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        st.error(f"Failed to initialize the AI assistant. Error: {str(e)}")
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