"""
DhanKanya - Financial Empowerment for Girls in India.

Main application entry point for the DhanKanya financial literacy application.
This file sets up the Streamlit interface and coordinates the various components.
"""

import streamlit as st
import logging
import sys
import traceback

from app.services.ai_service import create_llm_client
from app.utils.helpers import log_system_info
from app.components import home_page, templates_page, expense_tracker_page, auth_pages
from config.settings import APP_TITLE, APP_ICON, APP_LAYOUT, LLM_OPTIONS, DEFAULT_LLM_PROVIDER

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

    # Initialize authentication state if not present
    if "is_authenticated" not in st.session_state:
        st.session_state.is_authenticated = False
    if "user" not in st.session_state:
        st.session_state.user = None

    # Navigation menu items with icons
    menu_items = [
        {"name": "Financial Assistant", "icon": "💬"},
        {"name": "Build your Wealth", "icon": "💎"},
        {"name": "Savings and Budgeting", "icon": "💰"},
        {"name": "Profile", "icon": "👤"}
    ]
    
    # Initialize session state for navigation if not exists
    if "nav_selection" not in st.session_state:
        st.session_state.nav_selection = menu_items[0]["name"]
    
    # Initialize LLM selection if not exists
    if "selected_llm" not in st.session_state:
        st.session_state.selected_llm = DEFAULT_LLM_PROVIDER
    
    # Create sidebar navigation with modern styling
    with st.sidebar:
        # Logo and title
        st.image("./assets/images/logo.png", width=100)
        st.title("DhanKanya")
        
        # Display user status
        current_user = auth_pages.get_current_user()
        if current_user:
            # Show first name if available, otherwise show username
            first_name = current_user.get("first_name", "") or ""  # Ensure it's a string
            display_name = first_name.strip() or current_user["username"]
            st.write(f"Welcome, {display_name}!")
        else:
            st.write("👋 Welcome to DhanKanya!")
            st.caption("Sign in from the Profile section to access personalized features.")
        
        st.markdown("---")
        
        # LLM Model Selection
        st.subheader("🤖 AI Model")
        llm_options = list(LLM_OPTIONS.keys())
        llm_display_names = [LLM_OPTIONS[key]["display_name"] for key in llm_options]
        
        selected_index = st.selectbox(
            "Choose AI Model:",
            range(len(llm_options)),
            format_func=lambda x: llm_display_names[x],
            index=llm_options.index(st.session_state.selected_llm) if st.session_state.selected_llm in llm_options else 0,
            key="llm_selector"
        )
        
        # Update session state if selection changed
        new_selection = llm_options[selected_index]
        if new_selection != st.session_state.selected_llm:
            st.session_state.selected_llm = new_selection
            st.rerun()
        
        st.markdown("---")
        
        # Add navigation buttons with icons
        for item in menu_items:
            # Customize button text based on authentication status
            if item["name"] == "Profile":
                if st.session_state.get("is_authenticated", False):
                    button_text = f"{item['icon']} {item['name']}"
                else:
                    button_text = f"{item['icon']} Login / SignUp"
            else:
                button_text = f"{item['icon']} {item['name']}"
            
            # Create a button with icon and name
            if st.button(
                button_text,
                key=f"nav_{item['name']}",
                use_container_width=True,
                type="primary" if st.session_state.nav_selection == item["name"] else "secondary"
            ):
                st.session_state.nav_selection = item["name"]
                st.rerun()

    # Create the LLM client with error handling
    try:
        selected_provider = st.session_state.selected_llm
        logger.info(f"Attempting to create {selected_provider} client...")
        client = create_llm_client(selected_provider)
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
    selected_provider = st.session_state.selected_llm
    
    if choice == "Financial Assistant":
        home_page.render(selected_provider, client)
    elif choice == "Build your Wealth":
        templates_page.render(selected_provider, client)
    elif choice == "Savings and Budgeting":
        expense_tracker_page.render()
    elif choice == "Profile":
        # Handle authentication for profile page
        if not st.session_state.get("is_authenticated", False):
            # Show authentication forms in the profile section
            st.title("Profile")
            st.markdown("Please sign in or create an account to access your profile.")
            st.markdown("---")
            
            is_authenticated = auth_pages.render_auth_pages()
            
            # If just authenticated, refresh to show profile
            if is_authenticated:
                st.rerun()
        else:
            # User is authenticated, show profile page
            auth_pages.render_profile_page()

if __name__ == "__main__":
    main() 