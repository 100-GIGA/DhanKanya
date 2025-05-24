"""
Authentication UI components for DhanKanya.

This module provides UI components for user authentication, including
sign-in and sign-up pages.
"""

import streamlit as st
import re
from app.services.auth_service import (
    register_user, authenticate_user, get_user_profile, 
    update_user_password, update_user_profile, delete_user_account
)

def render_signin_page():
    """
    Render the sign-in page.
    
    Returns:
        bool: True if authentication was successful, False otherwise.
    """
    st.title("Sign In")
    
    with st.form("signin_form"):
        username_or_email = st.text_input("Username or Email")
        password = st.text_input("Password", type="password")
        signin_button = st.form_submit_button("Sign In")
        
        if signin_button:
            if not username_or_email or not password:
                st.error("Please enter both username/email and password.")
                return False
            
            user = authenticate_user(username_or_email, password)
            
            if user:
                st.success(f"Welcome back, {user['username']}!")
                # Set session state
                st.session_state.user = user
                st.session_state.is_authenticated = True
                return True
            else:
                st.error("Invalid username/email or password.")
                return False
    
    # Show sign-up link
    st.markdown("---")
    st.markdown("Don't have an account? [Sign Up](#signup)")
    
    return False

def render_signup_page():
    """
    Render the sign-up page.
    
    Returns:
        bool: True if registration was successful, False otherwise.
    """
    st.title("Create an Account")
    
    with st.form("signup_form"):
        username = st.text_input("Username (min 3 characters)")
        email = st.text_input("Email")
        first_name = st.text_input("First Name (optional)")
        last_name = st.text_input("Last Name (optional)")
        password = st.text_input("Password (min 8 characters)", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        signup_button = st.form_submit_button("Sign Up")
        
        if signup_button:
            # Validate inputs
            valid = True
            
            if len(username) < 3:
                st.error("Username must be at least 3 characters.")
                valid = False
            
            # Email validation using regex
            email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            if not re.match(email_pattern, email):
                st.error("Please enter a valid email address.")
                valid = False
            
            if len(password) < 8:
                st.error("Password must be at least 8 characters.")
                valid = False
            
            if password != confirm_password:
                st.error("Passwords do not match.")
                valid = False
            
            if valid:
                success = register_user(username, email, password, first_name, last_name)
                
                if success:
                    st.success("Registration successful! You can now sign in.")
                    # Automatically authenticate the user
                    user = authenticate_user(username, password)
                    if user:
                        st.session_state.user = user
                        st.session_state.is_authenticated = True
                        return True
                else:
                    st.error("Username or email already exists.")
                    return False
    
    # Show sign-in link
    st.markdown("---")
    st.markdown("Already have an account? [Sign In](#signin)")
    
    return False

def render_auth_pages():
    """
    Render the authentication pages.
    
    This function handles switching between sign-in and sign-up views
    and manages the authentication flow.
    
    Returns:
        bool: True if the user is authenticated, False otherwise.
    """
    # Create tabs for sign-in and sign-up
    tab1, tab2 = st.tabs(["Sign In", "Sign Up"])
    
    with tab1:
        # Show sign-in form
        if render_signin_page():
            return True
    
    with tab2:
        # Show sign-up form
        if render_signup_page():
            return True
    
    return st.session_state.get("is_authenticated", False)

def render_profile_page():
    """Render the user profile page."""
    # This function should only be called when user is authenticated
    # The authentication check is now handled in main.py
    user = st.session_state.user
    
    st.title("Your Profile")
    
    # Profile details section
    profile_tab, security_tab, danger_tab = st.tabs(["Profile Details", "Security", "Danger Zone"])
    
    with profile_tab:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image("https://www.gravatar.com/avatar/00000000000000000000000000000000?d=mp&f=y", width=150)
        
        with col2:
            st.header(user["username"])
            st.write(f"Email: {user['email']}")
            
            # Display first and last name if available
            first_name = user.get("first_name", "")
            last_name = user.get("last_name", "")
            if first_name or last_name:
                st.write(f"Name: {first_name} {last_name}")
            
            st.write(f"Member since: {user['created_at'][:10]}")
            
            if user.get("last_login"):
                st.write(f"Last login: {user['last_login'][:10]}")
        
        st.markdown("---")
        
        # Profile editing section
        st.subheader("Edit Profile")
        
        with st.form("edit_profile_form"):
            new_first_name = st.text_input("First Name", value=user.get("first_name", ""))
            new_last_name = st.text_input("Last Name", value=user.get("last_name", ""))
            new_email = st.text_input("Email", value=user["email"])
            
            submit_profile_button = st.form_submit_button("Update Profile")
            
            if submit_profile_button:
                # Only update if something changed
                if (new_first_name != user.get("first_name", "") or 
                    new_last_name != user.get("last_name", "") or 
                    new_email != user["email"]):
                    
                    success = update_user_profile(
                        user["username"], 
                        first_name=new_first_name, 
                        last_name=new_last_name,
                        email=new_email if new_email != user["email"] else None
                    )
                    
                    if success:
                        # Update the session state with new values
                        updated_user = get_user_profile(user["username"])
                        if updated_user:
                            st.session_state.user = updated_user
                            st.success("Profile updated successfully!")
                            st.rerun()
                        else:
                            st.error("Unable to refresh profile data.")
                    else:
                        st.error("Failed to update profile. Email may already be in use.")
                else:
                    st.info("No changes detected in profile information.")
    
    with security_tab:
        st.subheader("Change Password")
        
        with st.form("change_password_form"):
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password (min 8 characters)", type="password")
            confirm_new_password = st.text_input("Confirm New Password", type="password")
            
            submit_button = st.form_submit_button("Update Password")
            
            if submit_button:
                if not current_password or not new_password or not confirm_new_password:
                    st.error("Please fill in all fields.")
                elif len(new_password) < 8:
                    st.error("New password must be at least 8 characters.")
                elif new_password != confirm_new_password:
                    st.error("New passwords do not match.")
                else:
                    success = update_user_password(user["username"], current_password, new_password)
                    
                    if success:
                        st.success("Password updated successfully!")
                    else:
                        st.error("Failed to update password. Please check your current password.")
    
    with danger_tab:
        st.subheader("Delete Account")
        st.warning("⚠️ This action is irreversible. All your data will be permanently deleted.")
        
        with st.form("delete_account_form"):
            st.write("To confirm account deletion, please enter your password:")
            confirm_password = st.text_input("Password", type="password")
            
            # Add a confirmation checkbox
            confirm_delete = st.checkbox("I understand that this action cannot be undone")
            
            delete_button = st.form_submit_button("Delete My Account", type="primary")
            
            if delete_button:
                if not confirm_password:
                    st.error("Please enter your password to confirm.")
                elif not confirm_delete:
                    st.error("Please confirm that you understand the consequences.")
                else:
                    success = delete_user_account(user["username"], confirm_password)
                    
                    if success:
                        # Clear session state
                        st.session_state.is_authenticated = False
                        st.session_state.user = None
                        
                        st.success("Your account has been successfully deleted. Redirecting...")
                        st.rerun()
                    else:
                        st.error("Failed to delete account. Please check your password.")
    
    st.markdown("---")
    
    # Sign out button
    if st.button("Sign Out"):
        # Clear all authentication data
        st.session_state.is_authenticated = False
        st.session_state.user = None
        st.rerun()

def is_authenticated():
    """Check if the user is authenticated."""
    return st.session_state.get("is_authenticated", False)

def get_current_user():
    """Get the current logged-in user."""
    return st.session_state.get("user", None) 