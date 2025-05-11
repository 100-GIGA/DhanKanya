"""
Home page component for the DhanKanya application.

This module contains the UI components for the main home page,
which includes the chat interface for interacting with the AI assistant.
"""

import streamlit as st
import anthropic
from typing import List, Dict, Any, Optional, Tuple
import time
import re

from app.services.ai_service import get_response

# Constants
HINDI_FONT = "Noto Sans Devanagari"
ENGLISH_FONT = "Inter"
MAX_MESSAGE_LENGTH = 1024

# Supported Indian languages and their font mappings
INDIAN_LANGUAGES = {
    'hi': 'Noto Sans Devanagari',  # Hindi
    'mr': 'Noto Sans Devanagari',  # Marathi
    'ta': 'Noto Sans Tamil',       # Tamil
    'te': 'Noto Sans Telugu',      # Telugu
    'bn': 'Noto Sans Bengali',     # Bengali
    'gu': 'Noto Sans Gujarati',    # Gujarati
    'pa': 'Noto Sans Gurmukhi',    # Punjabi
    'ml': 'Noto Sans Malayalam',   # Malayalam
    'kn': 'Noto Sans Kannada',     # Kannada
    'en': 'Inter'                  # English
}

def detect_indian_language(text: str) -> Tuple[str, str]:
    """
    Detect if the text is in a supported Indian language or English.
    Uses a simple pattern-based approach to identify script.
    
    Args:
        text: The text to check
        
    Returns:
        Tuple[str, str]: A tuple containing (language_code, font_family)
        Defaults to English if no supported language is detected
    """
    # Devanagari Unicode range (Hindi, Marathi)
    if re.search(r'[\u0900-\u097F]', text):
        return 'hi', INDIAN_LANGUAGES['hi']
    
    # Bengali Unicode range
    if re.search(r'[\u0980-\u09FF]', text):
        return 'bn', INDIAN_LANGUAGES['bn']
    
    # Gurmukhi Unicode range (Punjabi)
    if re.search(r'[\u0A00-\u0A7F]', text):
        return 'pa', INDIAN_LANGUAGES['pa']
    
    # Gujarati Unicode range
    if re.search(r'[\u0A80-\u0AFF]', text):
        return 'gu', INDIAN_LANGUAGES['gu']
    
    # Tamil Unicode range
    if re.search(r'[\u0B80-\u0BFF]', text):
        return 'ta', INDIAN_LANGUAGES['ta']
    
    # Telugu Unicode range
    if re.search(r'[\u0C00-\u0C7F]', text):
        return 'te', INDIAN_LANGUAGES['te']
    
    # Kannada Unicode range
    if re.search(r'[\u0C80-\u0CFF]', text):
        return 'kn', INDIAN_LANGUAGES['kn']
    
    # Malayalam Unicode range
    if re.search(r'[\u0D00-\u0D7F]', text):
        return 'ml', INDIAN_LANGUAGES['ml']
    
    # Default to English for any other script
    return 'en', INDIAN_LANGUAGES['en']

def initialize_session_state() -> None:
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    if "claude_model" not in st.session_state:
        st.session_state.claude_model = "claude-3-haiku-20240307"

def render_message(message: Dict[str, Any]) -> None:
    """
    Render a single chat message with appropriate styling.
    
    Args:
        message: Dictionary containing message data
    """
    with st.chat_message(message["role"]):
        font_family = message.get("font_family", ENGLISH_FONT)
        st.markdown(
            f'<div style="font-family: {font_family}; font-size: 1.1em;">{message["content"]}</div>',
            unsafe_allow_html=True
        )

def render_header() -> None:
    """Render the header section with logo and welcome message."""
    # Title with custom styling
    st.markdown("# DhanKanya: Financial Empowerment for Girls in India")
    
    # Center the logo using columns
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("./assets/images/logo.png", width=200)
    
    # Welcome message
    st.markdown("### Welcome to our AI-powered financial literacy application!")
    st.markdown("Our mission is to empower girls in India with the knowledge and tools they need to achieve financial independence and success.")

def render_features() -> None:
    """Render the features section using Streamlit's native components."""
    st.markdown("### Key Features")
    
    # Create three columns for features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### ✨ Interactive Budgeting")
        st.markdown("Track your income and expenses with our user-friendly tools")
    
    with col2:
        st.markdown("#### 📚 Educational Resources")
        st.markdown("Learn essential financial literacy concepts like saving and investing")
    
    with col3:
        st.markdown("#### 🎯 Goal Setting")
        st.markdown("Plan and save for specific educational milestones")

def handle_text_input(prompt: str, client: anthropic.Anthropic) -> None:
    """Handle text input and process the response."""
    if prompt:
        st.session_state.is_processing = True
        lang_code, font_family = detect_indian_language(prompt)
        
        # Add user message with the detected language
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "font_family": font_family,
            "lang_code": lang_code
        })
        
        # Display a loading spinner while getting the response
        with st.spinner("✨ Thinking of a helpful response for you..."):
            # Get AI response with the same language context
            response = get_response(prompt, client, lang_code)
        
        # Add assistant message with the same font family
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "font_family": font_family,
            "lang_code": lang_code
        })
        
        st.session_state.is_processing = False
        st.rerun()

def render_chat_interface(client: anthropic.Anthropic) -> None:
    """Render the chat interface with input controls."""
    st.markdown("### Chat with DhanKanya assistant")
    st.markdown("You can ask questions in most of the Indian languages and English.")
    
    # Chat messages container
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            render_message(message)
    
    # Text input form with inline submit button
    with st.form(key="chat_form", clear_on_submit=True):
        # Create columns for input field and submit button
        input_col, button_col = st.columns([5, 1])
        
        with input_col:
            prompt = st.text_input(
                "Ask a question in any Indian language or English",
                key="chat_input",
                label_visibility="collapsed",
                disabled=st.session_state.is_processing
            )
        
        with button_col:
            submit_button = st.form_submit_button(
                "Send",
                use_container_width=True,
                disabled=st.session_state.is_processing
            )
        
        if submit_button and prompt:
            handle_text_input(prompt, client)

def render(client: anthropic.Anthropic) -> None:
    """
    Render the home page with chat interface.
    
    Args:
        client: The initialized Anthropic client for AI interaction.
    """
    # Initialize session state
    initialize_session_state()
    
    # Render page sections
    render_header()
    st.markdown("---")
    render_features()
    st.markdown("---")
    render_chat_interface(client) 