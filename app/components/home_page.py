"""
Home page component for the DhanKanya application.

This module contains the UI components for the main home page,
which includes the chat interface for interacting with the AI assistant.
"""

import streamlit as st
import anthropic
from typing import List, Dict, Any, Optional
from langdetect import detect
import time

from app.utils.voice_recognition import get_voice_input
from app.services.ai_service import get_response

# Constants
HINDI_FONT = "Noto Sans Devanagari"
ENGLISH_FONT = "Inter"
MAX_MESSAGE_LENGTH = 1024

def is_hindi(text: str) -> bool:
    """
    Check if the text is in Hindi.
    
    Args:
        text: The text to check
        
    Returns:
        bool: True if the text is in Hindi, False otherwise
    """
    try:
        return detect(text) == 'hi'
    except:
        return False

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
        if message.get("is_hindi", False):
            st.markdown(
                f'<div style="font-family: {HINDI_FONT}; font-size: 1.1em;">{message["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div style="font-family: {ENGLISH_FONT}; font-size: 1.1em;">{message["content"]}</div>',
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

def handle_voice_input(client: anthropic.Anthropic) -> None:
    """Handle voice input and process the response."""
    try:
        prompt = get_voice_input()
        if prompt:
            st.session_state.is_processing = True
            is_hindi_text = is_hindi(prompt)
            
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "is_hindi": is_hindi_text
            })
            
            # Get AI response
            response = get_response(prompt, client)
            
            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "is_hindi": is_hindi_text
            })
            
            st.session_state.is_processing = False
            st.rerun()
    except Exception as e:
        st.error(f"Error processing voice input: {str(e)}")
        st.session_state.is_processing = False

def handle_text_input(prompt: str, client: anthropic.Anthropic) -> None:
    """Handle text input and process the response."""
    if prompt:
        st.session_state.is_processing = True
        is_hindi_text = is_hindi(prompt)
        
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "is_hindi": is_hindi_text
        })
        
        # Get AI response
        response = get_response(prompt, client)
        
        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "is_hindi": is_hindi_text
        })
        
        st.session_state.is_processing = False
        st.rerun()

def render_chat_interface(client: anthropic.Anthropic) -> None:
    """Render the chat interface with input controls."""
    st.markdown("### Chat with DhanKanya")
    st.markdown("You can ask questions in Hindi using your voice or type them in English.")
    
    # Chat messages container
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            render_message(message)
    
    # Input area
    with st.form(key="chat_form", clear_on_submit=True):
        # Create a row with the input and button
        input_col1, input_col2 = st.columns([6, 1])
        
        with input_col1:
            prompt = st.text_input(
                "Ask a question in English",
                key="chat_input",
                label_visibility="collapsed",
                disabled=st.session_state.is_processing
            )
        
        with input_col2:
            voice_input = st.form_submit_button(
                "🎙️",
                use_container_width=True,
                disabled=st.session_state.is_processing
            )
        
        if voice_input:
            handle_voice_input(client)
        elif prompt:
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