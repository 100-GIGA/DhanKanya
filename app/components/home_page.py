"""
Home page component for the DhanKanya application.

This module contains the UI components for the main home page,
which includes the chat interface for interacting with the AI assistant.
"""

import streamlit as st
import anthropic
from typing import List, Dict, Any

from app.utils.voice_recognition import get_voice_input
from app.services.ai_service import get_response

def render(client: anthropic.Anthropic) -> None:
    """
    Render the home page with chat interface.
    
    Args:
        client: The initialized Anthropic client for AI interaction.
    """
    st.title("DhanKanya: Financial Empowerment for Girls in India")

    # Center the logo
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("./assets/images/logo.png", width=200)

    st.write("""
    ### Welcome to our AI-powered financial literacy application!

    Our mission is to empower girls in India with the knowledge and tools they need to achieve financial independence and success.

    With our user-friendly app, you'll have access to:
    """)

    st.write("- **Interactive Budgeting Tools** to help you track your income and expenses.")
    st.write("- **Educational Resources** on essential financial literacy concepts like saving and investing.")
    st.write("- **Goal Setting Functionality** to plan and save for specific educational milestones.")

    st.markdown("---")

    if "claude_model" not in st.session_state:
        st.session_state["claude_model"] = "claude-3-haiku-20240307"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Add voice input functionality
    st.write("You can ask questions in Hindi using your voice or type them in English.")
    voice_input = st.button("🎙️ Use voice input")

    if voice_input:
        prompt = get_voice_input()
    else:
        prompt = st.chat_input("Ask a question in English")

    # Process user input
    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            response = get_response(prompt, client)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response}) 