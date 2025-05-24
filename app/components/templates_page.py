"""
Templates page component for the DhanKanya application.

This module contains the UI components for the templates page,
which provides an AI assistant for state-specific financial guidance.
"""

import streamlit as st
import anthropic
from typing import Dict, List, Any

from app.services.template_service import (
    load_templates, 
    get_state_list, 
    get_template_by_state
)
from app.services.ai_service import query_anthropic

def initialize_session_state():
    """Initialize session state variables."""
    if 'selected_question' not in st.session_state:
        st.session_state.selected_question = ""
    if 'response_generated' not in st.session_state:
        st.session_state.response_generated = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'selected_state' not in st.session_state:
        st.session_state.selected_state = None

def render_state_selector(templates: Dict[str, Any]) -> str:
    """Render the state selector with a modern interface."""
    states = get_state_list(templates)
    
    # Create a bordered container for state selection
    with st.container():
        # Create two columns for the selector
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Select Your State")
        
        with col2:
            # Create a modern selectbox
            selected_state = st.selectbox(
                "",
                options=states,
                key="state_selector",
                label_visibility="collapsed"
            )
        st.markdown("---")
    
    return selected_state



def render_ai_assistant(selected_state: str, client: anthropic.Anthropic):
    """Render the AI assistant section with a chat-like interface."""
    st.subheader("💬 Ask Your Financial Questions")
    
    # Get sample prompts
    sample_prompts = get_template_by_state(load_templates(), selected_state).get('Sample Prompts', [])
    
    # Chat input at the top
    if prompt := st.chat_input("Type your question here..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Create context for the AI
        context = f"""
        I am providing financial advice for a user from {selected_state}, India.
        The user's question is specifically about opportunities available in {selected_state}.
        All information and recommendations should be tailored to {selected_state}'s specific 
        context, including state government schemes, educational institutions, scholarships,
        and financial assistance programs available in {selected_state}.
        
        When answering, always frame your response in the context of {selected_state} 
        and what is available or applicable there.
        """
        
        # Include sample prompts for context
        if sample_prompts:
            context += f"\n\nHere are some example questions users from {selected_state} typically ask: "
            context += "; ".join(sample_prompts[:3])
        
        # Ensure the query explicitly mentions the state
        if selected_state not in prompt:
            state_specific_query = f"For {selected_state}: {prompt}"
        else:
            state_specific_query = prompt
        
        # Get AI response
        with st.spinner("Thinking..."):
            response = query_anthropic(state_specific_query, client, system_prompt=context)
            
            # Add assistant message to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    # Display chat history in reverse order (most recent first)
    # Process messages in pairs (user + assistant)
    for i in range(len(st.session_state.chat_history) - 1, -1, -2):
        # Get the pair of messages (user + assistant)
        if i > 0:  # Ensure we have a pair
            user_msg = st.session_state.chat_history[i-1]
            assistant_msg = st.session_state.chat_history[i]
            
            # Display the pair in natural order (user then assistant)
            with st.chat_message(user_msg["role"]):
                st.write(user_msg["content"])
            with st.chat_message(assistant_msg["role"]):
                st.write(assistant_msg["content"])
    
    # Only show sample prompts if there's no chat history
    if not st.session_state.chat_history:
        st.caption("✨ Let's explore your financial journey together! Here are some questions you might want to ask.")
        
        # Create a container for sample prompts
        with st.container():
            for i, prompt in enumerate(sample_prompts):
                if st.button(prompt, key=f"prompt_{i}", use_container_width=True):
                    # Add user message to chat history
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    
                    # Create context for the AI
                    context = f"""
                    I am providing financial advice for a user from {selected_state}, India.
                    The user's question is specifically about opportunities available in {selected_state}.
                    All information and recommendations should be tailored to {selected_state}'s specific 
                    context, including state government schemes, educational institutions, scholarships,
                    and financial assistance programs available in {selected_state}.
                    
                    When answering, always frame your response in the context of {selected_state} 
                    and what is available or applicable there.
                    """
                    
                    # Include sample prompts for context
                    if sample_prompts:
                        context += f"\n\nHere are some example questions users from {selected_state} typically ask: "
                        context += "; ".join(sample_prompts[:3])
                    
                    # Ensure the query explicitly mentions the state
                    if selected_state not in prompt:
                        state_specific_query = f"For {selected_state}: {prompt}"
                    else:
                        state_specific_query = prompt
                    
                    # Get AI response
                    with st.spinner("Thinking..."):
                        response = query_anthropic(state_specific_query, client, system_prompt=context)
                        
                        # Add assistant message to chat history
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
                    # Force a rerun to update the UI
                    st.rerun()

def render(client: anthropic.Anthropic) -> None:
    """
    Render the templates page with state-specific financial information.
    
    Args:
        client: The initialized Anthropic client for AI interaction.
    """
    # Initialize session state
    initialize_session_state()
    
    # Page title
    st.title("Build Your Wealth")
    st.caption("Get personalized financial guidance through our AI assistant. Select your state to receive tailored advice on wealth building, scholarships, and financial planning specific to your region.")
    
    # Load templates
    templates = load_templates()
    
    if not templates:
        st.error("Failed to load state templates. Please try again later.")
        return
    
    # Render state selector
    selected_state = render_state_selector(templates)
    
    if not selected_state:
        st.warning("Please select a state to view available opportunities.")
        return
    
    # Render AI assistant section
    render_ai_assistant(selected_state, client) 