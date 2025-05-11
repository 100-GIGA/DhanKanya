"""
Templates page component for the DhanKanya application.

This module contains the UI components for the templates page,
which displays state-specific financial information and opportunities.
"""

import streamlit as st
import anthropic
from typing import Dict, List, Any
import pandas as pd
import plotly.express as px
import json
import os

from app.services.template_service import (
    load_templates, 
    get_state_list, 
    get_template_by_state, 
    format_template_for_display
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

def render_opportunities(formatted_template: Dict[str, Any]):
    """Render the opportunities section with modern cards."""
    st.subheader("💫 Available Opportunities")
    
    # Create tabs for different opportunity types
    tabs = st.tabs(["🎓 Scholarships", "💰 Educational Loans", "🏛️ Government Schemes"])
    
    # Scholarships tab
    with tabs[0]:
        scholarships = formatted_template.get('scholarships', [])
        if scholarships:
            for scholarship in scholarships:
                with st.expander(scholarship.get('name', 'Scholarship')):
                    st.write("**Provider:**", scholarship.get('provider', 'N/A'))
                    st.write("**Amount:**", scholarship.get('amount', 'Not specified'))
                    st.write("**Eligibility:**", scholarship.get('eligibility', 'Not specified'))
                    if scholarship.get('website'):
                        st.link_button("Visit Website", scholarship.get('website'))
        else:
            st.info("No scholarships available for this state yet.")
    
    # Educational Loans tab
    with tabs[1]:
        loans = formatted_template.get('educational_loans', [])
        if loans:
            for loan in loans:
                with st.expander(loan.get('name', 'Loan')):
                    st.write("**Provider:**", loan.get('provider', 'N/A'))
                    st.write("**Interest Rate:**", loan.get('interest_rate', 'Not specified'))
                    st.write("**Max Amount:**", loan.get('max_amount', 'Not specified'))
                    st.write("**Eligibility:**", loan.get('eligibility', 'Not specified'))
                    if loan.get('website'):
                        st.link_button("Visit Website", loan.get('website'))
        else:
            st.info("No educational loans available for this state yet.")
    
    # Government Schemes tab
    with tabs[2]:
        schemes = formatted_template.get('government_schemes', [])
        if schemes:
            for scheme in schemes:
                with st.expander(scheme.get('name', 'Scheme')):
                    st.write("**Description:**", scheme.get('description', 'No description available'))
                    st.write("**Eligibility:**", scheme.get('eligibility', 'Not specified'))
                    st.write("**Benefits:**", scheme.get('benefits', 'Not specified'))
                    if scheme.get('website'):
                        st.link_button("Visit Website", scheme.get('website'))
        else:
            st.info("No government schemes available for this state yet.")

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
    st.caption("Explore state-specific financial opportunities, scholarships, educational loans, and government schemes tailored to help you achieve your educational and financial goals. Select your state to get started.")
    
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
    
    # Get template for selected state
    state_template = get_template_by_state(templates, selected_state)
    
    if not state_template:
        st.warning(f"No information available for {selected_state}.")
        return
    
    # Format template for display
    formatted_template = format_template_for_display(state_template)
    
    # Create two columns for the main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Render opportunities section
        render_opportunities(formatted_template)
    
    with col2:
        # Render AI assistant section
        render_ai_assistant(selected_state, client) 