"""
Templates page component for the DhanKanya application.

This module contains the UI components for the templates page,
which displays state-specific financial information and opportunities.
"""

import streamlit as st
import anthropic
from typing import Dict, List, Any

from app.services.template_service import (
    load_templates, 
    get_state_list, 
    get_template_by_state, 
    format_template_for_display
)
from app.services.ai_service import query_anthropic

def render(client: anthropic.Anthropic) -> None:
    """
    Render the templates page with state-specific financial information.
    
    Args:
        client: The initialized Anthropic client for AI interaction.
    """
    st.title("Build your Wealth - State-Specific Opportunities")
    
    # Session state to track selected question
    if 'selected_question' not in st.session_state:
        st.session_state.selected_question = ""
    if 'response_generated' not in st.session_state:
        st.session_state.response_generated = False
    
    # Load templates
    templates = load_templates()
    
    if not templates:
        st.error("Failed to load state templates. Please try again later.")
        return
    
    # Get list of states
    states = get_state_list(templates)
    
    if not states:
        st.error("No states found in the templates data.")
        return
    
    # State selection dropdown
    selected_state = st.selectbox(
        "Select your state to discover financial opportunities:",
        options=states
    )
    
    # Get template for selected state
    state_template = get_template_by_state(templates, selected_state)
    
    if not state_template:
        st.warning(f"No information available for {selected_state}.")
        return
    
    # Format template for display
    formatted_template = format_template_for_display(state_template)
    
    # Display state information
    st.header(f"Financial Opportunities in {formatted_template.get('state', '')}")
    
    # Function to set selected question and trigger response
    def set_question(question):
        st.session_state.selected_question = question
        st.session_state.response_generated = True
    
    # Display sample prompts as clickable buttons
    st.subheader("Select a question or type your own below:")
    sample_prompts = formatted_template.get('sample_prompts', [])
    
    if sample_prompts:
        # Single column display with full text
        for i, prompt in enumerate(sample_prompts):
            st.button(
                prompt,  # Display full text without truncation
                key=f"prompt_{i}", 
                on_click=set_question, 
                args=(prompt,),
                use_container_width=True  # Make button full width
            )
    else:
        st.write("No sample questions available for this state.")
    
    # Display scholarships section if data exists
    scholarships = formatted_template.get('scholarships', [])
    if scholarships:
        st.subheader("Available Scholarships")
        for scholarship in scholarships:
            with st.expander(scholarship.get('name', 'Scholarship')):
                st.write(f"**Provider:** {scholarship.get('provider', 'N/A')}")
                st.write(f"**Eligibility:** {scholarship.get('eligibility', 'Not specified')}")
                st.write(f"**Amount:** {scholarship.get('amount', 'Not specified')}")
                if scholarship.get('website'):
                    st.write(f"**Website:** [{scholarship.get('website')}]({scholarship.get('website')})")
    
    # Display educational loans section if data exists
    loans = formatted_template.get('educational_loans', [])
    if loans:
        st.subheader("Educational Loans")
        for loan in loans:
            with st.expander(loan.get('name', 'Loan')):
                st.write(f"**Provider:** {loan.get('provider', 'N/A')}")
                st.write(f"**Eligibility:** {loan.get('eligibility', 'Not specified')}")
                st.write(f"**Interest Rate:** {loan.get('interest_rate', 'Not specified')}")
                st.write(f"**Max Amount:** {loan.get('max_amount', 'Not specified')}")
                if loan.get('website'):
                    st.write(f"**Website:** [{loan.get('website')}]({loan.get('website')})")
    
    # Display government schemes section if data exists
    schemes = formatted_template.get('government_schemes', [])
    if schemes:
        st.subheader("Government Schemes")
        for scheme in schemes:
            with st.expander(scheme.get('name', 'Scheme')):
                st.write(f"**Description:** {scheme.get('description', 'No description available')}")
                st.write(f"**Eligibility:** {scheme.get('eligibility', 'Not specified')}")
                st.write(f"**Benefits:** {scheme.get('benefits', 'Not specified')}")
                if scheme.get('website'):
                    st.write(f"**Website:** [{scheme.get('website')}]({scheme.get('website')})")
    
    # AI assistance for financial planning
    st.markdown("---")
    st.subheader("Financial Planning Advice")
    
    # Text input field that shows the selected question
    user_query = st.text_input(
        "Your question:", 
        value=st.session_state.selected_question
    )
    
    # Generate response if a question is selected or manually entered
    if user_query and (st.session_state.response_generated or st.button("Get Advice")):
        with st.spinner(f"Generating advice for {selected_state}..."):
            # Create a comprehensive state-specific context
            context = f"""
            I am providing financial advice for a user from {selected_state}, India.
            The user's question is specifically about opportunities available in {selected_state}.
            All information and recommendations should be tailored to {selected_state}'s specific 
            context, including state government schemes, educational institutions, scholarships,
            and financial assistance programs available in {selected_state}.
            
            When answering, always frame your response in the context of {selected_state} 
            and what is available or applicable there.
            """
            
            # Include sample prompts to give the AI model examples of state-specific questions
            if sample_prompts:
                context += f"\n\nHere are some example questions users from {selected_state} typically ask: "
                context += "; ".join(sample_prompts[:3])
            
            # Ensure the query explicitly mentions the state if it doesn't already
            if selected_state not in user_query:
                state_specific_query = f"For {selected_state}: {user_query}"
            else:
                state_specific_query = user_query
            
            # Query the AI with the enhanced context and state-specific query
            response = query_anthropic(state_specific_query, client, system_prompt=context)
            
            # Display the response
            st.write(f"### Financial Advice for {selected_state}")
            st.write(response)
            
            # Reset flag to prevent regeneration on rerun
            st.session_state.response_generated = False 