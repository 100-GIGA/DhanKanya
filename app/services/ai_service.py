"""
AI service for handling interactions with the Anthropic API.

This module provides functions for initializing the Anthropic client
and processing queries using the Claude model.
"""

import re
import anthropic
import streamlit as st
import logging
from typing import Optional, Dict, Any, List

from config.settings import ANTHROPIC_API_KEY, DEFAULT_CLAUDE_MODEL, INTRODUCTION_PROMPTS, PROMPT_TEMPLATE
from app.utils.helpers import log_exception, disable_proxies

logger = logging.getLogger(__name__)

def create_anthropic_client() -> anthropic.Anthropic:
    """
    Create and initialize the Anthropic client.
    
    Returns:
        An initialized Anthropic client instance.
        
    Raises:
        Exception: If client initialization fails.
    """
    try:
        logger.info("Attempting to create Anthropic client...")
        
        # Force disable any proxy settings
        disable_proxies()
        
        # Create client with minimal configuration
        client_config = {
            'api_key': ANTHROPIC_API_KEY
        }
        logger.info("Client configuration prepared (API key masked)")
        
        client = anthropic.Anthropic(**client_config)
        logger.info("Successfully created Anthropic client")
        return client
        
    except Exception as e:
        log_exception(e, "Anthropic Client Error")
        raise

def query_anthropic(query_text: str, client: anthropic.Anthropic, system_prompt: Optional[str] = None) -> str:
    """
    Query the Anthropic Claude model with the given text.
    
    Args:
        query_text: The user's query text.
        client: The initialized Anthropic client.
        system_prompt: Optional system prompt to provide context to the model.
        
    Returns:
        The response from the Claude model.
    """
    model = st.session_state.get("claude_model", DEFAULT_CLAUDE_MODEL)
    
    try:
        message = client.messages.create(
            model=model,
            max_tokens=1000,
            system=system_prompt if system_prompt else "You are a financial advisor for young women in India named DhanKanya. You provide clear, helpful advice focused on financial literacy, education planning, and building wealth. Be encouraging, informative, and tailored to the financial context in India.",
            messages=[
                {"role": "user", "content": query_text}
            ]
        )
        return message.content[0].text
    except Exception as e:
        log_exception(e, "Anthropic Query Error")
        return "I'm sorry, I encountered an error while processing your request. Please try again."

def is_introduction_query(query: str) -> bool:
    """
    Check if the query is asking for an introduction.
    
    Args:
        query: The user's query text.
        
    Returns:
        True if the query is asking for an introduction, False otherwise.
    """
    for pattern in INTRODUCTION_PROMPTS:
        if re.search(pattern, query.lower()):
            return True
    return False

def get_response(prompt: str, client: anthropic.Anthropic) -> str:
    """
    Process a user prompt and return an appropriate response.
    
    Args:
        prompt: The user's prompt text.
        client: The initialized Anthropic client.
        
    Returns:
        The response to the user's prompt.
    """
    if is_introduction_query(prompt):
        return query_anthropic(prompt, client)
    else:
        # Handle regular queries
        # In a real implementation, this might interact with a vector store for context
        return query_anthropic(prompt, client) 