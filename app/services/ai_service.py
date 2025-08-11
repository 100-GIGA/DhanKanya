"""
AI service for handling interactions with multiple LLM providers.

This module provides functions for initializing different LLM clients
and processing queries using Claude (Anthropic) and Gemini (Google) models.
"""

import re
import anthropic
import google.generativeai as genai
import streamlit as st
import logging
from typing import Optional, Dict, Any, List, Union

from config.settings import (
    ANTHROPIC_API_KEY, 
    GEMINI_API_KEY,
    DEFAULT_CLAUDE_MODEL, 
    DEFAULT_GEMINI_MODEL,
    DEFAULT_LLM_PROVIDER,
    LLM_OPTIONS,
    INTRODUCTION_PROMPTS, 
    PROMPT_TEMPLATE
)
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

def create_gemini_client() -> genai.GenerativeModel:
    """
    Create and initialize the Gemini client.
    
    Returns:
        An initialized Gemini GenerativeModel instance.
        
    Raises:
        Exception: If client initialization fails.
    """
    try:
        logger.info("Attempting to create Gemini client...")
        
        # Configure the Gemini API
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Use the same model selection logic as Claude
        model_name = st.session_state.get("gemini_model", DEFAULT_GEMINI_MODEL)
        
        # Create the model with consistent configuration
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 1000,
            }
        )
        logger.info("Successfully created Gemini client")
        return model
        
    except Exception as e:
        log_exception(e, "Gemini Client Error")
        raise

def create_llm_client(provider: str) -> Union[anthropic.Anthropic, genai.GenerativeModel]:
    """
    Create and initialize the appropriate LLM client based on provider.
    
    Args:
        provider: The LLM provider ('Claude' or 'Gemini')
        
    Returns:
        The initialized client instance for the specified provider.
        
    Raises:
        ValueError: If provider is not supported
        Exception: If client initialization fails
    """
    if provider == "Claude":
        return create_anthropic_client()
    elif provider == "Gemini":
        return create_gemini_client()
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

def query_anthropic(query_text: str, client: anthropic.Anthropic, system_prompt: Optional[str] = None, lang_code: str = 'en') -> str:
    """
    Query the Anthropic Claude model with the given text.
    
    Args:
        query_text: The user's query text.
        client: The initialized Anthropic client.
        system_prompt: Optional system prompt to provide context to the model.
        lang_code: The language code for the response (default: 'en').
        
    Returns:
        The response from the Claude model.
    """
    model = st.session_state.get("claude_model", DEFAULT_CLAUDE_MODEL)
    
    # Add language instruction to system prompt
    base_system_prompt = system_prompt if system_prompt else "You are a financial advisor for young women in India named DhanKanya. You provide clear, helpful advice focused on financial literacy, education planning, and building wealth. Be encouraging, informative, and tailored to the financial context in India."
    
    if lang_code != 'en':
        language_instruction = f"\n\nPlease respond in the same language as the user's query (language code: {lang_code})."
        system_prompt = base_system_prompt + language_instruction
    else:
        system_prompt = base_system_prompt
    
    try:
        message = client.messages.create(
            model=model,
            max_tokens=1000,
            system=system_prompt,
            messages=[
                {"role": "user", "content": query_text}
            ]
        )
        return message.content[0].text
    except Exception as e:
        log_exception(e, "Anthropic Query Error")
        return "I'm sorry, I encountered an error while processing your request. Please try again."

def query_gemini(query_text: str, client: genai.GenerativeModel, system_prompt: Optional[str] = None, lang_code: str = 'en') -> str:
    """
    Query the Google Gemini model with the given text.
    
    Args:
        query_text: The user's query text.
        client: The initialized Gemini GenerativeModel.
        system_prompt: Optional system prompt to provide context to the model.
        lang_code: The language code for the response (default: 'en').
        
    Returns:
        The response from the Gemini model.
    """
    # Use the same default system prompt as Claude
    base_system_prompt = system_prompt if system_prompt else "You are a financial advisor for young women in India named DhanKanya. You provide clear, helpful advice focused on financial literacy, education planning, and building wealth. Be encouraging, informative, and tailored to the financial context in India."
    
    # Add language instruction to system prompt (same as Claude)
    if lang_code != 'en':
        language_instruction = f"\n\nPlease respond in the same language as the user's query (language code: {lang_code})."
        final_system_prompt = base_system_prompt + language_instruction
    else:
        final_system_prompt = base_system_prompt
    
    # Combine system prompt with user query for Gemini (since Gemini doesn't have separate system/user like Claude)
    full_prompt = f"{final_system_prompt}\n\nUser Query: {query_text}"
    
    try:
        response = client.generate_content(full_prompt)
        return response.text
    except Exception as e:
        log_exception(e, "Gemini Query Error")
        return "I'm sorry, I encountered an error while processing your request. Please try again."

def query_llm(query_text: str, provider: str, client: Union[anthropic.Anthropic, genai.GenerativeModel], system_prompt: Optional[str] = None, lang_code: str = 'en') -> str:
    """
    Query the appropriate LLM based on provider.
    
    Args:
        query_text: The user's query text.
        provider: The LLM provider ('Claude' or 'Gemini').
        client: The initialized client instance.
        system_prompt: Optional system prompt to provide context to the model.
        lang_code: The language code for the response (default: 'en').
        
    Returns:
        The response from the specified LLM.
    """
    if provider == "Claude":
        return query_anthropic(query_text, client, system_prompt, lang_code)
    elif provider == "Gemini":
        return query_gemini(query_text, client, system_prompt, lang_code)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

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

def get_response(prompt: str, provider: str, client: Union[anthropic.Anthropic, genai.GenerativeModel], lang_code: str = 'en') -> str:
    """
    Process a user prompt and return an appropriate response using the specified LLM.
    
    Args:
        prompt: The user's prompt text.
        provider: The LLM provider ('Claude' or 'Gemini').
        client: The initialized client instance.
        lang_code: The language code for the response (default: 'en').
        
    Returns:
        The response to the user's prompt.
    """
    if is_introduction_query(prompt):
        return query_llm(prompt, provider, client, lang_code=lang_code)
    else:
        # Handle regular queries
        return query_llm(prompt, provider, client, lang_code=lang_code)

# Legacy function for backward compatibility
def get_response_legacy(prompt: str, client: anthropic.Anthropic, lang_code: str = 'en') -> str:
    """
    Legacy function for backward compatibility with existing code.
    
    Args:
        prompt: The user's prompt text.
        client: The initialized Anthropic client.
        lang_code: The language code for the response (default: 'en').
        
    Returns:
        The response to the user's prompt.
    """
    return get_response(prompt, "Claude", client, lang_code) 