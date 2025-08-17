"""
AI service for handling interactions with multiple LLM providers.

This module provides functions for initializing different LLM clients
and processing queries using Claude (Anthropic) and Gemini (Google) models.
Now includes Gemini Live API support for voice interactions.
"""

import re
import anthropic
import google.generativeai as genai
import streamlit as st
import logging
import asyncio
import json
from typing import Optional, Dict, Any, List, Union, Tuple
from datetime import datetime

# Import for Gemini Live
try:
    from google import genai as gemini_live
    GEMINI_LIVE_AVAILABLE = True
except ImportError:
    GEMINI_LIVE_AVAILABLE = False
    logging.warning("Gemini Live not available. Install with: pip install google-genai>=1.20.0")

from config.settings import (
    ANTHROPIC_API_KEY, 
    GEMINI_API_KEY,
    DEFAULT_CLAUDE_MODEL, 
    DEFAULT_GEMINI_MODEL,
    DEFAULT_LLM_PROVIDER,
    LLM_OPTIONS,
    INTRODUCTION_PROMPTS, 
    PROMPT_TEMPLATE,
    GEMINI_LIVE_MODEL,
    SUPPORTED_VOICE_LANGUAGES,
    DEFAULT_VOICE_LANGUAGE,
    LINKUP_ENABLED,
    LINKUP_SEARCH_DEPTH,
    LINKUP_MAX_SOURCES,
    FAST_MODE
)
from app.utils.helpers import log_exception, disable_proxies
from app.services.linkup_service import get_sources_for_query, linkup_service

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
                "temperature": 0,
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

def create_gemini_live_client() -> Optional[Any]:
    """
    Create and initialize the Gemini Live client for voice interactions.
    
    Returns:
        Initialized Gemini Live client or None if not available.
        
    Raises:
        Exception: If client initialization fails.
    """
    if not GEMINI_LIVE_AVAILABLE:
        logger.warning("Gemini Live not available")
        return None
    
    try:
        logger.info("Attempting to create Gemini Live client...")
        
        # Initialize Gemini Live client
        client = gemini_live.Client(api_key=GEMINI_API_KEY)
        
        logger.info("Successfully created Gemini Live client")
        return client
        
    except Exception as e:
        log_exception(e, "Gemini Live Client Error")
        return None

def get_voice_system_prompt(lang_code: str = 'en') -> str:
    """
    Get the system prompt for voice interactions in specified language.
    
    Args:
        lang_code: Language code ('en' for English, 'ta' for Tamil)
        
    Returns:
        System prompt for voice interactions
    """
    base_prompt = """You are DhanKanya, a friendly financial advisor for young women in India. You provide clear, helpful advice focused on financial literacy, education planning, and building wealth. Be encouraging, informative, and tailored to the financial context in India.

For voice interactions:
- Keep responses conversational and natural
- Use simple, clear language
- Be encouraging and supportive
- Provide practical, actionable advice
- Remember this is a voice conversation, so avoid complex formatting.
- Use the web scrapping tool available to give source links for any financial data you provide."""
    
    if lang_code == 'ta':
        tamil_instruction = "\n\nPlease respond in Tamil when the user speaks in Tamil. Mix Tamil and English naturally as appropriate for financial terms."
        return base_prompt + tamil_instruction
    else:
        return base_prompt

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

def query_llm_with_sources(query_text: str, provider: str, client: Union[anthropic.Anthropic, genai.GenerativeModel], system_prompt: Optional[str] = None, lang_code: str = 'en', include_sources: bool = True) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Query the appropriate LLM with optional Linkup sources integration.
    
    Args:
        query_text: The user's query text.
        provider: The LLM provider ('Claude' or 'Gemini').
        client: The initialized client instance.
        system_prompt: Optional system prompt to provide context to the model.
        lang_code: The language code for the response (default: 'en').
        include_sources: Whether to fetch sources from Linkup (default: True).
        
    Returns:
        Tuple of (LLM response, sources list)
    """
    # Get the standard LLM response first (prioritize speed)
    response = query_llm(query_text, provider, client, system_prompt, lang_code)
    
    # Get sources from Linkup only if explicitly requested and enabled
    sources = []
    if include_sources and LINKUP_ENABLED:
        try:
            # Use a shorter timeout for faster responses
            sources = get_sources_for_query(query_text)[:LINKUP_MAX_SOURCES]
            logger.info(f"Retrieved {len(sources)} sources from Linkup for query")
        except Exception as e:
            logger.error(f"Failed to get sources from Linkup: {e}")
            sources = []
    
    return response, sources

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

def get_response_with_sources(prompt: str, provider: str, client: Union[anthropic.Anthropic, genai.GenerativeModel], lang_code: str = 'en', include_sources: bool = None, system_prompt: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Process a user prompt and return response with optional sources using the specified LLM.
    Sources are controlled by FAST_MODE setting and include_sources parameter.
    
    Args:
        prompt: The user's prompt text.
        provider: The LLM provider ('Claude' or 'Gemini').
        client: The initialized client instance.
        lang_code: The language code for the response (default: 'en').
        include_sources: Whether to fetch sources from Linkup. If None, uses inverse of FAST_MODE.
        system_prompt: Optional system prompt to provide context to the model.
        
    Returns:
        Tuple of (response, sources_list)
    """
    # Determine whether to include sources based on settings
    if include_sources is None:
        include_sources = not FAST_MODE  # Fast mode = no sources by default
    
    if is_introduction_query(prompt):
        response = query_llm(prompt, provider, client, system_prompt, lang_code)
        return response, []  # No sources for introduction queries
    else:
        # Handle regular queries with optional sources
        return query_llm_with_sources(prompt, provider, client, system_prompt, lang_code, include_sources=include_sources)

def get_fast_response(prompt: str, provider: str, client: Union[anthropic.Anthropic, genai.GenerativeModel], lang_code: str = 'en') -> str:
    """
    Process a user prompt and return a fast response without sources.
    Optimized for speed by skipping Linkup API calls.
    
    Args:
        prompt: The user's prompt text.
        provider: The LLM provider ('Claude' or 'Gemini').
        client: The initialized client instance.
        lang_code: The language code for the response (default: 'en').
        
    Returns:
        LLM response string
    """
    return query_llm(prompt, provider, client, lang_code=lang_code)

async def process_voice_query(transcription: str, lang_code: str = 'en') -> Tuple[str, Dict[str, Any]]:
    """
    Process a voice query using Gemini Live API.
    
    Args:
        transcription: Transcribed user speech
        lang_code: Language code ('en' or 'ta')
        
    Returns:
        Tuple of (response_text, metadata)
    """
    try:
        # Use regular Gemini client for now (Live client for future streaming)
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_LIVE_MODEL)
        
        # Get voice-optimized system prompt
        system_prompt = get_voice_system_prompt(lang_code)
        
        # Combine system prompt with user query
        full_prompt = f"{system_prompt}\n\nUser said: {transcription}"
        
        # Generate response
        response = model.generate_content(full_prompt)
        response_text = response.text
        
        # Create metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'input_language': lang_code,
            'transcription': transcription,
            'response_text': response_text,
            'model_used': GEMINI_LIVE_MODEL,
            'mode': 'voice_hybrid'
        }
        
        logger.info(f"Voice query processed successfully for language: {lang_code}")
        return response_text, metadata
        
    except Exception as e:
        log_exception(e, "Voice Query Processing Error")
        error_response = "I'm sorry, I couldn't process your voice message. Please try again."
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'transcription': transcription,
            'mode': 'voice_hybrid'
        }
        return error_response, metadata

def detect_language_from_text(text: str) -> str:
    """
    Detect language from text (Tamil vs English).
    
    Args:
        text: Input text
        
    Returns:
        Language code ('ta' for Tamil, 'en' for English)
    """
    # Simple Tamil script detection using Unicode ranges
    tamil_chars = 0
    total_chars = len(text.replace(' ', ''))
    
    if total_chars == 0:
        return DEFAULT_VOICE_LANGUAGE
        
    for char in text:
        # Tamil Unicode range: U+0B80–U+0BFF
        if '\u0B80' <= char <= '\u0BFF':
            tamil_chars += 1
    
    # If more than 30% Tamil characters, consider it Tamil
    if tamil_chars / total_chars > 0.3:
        return 'ta'
    else:
        return 'en'

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