"""
Configuration settings for the DhanKanya application.

This module contains all the configuration parameters used across
the application, loaded from environment variables when appropriate.
"""

import os

# Check if streamlit is available and in proper context
try:
    import streamlit as st
    # Check if we're in streamlit context
    if hasattr(st, 'secrets'):
        STREAMLIT_AVAILABLE = True
    else:
        STREAMLIT_AVAILABLE = False
except (ImportError, AttributeError):
    STREAMLIT_AVAILABLE = False
    st = None

# API Keys - handle missing secrets gracefully for testing
if STREAMLIT_AVAILABLE:
    try:
        ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
    except (KeyError, FileNotFoundError, AttributeError):
        ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
else:
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

if STREAMLIT_AVAILABLE:
    try:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError, AttributeError):
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
else:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Linkup API Key
if STREAMLIT_AVAILABLE:
    try:
        LINKUP_API_KEY = st.secrets["LINKUP_API_KEY"]
    except (KeyError, FileNotFoundError, AttributeError):
        LINKUP_API_KEY = os.getenv("LINKUP_API_KEY", "")
else:
    LINKUP_API_KEY = os.getenv("LINKUP_API_KEY", "")

# Application settings
APP_TITLE = "DhanKanya: Financial Empowerment for Girls in India"
APP_ICON = ":moneybag:"
APP_LAYOUT = "wide"

# LLM model settings
DEFAULT_CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
DEFAULT_LLM_PROVIDER = "Claude"

# Available LLM options
LLM_OPTIONS = {
    "Claude": {
        "model": DEFAULT_CLAUDE_MODEL,
        "display_name": "Claude 3.5 Sonnet"
    },
    "Gemini": {
        "model": DEFAULT_GEMINI_MODEL,
        "display_name": "Gemini 2.0 Flash"
    }
}

# Vector database settings
CHROMA_PATH = 'chroma'

# Prompt templates
PROMPT_TEMPLATE = """
Answer the question so that it is easily understandable. The context is provided so that you can take reference from this. Please take inspiration from the context. You can also add things that you think are helpful for girls out there. Do not mention about the context provided. Answer as you usually answer.

{context}

---

{question}
"""

# Introduction prompts for pattern matching
INTRODUCTION_PROMPTS = [
    r'introduce yourself',
    r'tell me about yourself',
    r'who are you',
    r'what can you do',
    r'how can you help me',
    r'what are your capabilities',
    r'what kind of tasks can you assist with',
    r'what are you capable of',
    r'what can i ask you',
    r'what are you good at?',
    r'what are your specialties?',
    r'what is your purpose?',
    r'what are you designed for?',
    r'how do i get started with you?',
    r'how should i interact with you?',
    r'who created you',
    r'hello',
    r'hey',
    r'namaste'
]

# Voice settings for Tamil + English
VOICE_ENABLED = True
SUPPORTED_VOICE_LANGUAGES = {
    'en': 'English',
    'ta': 'Tamil'
}
DEFAULT_VOICE_LANGUAGE = 'en'

# Gemini Live settings
GEMINI_LIVE_MODEL = "gemini-2.0-flash-exp"
VOICE_RECORDING_MAX_DURATION = 30  # seconds
VOICE_CHUNK_SIZE = 1024
AUDIO_SAMPLE_RATE = 16000

# Voice database settings
VOICE_CONVERSATIONS_TABLE = "voice_conversations"
AUDIO_STORAGE_PATH = "audio_files"

# Linkup settings
LINKUP_ENABLED = True
LINKUP_SEARCH_DEPTH = "standard"  # "standard" or "deep"
LINKUP_MAX_SOURCES = 5 