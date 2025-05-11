"""
Configuration settings for the DhanKanya application.

This module contains all the configuration parameters used across
the application, loaded from environment variables when appropriate.
"""

import os
import streamlit as st

# API Keys
ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not found in streamlit secrets")

# Application settings
APP_TITLE = "DhanKanya: Financial Empowerment for Girls in India"
APP_ICON = ":moneybag:"
APP_LAYOUT = "wide"

# Claude model settings
DEFAULT_CLAUDE_MODEL = "claude-3-haiku-20240307"

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