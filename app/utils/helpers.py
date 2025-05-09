"""
Helper utility functions for the DhanKanya application.

This module contains various utility functions used across the application,
including formatting, validation, and environment checks.
"""

import os
import logging
import sys
import traceback
from babel.numbers import format_currency

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_env_file() -> None:
    """
    Validate that .env file exists and its structure is correct.
    
    Raises:
        FileNotFoundError: If .env file does not exist.
        ValueError: If .env file contains invalid entries.
    """
    logger.info("=== Checking .env File ===")
    if not os.path.exists('.env'):
        logger.error(".env file does not exist.")
        raise FileNotFoundError(".env file is missing.")

    with open('.env', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('#') or line.strip() == '':
                continue
            if '=' not in line:
                logger.error(f"Invalid line in .env file: {line}")
                raise ValueError(f"Invalid line in .env file: {line}")
        logger.info("Valid .env file structure.")

def log_system_info() -> None:
    """Log system and environment information for diagnostics."""
    logger.info("=== Starting Application Diagnostics ===")
    logger.info(f"Python version: {sys.version}")
    
    # Log environment info
    logger.info("=== Environment Check ===")
    is_cloud = os.getenv('STREAMLIT_DEPLOYMENT_RUNTIME') == 'cloud'
    logger.info(f"Running on Streamlit Cloud: {is_cloud}")
    
    # Check for proxy-related environment variables
    logger.info("=== Proxy Configuration Check ===")
    proxy_vars = {k: '***' for k, v in os.environ.items() 
                if 'PROXY' in k.upper() or 'HTTP' in k.upper()}
    logger.info(f"Proxy-related environment variables: {proxy_vars}")

def disable_proxies() -> None:
    """Disable any proxy settings in the environment."""
    os.environ.pop('HTTP_PROXY', None)
    os.environ.pop('HTTPS_PROXY', None)
    os.environ.pop('http_proxy', None)
    os.environ.pop('https_proxy', None)

def format_inr(amount: float) -> str:
    """
    Format an amount as Indian Rupees (INR).
    
    Args:
        amount: The amount to format.
        
    Returns:
        A string representation of the amount in INR format.
    """
    return format_currency(amount, 'INR', locale='en_IN')

def log_exception(e: Exception, context: str = "Error") -> None:
    """
    Log exception details for debugging.
    
    Args:
        e: The exception to log.
        context: Optional context string for the log entry.
    """
    logger.error(f"=== {context} ===")
    logger.error(f"Error type: {type(e)}")
    logger.error(f"Error message: {str(e)}")
    logger.error(f"Error args: {e.args}")
    logger.error(f"Traceback:\n{traceback.format_exc()}") 