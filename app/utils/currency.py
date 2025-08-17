"""Utility functions for currency formatting."""

from babel.numbers import format_currency

def format_inr(amount: float) -> str:
    """Format amount as Indian Rupees.
    
    Args:
        amount: The amount to format
        
    Returns:
        A string representing the amount in INR format
    """
    return format_currency(amount, 'INR', locale='en_IN') 