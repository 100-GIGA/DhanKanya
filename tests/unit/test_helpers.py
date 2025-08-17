"""
Unit tests for the helpers module.

This module contains tests for the helper utility functions
used throughout the application.
"""

import unittest
from unittest.mock import patch, mock_open
import os

from app.utils.helpers import format_inr

class TestHelpers(unittest.TestCase):
    """Test cases for helper utility functions."""
    
    def test_format_inr(self):
        """Test format_inr function for formatting currency values."""
        # Test with integer value
        self.assertEqual(format_inr(1000), '₹1,000.00')
        
        # Test with decimal value
        self.assertEqual(format_inr(1234.56), '₹1,234.56')
        
        # Test with zero
        self.assertEqual(format_inr(0), '₹0.00')
        
        # Test with negative value
        self.assertEqual(format_inr(-500), '-₹500.00')

if __name__ == '__main__':
    unittest.main() 