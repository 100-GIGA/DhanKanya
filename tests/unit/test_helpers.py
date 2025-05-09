"""
Unit tests for the helpers module.

This module contains tests for the helper utility functions
used throughout the application.
"""

import unittest
from unittest.mock import patch, mock_open
import os

from app.utils.helpers import check_env_file, format_inr

class TestHelpers(unittest.TestCase):
    """Test cases for helper utility functions."""
    
    @patch('os.path.exists')
    def test_check_env_file_not_found(self, mock_exists):
        """Test check_env_file function when the .env file does not exist."""
        # Configure mock to return False for os.path.exists
        mock_exists.return_value = False
        
        # Check that the function raises FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            check_env_file()
    
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='ANTHROPIC_API_KEY=test_key\n')
    def test_check_env_file_valid(self, mock_file, mock_exists):
        """Test check_env_file function with a valid .env file."""
        # Configure mock to return True for os.path.exists
        mock_exists.return_value = True
        
        # Should not raise any exception
        check_env_file()
    
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='INVALID_LINE\n')
    def test_check_env_file_invalid(self, mock_file, mock_exists):
        """Test check_env_file function with an invalid .env file."""
        # Configure mock to return True for os.path.exists
        mock_exists.return_value = True
        
        # Check that the function raises ValueError
        with self.assertRaises(ValueError):
            check_env_file()
    
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