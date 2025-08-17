"""
Template service for the DhanKanya application.

This module provides functions for loading and processing financial templates
that can be used to guide users through various financial scenarios.
"""

import json
import os
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

def load_templates() -> Dict[str, Any]:
    """
    Load financial templates from the JSON file.
    
    Returns:
        A dictionary containing the loaded templates,
        or an empty dictionary if loading fails.
    """
    try:
        with open('assets/state_templates.json', 'r') as f:
            templates = json.load(f)
        return templates
    except FileNotFoundError:
        logger.error("Templates file 'assets/state_templates.json' not found")
        return {}
    except json.JSONDecodeError:
        logger.error("Error decoding JSON from templates file")
        return {}
    except Exception as e:
        logger.error(f"Error loading templates: {str(e)}")
        return {}

def get_template_by_state(templates: Dict[str, Any], state: str) -> Optional[Dict[str, Any]]:
    """
    Get template data for a specific Indian state.
    
    Args:
        templates: The dictionary of all templates.
        state: The name of the Indian state.
        
    Returns:
        The template data for the specified state, or None if not found.
    """
    if not templates:
        return None
    
    # Direct key access since states are keys in the JSON
    if state in templates:
        # Convert to a standard format that includes the state name
        state_data = templates[state]
        state_data['name'] = state
        return state_data
    
    return None

def get_state_list(templates: Dict[str, Any]) -> List[str]:
    """
    Get a list of all available Indian states from the templates.
    
    Args:
        templates: The dictionary of all templates.
        
    Returns:
        A list of state names.
    """
    if not templates:
        return []
    
    # States are the keys in the JSON
    return list(templates.keys())

def format_template_for_display(template: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a template for display in the UI.
    
    Args:
        template: The raw template data.
        
    Returns:
        A dictionary with formatted template data for display.
    """
    if not template:
        return {}
    
    # Extract sample prompts and other data to fit the display format
    result = {
        'state': template.get('name', ''),
        'sample_prompts': template.get('Sample Prompts', []),
        # These fields aren't in the current JSON but keeping the structure
        # to avoid breaking the UI if they're added later
        'scholarships': template.get('scholarships', []),
        'educational_loans': template.get('educational_loans', []),
        'government_schemes': template.get('government_schemes', [])
    }
    
    return result 