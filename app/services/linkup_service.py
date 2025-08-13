"""
Linkup service for providing source links with AI responses.

This module integrates with Linkup API to provide high-quality source links
and context for financial information queries.
"""

import logging
import streamlit as st
from typing import Dict, List, Optional, Any, Tuple
from linkup import LinkupClient

logger = logging.getLogger(__name__)

class LinkupService:
    """Service for integrating Linkup API to provide source links with AI responses."""
    
    def __init__(self):
        """Initialize Linkup service with API key."""
        try:
            self.api_key = st.secrets.get("LINKUP_API_KEY", "")
            if not self.api_key:
                raise ValueError("LINKUP_API_KEY not found in secrets")
            
            self.client = LinkupClient(api_key=self.api_key)
            self.enabled = True
            logger.info("Linkup service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Linkup service: {e}")
            self.enabled = False
            self.client = None
    
    def search_with_sources(self, query: str, depth: str = "standard", timeout: int = 30) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Search for information and return sourced answer with links.
        
        Args:
            query: The search query
            depth: Search depth ("standard" or "deep")
            timeout: Request timeout in seconds (default: 10)
            
        Returns:
            Tuple of (answer_text, sources_list)
        """
        if not self.enabled or not self.client:
            logger.warning("Linkup service not available")
            return "", []

        try:
            # Make request to Linkup API
            response = self.client.search(
                query=query,
                depth=depth,
                output_type="sourcedAnswer"
            )
            
            # Extract answer and sources from LinkupSourcedAnswer object
            answer = getattr(response, "answer", "")
            sources = getattr(response, "sources", [])
            
            # Format sources for better display
            formatted_sources = []
            for source in sources:
                # Handle both dict and object types for sources
                if hasattr(source, '__dict__'):
                    # Object with attributes
                    name = getattr(source, "name", "Unknown Source")
                    url = getattr(source, "url", "")
                    snippet = getattr(source, "snippet", "")
                else:
                    # Dictionary
                    name = source.get("name", "Unknown Source")
                    url = source.get("url", "")
                    snippet = source.get("snippet", "")
                
                # Truncate long snippets
                if snippet and len(snippet) > 200:
                    snippet = snippet[:200] + "..."
                
                formatted_source = {
                    "name": name,
                    "url": url,
                    "snippet": snippet
                }
                formatted_sources.append(formatted_source)
            
            logger.info(f"Linkup search successful: {len(formatted_sources)} sources found")
            return answer, formatted_sources
            
        except Exception as e:
            logger.error(f"Linkup search failed: {e}")
            return "", []

    def enhance_financial_query(self, user_query: str, ai_response: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Enhance a financial query by adding relevant sources.
        
        Args:
            user_query: The original user query
            ai_response: The AI-generated response
            
        Returns:
            Tuple of (enhanced_context, sources_list)
        """
        if not self.enabled:
            return "", []
        
        # Create search query focused on financial information
        search_query = f"financial advice India {user_query}"
        
        # Get sources from Linkup
        _, sources = self.search_with_sources(search_query, depth="standard")
        
        return "", sources  # Return empty context as we're using AI response, just adding sources
    
    def get_financial_context(self, topic: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Get financial context and sources for a specific topic.
        
        Args:
            topic: The financial topic to search for
            
        Returns:
            Tuple of (context_text, sources_list)
        """
        if not self.enabled:
            return "", []
        
        # Create comprehensive search query
        search_query = f"India financial information {topic} government schemes investment options"
        
        return self.search_with_sources(search_query, depth="deep")

# Global instance
linkup_service = LinkupService()

def get_sources_for_query(query: str) -> List[Dict[str, Any]]:
    """
    Get sources for a financial query.
    
    Args:
        query: The user's query
        
    Returns:
        List of source dictionaries
    """
    _, sources = linkup_service.enhance_financial_query(query, "")
    return sources

def get_linkup_context(query: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Get Linkup context and sources for a query.
    
    Args:
        query: The search query
        
    Returns:
        Tuple of (context, sources)
    """
    return linkup_service.search_with_sources(query)
