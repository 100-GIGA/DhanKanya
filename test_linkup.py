"""
Test script for Linkup integration
"""
import os
import sys
sys.path.append('/home/pritamv/DK')

# Test the Linkup service
try:
    from app.services.linkup_service import linkup_service
    
    print("🔗 Testing Linkup Integration...")
    
    # Test query
    test_query = "investment options for women in India 2024"
    print(f"Testing query: {test_query}")
    
    # Get sources
    sources = linkup_service.enhance_financial_query(test_query, "")
    
    if sources[1]:  # Check if sources were returned
        print(f"✅ Success! Found {len(sources[1])} sources:")
        for i, source in enumerate(sources[1][:3], 1):  # Show first 3
            print(f"{i}. {source.get('name', 'Unknown')}")
            print(f"   URL: {source.get('url', 'No URL')}")
            print(f"   Snippet: {source.get('snippet', 'No snippet')[:100]}...")
            print()
    else:
        print("❌ No sources found")
        
except Exception as e:
    print(f"❌ Error testing Linkup: {e}")
    import traceback
    traceback.print_exc()
