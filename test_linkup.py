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
    
    # Test the search function directly
    print("Testing direct search...")
    answer, sources = linkup_service.search_with_sources(test_query)
    
    print(f"Answer length: {len(answer)}")
    print(f"Sources count: {len(sources)}")
    
    if sources:
        print(f"✅ Success! Found {len(sources)} sources:")
        for i, source in enumerate(sources[:3], 1):  # Show first 3
            print(f"{i}. {source.get('name', 'Unknown')}")
            print(f"   URL: {source.get('url', 'No URL')}")
            print(f"   Snippet: {source.get('snippet', 'No snippet')[:100]}...")
            print()
    else:
        print("❌ No sources found")
        
    # Also test the enhance function
    print("\nTesting enhance function...")
    _, sources2 = linkup_service.enhance_financial_query(test_query, "")
    print(f"Enhanced sources count: {len(sources2)}")
        
except Exception as e:
    print(f"❌ Error testing Linkup: {e}")
    import traceback
    traceback.print_exc()
