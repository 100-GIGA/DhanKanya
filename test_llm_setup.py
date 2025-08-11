#!/usr/bin/env python3
"""
Test script to verify LLM setup and functionality.
Run this script to test if both Claude and Gemini are properly configured.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required imports work."""
    try:
        import anthropic
        print("✅ Anthropic library imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Anthropic: {e}")
        return False

    try:
        import google.generativeai as genai
        print("✅ Google GenerativeAI library imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Google GenerativeAI: {e}")
        return False

    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Streamlit: {e}")
        return False

    return True

def test_config():
    """Test if configuration loads properly."""
    try:
        # Mock streamlit secrets for testing
        class MockSecrets:
            def __getitem__(self, key):
                if key == "ANTHROPIC_API_KEY":
                    return "test_anthropic_key"
                elif key == "GEMINI_API_KEY":
                    return "test_gemini_key"
                else:
                    raise KeyError(f"Secret {key} not found")

        # Temporarily replace streamlit secrets
        import streamlit as st
        original_secrets = getattr(st, 'secrets', None)
        st.secrets = MockSecrets()

        # Test config import
        from config.settings import LLM_OPTIONS, DEFAULT_LLM_PROVIDER
        print("✅ Configuration loaded successfully")
        print(f"   Default LLM Provider: {DEFAULT_LLM_PROVIDER}")
        print(f"   Available LLM Options: {list(LLM_OPTIONS.keys())}")

        # Restore original secrets
        if original_secrets:
            st.secrets = original_secrets

        return True
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return False

def test_ai_service():
    """Test if AI service can be imported."""
    try:
        from app.services.ai_service import create_llm_client, query_llm
        print("✅ AI service imported successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to import AI service: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing DhanKanya Multi-LLM Setup")
    print("=" * 50)

    tests = [
        ("Testing imports", test_imports),
        ("Testing configuration", test_config),
        ("Testing AI service", test_ai_service),
    ]

    all_passed = True
    for test_name, test_func in tests:
        print(f"\n{test_name}...")
        if not test_func():
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Multi-LLM setup is ready.")
        print("\nNext steps:")
        print("1. Make sure you have valid API keys in .streamlit/secrets.toml")
        print("2. Run: streamlit run main.py")
        print("3. Test switching between Claude and Gemini models")
    else:
        print("❌ Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
