#!/usr/bin/env python3
"""
Unified API testing script for SHUBI API
Combines API connection test and model availability check
"""

import os
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# Load environment variables
load_dotenv()

def test_api_connection():
    """Test basic API connection"""
    api_key = os.environ.get("SHUBI_API_KEY")
    base_url = os.environ.get("SHUBI_URL")
    
    print("🔑 Testing API Configuration...")
    print(f"API Key: {'✅ Set' if api_key else '❌ Missing'}")
    print(f"Base URL: {'✅ Set' if base_url else '❌ Missing'}")
    
    if not api_key or not base_url:
        print("\n❌ Missing API key or URL. Check .env file")
        return False
    
    return True

def get_available_models():
    """Get list of available models from API"""
    try:
        api_key = os.environ.get("SHUBI_API_KEY")
        base_url = os.environ.get("SHUBI_URL")
        
        models_url = f"{base_url.rstrip('/')}/models"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(models_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            models_data = response.json()
            if "data" in models_data:
                return [model["id"] for model in models_data["data"]]
    except Exception as e:
        print(f"Error getting model list: {e}")
    
    return []

def test_model_connection(model_name):
    """Test if a specific model is working"""
    print(f"Testing {model_name}...", end=" ")
    
    try:
        llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=os.environ.get("SHUBI_API_KEY"),
            base_url=os.environ.get("SHUBI_URL"),
            max_retries=1
        )
        
        # Simple test
        response = llm.invoke([HumanMessage(content="Hello")])
        print("✅ WORKING")
        return True
        
    except Exception as e:
        error_msg = str(e)
        if "503" in error_msg:
            print("❌ Not available (503)")
        elif "404" in error_msg:
            print("❌ Not found (404)")
        elif "401" in error_msg:
            print("❌ Unauthorized (401)")
        elif "429" in error_msg:
            print("⚠️ Rate limited (429)")
        else:
            print(f"❌ Error: {error_msg[:50]}...")
        return False

def test_clickbait_classification(model_name):
    """Test clickbait classification with working model"""
    print(f"\n🎯 Testing clickbait classification with {model_name}...")
    print("=" * 50)
    
    try:
        llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=os.environ.get("SHUBI_API_KEY"),
            base_url=os.environ.get("SHUBI_URL")
        )
        
        # Test prompts
        test_titles = [
            "SHOCK: You won't believe what happened next!",
            "Apple announces iPhone 15 with new features and pricing"
        ]
        
        for title in test_titles:
            print(f"\nTesting: {title}")
            
            prompt = f"""Classify this headline as clickbait (1) or not clickbait (0):
"{title}"

Respond with only 0 or 1:"""
            
            response = llm.invoke([HumanMessage(content=prompt)])
            result = response.content.strip()
            
            label = "CLICKBAIT" if "1" in result else "NOT CLICKBAIT"
            print(f"Result: {result} -> {label}")
        
        print("\n✅ Clickbait classification test successful!")
        print(f"👍 Recommended model: {model_name}")
        return True
        
    except Exception as e:
        print(f"❌ Clickbait test failed: {e}")
        return False

def test_priority_models():
    """Test priority models for clickbait classification"""
    
    # Priority models to test
    priority_models = [
        "claude-3-7-sonnet-20250219",
        "claude-opus-4-20250514", 
        "claude-sonnet-4-20250514",
        "deepseek-r1",
        "deepseek-v3",
        "gemini-1.5-flash",
        "gemini-1.5-flash-002",
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4"
    ]
    
    print(f"\n🧪 Testing {len(priority_models)} priority models...")
    print("=" * 50)
    
    working_models = []
    
    for model in priority_models:
        if test_model_connection(model):
            working_models.append(model)
    
    return working_models

def main():
    print("🚀 UNIFIED SHUBI API TEST")
    print("=" * 50)
    
    # 1. Test basic connection
    if not test_api_connection():
        return
    
    # 2. Get available models from API
    print("\n🔍 Getting available models from API...")
    available_models = get_available_models()
    
    if available_models:
        print(f"✅ Found {len(available_models)} available models")
        print("First 10 models:")
        for i, model in enumerate(available_models[:10]):
            print(f"  {i+1}. {model}")
        if len(available_models) > 10:
            print(f"  ... and {len(available_models) - 10} more")
    else:
        print("❌ Could not get model list from API")
    
    # 3. Test priority models
    working_models = test_priority_models()
    
    # 4. Display results
    print("\n" + "=" * 50)
    print("📊 SUMMARY:")
    print(f"✅ Working models: {len(working_models)}")
    
    if working_models:
        print("Available models:")
        for model in working_models:
            print(f"  - {model}")
        
        # Test clickbait classification with first working model
        test_clickbait_classification(working_models[0])
        
        print(f"\n🏆 RECOMMENDED MODEL: {working_models[0]}")
        print(f"\n💡 To use this model, update your prompting scripts:")
        print(f'   Change model="gpt-4o-mini" to model="{working_models[0]}"')
        
    else:
        print("❌ No working models found!")
        print("\nPossible issues:")
        print("1. Incorrect API key")
        print("2. Incorrect base URL") 
        print("3. No available models in your subscription")
        print("4. Network connectivity issues")
    
    print("\n" + "=" * 50)
    print("🎉 API test completed!")

if __name__ == "__main__":
    main() 