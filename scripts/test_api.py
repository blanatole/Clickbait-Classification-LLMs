"""
Script test API key và các model có sẵn trên SHUBI
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
        print("\n❌ Thiếu API key hoặc URL. Kiểm tra file .env")
        return False
    
    return True

def test_models():
    """Test các model phổ biến"""
    
    if not test_api_connection():
        return
    
    # Danh sách các model thường có trên OpenAI-compatible APIs
    models_to_test = [
        "gpt-4o-mini",
        "gpt-4o", 
        "gpt-4",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
        "claude-3-haiku",
        "claude-3-sonnet", 
        "claude-3-opus"
    ]
    
    print(f"\n🧪 Testing {len(models_to_test)} models...")
    print("=" * 50)
    
    working_models = []
    
    for model in models_to_test:
        print(f"Testing {model}...", end=" ")
        
        try:
            llm = ChatOpenAI(
                model=model,
                temperature=0,
                api_key=os.environ.get("SHUBI_API_KEY"),
                base_url=os.environ.get("SHUBI_URL"),
                max_retries=1
            )
            
            # Test với một prompt đơn giản
            response = llm.invoke([HumanMessage(content="Hello")])
            
            print("✅ WORKING")
            working_models.append(model)
            
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
    
    print("\n" + "=" * 50)
    print("📊 SUMMARY:")
    print(f"✅ Working models: {len(working_models)}")
    
    if working_models:
        print("Available models:")
        for model in working_models:
            print(f"  - {model}")
        
        # Test phân loại clickbait với model đầu tiên
        test_clickbait_classification(working_models[0])
    else:
        print("❌ No working models found!")
        print("\nPossible issues:")
        print("1. Incorrect API key")
        print("2. Incorrect base URL") 
        print("3. No available models in your subscription")
        print("4. Network connectivity issues")

def test_clickbait_classification(model_name):
    """Test phân loại clickbait với model working"""
    
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
            "SHOCK: Bạn sẽ không tin điều này!",
            "Apple công bố iPhone 15 với giá 999 USD"
        ]
        
        for title in test_titles:
            print(f"\nTesting: {title}")
            
            prompt = f"""Phân loại tiêu đề sau thành clickbait (1) hoặc not clickbait (0):
"{title}"

Trả lời chỉ số 0 hoặc 1:"""
            
            response = llm.invoke([HumanMessage(content=prompt)])
            result = response.content.strip()
            
            label = "CLICKBAIT" if "1" in result else "NOT CLICKBAIT"
            print(f"Result: {result} -> {label}")
        
        print("\n✅ Clickbait classification test successful!")
        print(f"👍 Recommended model: {model_name}")
        
    except Exception as e:
        print(f"❌ Clickbait test failed: {e}")

def get_available_models_api():
    """Thử lấy danh sách model từ API endpoint"""
    
    print("\n🔍 Trying to get available models from API...")
    
    try:
        api_key = os.environ.get("SHUBI_API_KEY")
        base_url = os.environ.get("SHUBI_URL")
        
        # Thử endpoint /models (OpenAI standard)
        models_url = f"{base_url.rstrip('/')}/models"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(models_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            models_data = response.json()
            
            if "data" in models_data:
                models = [model["id"] for model in models_data["data"]]
                print(f"✅ Found {len(models)} available models:")
                for model in models[:10]:  # Show first 10
                    print(f"  - {model}")
                if len(models) > 10:
                    print(f"  ... and {len(models) - 10} more")
                return models
            else:
                print("⚠️ Unexpected API response format")
        else:
            print(f"❌ API request failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Failed to get models: {e}")
    
    return []

def main():
    print("🚀 SHUBI API CONNECTION TEST")
    print("=" * 50)
    
    # 1. Test basic connection
    if not test_api_connection():
        return
    
    # 2. Try to get available models from API
    available_models = get_available_models_api()
    
    # 3. Test common models
    test_models()
    
    print("\n" + "=" * 50)
    print("🎉 API test completed!")

if __name__ == "__main__":
    main() 