"""
Test với các model thực tế có sẵn trên SHUBI API
"""

import os
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# Load environment variables
load_dotenv()

def get_available_models():
    """Lấy danh sách model từ API"""
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
        print(f"Lỗi lấy danh sách model: {e}")
    
    return []

def test_model_for_clickbait(model_name):
    """Test một model với bài toán phân loại clickbait"""
    
    print(f"\n🧪 Testing {model_name}...")
    
    try:
        llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=os.environ.get("SHUBI_API_KEY"),
            base_url=os.environ.get("SHUBI_URL"),
            max_retries=1
        )
        
        # Test prompt đơn giản
        test_prompt = """Phân loại tiêu đề sau thành clickbait (1) hoặc not clickbait (0):
"SHOCK: Sự thật mà 99% người không biết!"

Trả lời chỉ số 0 hoặc 1:"""
        
        response = llm.invoke([HumanMessage(content=test_prompt)])
        result = response.content.strip()
        
        print(f"✅ {model_name}: {result}")
        return True, result
        
    except Exception as e:
        error_msg = str(e)
        if "503" in error_msg:
            print(f"❌ {model_name}: Not available (503)")
        elif "404" in error_msg:
            print(f"❌ {model_name}: Not found (404)")
        elif "401" in error_msg:
            print(f"❌ {model_name}: Unauthorized (401)")
        else:
            print(f"❌ {model_name}: {error_msg[:60]}...")
        return False, None

def main():
    print("🚀 TESTING AVAILABLE MODELS FOR CLICKBAIT CLASSIFICATION")
    print("=" * 70)
    
    # Lấy danh sách models
    available_models = get_available_models()
    
    if not available_models:
        print("❌ Không thể lấy danh sách models")
        return
    
    print(f"📊 Found {len(available_models)} available models")
    
    # Test một số models phổ biến trước
    priority_models = [
        "claude-3-7-sonnet-20250219",
        "claude-opus-4-20250514", 
        "claude-sonnet-4-20250514",
        "deepseek-r1",
        "deepseek-v3",
        "gemini-1.5-flash",
        "gemini-1.5-flash-002"
    ]
    
    working_models = []
    
    print("\n🎯 Testing priority models first:")
    for model in priority_models:
        if model in available_models:
            success, result = test_model_for_clickbait(model)
            if success:
                working_models.append((model, result))
    
    # Nếu không có model nào hoạt động, test thêm một vài models khác
    if not working_models:
        print("\n🔍 Testing more models...")
        for model in available_models[:10]:  # Test 10 models đầu tiên
            if model not in priority_models:
                success, result = test_model_for_clickbait(model)
                if success:
                    working_models.append((model, result))
                    break  # Chỉ cần tìm một model hoạt động
    
    # Hiển thị kết quả
    print("\n" + "=" * 70)
    print("📈 RESULTS:")
    
    if working_models:
        print(f"✅ Found {len(working_models)} working models:")
        for model, result in working_models:
            print(f"  - {model}: Response = '{result}'")
        
        # Recommend model tốt nhất
        recommended = working_models[0][0]
        print(f"\n🏆 RECOMMENDED MODEL: {recommended}")
        
        print(f"\n💡 To use this model, update your prompting_example.py:")
        print(f'   Change model="gpt-4o-mini" to model="{recommended}"')
        
    else:
        print("❌ No working models found for clickbait classification")
        print("\nAll available models:")
        for i, model in enumerate(available_models[:20]):
            print(f"  {i+1}. {model}")
        if len(available_models) > 20:
            print(f"  ... and {len(available_models) - 20} more")

if __name__ == "__main__":
    main() 