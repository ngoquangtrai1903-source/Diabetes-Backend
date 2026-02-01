"""
Script Test Gemini API Connection (Fixed)
Sử dụng thư viện google-genai để đồng bộ với Backend FastAPI
"""

import sys
import os
from dotenv import load_dotenv

# Nạp file .env với override=True để đảm bảo lấy key mới nhất
load_dotenv(override=True)

def test_import():
    """Test 1: Kiểm tra package đã cài đặt"""
    print("\n" + "=" * 60)
    print("TEST 1: Kiểm tra google-genai package")
    print("=" * 60)

    try:
        from google import genai
        print("✅ Package 'google-genai' đã được cài đặt")
        return True
    except ImportError:
        print("❌ Chưa cài đặt 'google-genai'")
        print("👉 Chạy lệnh: pip install google-genai")
        return False

def test_api_key():
    """Test 2: Kiểm tra API key từ .env"""
    print("\n" + "=" * 60)
    print("TEST 2: Kiểm tra nạp API Key từ .env")
    print("=" * 60)

    # LỖI CŨ CỦA BẠN: API_KEY = "GEMINI_API_KEY"
    # SỬA THÀNH:
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("❌ KHÔNG TÌM THẤY GEMINI_API_KEY trong file .env")
        return False, None

    try:
        from google import genai
        # In ra 5 ký tự cuối để bạn đối chiếu xem có đúng Key mới không
        print(f"📝 Đã nạp Key kết thúc bằng: ...{api_key[-5:]}")

        client = genai.Client(api_key=api_key)
        print("✅ Client khởi tạo thành công")
        return True, client
    except Exception as e:
        print(f"❌ Lỗi khởi tạo client: {e}")
        return False, None

def test_models(client):
    """Test 3: Gọi thử các model (Bao gồm cả bản 2.5 bạn yêu cầu)"""
    print("\n" + "=" * 60)
    print("TEST 3: Thử nghiệm các Model ID")
    print("=" * 60)

    # Thêm model 2.5 vào danh sách test nếu bạn đang có quyền truy cập
    models_to_test = [
        "gemini-2.5-flash",
        "gemini-2.0-flash-exp",
        "gemini-1.5-flash"
    ]

    working_model = None

    for model_name in models_to_test:
        print(f"\n🧪 Đang thử model: {model_name}...")
        try:
            response = client.models.generate_content(
                model=model_name,
                contents="Hello, this is a connection test."
            )
            if response.text:
                print(f"✅ Model {model_name} HOẠT ĐỘNG!")
                working_model = model_name
                break
        except Exception as e:
            print(f"❌ Model {model_name} không phản hồi: {e}")

    return working_model

def run_all_tests():
    print("🧬 " + "="*50 + " 🧬")
    print("GEMINI API TEST SUITE FOR DIABETWIN")
    print("🧬 " + "="*50 + " 🧬")

    if not test_import(): return

    api_ok, client = test_api_key()
    if not api_ok: return

    working_model = test_models(client)

    if working_model:
        print("\n" + "=" * 60)
        print(f"🎉 KẾT QUẢ: API HOẠT ĐỘNG VỚI MODEL: {working_model}")
        print("🚀 Bạn có thể quay lại Backend và sửa GEMINI_MODEL_ID")
        print("=" * 60)
    else:
        print("\n❌ TẤT CẢ MODEL ĐỀU THẤT BẠI")
        print("💡 Hãy kiểm tra lại xem Key có bị giới hạn vùng (Region) không.")

if __name__ == "__main__":
    run_all_tests()