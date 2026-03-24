"""
Script Test Backend API - DiabeTwin
Chạy script này để kiểm tra backend hoạt động đúng
"""

import requests
import json
from typing import Dict, Any

# Cấu hình
API_BASE_URL = "http://localhost:8000"


def test_health_endpoint():
    """Test health check endpoint"""
    print("\n" + "=" * 60)
    print("TEST 1: Health Check")
    print("=" * 60)

    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        print(f"✅ Status Code: {response.status_code}")
        print(f"✅ Response: {response.json()}")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("❌ Không thể kết nối đến backend!")
        print("   Hãy chạy: python -m uvicorn clinical-input:app --reload")
        return False
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return False


def test_clinical_prediction():
    """Test clinical prediction endpoint (Doctor mode)"""
    print("\n" + "=" * 60)
    print("TEST 2: Clinical Prediction (Doctor Mode)")
    print("=" * 60)

    test_data = {
        "gender": "Male",
        "age": 45,
        "smoking_history": "never",
        "hypertension": 1,
        "heart_disease": 0,
        "bmi": 25.5,
        "hba1c": 5.8,
        "glucose": 110
    }

    print("📤 Sending data:")
    print(json.dumps(test_data, indent=2))

    try:
        response = requests.post(
            f"{API_BASE_URL}/api/predict/clinical",
            json=test_data,
            timeout=30
        )

        print(f"\n✅ Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("\n📥 Response:")
            print(f"  - Probability: {result['probability']}%")
            print(f"  - Status: {result['status']}")
            print(f"  - Risk Level: {result['risk_level']}")
            print(f"  - Number of impacts: {len(result['impacts'])}")
            print(f"  - AI Advice length: {len(result['ai_advice'])} chars")

            print("\n🔍 Top 3 SHAP Impacts:")
            for impact in result['impacts'][:3]:
                print(f"  - {impact['feature']}: {impact['impact']}%")

            return True
        else:
            print(f"❌ Error: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return False


def test_home_prediction():
    """Test home prediction endpoint (User mode)"""
    print("\n" + "=" * 60)
    print("TEST 3: Home Prediction (User Mode)")
    print("=" * 60)

    test_data = {
        "HighBP": 1,
        "HighChol": 1,
        "CholCheck": 1,
        "BMI": 25.5,
        "Smoker": 0,
        "Stroke": 0,
        "HeartDiseaseorAttack": 0,
        "PhysActivity": 1,
        "Fruits": 1,
        "Veggies": 1,
        "HvyAlcoholConsump": 0,
        "GenHlth": 2,
        "MentHlth": 0,
        "PhysHlth": 0,
        "DiffWalk": 0,
        "Sex": 1,
        "Age": 9
    }

    print("📤 Sending data:")
    print(json.dumps(test_data, indent=2))

    try:
        response = requests.post(
            f"{API_BASE_URL}/api/predict/home",
            json=test_data,
            timeout=30
        )

        print(f"\n✅ Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("\n📥 Response:")
            print(f"  - Probability: {result['probability']}%")
            print(f"  - Status: {result['status']}")
            print(f"  - Number of impacts: {len(result['impacts'])}")
            print(f"  - AI Advice length: {len(result['ai_advice'])} chars")

            print("\n🔍 Top 3 SHAP Impacts:")
            for impact in result['impacts'][:3]:
                print(f"  - {impact['feature']}: {impact['impact']}%")

            return True
        else:
            print(f"❌ Error: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return False


def test_cors():
    """Test CORS headers"""
    print("\n" + "=" * 60)
    print("TEST 4: CORS Configuration")
    print("=" * 60)

    try:
        response = requests.options(
            f"{API_BASE_URL}/api/predict/clinical",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type"
            },
            timeout=5
        )

        print(f"✅ Status Code: {response.status_code}")

        cors_headers = {
            "Access-Control-Allow-Origin": response.headers.get("Access-Control-Allow-Origin"),
            "Access-Control-Allow-Methods": response.headers.get("Access-Control-Allow-Methods"),
            "Access-Control-Allow-Headers": response.headers.get("Access-Control-Allow-Headers")
        }

        print("\n📋 CORS Headers:")
        for key, value in cors_headers.items():
            print(f"  - {key}: {value}")

        if cors_headers["Access-Control-Allow-Origin"]:
            print("\n✅ CORS đã được cấu hình đúng!")
            return True
        else:
            print("\n❌ CORS chưa được cấu hình!")
            return False

    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return False


def run_all_tests():
    """Chạy tất cả tests"""
    print("\n" + "🚀" * 30)
    print("DIABETWIN BACKEND API TEST SUITE")
    print("🚀" * 30)

    results = {
        "Health Check": test_health_endpoint(),
        "Clinical Prediction": False,
        "Home Prediction": False,
        "CORS": False
    }

    # Chỉ chạy các test còn lại nếu health check pass
    if results["Health Check"]:
        results["Clinical Prediction"] = test_clinical_prediction()
        results["Home Prediction"] = test_home_prediction()
        results["CORS"] = test_cors()

    # Tổng kết
    print("\n" + "=" * 60)
    print("📊 KẾT QUẢ TỔNG HỢP")
    print("=" * 60)

    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")

    print("\n" + "=" * 60)
    print(f"Tổng số tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success rate: {(passed_tests / total_tests) * 100:.1f}%")
    print("=" * 60)

    if passed_tests == total_tests:
        print("\n🎉 TẤT CẢ TESTS ĐỀU PASS! Backend hoạt động hoàn hảo!")
        print("👉 Bây giờ bạn có thể chạy frontend và test trên UI")
    else:
        print("\n⚠️ CÓ TESTS FAIL! Hãy kiểm tra:")
        if not results["Health Check"]:
            print("  - Backend có đang chạy không?")
            print("  - Chạy: python -m uvicorn clinical-input:app --reload")
        if not results["CORS"]:
            print("  - CORS middleware có được cấu hình đúng không?")
        if not results["Clinical Prediction"] or not results["Home Prediction"]:
            print("  - Model files có tồn tại không?")
            print("  - Dependencies có đầy đủ không?")


if __name__ == "__main__":
    run_all_tests()