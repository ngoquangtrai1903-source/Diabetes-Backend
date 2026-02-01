import joblib
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import logging
import os
from dotenv import load_dotenv
load_dotenv(override=True)
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. CẤU HÌNH AI & BIẾN TOÀN CỤC ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_ID = "gemini-2.5-flash"  # Sử dụng model mới nhất
GEMINI_CLIENT = None
print(f"DEBUG: Key đang dùng là: {GEMINI_API_KEY[:10]}...")

# Khởi tạo Gemini client với error handling
def init_gemini_client():
    """Khởi tạo Gemini client với xử lý lỗi"""
    global GEMINI_CLIENT
    try:
        from google import genai
        GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("✅ Gemini API client initialized successfully")
        return True
    except ImportError:
        logger.error("❌ google-generativeai package not installed")
        logger.info("Install with: pip install google-generativeai")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to initialize Gemini client: {e}")
        return False


MODELS = {}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def get_path(filename):
    return os.path.join(BASE_DIR, filename)

# --- 2. QUẢN LÝ VÒNG ĐỜI (LIFESPAN) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # [STARTUP]: Chạy khi server bắt đầu
    try:
        MODELS['clinical_model'] = joblib.load(get_path('diabetes_model.pkl'))
        MODELS['clinical_scaler'] = joblib.load(get_path('scaler_diabetes.pkl'))
        MODELS['clinical_encoders'] = joblib.load(get_path('label_encoders.pkl'))
        MODELS['clinical_background'] = joblib.load(get_path('x_train_sample.pkl'))

        # Load ML models cho Người dùng (Home)
        MODELS['home_model'] = joblib.load(get_path('diabetes_model_home.pkl'))
        MODELS['home_background'] = joblib.load(get_path('x_train_sample_home.pkl'))

        logger.info("✅ [Lifespan] Đã tải tất cả ML Models thành công!")

        # Initialize Gemini (non-blocking)
        gemini_ok = init_gemini_client()
        if not gemini_ok:
            logger.warning("⚠️ Gemini API không khả dụng - sẽ sử dụng AI advice mặc định")

    except Exception as e:
        logger.error(f"❌ [Lifespan] Lỗi tải model: {e}")
        raise

    yield
    MODELS.clear()
    logger.info("🧹 [Lifespan] Đã giải phóng bộ nhớ.")


# --- 3. KHỞI TẠO APP & CORS ---
app = FastAPI(title="DiabeTwin AI Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://v0-frontend-ai-one.vercel.app",
                  "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 4. SCHEMA DỮ LIỆU ---
class ClinicalInput(BaseModel):
    gender: str
    age: int
    smoking_history: str
    hypertension: int
    heart_disease: int
    bmi: float
    hba1c: float
    glucose: int


class HomeInput(BaseModel):
    HighBP: int
    HighChol: int
    CholCheck: int
    BMI: float
    Smoker: int
    Stroke: int
    HeartDiseaseorAttack: int
    PhysActivity: int
    Fruits: int
    Veggies: int
    HvyAlcoholConsump: int
    GenHlth: int
    MentHlth: int
    PhysHlth: int
    DiffWalk: int
    Sex: int
    Age: int


# --- 5. HELPER FUNCTIONS ---

def generate_fallback_advice(prob: float, impacts: list, role: str, raw_data: dict) -> str:
    """Tạo lời khuyên mặc định khi Gemini không khả dụng"""

    top_3 = sorted(impacts, key=lambda x: abs(x['impact']), reverse=True)[:3]
    risk_level = "cao" if prob > 0.7 else "trung bình" if prob > 0.4 else "thấp"

    advice = f"""**Phân tích nguy cơ tiểu đường**

🎯 **Kết quả:** Nguy cơ ở mức **{risk_level}** với xác suất {prob * 100:.1f}%

📊 **3 Yếu tố ảnh hưởng lớn nhất:**
"""

    for i, impact in enumerate(top_3, 1):
        direction = "tăng" if impact['impact'] > 0 else "giảm"
        advice += f"\n{i}. **{impact['feature']}**: {direction} {abs(impact['impact']):.1f}% nguy cơ"

    advice += "\n\n💡 **Khuyến nghị:**\n"

    # Khuyến nghị dựa trên risk level
    if prob > 0.7:
        advice += """
1. **Khẩn cấp:** Cần gặp bác sĩ chuyên khoa tiểu đường trong vòng 1 tuần
2. **Kiểm tra:** Xét nghiệm HbA1c và đường huyết đói ngay
3. **Lối sống:** Bắt đầu chế độ ăn kiêng ít đường, tăng vận động ngay lập tức
"""
    elif prob > 0.4:
        advice += """
1. **Theo dõi:** Đặt lịch khám sức khỏe định kỳ 3-6 tháng/lần
2. **Phòng ngừa:** Điều chỉnh chế độ ăn, tăng vận động 30 phút/ngày
3. **Kiểm tra:** Theo dõi các chỉ số sức khỏe tại nhà
"""
    else:
        advice += """
1. **Duy trì:** Tiếp tục lối sống lành mạnh hiện tại
2. **Kiểm tra:** Khám sức khỏe định kỳ hàng năm
3. **Phòng ngừa:** Giữ cân nặng ổn định, vận động đều đặn
"""

    advice += "\n\n⚠️ *Lưu ý: Kết quả chỉ mang tính tham khảo. Vui lòng tham khảo ý kiến bác sĩ chuyên khoa.*"

    return advice


async def get_gemini_advice(prob: float, impacts: list, role: str, raw_data: dict) -> str:
    # 1. Tạo bối cảnh dựa trên Role
    if role == "Bác sĩ":
        context = """Bạn là một chuyên gia nội tiết hỗ trợ bác sĩ. 
        Hãy phân tích dữ liệu lâm sàng dưới góc độ chuyên môn y khoa, sử dụng thuật ngữ y học."""
        requirement = "Hãy đưa ra nhận định chuyên môn ngắn gọn nhất có thể, lưu ý các chỉ số nguy hiểm và gợi ý hướng điều trị/xét nghiệm tiếp theo."
    else:
        context = """Bạn là một bác sĩ gia đình ảo thân thiện. 
        Hãy giải thích kết quả sàng lọc cho người dùng bình thường bằng ngôn ngữ dễ hiểu, gần gũi."""
        requirement = "Hãy giải thích các chỉ số một cách đơn giản và đưa ra 3-4 lời khuyên thay đổi lối sống, thực đơn ăn uống cụ thể."

    # 2. Xây dựng Prompt tổng hợp
    prompt = f"""
    {context}

    **Kết quả dự đoán:**
    - Nguy cơ: {prob * 100:.1f}%
    - Các yếu tố ảnh hưởng (SHAP): {impacts}
    - Dữ liệu chi tiết: {raw_data}

    **Yêu cầu:**
    {requirement}

    Trả lời bằng tiếng Việt, định dạng Markdown rõ ràng.
    """

    try:
        # Gọi Gemini API (giữ nguyên logic đã chạy OK của bạn)
        response = GEMINI_CLIENT.models.generate_content(
            model=GEMINI_MODEL_ID,
            contents=prompt
        )
        return response.candidates[0].content.parts[0].text
    except Exception as e:
        logger.error(f"Error: {e}")
        return "Xin lỗi, tôi không thể đưa ra lời khuyên lúc này."

# --- 6. ENDPOINTS ---

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(MODELS) > 0,
        "gemini_available": GEMINI_CLIENT is not None
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "DiabeTwin AI Backend",
        "version": "2.0",
        "endpoints": {
            "health": "/health",
            "clinical": "/api/predict/clinical",
            "home": "/api/predict/home",
            "docs": "/docs"
        }
    }


@app.post("/api/predict/clinical")
async def predict_clinical(data: ClinicalInput):
    """Clinical prediction endpoint (Doctor mode)"""
    try:
        logger.info(f"Clinical prediction request: {data.model_dump()}")

        encoders = MODELS['clinical_encoders']
        scaler = MODELS['clinical_scaler']

        input_list = [
            encoders['gender'].transform([data.gender])[0],
            data.age, data.hypertension, data.heart_disease,
            encoders['smoking_history'].transform([data.smoking_history])[0],
            data.bmi, data.hba1c, data.glucose
        ]

        df = pd.DataFrame([input_list], columns=MODELS['clinical_background'].columns)
        scaled_df = pd.DataFrame(scaler.transform(df), columns=df.columns)

        prob = float(MODELS['clinical_model'].predict_proba(scaled_df)[0][1])

        # SHAP Calculation
        f = lambda x: MODELS['clinical_model'].predict_proba(x)[:, 1]
        background = scaler.transform(MODELS['clinical_background'].sample(100))
        explainer = shap.Explainer(f, background)
        shap_values = explainer(scaled_df)

        impacts = []
        features = ["Giới tính", "Tuổi", "Huyết áp", "Bệnh tim", "Hút thuốc", "BMI", "HbA1c", "Đường huyết"]
        for i, val in enumerate(shap_values.values[0]):
            impacts.append({"feature": features[i], "impact": round(val * 100, 2)})

        advice = await get_gemini_advice(prob, impacts, "Bác sĩ", data.model_dump())

        excluded_features = ["Giới tính", "Hút thuốc"]
        frontend_impacts = [i for i in impacts if i["feature"] not in excluded_features]

        result = {
            "probability": round(prob * 100, 2),
            "status": "DƯƠNG TÍNH" if prob > 0.5 else "ÂM TÍNH",
            "risk_level": "🔴" if prob > 0.7 else "🟡" if prob > 0.3 else "🟢",
            "impacts": frontend_impacts,
            "ai_advice": advice
        }

        logger.info(f"Clinical prediction successful: {prob * 100:.1f}%")
        return result

    except Exception as e:
        logger.error(f"Clinical prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/home")
async def predict_home(data: HomeInput):
    """Home prediction endpoint (User mode)"""
    try:
        logger.info(f"Home prediction request received")

        df = pd.DataFrame([data.dict()])
        prob = float(MODELS['home_model'].predict_proba(df)[0][1])

        f = lambda x: MODELS['home_model'].predict_proba(x)[:, 1]
        explainer = shap.Explainer(f, MODELS['home_background'])
        shap_values = explainer(df)

        display_names = [
            "Huyết áp cao", "Cholesterol cao", "Kiểm tra Chol", "Chỉ số BMI",
            "Hút thuốc", "Đột quỵ", "Bệnh tim", "Vận động", "Trái cây",
            "Rau xanh", "Rượu bia", "Sức khỏe tổng quát", "Sức khỏe tâm thần",
            "Sức khỏe thể chất", "Đi lại khó", "Giới tính", "Nhóm tuổi"
        ]

        impacts = []
        for i, val in enumerate(shap_values.values[0]):
            impacts.append({"feature": display_names[i], "impact": round(val * 100, 2)})

        advice = await get_gemini_advice(prob, impacts, "Người dùng", data.model_dump())
        excluded_features_home = ["Giới tính", "Rượu bia", "Sức khỏe tâm thần", "Hút thuốc", "Trái cây", "Rau xanh"]
        frontend_impacts_home = [i for i in impacts if i["feature"] not in excluded_features_home]
        result = {
            "probability": round(prob * 100, 2),
            "status": "NGUY CƠ CAO" if prob > 0.5 else "AN TOÀN",
            "impacts": frontend_impacts_home,
            "ai_advice": advice
        }

        logger.info(f"Home prediction successful: {prob * 100:.1f}%")
        return result

    except Exception as e:
        logger.error(f"Home prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
