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

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

# BẮT BUỘC: Phải có định nghĩa này ở file Backend
class OutlierClipper(BaseEstimator, TransformerMixin):
    """Học ngưỡng clip từ train, áp dụng cho mọi tập. An toàn với joblib."""
    def __init__(self, cols, lower_q=0.01, upper_q=0.99):
        self.cols    = cols
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, X, y=None):
        # Khi load từ pkl, hàm này không chạy, nhưng class vẫn cần có cấu trúc này
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        if not isinstance(X_, pd.DataFrame):
            X_ = pd.DataFrame(X_)
        # Lưu ý: Khi load từ joblib, self.clip_limits_ đã có sẵn dữ liệu từ lúc train
        for col, (lo, hi) in self.clip_limits_.items():
            if col in X_.columns:
                X_[col] = X_[col].clip(lo, hi)
        return X_

    def get_feature_names_out(self, input_features=None):
        return input_features

MODELS = {}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def get_path(filename):
    return os.path.join(BASE_DIR, filename)

# --- 2. QUẢN LÝ VÒNG ĐỜI (LIFESPAN) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # [STARTUP]: Chạy khi server bắt đầu
    try:
        MODELS['preprocessor_clinical'] = joblib.load(get_path('preprocessor_clinical.pkl'))
        MODELS['model_clinical'] = joblib.load(get_path('model_clinical.pkl'))
        MODELS['clinical_background_processed'] = joblib.load(get_path('clinical_background_processed.pkl'))

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
1. Khẩn cấp: Cần gặp bác sĩ chuyên khoa tiểu đường trong vòng 1 tuần
2. Kiểm tra: Xét nghiệm HbA1c và đường huyết đói ngay
3. Lối sống: Bắt đầu chế độ ăn kiêng ít đường, tăng vận động ngay lập tức
"""
    elif prob > 0.4:
        advice += """
1. Theo dõi: Đặt lịch khám sức khỏe định kỳ 3-6 tháng/lần
2. Phòng ngừa: Điều chỉnh chế độ ăn, tăng vận động 30 phút/ngày
3. Kiểm tra: Theo dõi các chỉ số sức khỏe tại nhà
"""
    else:
        advice += """
1. Duy trì: Tiếp tục lối sống lành mạnh hiện tại
2. Kiểm tra: Khám sức khỏe định kỳ hàng năm
3. Phòng ngừa: Giữ cân nặng ổn định, vận động đều đặn
"""

    advice += "\n\n⚠️ *Lưu ý: Kết quả chỉ mang tính tham khảo. Vui lòng tham khảo ý kiến bác sĩ chuyên khoa.*"

    return advice


async def get_gemini_advice(prob: float, impacts: list, role: str, raw_data: dict) -> str:
    # 1. Tạo bối cảnh dựa trên Role
    if role == "Bác sĩ":
        context = """Bạn là một chuyên gia nội tiết hỗ trợ bác sĩ. 
            Hãy phân tích dữ liệu lâm sàng dưới góc độ chuyên môn y khoa."""

        requirement = """
            Hãy trả lời theo CHÍNH XÁC format sau (mỗi mục trên 1 dòng):

    **Đánh giá nguy cơ**
    - [Nhận định về mức độ nguy cơ]
    - [Giải thích về các chỉ số quan trọng]

    **Khuyến nghị lâm sàng**
    - [Xét nghiệm cần làm thêm]
    - [Hướng điều trị đề xuất]
    - [Theo dõi cần thiết]

    **Lưu ý đặc biệt**
    - [Các chỉ số cần chú ý khẩn cấp nếu có]
            """
    else:
        context = """Bạn là Bác sĩ gia đình hỗ trợ người dùng tại nhà. 
Hãy giải thích kết quả dự đoán tiểu đường một cách dễ hiểu, gần gũi và đầy đủ định lượng."""

        requirement = f"""
        Hãy trả lời theo CHÍNH XÁC format sau (mỗi mục trên 1 dòng):

**Đánh giá sức khỏe**
- [Nhận định nguy cơ dựa trên xác suất]
- [Giải thích ý nghĩa các chỉ số ảnh hưởng chính đến dự đoán]

**Lời khuyên hành động (Cụ thể số liệu)**
- [Chế độ ăn: Ăn gì, bỏ gì, định lượng gram/bữa thế nào?]
- [Vận động: Tập môn gì, bao nhiêu phút/ngày, bao nhiêu ngày/tuần?]
- [Mục tiêu: Cần giảm bao nhiêu kg, đưa chỉ số về mức bao nhiêu?]

**Lưu ý quan trọng**
- [Dấu hiệu cần đi khám ngay hoặc lời nhắc tái khám]

# QUY TẮC CỐ ĐỊNH
1. KHÔNG chào hỏi xã giao. Bắt đầu ngay bằng phần Đánh giá.
2. Mọi lời khuyên PHẢI gắn liền với con số cụ thể của bệnh nhân (BMI, Glucose...).
3. Ít nhất 9-10 gợi ý chi tiết chia đều cho các mục.
                """

    prompt = f"""
    {context}

    **Dữ liệu bệnh nhân:**
    - Nguy cơ tiểu đường: {prob * 100:.1f}%
    - Các yếu tố ảnh hưởng chính: {impacts}
    - Thông tin chi tiết: {raw_data}

    **YÊU CẦU QUAN TRỌNG:**
    {requirement}

    CHÚ Ý: 
    - Mỗi gợi ý phải là 1 câu hoàn chỉnh, cụ thể, có thể thực hiện được
    - Sử dụng dấu - ở đầu mỗi dòng
    - KHÔNG thêm số thứ tự, KHÔNG thêm header phức tạp
    - Trả lời bằng tiếng Việt
    """

    try:

        response = GEMINI_CLIENT.models.generate_content(
            model=GEMINI_MODEL_ID,
            contents=prompt
        )
        print("=" * 50, flush=True)
        print("GEMINI RESPONSE:", flush=True)
        print("=" * 50, flush=True)
        print(response.candidates[0].content.parts[0].text, flush=True)
        print("=" * 50, flush=True)

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
    """Clinical prediction endpoint (Doctor mode) - Chỉ thay đổi máy học, giữ nguyên nghiệp vụ"""
    try:
        # LOGIC NGHIỆP VỤ: Logger giữ nguyên
        logger.info(f"Clinical prediction request: {data.model_dump()}")
        clean_smoking = data.smoking_history.replace('not current', 'not_current').replace('No Info', 'no_info')
        # THAY ĐỔI CÁCH XỬ LÝ: Sử dụng Pipeline tập trung thay vì bóc tách scaler/encoder thủ công
        # Điều này đảm bảo OutlierClipper (nghiệp vụ xử lý ngoại lệ mới) được áp dụng
        raw_input = pd.DataFrame([{
            'gender': data.gender,
            'age': data.age,
            'hypertension': data.hypertension,
            'heart_disease': data.heart_disease,
            'smoking_history': clean_smoking,
            'bmi': data.bmi,
            'HbA1c_level': data.hba1c,  # Map đúng tên cột tập train
            'blood_glucose_level': data.glucose  # Map đúng tên cột tập train
        }])

        # Chạy qua bộ tiền xử lý mới (bao gồm Clipping + Scaling + Encoding)
        processed_data = MODELS['preprocessor_clinical'].transform(raw_input)

        # DỰ ĐOÁN: Sử dụng model AdaBoost mới đã train
        model = MODELS['model_clinical']
        prob = float(model.predict_proba(processed_data)[0][1])

        # LOGIC SHAP: Giữ nguyên cách tính nhưng cập nhật dữ liệu đầu vào
        f = lambda x: model.predict_proba(x)[:, 1]
        # Background đã được xử lý sẵn khi khởi động server để tối ưu tốc độ
        background = MODELS['clinical_background_processed']
        explainer = shap.Explainer(f, background)
        shap_values = explainer(processed_data)

        # NGHIỆP VỤ CŨ: Map tên tiếng Việt cho các tính năng
        impacts = []
        # Lưu ý: Vì dùng OHE nên số lượng feature sau transform sẽ nhiều hơn 8.
        # Chúng ta sẽ map lại theo logic nghiệp vụ hiển thị của bạn.
        feature_names_out = MODELS['preprocessor_clinical'].get_feature_names_out()

        for i, val in enumerate(shap_values.values[0]):
            impacts.append({
                "feature": feature_names_out[i],
                "impact": round(val * 100, 2)
            })

        # NGHIỆP VỤ CŨ: Gọi Gemini tư vấn
        advice = await get_gemini_advice(prob, impacts, "Bác sĩ", data.model_dump())

        # NGHIỆP VỤ CŨ: Loại bỏ Giới tính và Hút thuốc khỏi impacts hiển thị frontend
        # (Giữ nguyên logic excluded_features cũ của bạn)
        excluded_keywords = ["gender", "smoking_history", "Giới tính", "Hút thuốc"]
        frontend_impacts = [
            i for i in impacts
            if not any(key in i["feature"] for key in excluded_keywords)
        ]

        # NGHIỆP VỤ CŨ: Cấu trúc kết quả trả về không đổi
        result = {
            "probability": round(prob * 100, 2),
            # Cập nhật Threshold mới 0.4945 để status chính xác theo model mới
            "status": "DƯƠNG TÍNH" if prob > 0.4945 else "ÂM TÍNH",
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
