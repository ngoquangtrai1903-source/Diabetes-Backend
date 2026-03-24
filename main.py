"""
DiabeTwin AI Backend v3.2
Luồng: ML → SHAP → RAG (Firestore Vector Search) → LLM (OpenRouter/Gemini)

Fixes so với v3.1:
- Thêm fetch_benchmark_doc(): lấy trực tiếp Registry 2026 bằng document ID cứng
  → Bước Validate/Critique luôn có đủ Thresholds & Mechanisms từ đúng document
- synthesize_advice() nhận thêm benchmark_doc (str) tách riêng khỏi context_docs
- System prompt mới: 3 bước suy luận Verify → Critique → Synthesis
- User prompt mới: output phân tầng [PHÂN TÍCH ĐỐI CHIẾU] / [CƠ CHẾ Y SINH] /
  [HƯỚNG XỬ TRÍ] cho bác sĩ; [KIỂM CHỨNG SỨC KHỎE] / [ĐÁNH GIÁ DỰ ĐOÁN] /
  [KẾ HOẠCH HÀNH ĐỘNG] cho người dùng
- Safety guardrail: LLM bắt buộc cảnh báo khi phát hiện mâu thuẫn ML vs RAG
- retrieve_context top_k tăng lên 5 cho phần khuyến nghị
- audience_type luôn được kiểm tra nghiêm ngặt trước khi truyền vào Firestore
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import joblib
import pandas as pd
import shap
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# CẤU HÌNH
# ==============================================================================

OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
FIREBASE_SERVICE_ACCOUNT_PATH: str = os.getenv(
    "FIREBASE_SERVICE_ACCOUNT_PATH", "service-account.json"
)
PROJECT_ID: str = os.getenv("PROJECT_ID", "medical-database-1da52")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = "openai/text-embedding-3-small"
LLM_MODEL = "google/gemini-2.0-flash-001"
FIRESTORE_COLLECTION = "diabetes_knowledge_base"

# ID cứng của 2 document Registry 2026 dùng cho bước Benchmark Validation
# (fetch trực tiếp, không qua vector search để đảm bảo luôn lấy đúng document)
BENCHMARK_DOC_ID_DOCTOR = "83040592-d71f-4dd7-85b3-fd382a8262e8"
BENCHMARK_DOC_ID_USER   = "ba39492f-5641-47d5-a3e2-c3be0d6a0058"

# Map tên kỹ thuật preprocessor → tên tiếng Việt cho RAG query
CLINICAL_FEATURE_VI_MAP = {
    "num__blood_glucose_level": "đường huyết",
    "num__HbA1c_level": "HbA1c",
    "num__age": "tuổi",
    "num__bmi": "chỉ số BMI",
    "num__hypertension": "huyết áp",
    "num__heart_disease": "bệnh tim",
    "cat__gender_Female": "giới tính nữ",
    "cat__gender_Male": "giới tính nam",
    "cat__smoking_history_current": "hút thuốc",
    "cat__smoking_history_ever": "hút thuốc",
    "cat__smoking_history_former": "hút thuốc",
    "cat__smoking_history_never": "không hút thuốc",
    "cat__smoking_history_not_current": "không hút thuốc",
    "cat__smoking_history_no_info": "không rõ hút thuốc",
}

MODELS: dict = {}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(filename: str) -> str:
    return os.path.join(BASE_DIR, filename)


# ==============================================================================
# RAG SERVICE  (async-native với httpx)
# ==============================================================================

class RAGService:
    """
    Xử lý toàn bộ pipeline RAG:
      1. Embedding  → httpx.AsyncClient  → OpenRouter /embeddings
      2. Retrieval  → Firestore find_nearest  (sync trong executor)
      3. Synthesis  → httpx.AsyncClient  → OpenRouter /chat/completions
    """

    def __init__(self) -> None:
        self._http: Optional[httpx.AsyncClient] = None   # dùng chung cho embed + chat
        self._db = None                                   # Firestore client (sync)
        self.initialized = False

    # ── Khởi tạo ──────────────────────────────────────────────────────────────

    def initialize(self) -> bool:
        """Gọi trong lifespan (sync context). Trả True nếu cả hai kết nối OK."""
        fs_ok = self._init_firestore()
        if not OPENROUTER_API_KEY:
            logger.error("❌ OPENROUTER_API_KEY chưa được thiết lập.")
            self.initialized = False
            return False
        # httpx.AsyncClient tạo lazy trong coroutine đầu tiên
        self.initialized = fs_ok
        status = "✅" if self.initialized else "⚠️"
        logger.info(f"{status} RAGService initialized (Firestore={fs_ok})")
        return self.initialized

    def _init_firestore(self) -> bool:
        try:
            from google.cloud import firestore
            from google.oauth2 import service_account

            if not os.path.exists(FIREBASE_SERVICE_ACCOUNT_PATH):
                logger.error(f"❌ Không tìm thấy service account: {FIREBASE_SERVICE_ACCOUNT_PATH}")
                return False
            cred = service_account.Credentials.from_service_account_file(
                FIREBASE_SERVICE_ACCOUNT_PATH
            )
            self._db = firestore.Client(credentials=cred, project=PROJECT_ID)
            logger.info("✅ Firestore client initialized.")
            return True
        except Exception as e:
            logger.error(f"❌ Firestore init failed: {e}")
            return False

    def _get_http_client(self) -> httpx.AsyncClient:
        """Tạo AsyncClient lazy (phải gọi trong coroutine)."""
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                base_url=OPENROUTER_BASE_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                timeout=60.0,
            )
        return self._http

    async def close(self) -> None:
        if self._http and not self._http.is_closed:
            await self._http.aclose()

    # ── BƯỚC 3a: Embedding (async native) ─────────────────────────────────────

    async def get_embedding(self, text: str) -> list[float]:
        client = self._get_http_client()
        payload = {
            "model": EMBEDDING_MODEL,
            "input": [text.replace("\n", " ")],
        }
        resp = await client.post("/embeddings", json=payload)
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    # ── BƯỚC 3b: Vector Search Firestore (sync → executor) ────────────────────

    async def retrieve_context(
        self,
        query_text: str,
        audience_type: str,   # "doctor" | "user"
        top_k: int = 4,
    ) -> list[dict]:
        """
        Trả về list[{"index": int, "content": str, "header": str}]
        audience_type:
          - clinical endpoint → "doctor"
          - home endpoint     → "user"
        """
        query_vector = await self.get_embedding(query_text)

        def _sync_search() -> list[dict]:
            from google.cloud.firestore_v1.base_query import FieldFilter
            from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
            from google.cloud.firestore_v1.vector import Vector

            collection = self._db.collection(FIRESTORE_COLLECTION)
            results = (
                collection
                .where(filter=FieldFilter("audience", "==", audience_type))
                .find_nearest(
                    vector_field="embedding",
                    query_vector=Vector(query_vector),
                    distance_measure=DistanceMeasure.COSINE,
                    limit=top_k,
                )
                .get()
            )
            docs = []
            for i, doc in enumerate(results, start=1):
                data = doc.to_dict()
                content = data.get("content", "")
                metadata = data.get("metadata", {})
                header = (
                        metadata.get("Header_3") or
                        metadata.get("Header_2") or
                        metadata.get("Header_1") or
                        "Tài liệu tổng hợp"
                )
                logger.info(
                    f"  [RAG][{audience_type}] Đoạn {i} ({header}): {content[:80]}..."
                )
                docs.append({"index": i, "content": content, "header": header})
            return docs

        return await asyncio.get_event_loop().run_in_executor(None, _sync_search)

    # ── BƯỚC 3c: Fetch Benchmark Document (direct ID, sync → executor) ────────

    async def fetch_benchmark_doc(self, audience_type: str) -> str:
        """
        Lấy trực tiếp document Registry 2026 theo ID cứng từ Firestore.
        Dùng cho bước Benchmark Validation trong synthesize_advice.
        audience_type: "doctor" → BENCHMARK_DOC_ID_DOCTOR
                       "user"   → BENCHMARK_DOC_ID_USER
        Trả về nội dung (str) hoặc chuỗi rỗng nếu lỗi.
        """
        doc_id = (
            BENCHMARK_DOC_ID_DOCTOR if audience_type == "doctor"
            else BENCHMARK_DOC_ID_USER
        )

        def _sync_fetch() -> str:
            try:
                doc_ref = self._db.collection(FIRESTORE_COLLECTION).document(doc_id)
                doc = doc_ref.get()
                if not doc.exists:
                    logger.warning(f"⚠️ [Benchmark] Document {doc_id} không tồn tại.")
                    return ""
                content = doc.to_dict().get("content", "")
                logger.info(
                    f"✅ [Benchmark][{audience_type}] Đã lấy Registry 2026 "
                    f"({len(content)} ký tự)"
                )
                return content
            except Exception as e:
                logger.error(f"❌ [Benchmark] Fetch failed ({doc_id}): {e}")
                return ""

        return await asyncio.get_event_loop().run_in_executor(None, _sync_fetch)

    # ── BƯỚC 4: LLM Synthesis (async native) ──────────────────────────────────

    async def synthesize_advice(
        self,
        prob: float,
        impacts: list[dict],
        role: str,          # "Bác sĩ" | "Người dùng"
        raw_data: dict,
        context_docs: list[dict],
        benchmark_doc: str = "",   # ← Nội dung Registry 2026 để Validate/Critique
    ) -> str:
        """
        Tổng hợp lời khuyên theo quy trình 3 bước:
          Bước 1 – Verify:   Đối chiếu chỉ số bệnh nhân vs Thresholds trong benchmark_doc
          Bước 2 – Critique: Đánh giá tính hợp lý của ML prob & SHAP impacts vs RAG
          Bước 3 – Synthesis: Đưa ra khuyến nghị dựa trên context_docs (vector search)
        """

        # ── Chuẩn bị các khối ngữ cảnh ──────────────────────────────────────────

        # Benchmark block: Registry 2026 dùng cho Bước 1 & 2 (Verify + Critique)
        benchmark_block = (
            benchmark_doc.strip()
            if benchmark_doc.strip()
            else "Không có dữ liệu Registry 2026 được tải."
        )

        # Context block: kết quả vector search dùng cho Bước 3 (Synthesis)
        context_block = (
            "\n\n".join(
                f"[Đoạn {d['index']}] (Nguồn: {d['header']}):\n{d['content']}"
                for d in context_docs
            )
            if context_docs
            else "Không có ngữ cảnh y khoa bổ sung được truy xuất."
        )

        is_doctor = (role == "Bác sĩ")

        # ── System Prompt: định nghĩa vai trò & quy trình 3 bước ─────────────
        system_prompt = f"""Bạn là {'chuyên gia nội tiết hỗ trợ bác sĩ lâm sàng' if is_doctor else 'Bác sĩ gia đình hỗ trợ người dùng phổ thông'}.

Bạn nhận được 2 nguồn dữ liệu y khoa riêng biệt:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[REGISTRY 2026 – BENCHMARK CHUẨN]
(Dùng cho Bước 1 & 2: Đối chiếu ngưỡng & Phê bình ML)
{benchmark_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[NGỮ CẢNH BỔ SUNG – VECTOR SEARCH]
(Dùng cho Bước 3: Tổng hợp khuyến nghị chi tiết)
{context_block}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUY TRÌNH BẮT BUỘC – 3 BƯỚC SUY LUẬN:

BƯỚC 1 – ĐỐI CHIẾU BENCHMARK (Verify):
  • Lấy từng chỉ số của bệnh nhân (BMI, Glucose, HbA1c, tuổi, huyết áp...)
  • So sánh trực tiếp với các ngưỡng (Thresholds) trong [REGISTRY 2026]
  • Ghi rõ chỉ số nào VƯỢT NGƯỠNG, chỉ số nào trong giới hạn bình thường

BƯỚC 2 – PHÊ BÌNH MODEL ML & SHAP (Critique):
  • Đánh giá xác suất ML có HỢP LÝ so với kết quả Bước 1 không?
    – Nếu ML nói nguy cơ cao nhưng các chỉ số đều bình thường: chỉ ra mâu thuẫn
    – Nếu ML nói nguy cơ thấp nhưng có chỉ số vượt ngưỡng: cảnh báo bổ sung
  • Đánh giá các yếu tố SHAP hàng đầu có khớp với cơ chế y sinh trong [REGISTRY 2026] không?
  ⚠️ SAFETY GUARDRAIL: Nếu phát hiện ML dự đoán KHÔNG TƯƠNG XỨNG với dữ liệu lâm sàng,
     BẮT BUỘC ghi: "⚠️ Lưu ý: Kết quả dự báo kỹ thuật có dấu hiệu chưa tương xứng
     với dữ liệu lâm sàng hiện tại, cần ưu tiên kiểm tra lại [tên chỉ số cụ thể]."

BƯỚC 3 – TỔNG HỢP TƯ VẤN (Synthesis):
  • Sau khi đã xác minh độ tin cậy, đưa ra khuyến nghị dựa trên [NGỮ CẢNH BỔ SUNG]
  • Mọi lời khuyên phải gắn với số liệu thực tế của bệnh nhân
  • Trích dẫn nguồn bằng [Đoạn N] khi dùng thông tin từ NGỮ CẢNH BỔ SUNG

QUY TẮC CHUNG:
  • KHÔNG chào hỏi xã giao, bắt đầu trả lời ngay
  • Nếu Registry không đề cập đến một vấn đề: ghi "Dữ liệu hiện tại không đề cập đến vấn đề này."
  • Độ chính xác > Độ dài: ưu tiên thông tin có căn cứ từ RAG
"""

        # ── User Prompt: định nghĩa output format theo vai trò ────────────────
        if is_doctor:
            user_prompt = f"""DỮ LIỆU ĐẦU VÀO:
- Xác suất tiểu đường (ML): {prob * 100:.1f}%
- Top SHAP impacts: {impacts}
- Thông tin lâm sàng chi tiết: {raw_data}

Hãy trả lời theo đúng 3 section sau (KHÔNG thêm section khác):

---
**[PHÂN TÍCH ĐỐI CHIẾU LÂM SÀNG]**
*(Bước 1 + 2: Verify & Critique)*
- Liệt kê từng chỉ số bệnh nhân và so sánh với ngưỡng trong Registry 2026
- Đánh giá tính hợp lý của xác suất ML {prob * 100:.1f}% dựa trên kết quả đối chiếu
- Đánh giá các yếu tố SHAP hàng đầu có nhất quán với cơ chế y sinh trong Registry không?
- Nếu phát hiện mâu thuẫn: ghi cảnh báo Safety Guardrail bắt buộc

**[CƠ CHẾ Y SINH]**
*(Giải thích cơ chế tại sao các yếu tố SHAP hàng đầu dẫn đến kết quả này)*
- Dựa trên cơ chế trong Registry 2026 và NGỮ CẢNH BỔ SUNG

**[HƯỚNG XỬ TRÍ]**
*(Bước 3: Synthesis – khuyến nghị lâm sàng chuyên sâu dựa trên vector search)*
- Xét nghiệm bổ sung cần chỉ định (với lý do cụ thể)
- Hướng điều trị / can thiệp đề xuất
- Lịch theo dõi và các mốc tái đánh giá
- Lưu ý khẩn cấp nếu có chỉ số nguy hiểm
"""
        else:
            user_prompt = f"""DỮ LIỆU ĐẦU VÀO:
- Xác suất tiểu đường (ML): {prob * 100:.1f}%
- Các yếu tố ảnh hưởng chính (SHAP): {impacts}
- Thông tin sức khỏe: {raw_data}

QUY TẮC BỔ SUNG:
1. Dùng ngôn ngữ đơn giản, dễ hiểu cho người không có chuyên môn y tế
2. Mỗi số liệu phải được giải thích ý nghĩa (ví dụ: "BMI 28 của bạn tương đương Béo phì độ I theo chuẩn Châu Á")
3. Tối thiểu 3 lời khuyên cụ thể có số liệu cho mỗi mục hành động

Hãy trả lời theo đúng 3 section sau (KHÔNG thêm section khác):

---
**[KIỂM CHỨNG SỨC KHỎE]**
*(Bước 1 + 2: Giải thích đơn giản các chỉ số & đánh giá kết quả AI)*
- Giải thích từng chỉ số của bạn theo ngưỡng chuẩn trong Registry 2026 (dùng ngôn ngữ thân thiện)
- Xác nhận kết quả dự đoán {prob * 100:.1f}% có phù hợp với lối sống/chỉ số hiện tại không?
- Nếu phát hiện mâu thuẫn: giải thích nhẹ nhàng bằng ngôn ngữ đơn giản (không dùng thuật ngữ kỹ thuật)
- Chú ý tuổi thật sau khi tham chiếu là: 1(18-24), 2(25-29), 3(30-34), 4(35-39), 5(40-44), 6(45-49), 7(50-54), 8(55-59), 9(60-64), 10(65-69), 11(70-74), 12(75-79), 13(80+)

**[ĐÁNH GIÁ DỰ ĐOÁN]**
*(Ý nghĩa thực tế của kết quả – tại sao các yếu tố này quan trọng với bạn)*
- Giải thích tại sao các yếu tố SHAP hàng đầu ảnh hưởng đến nguy cơ của bạn
- Kết nối với lối sống thực tế của bạn (dựa trên raw_data)

**[KẾ HOẠCH HÀNH ĐỘNG]**
*(Bước 3: Synthesis – lời khuyên thực tế, cụ thể số liệu)*
- Chế độ ăn: ăn gì, kiêng gì, định lượng gram hoặc khẩu phần/bữa
- Vận động: môn cụ thể, bao nhiêu phút/ngày, bao nhiêu ngày/tuần
- Mục tiêu chỉ số: cần đưa BMI/Glucose/HbA1c về mức nào, trong bao lâu
- Khi nào cần đi khám ngay và lịch tái khám định kỳ
"""

        try:
            client = self._get_http_client()
            payload = {
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            resp = await client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"❌ LLM synthesis failed: {e}")
            return self._fallback_advice(prob, impacts)

    # ── Tiện ích ───────────────────────────────────────────────────────────────

    @staticmethod
    def build_rag_query(
        impacts: list[dict],
        feature_vi_map: Optional[dict] = None,
        top_n: int = 3,
    ) -> str:
        """
        Tạo câu truy vấn ngữ nghĩa từ top N SHAP impacts.
        Nếu feature name là tên kỹ thuật (num__xxx / cat__xxx),
        dịch sang tiếng Việt qua feature_vi_map trước khi build query.
        """
        top = sorted(impacts, key=lambda x: abs(x["impact"]), reverse=True)[:top_n]
        names = []
        for item in top:
            raw = item["feature"]
            if feature_vi_map and raw in feature_vi_map:
                names.append(feature_vi_map[raw])
            else:
                clean = raw.replace("num__", "").replace("cat__", "")
                names.append(clean)
        # Bỏ duplicate (ví dụ nhiều OHE smoking → "hút thuốc" x3)
        seen: set[str] = set()
        unique: list[str] = []
        for n in names:
            if n not in seen:
                seen.add(n)
                unique.append(n)
        return (
            "Ảnh hưởng của {} đến nguy cơ và phòng ngừa tiểu đường"
            .format(", ".join(unique))
        )

    @staticmethod
    def _fallback_advice(prob: float, impacts: list[dict]) -> str:
        top3 = sorted(impacts, key=lambda x: abs(x["impact"]), reverse=True)[:3]
        risk = "cao" if prob > 0.7 else "trung bình" if prob > 0.4 else "thấp"
        lines = [
            "**Phân tích nguy cơ tiểu đường**\n",
            f"🎯 **Kết quả:** Nguy cơ ở mức **{risk}** với xác suất {prob * 100:.1f}%\n",
            "📊 **3 Yếu tố ảnh hưởng lớn nhất:**",
        ]
        for i, item in enumerate(top3, 1):
            direction = "tăng" if item["impact"] > 0 else "giảm"
            lines.append(
                f"\n{i}. **{item['feature']}**: {direction} {abs(item['impact']):.1f}% nguy cơ"
            )
        lines.append("\n\n💡 **Khuyến nghị:**\n")
        if prob > 0.7:
            lines += [
                "\n1. Khẩn cấp: Gặp bác sĩ chuyên khoa tiểu đường trong vòng 1 tuần",
                "\n2. Xét nghiệm HbA1c và đường huyết đói ngay",
                "\n3. Bắt đầu chế độ ăn ít đường, tăng vận động ngay lập tức",
            ]
        elif prob > 0.4:
            lines += [
                "\n1. Đặt lịch khám sức khỏe định kỳ 3-6 tháng/lần",
                "\n2. Điều chỉnh chế độ ăn, tăng vận động 30 phút/ngày",
                "\n3. Theo dõi các chỉ số sức khỏe tại nhà",
            ]
        else:
            lines += [
                "\n1. Duy trì lối sống lành mạnh hiện tại",
                "\n2. Khám sức khỏe định kỳ hàng năm",
                "\n3. Giữ cân nặng ổn định, vận động đều đặn",
            ]
        lines.append(
            "\n\n⚠️ *Lưu ý: Kết quả chỉ mang tính tham khảo. "
            "Vui lòng tham khảo ý kiến bác sĩ chuyên khoa.*"
        )
        return "".join(lines)


# Singleton
rag_service = RAGService()


# ==============================================================================
# LIFESPAN
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 [Lifespan] Đang khởi động DiabeTwin AI Backend...")
    try:
        MODELS["preprocessor_clinical"] = joblib.load(get_path("preprocessor_clinical.pkl"))
        MODELS["model_clinical"] = joblib.load(get_path("model_clinical.pkl"))
        MODELS["clinical_background_processed"] = joblib.load(
            get_path("clinical_background_processed.pkl")
        )
        MODELS["home_model"] = joblib.load(get_path("diabetes_model_home.pkl"))
        MODELS["home_background"] = joblib.load(get_path("x_train_sample_home.pkl"))
        logger.info("✅ [Lifespan] Đã tải tất cả ML Models thành công!")
    except Exception as e:
        logger.error(f"❌ [Lifespan] Lỗi tải ML model: {e}")
        raise

    rag_ok = rag_service.initialize()
    if not rag_ok:
        logger.warning("⚠️ RAGService không khả dụng - sẽ dùng fallback advice.")

    yield

    await rag_service.close()
    MODELS.clear()
    logger.info("🧹 [Lifespan] Đã giải phóng bộ nhớ.")


# ==============================================================================
# APP & CORS
# ==============================================================================

app = FastAPI(title="DiabeTwin AI Backend", version="3.2", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://v0-frontend-ai-one.vercel.app",
        "https://connor-apostolic-jaye.ngrok-free.dev",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# SCHEMAS (giữ nguyên)
# ==============================================================================

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


# ==============================================================================
# HELPER: SHAP
# ==============================================================================

def compute_shap_impacts(
    model,
    processed_data,
    background,
    display_names: list[str],
) -> list[dict]:
    f = lambda x: model.predict_proba(x)[:, 1]
    explainer = shap.Explainer(f, background)
    shap_values = explainer(processed_data)
    return [
        {
            "feature": display_names[i] if i < len(display_names) else f"feature_{i}",
            "impact": round(float(val) * 100, 2),
        }
        for i, val in enumerate(shap_values.values[0])
    ]


# ==============================================================================
# ENDPOINTS
# ==============================================================================

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": len(MODELS) > 0,
        "rag_service_available": rag_service.initialized,
    }


@app.get("/")
async def root():
    return {
        "message": "DiabeTwin AI Backend",
        "version": "3.2",
        "endpoints": {
            "health": "/health",
            "clinical": "/api/predict/clinical",
            "home": "/api/predict/home",
            "docs": "/docs",
        },
    }


@app.post("/api/predict/clinical")
async def predict_clinical(data: ClinicalInput):
    """
    Doctor mode.
    RAG audience = "doctor"  (khớp với Firestore field).
    """
    try:
        logger.info(f"[Clinical] Request: {data.model_dump()}")

        # ── BƯỚC 1: Tiền xử lý & ML ───────────────────────────────────────
        clean_smoking = (
            data.smoking_history
            .replace("not current", "not_current")
            .replace("No Info", "no_info")
        )
        raw_input = pd.DataFrame([{
            "gender": data.gender,
            "age": data.age,
            "hypertension": data.hypertension,
            "heart_disease": data.heart_disease,
            "smoking_history": clean_smoking,
            "bmi": data.bmi,
            "HbA1c_level": data.hba1c,
            "blood_glucose_level": data.glucose,
        }])
        processed_data = MODELS["preprocessor_clinical"].transform(raw_input)
        model = MODELS["model_clinical"]
        prob = float(model.predict_proba(processed_data)[0][1])
        logger.info(f"[Clinical] ML prob: {prob * 100:.1f}%")

        # ── BƯỚC 2: SHAP ──────────────────────────────────────────────────
        feature_names_out = list(MODELS["preprocessor_clinical"].get_feature_names_out())
        impacts = compute_shap_impacts(
            model=model,
            processed_data=processed_data,
            background=MODELS["clinical_background_processed"],
            display_names=feature_names_out,  # tên kỹ thuật → map khi build query
        )

        # ── BƯỚC 3: RAG ───────────────────────────────────────────────────────
        benchmark_doc: str = ""
        context_docs: list[dict] = []

        if rag_service.initialized:
            # 3a. Lấy benchmark cứng (Registry 2026) cho bước Verify + Critique
            try:
                benchmark_doc = await rag_service.fetch_benchmark_doc(
                    audience_type="doctor"
                )
            except Exception as e:
                logger.warning(f"⚠️ [Clinical Benchmark] Failed: {e}")

            # 3b. Vector search bình thường cho bước Synthesis (khuyến nghị)
            try:
                rag_query = rag_service.build_rag_query(
                    impacts,
                    feature_vi_map=CLINICAL_FEATURE_VI_MAP,
                    top_n=3,
                )
                logger.info(f"[Clinical RAG] Query: {rag_query}")
                context_docs = await rag_service.retrieve_context(
                    query_text=rag_query,
                    audience_type="doctor",
                    top_k=5,
                )
                logger.info(f"[Clinical RAG] Retrieved {len(context_docs)} docs.")
            except Exception as e:
                logger.warning(f"⚠️ [Clinical RAG] Failed: {e}")

        # ── BƯỚC 4: LLM ───────────────────────────────────────────────────────
        ai_advice = await rag_service.synthesize_advice(
            prob=prob,
            impacts=impacts,
            role="Bác sĩ",
            raw_data=data.model_dump(),
            context_docs=context_docs,
            benchmark_doc=benchmark_doc,
        )

        # ── Lọc impacts cho Frontend (bỏ gender, smoking) ─────────────────
        excluded_kw = ["gender", "smoking_history"]
        frontend_impacts = [
            i for i in impacts
            if not any(kw in i["feature"] for kw in excluded_kw)
        ]

        return {
            "probability": round(prob * 100, 2),
            "status": "DƯƠNG TÍNH" if prob > 0.4945 else "ÂM TÍNH",
            "risk_level": "🔴" if prob > 0.7 else "🟡" if prob > 0.3 else "🟢",
            "impacts": frontend_impacts,
            "ai_advice": ai_advice,
        }

    except Exception as e:
        logger.error(f"[Clinical] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/home")
async def predict_home(data: HomeInput):
    """
    User/Home mode.
    RAG audience = "user"  (khớp với Firestore field).
    """
    try:
        logger.info("[Home] Prediction request received.")

        # ── BƯỚC 1: ML ────────────────────────────────────────────────────
        df = pd.DataFrame([data.dict()])
        model = MODELS["home_model"]
        prob = float(model.predict_proba(df)[0][1])
        logger.info(f"[Home] ML prob: {prob * 100:.1f}%")

        # ── BƯỚC 2: SHAP ──────────────────────────────────────────────────
        display_names = [
            "Huyết áp cao", "Cholesterol cao", "Kiểm tra Chol", "Chỉ số BMI",
            "Hút thuốc", "Đột quỵ", "Bệnh tim", "Vận động", "Trái cây",
            "Rau xanh", "Rượu bia", "Sức khỏe tổng quát", "Sức khỏe tâm thần",
            "Sức khỏe thể chất", "Đi lại khó", "Giới tính", "Nhóm tuổi",
        ]
        impacts = compute_shap_impacts(
            model=model,
            processed_data=df,
            background=MODELS["home_background"],
            display_names=display_names,  # đã là tiếng Việt → không cần map
        )

        # ── BƯỚC 3: RAG ───────────────────────────────────────────────────────
        benchmark_doc: str = ""
        context_docs: list[dict] = []

        if rag_service.initialized:
            # 3a. Lấy benchmark cứng (Registry 2026) cho bước Verify + Critique
            try:
                benchmark_doc = await rag_service.fetch_benchmark_doc(
                    audience_type="user"
                )
            except Exception as e:
                logger.warning(f"⚠️ [Home Benchmark] Failed: {e}")

            # 3b. Vector search bình thường cho bước Synthesis (khuyến nghị)
            try:
                rag_query = rag_service.build_rag_query(
                    impacts,
                    feature_vi_map=None,   # display_names đã là tiếng Việt
                    top_n=3,
                )
                logger.info(f"[Home RAG] Query: {rag_query}")
                context_docs = await rag_service.retrieve_context(
                    query_text=rag_query,
                    audience_type="user",
                    top_k=5,
                )
                logger.info(f"[Home RAG] Retrieved {len(context_docs)} docs.")
            except Exception as e:
                logger.warning(f"⚠️ [Home RAG] Failed: {e}")

        # ── BƯỚC 4: LLM ───────────────────────────────────────────────────────
        ai_advice = await rag_service.synthesize_advice(
            prob=prob,
            impacts=impacts,
            role="Người dùng",
            raw_data=data.model_dump(),
            context_docs=context_docs,
            benchmark_doc=benchmark_doc,
        )

        # ── Lọc impacts cho Frontend ───────────────────────────────────────
        excluded = {
            "Giới tính", "Rượu bia", "Sức khỏe tâm thần",
            "Hút thuốc", "Trái cây", "Rau xanh",
        }
        frontend_impacts = [i for i in impacts if i["feature"] not in excluded]
        #print(ai_advice)
        return {
            "probability": round(prob * 100, 2),
            "status": "NGUY CƠ CAO" if prob > 0.5 else "AN TOÀN",
            "impacts": frontend_impacts,
            "ai_advice": ai_advice,
        }

    except Exception as e:
        logger.error(f"[Home] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)