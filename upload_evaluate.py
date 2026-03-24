import os
import uuid
import re
from openai import OpenAI
from google.cloud import firestore
# Sử dụng class Vector chuẩn xác từ SDK
from google.cloud.firestore_v1.vector import Vector
from google.oauth2 import service_account
from dotenv import load_dotenv

# --- 1. CẤU HÌNH ---
load_dotenv()
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

project_id = "medical-database-1da52"
cred = service_account.Credentials.from_service_account_file("service-account.json")
db = firestore.Client(credentials=cred, project=project_id)


# --- 2. HÀM LÀM SẠCH VĂN BẢN (DỌN DẸP LATEX) ---
def clean_medical_text(text):
    """Chuyển đổi các ký tự LaTeX sang Unicode chuẩn và dọn dẹp format"""
    if not text:
        return ""

    replacements = {
        r'\$\\geq\$': '≥',
        r'\$\\leq\$': '≤',
        r'\$\\beta\$': 'β',
        r'\$\\alpha\$': 'α',
        r'\$\\gamma\$': 'γ',
        r'\^2': '²',  # Cho BMI kg/m²
        r'\$': ''  # Loại bỏ dấu $ bao quanh ký tự
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)

    # Loại bỏ các khoảng trắng thừa do format Markdown
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# --- 3. HÀM TẠO EMBEDDING ---
def get_embedding(text):
    """Tạo vector cho văn bản đã làm sạch"""
    # Thay thế xuống dòng bằng khoảng trắng để embedding chuẩn hơn
    processed_text = text.replace("\n", " ")
    response = client.embeddings.create(
        model="openai/text-embedding-3-small",
        input=[processed_text]
    )
    return response.data[0].embedding


# --- 4. LUỒNG XỬ LÝ CHÍNH ---
def process_and_upload_unified():
    file_path = "EVALUATE_ML.md"

    if not os.path.exists(file_path):
        print(f"❌ Lỗi: Không tìm thấy file {file_path}")
        return

    print(f"📖 Đang đọc file: {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        full_content = f.read()

    # Tách nội dung dựa trên từ khóa GROUP
    parts = full_content.split("## GROUP")

    doctor_raw = ""
    user_raw = ""

    for part in parts:
        if part.startswith(" 1: CLINICAL"):
            doctor_raw = "## GROUP" + part
        elif part.startswith(" 2: COMMUNITY"):
            user_raw = "## GROUP" + part

    # Làm sạch dữ liệu trước khi xử lý tiếp
    configs = [
        {
            "content": clean_medical_text(doctor_raw),
            "audience": "doctor",
            "label": "Clinical Registry 2026"
        },
        {
            "content": clean_medical_text(user_raw),
            "audience": "user",
            "label": "Lifestyle Registry 2026"
        }
    ]

    collection = db.collection("diabetes_knowledge_base")

    for config in configs:
        if not config["content"].strip():
            print(f"⚠️ Bỏ qua {config['label']} vì nội dung trống.")
            continue

        print(f"--- ⚙️ Đang xử lý khối dữ liệu: {config['label']} ---")

        # Tạo embedding từ nội dung ĐÃ LÀM SẠCH
        embedding_vector = get_embedding(config["content"])

        doc_id = str(uuid.uuid4())
        doc_data = {
            "id": doc_id,
            "content": config["content"],  # Nội dung sạch hiển thị trên UI
            "embedding": Vector(embedding_vector),  # Vector chuẩn cho search
            "audience": config["audience"],
            "metadata": {
                "source": "EVALUATE_ML.md",
                "type": "unified_registry_v2",
                "version": "2026.1",
                "cleaned": True  # Đánh dấu dữ liệu đã qua xử lý
            },
            "created_at": firestore.SERVER_TIMESTAMP
        }

        collection.document(doc_id).set(doc_data)
        print(f"✅ Đã đẩy thành công chế độ: {config['audience']}")


if __name__ == "__main__":
    try:
        process_and_upload_unified()
        print("\n🚀 [SUCCESS] Dữ liệu chuẩn đã sẵn sàng trên Firebase!")
    except Exception as e:
        print(f"❌ Lỗi hệ thống: {e}")