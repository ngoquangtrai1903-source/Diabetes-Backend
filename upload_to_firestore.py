import os
from openai import OpenAI
from google.cloud import firestore
# Import chính xác class Vector từ SDK
from google.cloud.firestore_v1.vector import Vector
from google.oauth2 import service_account
from langchain_text_splitters import MarkdownHeaderTextSplitter
from dotenv import load_dotenv

# --- CẤU HÌNH API ---
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    default_headers={
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Diabetes Research AI",
    }
)
project_id_real="medical-database-1da52"
# Kết nối Firestore
cred = service_account.Credentials.from_service_account_file("service-account.json")
db = firestore.Client(credentials=cred, project=project_id_real)
def get_embedding(text):
    """Gọi OpenAI text-embedding-3-small qua OpenRouter"""
    text = text.replace("\n", " ")
    response = client.embeddings.create(
        model="openai/text-embedding-3-small",
        input=[text]
    )
    return response.data[0].embedding

def upload_markdown_to_firestore(file_path, audience):
    if not os.path.exists(file_path):
        print(f"Lỗi: Không tìm thấy file {file_path}")
        return

    print(f"\n>>> Đang xử lý: {file_path} (Audience: {audience})")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    chunks = markdown_splitter.split_text(content)

    collection_ref = db.collection("diabetes_knowledge_base")

    for i, chunk in enumerate(chunks):
        text_data = chunk.page_content
        metadata = chunk.metadata

        print(f"--- Đang tạo vector cho đoạn {i + 1}/{len(chunks)}...")
        vector = get_embedding(text_data)

        # Sử dụng class Vector chuẩn xác
        doc_data = {
            "content": text_data,
            "embedding": Vector(vector),
            "metadata": metadata,
            "audience": audience,
            "source": file_path,
            "created_at": firestore.SERVER_TIMESTAMP
        }

        collection_ref.add(doc_data)
        print(f"--- Đã lưu đoạn {i + 1} vào Firebase.")

if __name__ == "__main__":
    upload_markdown_to_firestore("CLINICAL_KNOWLEDGE_BASE.md", "doctor")
    upload_markdown_to_firestore("HOME_SCREENING_GUIDELINES.md", "user")
    print("\n[SUCCESS] Đã đẩy toàn bộ dữ liệu lên Firebase thành công!")