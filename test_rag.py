import os
from openai import OpenAI
from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.base_query import FieldFilter  # Sửa lỗi Warning
from google.oauth2 import service_account
from dotenv import load_dotenv

# --- CẤU HÌNH ---
load_dotenv()
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))

project_id = "medical-database-1da52"
cred = service_account.Credentials.from_service_account_file("service-account.json")
db = firestore.Client(credentials=cred, project=project_id)


def get_embedding(text):
    response = client.embeddings.create(model="openai/text-embedding-3-small", input=[text.replace("\n", " ")])
    return response.data[0].embedding


def retrieve_context_debug(query_text, audience_type, top_k=5):  # Tăng k lên 5
    print(f"\n[STEP 1] Đang tìm kiếm tài liệu cho đối tượng: {audience_type}...")
    query_vector = get_embedding(query_text)
    collection = db.collection("diabetes_knowledge_base")

    # Sử dụng FieldFilter để loại bỏ Warning và tăng độ chính xác
    results = collection.where(filter=FieldFilter("audience", "==", audience_type)).find_nearest(
        vector_field="embedding",
        query_vector=Vector(query_vector),
        distance_measure=DistanceMeasure.COSINE,
        limit=top_k
    ).get()

    context_parts = []
    print(f"\n--- CÁC ĐOẠN DỮ LIỆU TÌM THẤY TRONG DATABASE ---")
    for i, doc in enumerate(results):
        data = doc.to_dict()
        content = data.get("content", "")
        # Lấy thông tin Header từ metadata để biết nội dung thuộc mục nào
        header = data.get("metadata", {}).get("Header_3", "Không rõ mục")

        print(f"\n>>> ĐOẠN {i + 1} (Nguồn: {header}):")
        print(f"{content[:300]}...")  # In 300 ký tự đầu để kiểm tra

        # Đánh số thứ tự cho từng đoạn để AI dễ trích dẫn
        context_parts.append(f"[Đoạn {i + 1}]: {content}")

    return "\n\n".join(context_parts)


def ask_gemini_with_source(question, context, audience_type):
    print(f"\n[STEP 2] Đang gửi câu hỏi và ngữ cảnh cho Gemini...")

    system_prompt = f"""
    Bạn là chuyên gia y tế về tiểu đường cho {audience_type}.
    Dưới đây là các đoạn dữ liệu trích xuất từ database (được đánh dấu [Đoạn 1], [Đoạn 2]...).

    Nhiệm vụ:
    1. Trả lời câu hỏi dựa TRÊN CƠ SỞ các đoạn dữ liệu được cung cấp.
    2. Ở cuối câu trả lời, hãy liệt kê rõ bạn đã dùng thông tin từ [Đoạn mấy].
    3. Nếu các đoạn dữ liệu không chứa thông tin về câu hỏi, hãy nói: 'Dữ liệu hiện tại không đề cập đến vấn đề này'.

    NGỮ CẢNH CUNG CẤP:
    {context}
    """

    response = client.chat.completions.create(
        model="google/gemini-2.0-flash-001",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    query = "Tiêu chuẩn chẩn đoán đái tháo đường thai kỳ (GDM) là gì?"
    audience = "doctor"

    # Lấy ngữ cảnh và in ra màn hình để kiểm tra
    context_data = retrieve_context_debug(query, audience)

    if not context_data:
        print("Lỗi: Không tìm thấy dữ liệu nào liên quan!")
    else:
        # AI trả lời
        final_answer = ask_gemini_with_source(query, context_data, audience)
        print("\n" + "=" * 50)
        print("KẾT QUẢ CUỐI CÙNG TỪ AI:")
        print(final_answer)
        print("=" * 50)