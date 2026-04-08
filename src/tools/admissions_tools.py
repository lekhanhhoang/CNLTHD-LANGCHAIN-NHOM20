import json
import os
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Đường dẫn đến cơ sở dữ liệu vector (phải khớp với ingest.py)
DB_DIR = "./chroma_db"

@tool
def tra_cuu_thong_tin(query: str) -> str:
    """
    Tra cứu thông tin chung về đề án tuyển sinh, quy chế, học phí và các thông tin phi cấu trúc khác từ tài liệu PDF.
    Sử dụng công cụ này khi thí sinh hỏi về chính sách hoặc các thông tin chung.
    """
    try:
        if not os.path.exists(DB_DIR):
            return "Dữ liệu tra cứu chưa được khởi tạo. Vui lòng chạy script nạp dữ liệu (ingest.py) trước."

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        
        # Tìm kiếm top 3 đoạn văn bản liên quan nhất
        docs = vectorstore.similarity_search(query, k=3)
        
        if not docs:
            return "Không tìm thấy thông tin liên quan trong tài liệu tuyển sinh."
            
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        return f"Lỗi khi tra cứu thông tin: {str(e)}"

@tool
def tra_cuu_diem_chuan(query: str) -> str:
    """
    Tra cứu điểm chuẩn, mã ngành, tổ hợp xét tuyển và chỉ tiêu từ cơ sở dữ liệu có cấu trúc.
    Sử dụng công cụ này khi thí sinh hỏi về điểm số hoặc mã ngành cụ thể.
    """
    json_path = "data/diem_chuan_2025.json"
    try:
        if not os.path.exists(json_path):
            return "Chưa có dữ liệu điểm chuẩn trong hệ thống."

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Chuyển đổi query về chữ thường để tìm kiếm linh hoạt (partial matching)
        search_term = query.lower()
        results = []

        for item in data:
            # Tìm kiếm theo tên ngành (partial match) hoặc mã ngành (exact match)
            if search_term in item["ten_nganh"].lower() or search_term == item["ma_nganh"].lower():
                res = (
                    f"📍 Ngành: {item['ten_nganh']} ({item['ma_nganh']})\n"
                    f"- Khối xét tuyển: {', '.join(item['to_hop_xet_tuyen'])}\n"
                    f"- Điểm chuẩn năm trước: {item['diem_chuan_nam_truoc']}\n"
                    f"- Chỉ tiêu: {item['chi_tieu']}"
                )
                results.append(res)

        if not results:
            return f"Không tìm thấy mã ngành hoặc tên ngành '{query}' trong hệ thống."

        return "\n---\n".join(results)
    except Exception as e:
        return f"Lỗi khi tra cứu điểm chuẩn: {str(e)}"

# Danh sách các tool để đăng ký với Agent
admissions_tools = [tra_cuu_thong_tin, tra_cuu_diem_chuan]
