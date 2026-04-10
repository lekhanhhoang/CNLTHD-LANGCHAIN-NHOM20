import json
import os
import difflib
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
    Tuyệt đối KHÔNG dùng công cụ này để tra cứu điểm chuẩn.
    """
    try:
        if not os.path.exists(DB_DIR):
            return "Dữ liệu tra cứu chưa được khởi tạo. Vui lòng chạy script nạp dữ liệu (ingest.py) trước."

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        
        # Tìm kiếm top 3 đoạn văn bản liên quan nhất
        docs = vectorstore.similarity_search(query, k=3)
        
        if not docs:
            return "Hệ thống không tìm thấy thông tin liên quan trong tài liệu tuyển sinh. Yêu cầu tham khảo thêm nguồn khác."
            
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

        # Chuyển đổi query về chữ thường
        search_term = query.lower().strip()
        
        # Lớp 1: Từ điển viết tắt / từ lóng
        tu_dien = {
            "cntt": "công nghệ",
            "maketing": "marketing",
            "qtkd": "quản trị kinh doanh"
        }
        # Chỉ áp dụng từ điển nếu tìm thấy từ khóa
        for k, v in tu_dien.items():
            if k in search_term:
                search_term = search_term.replace(k, v)

        results = []
        
        # Lớp 2: Fuzzy Match để bắt lỗi chính tả, thiếu dấu
        danh_sach_ten_nganh = [item["ten_nganh"].lower() for item in data]
        # difflib.get_close_matches sẽ tìm các từ có độ tương đồng >= 60%
        fuzzy_matches = difflib.get_close_matches(search_term, danh_sach_ten_nganh, n=2, cutoff=0.5)

        for item in data:
            ten_nganh = item["ten_nganh"].lower()
            ma_nganh = item["ma_nganh"].lower()
            
            # Khớp chuỗi trực tiếp HOẶC khớp mã trực tiếp HOẶC nằm trong gợi ý mờ (Fuzzy Match)
            if search_term in ten_nganh or search_term == ma_nganh or ten_nganh in fuzzy_matches:
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
