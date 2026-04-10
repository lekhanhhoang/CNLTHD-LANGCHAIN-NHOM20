# 🎓 Hệ thống Tư vấn Tuyển sinh Giáo dục Đại học

[![Framework](https://img.shields.io/badge/Framework-LangChain-green)](https://python.langchain.com/)
[![Graph](https://img.shields.io/badge/Orchestration-LangGraph-blue)](https://langchain-ai.github.io/langgraph/)
[![Database](https://img.shields.io/badge/VectorDB-ChromaDB-orange)](https://www.trychroma.com/)
[![UI](https://img.shields.io/badge/Frontend-Streamlit-red)](https://streamlit.io/)

## 📖 Giới thiệu
Dự án xây dựng một **AI Agent** chuyên dụng để hỗ trợ tư vấn tuyển sinh cho trường Đại học Công nghệ XYZ. Hệ thống sử dụng kiến trúc **RAG (Retrieval-Augmented Generation)** để cung cấp thông tin chính xác từ các tài liệu đề án tuyển sinh (PDF) và tra cứu điểm chuẩn từ cơ sở dữ liệu có cấu trúc (JSON).

Hệ thống được thiết kế để hoạt động như một chuyên viên tư vấn chuyên nghiệp, chính xác và thân thiện, giúp thí sinh và phụ huynh giải đáp các thắc mắc về:
*   Điểm chuẩn các năm.
*   Chỉ tiêu và tổ hợp xét tuyển.
*   Học phí và các chính sách ưu đãi.
*   Quy chế tuyển sinh và các mốc thời gian quan trọng.

---

## 📋 Mục lục
*   [Giới thiệu](#-giới-thiệu)
*   [Thành viên thực hiện](#-thành-viên-thực-hiện)
*   [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
*   [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
*   [Hướng dẫn cài đặt](#-hướng-dẫn-cài-đặt)
*   [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
*   [Công nghệ sử dụng](#-công-nghệ-sử-dụng)

---

## 👤 Thành viên thực hiện
| Họ tên | Vai trò |
| :--- | :--- |
| **Lê Khánh Hoàng** | Phát triển hệ thống (AI Engineer) |
| **Trương Xuân Hưng** | Phát triển hệ thống (AI Engineer) |
| **Lê Quốc Nam** | Phát triển hệ thống (AI Engineer) |
---

## 🏗️ Kiến trúc hệ thống
Hệ thống được xây dựng dựa trên quy trình **Reasoner-ToolNode** của LangGraph:
1.  **Reasoner (Bộ não)**: Sử dụng mô hình LLM Qwen2.5 để phân tích câu hỏi của người dùng.
2.  **Tools (Công cụ)**:
    *   `tra_cuu_thong_tin`: Truy xuất dữ liệu từ mã nguồn PDF (RAG).
    *   `tra_cuu_diem_chuan`: Tra cứu dữ liệu có cấu trúc từ tệp JSON.
3.  **State Management**: Quản lý lịch sử hội thoại và thông tin hồ sơ thí sinh theo thời gian thực.

---

## 📂 Cấu trúc thư mục
```text
CNLTHD-LANGCHAIN-NHOM20/
├── data/
│   ├── admissions/      # Chứa tệp PDF đề án tuyển sinh
│   └── diem_chuan_2025.json # Dữ liệu điểm chuẩn có cấu trúc
├── src/
│   ├── agents/          # Định nghĩa persona và logic điều phối
│   ├── graph/           # Xây dựng luồng workflow (State, Nodes, Graph)
│   └── tools/           # Các công cụ tra cứu dữ liệu
├── chroma_db/           # Cơ sở dữ liệu vector lưu trữ embeddings
├── frontend1.py         # Giao diện người dùng Streamlit
├── ingest.py            # Script nạp dữ liệu từ PDF vào VectorDB
├── requirements.txt     # Danh sách thư viện phụ thuộc
└── .env                 # Cấu hình biến môi trường (HF_TOKEN)
```

---

## 🛠️ Hướng dẫn cài đặt

### 1. Chuẩn bị môi trường
Yêu cầu Python 3.10 trở lên. Khuyến khích sử dụng [uv](https://github.com/astral-sh/uv) để quản lý gói nhanh hơn.

### 2. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 3. Cấu hình biến môi trường
Tạo tệp `.env` tại thư mục gốc và thêm mã Hugging Face Token:
```env
HF_TOKEN=your_huggingface_token_here
```

---

## 🚀 Hướng dẫn sử dụng

### Bước 1: Nạp dữ liệu tuyển sinh
Bỏ tệp PDF đề án vào `data/admissions/De_an_tuyen_sinh_2026.pdf` và chạy:
```bash
python ingest.py
```

### Bước 2: Khởi chạy Chatbot
```bash
streamlit run frontend1.py
```

---

## 💻 Công nghệ sử dụng
*   **Ngôn ngữ**: Python
*   **LLM**: Qwen2.5 (via Hugging Face API)
*   **Framework**: LangChain, LangGraph
*   **Vector Database**: ChromaDB (với HuggingFace Embeddings)
*   **Interface**: Streamlit
