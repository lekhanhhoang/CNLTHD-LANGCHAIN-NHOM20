import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

def ingest_documents(file_path: str, persist_directory: str = "./chroma_db"):
    """
    Xử lý tệp PDF tuyển sinh, chia nhỏ văn bản và lưu vào ChromaDB.
    """
    if not os.path.exists(file_path):
        print(f"⚠️ Lỗi: Không tìm thấy tệp {file_path}. Vui lòng kiểm tra lại đường dẫn.")
        return

    print(f"🚀 Bắt đầu nạp dữ liệu từ: {file_path}...")

    # 1. Load PDF using Unstructured (Yêu cầu cài đặt: pip install unstructured)
    loader = UnstructuredPDFLoader(file_path)
    data = loader.load()

    # 2. Split text into chunks using sliding window (overlap)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    all_splits = text_splitter.split_documents(data)
    print(f"✅ Đã chia nhỏ tài liệu thành {len(all_splits)} đoạn văn bản.")

    # 3. Create Embeddings
    print("🧠 Đang tạo embeddings (sử dụng HuggingFace)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Store in ChromaDB
    print(f"💾 Đang lưu vào ChromaDB tại: {persist_directory}...")
    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print("✨ Hoàn tất! Dữ liệu tuyển sinh đã sẵn sàng để tra cứu.")

if __name__ == "__main__":
    # Đường dẫn tệp placeholder (Người dùng sẽ thay bằng file .pdf thật)
    SOURCE_PDF = "data/admissions/De_an_tuyen_sinh_2026.pdf"
    DB_DIR = "./chroma_db"
    
    # Tạo thư mục data/admissions nếu chưa có
    os.makedirs(os.path.dirname(SOURCE_PDF), exist_ok=True)
    
    # Nếu file chưa tồn tại, thông báo để người dùng bỏ file vào
    if not os.path.exists(SOURCE_PDF):
        print(f"📌 Ghi chú: Hãy đặt tệp PDF tuyển sinh của bạn vào: {SOURCE_PDF}")
        print("Sau đó chạy lại script này.")
    else:
        ingest_documents(SOURCE_PDF, DB_DIR)
