import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

def ingest_directory(directory_path: str, persist_directory: str = "./chroma_db"):
    """
    Xử lý tất cả tệp PDF trong thư mục tuyển sinh, chia nhỏ văn bản và lưu vào ChromaDB.
    """
    if not os.path.exists(directory_path):
        print(f"⚠️ Lỗi: Không tìm thấy thư mục {directory_path}. Vui lòng tạo thư mục.")
        return

    print(f"🚀 Bắt đầu nạp toàn bộ dữ liệu PDF từ thư mục: {directory_path}...")

    # 1. Load PDF from directory
    loader = PyPDFDirectoryLoader(directory_path)
    data = loader.load()

    if not data:
        print("⚠️ Không tìm thấy file PDF nào trong thư mục để nạp.")
        return

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
    
    print("✨ Đã lưu trữ thành công! Dữ liệu tuyển sinh đã sẵn sàng để tra cứu.")

if __name__ == "__main__":
    # Đường dẫn thư mục PDF tuyển sinh
    SOURCE_DIR = "data/admissions"
    DB_DIR = "./chroma_db"
    
    # Tạo thư mục data/admissions nếu chưa có
    os.makedirs(SOURCE_DIR, exist_ok=True)
    
    # Chạy script nạp liệu
    ingest_directory(SOURCE_DIR, DB_DIR)
