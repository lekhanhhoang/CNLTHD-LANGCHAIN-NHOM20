import streamlit as st
import uuid

# --- CẤU HÌNH TRANG VÀ UI (ĐẶT Ở ĐẦU ĐỂ CHẠY NHANH NHẤT) ---
st.set_page_config(page_title="Hệ thống Tư vấn Tuyển sinh", page_icon="🎓")

# --- CUSTOM CSS FOR LOADING ANIMATION ---
st.markdown("""
    <style>
    /* 1. Ẩn hoạt ảnh thể thao mặc định của Streamlit (nếu có) */
    [data-testid="stStatusWidget"] svg, 
    .st-emotion-cache-1v0vhou svg,
    img[src*="loading_animation"] {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* 2. Thêm hoạt ảnh đồng hồ cát xoay thay thế */
    [data-testid="stStatusWidget"]::before {
        content: "";
        display: inline-block;
        width: 24px;
        height: 24px;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%23FF4B4B' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M5 2h14'/%3E%3Cpath d='M5 22h14'/%3E%3Cpath d='M19 2v6l-7 4-7-4V2'/%3E%3Cpath d='M5 22v-6l7-4 7 4v6'/%3E%3C/svg%3E");
        background-size: contain;
        background-repeat: no-repeat;
        animation: rotate_hourglass 2s linear infinite;
        margin-right: 10px;
    }

    @keyframes rotate_hourglass {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Làm đẹp cụm trạng thái */
    [data-testid="stStatusWidget"] {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 5px 12px;
        border-radius: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        border: 1px solid #f0f2f6;
    }
    
    /* Ẩn logo 'Made with Streamlit' để giao diện sạch hơn */
    footer {visibility: hidden;}
    #MainMenu {visibility: visible;}
    </style>
    """, unsafe_allow_html=True)

# --- CÁC IMPORT NẶNG (ĐẶT SAU ĐỂ UI HIỆN LÊN TRƯỚC) ---
from langchain_core.messages import HumanMessage, AIMessage
from src.graph.workflow import app

st.title("🎓 Hệ thống Tư vấn Tuyển sinh Giáo dục Đại học")
st.markdown("---")

# Khởi tạo session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Cấu hình thread_id cho LangGraph
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# Hiển thị lịch sử hội thoại
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Ô nhập yêu cầu
if prompt := st.chat_input("Hỏi về điểm chuẩn, học phí, ngành học hoặc đề án tuyển sinh..."):
    # Lưu tin nhắn người dùng
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Phản hồi từ AI
    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu cơ sở dữ liệu và xử lý..."):
            try:
                # Chạy luồng LangGraph
                initial_state = {
                    "messages": [HumanMessage(content=prompt)],
                    "current_intent": "tra_cuu",
                    "student_profile": {}
                }
                
                events = app.stream(
                    initial_state,
                    config=config,
                    stream_mode="values"
                )

                final_response = ""
                for event in events:
                    if "messages" in event:
                        latest_msg = event["messages"][-1]
                        # Bỏ qua tin nhắn của người dùng trong luồng stream
                        if latest_msg.type == "ai" and latest_msg.content:
                            final_response = latest_msg.content

                if final_response:
                    st.markdown(final_response)
                    st.session_state.messages.append(AIMessage(content=final_response))
                else:
                    st.warning("Hệ thống không tìm thấy nội dung phản hồi phù hợp.")
                    
            except Exception as e:
                st.error(f"Lỗi hệ thống: {str(e)}")
                st.info("Vui lòng đảm bảo bạn đã cài đặt đủ thư viện và cấu hình HF_TOKEN trong file .env")