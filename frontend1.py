import streamlit as st
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from src.graph.workflow import app

# Cấu hình trang
st.set_page_config(page_title="Tư vấn Tuyển sinh XYZ", page_icon="🎓")
st.title("🎓 Hệ thống Tư vấn Tuyển sinh Đại học Công nghệ XYZ")
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
                # Chúng ta truyền thêm profile trống nếu cần, hoặc để Agent tự thu thập
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