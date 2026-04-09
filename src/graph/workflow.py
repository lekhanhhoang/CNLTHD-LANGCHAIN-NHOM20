import os
from dotenv import load_dotenv
from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from src.graph.state import AgentState
from src.tools.admissions_tools import admissions_tools

# Load environment variables
load_dotenv()

# 1. Khởi tạo LLM và Bind Tools
hf_token = os.environ.get("HF_TOKEN")
llm = ChatOpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=hf_token,
    model="Qwen/Qwen2.5-72B-Instruct",
    temperature=0.1
)

# Gắn công cụ vào mô hình
llm_with_tools = llm.bind_tools(admissions_tools)

# 2. Định nghĩa các Node trong Graph

def reasoner(state: AgentState):
    """
    Node xử lý chính: Nhận câu hỏi, áp dụng Persona và quyết định dùng Tool hay trả lời.
    """
    system_prompt = SystemMessage(content=(
        "Bạn là Chuyên viên Tư vấn Tuyển sinh Giáo dục Đại học chuyên nghiệp.\n"
        "Nhiệm vụ của bạn là tư vấn thông tin tuyển sinh, điểm chuẩn, học phí và thông tin xét tuyển của nhiều trường đại học khác nhau dựa trên tập dữ liệu RAG đề án tuyển sinh mà bạn kết nối.\n"
        "Phong cách: Chuyên nghiệp, khách quan, chào đón và cực kỳ chính xác.\n\n"
        "QUY TẮC CỐT LÕI:\n"
        "1. Bạn PHẢI sử dụng công cụ tra cứu dữ liệu để tìm thông tin tuyển sinh cụ thể của từng trường đại học mà người dùng nhắc tới. KHÔNG ĐƯỢC tự bịa con số.\n"
        "2. Nếu công cụ không trả về dữ liệu về trường đó, hãy trả lời trung thực là bạn chưa có dữ liệu của trường đó trong hệ thống.\n"
        "3. Luôn phản hồi bằng Tiếng Việt lịch sự, định dạng câu trả lời rõ ràng."
    ))
    
    # Kết hợp persona và lịch sử hội thoại
    messages = [system_prompt] + state["messages"]
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response]}

# Node chạy Tool tự động
tool_node = ToolNode(admissions_tools)

# 3. Định nghĩa Logic điều hướng (Routing)

def should_continue(state: AgentState) -> Literal["tools", END]:
    """
    Kiểm tra xem LLM có yêu cầu gọi Tool hay không.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # Nếu có tool_calls, đi tiếp sang node 'tools'
    if last_message.tool_calls:
        return "tools"
    
    # Nếu không, kết thúc hội thoại
    return END

# 4. Xây dựng Graph

workflow = StateGraph(AgentState)

# Thêm các nút
workflow.add_node("reasoner", reasoner)
workflow.add_node("tools", tool_node)

# Thiết lập điểm bắt đầu
workflow.add_edge(START, "reasoner")

# Thiết lập quan hệ rẽ nhánh từ reasoner
workflow.add_conditional_edges(
    "reasoner",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

# Sau khi chạy tool xong, bắt buộc quay lại reasoner để tổng hợp kết quả
workflow.add_edge("tools", "reasoner")

# 5. Biên dịch Workflow
app = workflow.compile(checkpointer=MemorySaver())