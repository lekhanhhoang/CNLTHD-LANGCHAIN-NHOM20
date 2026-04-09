from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel
from typing import Literal
from langchain_core.output_parsers import StrOutputParser

# Định nghĩa các hướng đi mới cho hệ thống tư vấn tuyển sinh
options = ["FINISH", "AdmissionsAgent"]

class Route(BaseModel):
    next: Literal["FINISH", "AdmissionsAgent"]

def supervisor_node(state, llm):
    # System Prompt cho Chuyên viên tư vấn tuyển sinh
    system_prompt = (
        "Bạn là nhân viên điều phối Tư vấn Tuyển sinh đa trường đại học.\n"
        "Phong cách làm việc: Chuyên nghiệp, thân thiện, và cực kỳ chính xác.\n\n"
        "Nhiệm vụ của bạn là điều phối cuộc hội thoại:\n"
        "- Nếu người dùng hỏi về điểm chuẩn, học phí, ngành học, đề án tuyển sinh của bất kỳ trường nào -> Trả về đúng chữ: AdmissionsAgent\n"
        "- Nếu là lời chào, cảm ơn hoặc người dùng không cần tư vấn thêm -> Trả về đúng chữ: FINISH\n\n"
        "QUY TẮC QUAN TRỌNG:\n"
        "1. Bạn phải luôn định hướng người dùng cho AdmissionsAgent tra cứu thông tin chính xác.\n"
        "2. TUYỆT ĐỐI CHỈ TRẢ VỀ 1 TRONG 2 TỪ: AdmissionsAgent HOẶC FINISH. KHÔNG GIẢI THÍCH THÊM."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Ai sẽ xử lý câu hỏi này tiếp theo? (AdmissionsAgent/FINISH)")
    ])

    chain = prompt | llm | StrOutputParser()
    raw_result = chain.invoke({"messages": state["messages"]})

    # Làm sạch dữ liệu
    clean_result = raw_result.strip().strip("'\"").replace(" ", "")

    # Fallback an toàn
    if clean_result not in options:
        print(f"[Supervisor] Model trả về sai format: '{clean_result}'. Mặc định chọn FINISH.")
        clean_result = "FINISH"

    print(f"[Supervisor] Quyết định: {clean_result}")
    return {"next": clean_result}