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
        "Bạn là Chuyên viên Tư vấn Tuyển sinh chính thức của Trường Đại học Công nghệ XYZ cho mùa tuyển sinh năm 2026.\n"
        "Phong cách làm việc: Chuyên nghiệp, thân thiện, và cực kỳ chính xác.\n\n"
        "Nhiệm vụ của bạn là điều phối cuộc hội thoại:\n"
        "- Nếu thí sinh hỏi về điểm chuẩn, học phí, ngành học, đề án tuyển sinh hoặc các thông tin liên quan -> Trả về đúng chữ: AdmissionsAgent\n"
        "- Nếu là lời chào, lời cảm ơn hoặc thí sinh đã nhận được thông tin mình cần và không hỏi thêm -> Trả về đúng chữ: FINISH\n\n"
        "QUY TẮC QUAN TRỌNG:\n"
        "1. Bạn phải luôn ưu tiên sử dụng công cụ để tra cứu thông tin chính xác. KHÔNG BAO GIỜ tự bịa ra con số.\n"
        "2. Nếu không tìm thấy thông tin trong hệ thống tra cứu, hãy trả lời trung thực: 'Tôi không có thông tin chính thức về vấn đề này'.\n\n"
        "TUYỆT ĐỐI CHỈ TRẢ VỀ 1 TRONG 2 TỪ: AdmissionsAgent HOẶC FINISH. KHÔNG GIẢI THÍCH THÊM."
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