from typing import Annotated, TypedDict, Dict, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    # Sử dụng add_messages để tự động quản lý lịch sử trò chuyện
    messages: Annotated[list[BaseMessage], add_messages]
    
    # Theo dõi ý định hiện tại của thí sinh (vd: diem_chuan, hoc_phi, nganh_hoc)
    current_intent: str
    
    # Thông tin hồ sơ học sinh (nguyện vọng, khối xét tuyển, điểm thi dự kiến, v.v.)
    student_profile: Dict[str, Any]
    
    # Tác nhân tiếp theo sẽ được gọi
    next: str