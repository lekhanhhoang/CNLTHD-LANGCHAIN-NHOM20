from src.graph.workflow import app
from langchain_core.messages import HumanMessage
import uuid
import sys

def run_trace():
    initial_state = {
        "messages": [HumanMessage(content="Điểm chuẩn KTPM là bao nhiêu?")],
        "current_intent": "tra_cuu",
        "student_profile": {}
    }

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print("\n" + "="*70)
    print("Cau hoi dau vao: 'Diem chuan KTPM la bao nhieu?'\n")
    print(" -> DONG CO BAT DAU [START]")

    try:
        events = app.stream(initial_state, config=config, stream_mode="updates")
        for event in events:
            for node_name, node_state in event.items():
                print(f" -> Da chay qua Node: [{node_name}]")
        
        print(" -> Da hoi tu ve [END]")
        print("\n>> KIEM CHUNG: Do thi da bien dich va luan chuyen logic thanh cong!")
        print("="*70 + "\n")
    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    run_trace()
