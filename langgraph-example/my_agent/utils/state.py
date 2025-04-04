from typing import TypedDict, List
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: List[BaseMessage]
    doc_id: str
    model_name: str  # 추가

# from langgraph.graph import add_messages
# from langchain_core.messages import BaseMessage
# from typing import TypedDict, Annotated, Sequence

# class AgentState(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], add_messages]

# from typing import TypedDict, List
# from langchain_core.messages import BaseMessage

# class AgentState(TypedDict):
#     messages: List[BaseMessage]
#     doc_id: str  # doc_id 필드 추가