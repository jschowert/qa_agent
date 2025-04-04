from dotenv import load_dotenv
import os
from typing import Any, Dict, TypedDict
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from utils.nodes import analyze_intent_node, summary_node, purpose_node
from utils.tools import tools
from utils.models import model_provider
from langchain_community.tools.tavily_search import TavilySearchResults





class AgentState(TypedDict):
    """상태 타입 정의"""
    messages: list[Any]
    intent: str | None

def create_workflow():
    """워크플로우 그래프 생성"""
    # 그래프 초기화
    graph = StateGraph(AgentState)
    
    # ReAct 에이전트 생성 (기본 처리용)
    react_agent = create_react_agent(
        model=model_provider.get_model(),
        tools=tools,
        checkpointer=MemorySaver()
    )
    
    # 기본 처리 노드 정의
    def default_node(state: AgentState) -> AgentState:
        """기본 ReAct 에이전트를 사용하는 노드"""
        return react_agent.invoke(state)
    
    # 노드 추가
    graph.add_node("analyze_intent", analyze_intent_node)
    graph.add_node("summarize", summary_node)
    graph.add_node("analyze_purpose", purpose_node)
    graph.add_node("default", default_node)
    
    # 라우터 함수 정의
    def router(state: AgentState) -> str:
        """의도에 따라 다음 노드를 결정하는 라우터"""
        intent = state.get("intent", "OTHER")
        intent_to_node = {
            "SUMMARY": "summarize",
            "PURPOSE": "analyze_purpose",
            "OTHER": "default"
        }
        return intent_to_node.get(intent, "default")
    
    # 조건부 엣지 추가
    graph.add_conditional_edges(
        "analyze_intent",
        router,
        {
            "summarize": "summarize",
            "analyze_purpose": "analyze_purpose",
            "default": "default"
        }
    )
    
    # 최종 노드들에서 END로 연결
    graph.add_edge("summarize", END)
    graph.add_edge("analyze_purpose", END)
    graph.add_edge("default", END)
    
    # 시작 노드 설정
    graph.set_entry_point("analyze_intent")
    
    return graph.compile()

def main():
    """메인 실행 함수"""
    # 워크플로우 생성
    workflow = create_workflow()
    
    print("\n=== AI 어시스턴트 시작 ===")
    print("- 텍스트 요약이 필요하면 '요약해줘'와 같이 요청하세요.")
    print("- 목적이나 의도를 알고 싶으면 '목적이 뭐야?'와 같이 질문하세요.")
    print("- 종료하려면 'quit' 또는 'exit'를 입력하세요.")
    print("========================\n")

    while True:
        try:
            # 사용자 입력 받기
            user_input = input("\n사용자: ").strip()
            if user_input.lower() in ['quit', 'exit']:
                print("\n대화를 종료합니다.")
                break
            
            # 상태 초기화
            state = {
                "messages": [HumanMessage(content=user_input)],
                "intent": None
            }
            
            # 워크플로우 실행
            result = workflow.invoke(state)
            
            # 마지막 AI 응답 출력
            for message in reversed(result["messages"]):
                if isinstance(message, AIMessage):
                    print(f"\nAI: {message.content}")
                    break
                    
        except Exception as e:
            print(f"\n에러 발생: {str(e)}")
            print("다시 시도해주세요.")
            continue
    
if __name__ == "__main__":
    main()