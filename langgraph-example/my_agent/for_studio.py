from typing import Annotated, Sequence
from langgraph.graph import StateGraph
from langchain_core.messages import BaseMessage, HumanMessage
# from my_agent.utils.nodes import (
#     analyze_intent, 
#     generate_summary_response,
#     generate_purpose_response,
#     generate_body_response,
#     generate_database_response,
#     generate_default_response,
#     generate_search_response,  # 추가
#     router
# )
from utils.nodes import (
    analyze_intent, 
    generate_body_response,
    generate_database_response,
    generate_default_response,
    generate_search_response,  # 추가
    router
)
from utils.state import AgentState

import graphviz



# def create_graph():
#     # 워크플로우 정의
#     workflow = StateGraph(AgentState)
    
#     # 노드 추가
#     workflow.add_node("의도분석", analyze_intent)
#     workflow.add_node("의도분석 라우터", router)
#     workflow.add_node("요약_생성", generate_summary_response)
#     workflow.add_node("목적_생성", generate_purpose_response)
#     workflow.add_node("검색_생성", generate_search_response)  # 추가
#     workflow.add_node("그_외_생성", generate_default_response)
    
#     # 엣지 설정
#     workflow.set_entry_point("의도분석")
#     workflow.add_edge("의도분석", "의도분석 라우터")
    
#     # 라우터에서 각 생성기로의 조건부 엣지 설정
#     workflow.add_conditional_edges(
#         "의도분석 라우터",
#         lambda x: x["next"],
#         {
#             "summary": "요약_생성",
#             "purpose": "목적_생성",
#             "search": "검색_생성",    # 추가
#             "default": "그_외_생성"
#         }
#     )
    
#     # 각 생성기 노드에서 종료
#     workflow.set_finish_point("요약_생성")
#     workflow.set_finish_point("목적_생성")
#     workflow.set_finish_point("검색_생성") 
#     workflow.set_finish_point("그_외_생성")
    
#     # 그래프 컴파일
#     return workflow.compile()

# # LangGraph Studio를 위한 graph 변수 export
# graph = create_graph()


def create_graph():
    # 워크플로우 정의
    workflow = StateGraph(AgentState)
    
    # 노드 추가
    workflow.add_node("의도분석", analyze_intent)
    workflow.add_node("의도분석 라우터", router)
    workflow.add_node("본문_처리", generate_body_response)
    workflow.add_node("DB_처리", generate_database_response)
    workflow.add_node("검색_생성", generate_search_response)  # 추가
    workflow.add_node("그_외_생성", generate_default_response)
    
    # 엣지 설정
    workflow.set_entry_point("의도분석")
    workflow.add_edge("의도분석", "의도분석 라우터")
    
    # 라우터에서 각 생성기로의 조건부 엣지 설정
    workflow.add_conditional_edges(
        "의도분석 라우터",
        lambda x: x["next"],
        {
            "body": "본문_처리",
            "database": "DB_처리",
            "search": "검색_생성",    # 추가
            "default": "그_외_생성"
        }
    )
    
    # 각 생성기 노드에서 종료
    workflow.set_finish_point("본문_처리")
    workflow.set_finish_point("DB_처리")
    workflow.set_finish_point("검색_생성") 
    workflow.set_finish_point("그_외_생성")
    
    # 그래프 컴파일
    return workflow.compile()

# LangGraph Studio를 위한 graph 변수 export
graph = create_graph()

from langgraph.graph import StateGraph, END

import graphviz

import graphviz

def visualize_current_graph():
    # 그래프 생성
    dot = graphviz.Digraph('Patent Agent Flow')
    dot.attr(rankdir='LR')  # 왼쪽에서 오른쪽으로 진행
    
    # 전체 그래프 스타일
    dot.attr('node', 
            style='filled',
            shape='box',
            fontname='Arial',
            fontsize='12',
            margin='0.2')
    
    # 노드 추가 (실제 graph의 구조대로)
    dot.node('start', 'Start', 
            fillcolor='lightblue', 
            shape='oval')
    
    dot.node('route', 'Route\nFunction', 
            fillcolor='lightgreen')
    
    dot.node('generate_database_response', 'Generate\nDatabase Response', 
            fillcolor='lightyellow')
    
    dot.node('generate_body_response', 'Generate\nBody Response', 
            fillcolor='lightpink')
    
    dot.node('end', 'End', 
            fillcolor='lightgray',
            shape='oval')
    
    # 엣지 추가 (실제 flow대로)
    dot.edge('start', 'route')
    dot.edge('route', 'generate_database_response', 'database query')
    dot.edge('route', 'generate_body_response', 'body query')
    dot.edge('generate_database_response', 'end')
    dot.edge('generate_body_response', 'end')
    
    # 그래프 저장
    dot.render("patent_agent_flow", format="png", cleanup=True)
    print("Graph has been saved as 'patent_agent_flow.png'")


# if __name__ == "__main__":
#     # test_messages = [
#     #     "이 내용을 간단히 요약해줘: 인공지능은 컴퓨터가 인간처럼 생각하고 학습하는 기술입니다.",
#     #     "왜 인공지능이 중요한가요?",
#     #     "인공지능에 대해서 검색해줘",  # 검색 테스트 추가
#     #     "파이썬 코드 어떻게 작성하나요?"
#     # ]
#     test_messages = ["""
# 정보 교류 기능을 가진 단말기, 이를 이용한 정보 교류서버시스템 및 방법(a wireless terminal having information exchange facility, information exchange system and method using the wireless terminal)
# 이 특허와 비슷한 특허가 있는지 찾아줘.
# """]
    
#     print("\n======== 테스트 시작 ========")
    
#     for i, message in enumerate(test_messages, 1):
#         print(f"\n{'='*20} 테스트 케이스 {i} {'='*20}")
#         print(f"입력 메시지: {message}")
        
#         input_state = {
#             "messages": [HumanMessage(content=message)]
#         }
        
#         result = graph.invoke(input_state)
        
#         print("\n--- 최종 결과 ---")
#         final_message = result["messages"][-1]
#         print(f"최종 응답: {final_message.content}")
#         print("=" * 50)

if __name__ == "__main__":
    visualize_current_graph()    # test_messages = [
    #     # 본문 관련 질문
    #     "이 특허의 목적은 무엇인가요?",
    #     "이 기술의 주요 특징을 설명해주세요",
    #     "이 특허의 효과는 무엇인가요?",
        
    #     # 메타데이터(DB) 관련 질문
    #     "이 특허의 출원인이 누구야?",
    #     "청구항은 몇 개야?",
    #     "청구항 1번이 뭐야?",
        
    #     # 검색 관련 질문
    #     "이 특허와 비슷한 특허가 있는지 찾아줘",
    #     "이 기술이 실제로 사용되고 있어?",
        
    #     # 기타 질문
    #     "이 특허에 대해서 설명해줘"
    # ]
    
    # print("\n======== 특허 분석 시스템 테스트 시작 ========")
    
    # for i, message in enumerate(test_messages, 1):
    #     print(f"\n{'='*20} 테스트 케이스 {i} {'='*20}")
    #     print(f"입력 질문: {message}")
        
    #     input_state = {
    #         "messages": [HumanMessage(content=message)]
    #     }
        
    #     try:
    #         result = graph.invoke(input_state)
            
    #         print("\n--- 결과 ---")
    #         final_message = result["messages"][-1]
    #         print(f"응답: {final_message.content}")
            
    #     except Exception as e:
    #         print(f"\n[오류 발생] {str(e)}")
            
    #     print("=" * 50)