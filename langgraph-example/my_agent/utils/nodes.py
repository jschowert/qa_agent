from functools import lru_cache
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from .tools import tools
from langgraph.prebuilt import ToolNode
from .state import AgentState
from .models import get_model
from .prompts import BODY_PROMPT, DATABASE_PROMPT, SEARCH_PROMPT, DEFAULT_PROMPT
from .data_parser import DataParser
from .rag import (
    read_json_file, 
    split_json_into_chunks, 
    save_chunks_to_file,
    create_vectordb, 
    hybrid_search
)
from .get_data import get_elasticsearch_data
from .tools import tavily_search
from langchain_core.messages import AIMessage, HumanMessage


tool_node = ToolNode(tools)


###########코드추가###########
# from my_agent.utils.models import get_model
# from my_agent.utils.prompts import PURPOSE_PROMPT, SUMMARY_PROMPT, DEFAULT_PROMPT, SEARCH_PROMPT

# def analyze_intent(state):
#     current_message = state["messages"][-1].content
#     print("\n" + "="*50)
#     print(f"[의도 분석] 현재 입력 메시지: {current_message}")
    
#     intent_prompt = """
#     다음 메시지 하나만 보고 의도를 파악해주세요.
    
#     반드시 다음 중 하나로만 답변하세요:
#     1. 이 글의 전체적인 내용을 요약하길 원하는거면 summary
#     2. 왜 이 기술이 만들었는지, 목적이 무엇인지, 무엇을 해결할 수 있는지를 묻는 의도라면 purpose
#     3. 이 기술이 현재 산업에서 실제 쓰이고 있는지, 비슷한 특허가 있는지 등 검색이 필요할거 같으면 search
#     4. 그 외의 모든 경우는 default
    
#     이 4가지 중 이 메세지는 무엇을 의도하는지 오직 summary, purpose, search, default 중 한 단어로만 답변하세요.
    
#     메시지: {message}
#     """
    
#     model = get_model()
#     response = model.invoke(intent_prompt.format(message=current_message))
#     raw_intent = response.content.strip().lower()
    
#     print(f"[의도 분석] 모델 원본 응답: '{raw_intent}'")
    
#     ALLOWED_INTENTS = {"summary", "purpose", "search", "default"}
#     if raw_intent not in ALLOWED_INTENTS:
#         print(f"[의도 분석] 경고: 잘못된 의도 감지됨. '{raw_intent}' → 'default'로 변경")
#         intent = "default"
#     else:
#         intent = raw_intent
#         print(f"[의도 분석] 감지된 의도: '{intent}'")
    
#     # additional_kwargs에 intent 저장
#     state["messages"][-1].additional_kwargs["intent"] = intent
#     print(f"[의도 분석] 최종 설정된 intent: {intent}")
#     print(f"[의도 분석] 현재 메시지 additional_kwargs: {state['messages'][-1].additional_kwargs}")
#     print("="*50)
#     return state

# def router(state):
#     current_message = state["messages"][-1].content
#     intent = state["messages"][-1].additional_kwargs.get("intent", "default")
    
#     print("\n" + "="*50)
#     print(f"[라우터] 현재 메시지: {current_message}")
#     print(f"[라우터] additional_kwargs: {state['messages'][-1].additional_kwargs}")
#     print(f"[라우터] 전달받은 의도: '{intent}'")
    
#     ALLOWED_INTENTS = {"summary", "purpose", "search", "default"}
#     if intent not in ALLOWED_INTENTS:
#         print(f"[라우터] 경고: 잘못된 의도 '{intent}' → 'default'로 변경")
#         intent = "default"
    
#     print(f"[라우터] 최종 라우팅 의도: '{intent}'")
#     print(f"[라우터] 다음 노드: {intent}_generator")
#     print("="*50)
    
#     return {"next": intent}

# def generate_summary_response(state):
#     current_message = state["messages"][-1]
#     print("\n" + "="*50)
#     print(f"[요약 생성기] 입력 메시지: {current_message.content}")
    
#     prompt = SUMMARY_PROMPT.format(question=current_message.content)
#     model = get_model()
#     response = model.invoke(prompt)
    
#     print(f"[요약 생성기] 생성된 응답: {response.content}")
#     print("="*50)
#     return {"messages": [response]}

# def generate_purpose_response(state):
#     current_message = state["messages"][-1]
#     print("\n" + "="*50)
#     print(f"[목적 생성기] 입력 메시지: {current_message.content}")
    
#     prompt = PURPOSE_PROMPT.format(question=current_message.content) 
#     model = get_model()
#     response = model.invoke(prompt)
    
#     print(f"[목적 생성기] 생성된 응답: {response.content}")
#     print("="*50)
#     return {"messages": [response]}

# def generate_default_response(state):
#     current_message = state["messages"][-1]
#     print("\n" + "="*50)
#     print(f"[기본 생성기] 입력 메시지: {current_message.content}")
    
#     prompt = DEFAULT_PROMPT.format(question=current_message.content)
#     model = get_model()
#     response = model.invoke(prompt)
    
#     print(f"[기본 생성기] 생성된 응답: {response.content}")
#     print("="*50)
#     return {"messages": [response]}

# # ... existing code ...

# from langchain_core.messages import AIMessage, HumanMessage  # 추가

# from .tools import tavily_search  # tavily_search 함수 직접 import

# def generate_search_response(state):
#     current_message = state["messages"][-1]
#     print("\n" + "="*50)
#     print(f"[검색 생성기] 입력 메시지: {current_message.content}")
    
#     # Tavily 검색 직접 수행
#     search_results = tavily_search(current_message.content)
#     print(f"[검색 생성기] 검색 결과: {search_results}")
    
#     if not search_results or "검색 중 오류 발생" in str(search_results):
#         response_content = "죄송합니다. 검색 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
#         return {"messages": [AIMessage(content=response_content)]}
    
#     # 검색 결과를 바탕으로 응답 생성
#     prompt = f"""
#     다음 검색 결과를 바탕으로 사용자의 질문에 대해 상세하게 답변해주세요.
#     가능한 한 많은 정보를 포함시켜 주세요.
    
#     질문: {current_message.content}
    
#     검색 결과:
#     {search_results}
#     """
    
#     model = get_model()
#     response = model.invoke(prompt)
    
#     print(f"[검색 생성기] 생성된 응답: {response.content}")
#     print("="*50)
#     return {"messages": [response]}



############새로운 코드############

###########코드추가###########
from .models import get_model
from .prompts import BODY_PROMPT, DATABASE_PROMPT, SEARCH_PROMPT, DEFAULT_PROMPT
from .data_parser import DataParser

def analyze_intent(state: AgentState) -> AgentState:
    current_message = state["messages"][-1]
    # BaseMessage 객체이면 .content를, 딕셔너리면 ['content']를 사용
    query = current_message.content if hasattr(current_message, 'content') else current_message['content']
    model_name = state.get("model_name", "exaone3.5:7.8b")  # 추가
    
    print(f"\n[의도 분석] 입력: {query}")
    
    intent_prompt = """
    다음 메시지 하나만 보고 의도를 파악해주세요.
    
    반드시 다음 중 하나로만 답변하세요:
    1. 이 특허의 기술분야, 해결과제, 해결 수단, 특징, 효과 등(왜 이 기술이 만들었는지, 목적이 무엇인지, 무엇을 해결할 수 있는지를 묻는 등) 본문 내용을 요약해야 할 것 같으면 body 
    2. 이 특허의 본문 외 모든 내용(청구일, 출원일, 청구항 등) 본문 내용안에 없을 것 같은 정보들을 알고싶으면 database
    3. 이 기술이 현재 산업에서 실제 쓰이고 있는지, 비슷한 특허가 있는지 등 한 특허를 보고 답을 할 수 없고 검색이 필요할거 같으면 search
    4. 그 외의 모든 경우는 default
    최대한 body or database 의도로 생각해주세요
  
    
    이 3가지 중 이 메세지는 무엇을 의도하는지 오직 body, database, search, default 중 한 단어로만 답변하세요.
    
    메시지: {message}
    """
    
    model = get_model(model_name)
    response = model.invoke(intent_prompt.format(message=current_message))
    raw_intent = response.content.strip().lower()
    
    print(f"[의도 분석] 모델 원본 응답: '{raw_intent}'")
    
    ALLOWED_INTENTS = {"body", "database", "search", "default"}
    if raw_intent not in ALLOWED_INTENTS:
        print(f"[의도 분석] 경고: 잘못된 의도 감지됨. '{raw_intent}' → 'default'로 변경")
        intent = "default"
    else:
        intent = raw_intent
        print(f"[의도 분석] 감지된 의도: '{intent}'")
    
    # additional_kwargs에 intent 저장
    state["messages"][-1].additional_kwargs["intent"] = intent
    print(f"[의도 분석] 최종 설정된 intent: {intent}")
    print(f"[의도 분석] 현재 메시지 additional_kwargs: {state['messages'][-1].additional_kwargs}")
    print("="*50)
    return state

def router(state):
    current_message = state["messages"][-1].content
    intent = state["messages"][-1].additional_kwargs.get("intent", "default")
    
    print("\n" + "="*50)
    print(f"[라우터] 현재 메시지: {current_message}")
    print(f"[라우터] additional_kwargs: {state['messages'][-1].additional_kwargs}")
    print(f"[라우터] 전달받은 의도: '{intent}'")
    
    ALLOWED_INTENTS = {"body", "database", "search", "default"}
    if intent not in ALLOWED_INTENTS:
        print(f"[라우터] 경고: 잘못된 의도 '{intent}' → 'default'로 변경")
        intent = "default"
    
    print(f"[라우터] 최종 라우팅 의도: '{intent}'")
    print(f"[라우터] 다음 노드: {intent}_generator")
    print("="*50)
    
    return {"next": intent}

from .data_parser import DataParser  # 상단에 import 추가


def generate_body_response(state):
    try:
        # 현재 메시지와 doc_id 가져오기
        current_message = state["messages"][-1]
        doc_id = state.get("doc_id", "kr20240172646b1")
        model_name = state.get("model_name", "exaone3.5:7.8b")  # 추가
        
        print("\n" + "="*50)
        print(f"[특허 본문 분석] 입력 질문: {current_message.content}")
        print(f"[특허 ID]: {doc_id}")
        
        # DataParser를 사용하여 특허 데이터 파싱
        parsed_data = DataParser.parse_for_summary(doc_id)
        
        if "error" in parsed_data:
            print(f"[파싱 오류]: {parsed_data['error']}")
            return {"messages": [AIMessage(content=f"죄송합니다. 특허 데이터를 불러오는 중 오류가 발생했습니다. (ID: {doc_id})")]}
        
        print("\n[파싱된 특허 본문]")
        for key, value in parsed_data.items():
            print(f"\n=== {key} ===")
            print(value[:200] + "..." if len(value) > 200 else value)
        
        # 주요 섹션 선택
        relevant_sections = {
            "발명의 명칭": parsed_data.get("발명의 명칭", ""),
            "기술 분야": parsed_data.get("기술 분야", "")[:1000],
            "발명의 요약": parsed_data.get("발명의 요약", "")[:1000],
            "기술적 과제": parsed_data.get("기술적 과제", "")[:1000],
            "해결 방안": parsed_data.get("해결 방안", "")[:1000],
            "효과": parsed_data.get("효과", "")[:1000]
        }
        
        # 분석할 컨텍스트 구성
        context = "\n\n".join([f"[{key}]\n{value}" for key, value in relevant_sections.items() if value.strip()])
        
        print("\n[분석할 특허 컨텍스트]")
        print(context[:500] + "..." if len(context) > 500 else context)
        
        # LLM에 전달할 프롬프트 구성
        prompt = BODY_PROMPT.format(
            context=context,
            question=current_message.content
        )
        
        print("\n[LLM 프롬프트]")
        print(prompt)
        
        # LLM으로 응답 생성
        model = get_model(model_name)
        response = model.invoke(prompt)
        
        print(f"\n[LLM 응답]")
        print(response.content)
        print("="*50)
        
        return {"messages": [response]}
        
    except Exception as e:
        logging.error(f"Error in generate_body_response: {str(e)}")
        return {"messages": [AIMessage(content=f"죄송합니다. 처리 중 오류가 발생했습니다: {str(e)}")]}

from .rag import (
    read_json_file, 
    split_json_into_chunks, 
    save_chunks_to_file,
    create_vectordb, 
    hybrid_search
)

from functools import lru_cache
import os
from .get_data import get_elasticsearch_data  # 상단에 추가
import json
# 캐시된 결과를 저장할 딕셔너리
_cache = {}

def generate_database_response(state):
    current_message = state["messages"][-1]
    doc_id = state.get("doc_id", "kr20240172646b1")  # state에서 doc_id 가져오기
    model_name = state.get("model_name", "exaone3.5:7.8b")  # 추가
    
    print("\n" + "="*50)
    print(f"[데이터베이스 검색] 입력 질문: {current_message.content}")
    print(f"[특허 ID]: {doc_id}")
    
    try:
        # 캐시된 데이터가 있으면 재사용
        if doc_id not in _cache:
            # Elasticsearch에서 데이터 가져오기
            data = get_elasticsearch_data(doc_id)
            if not data:
                return {"messages": [AIMessage(content=f"특허 데이터(ID: {doc_id})를 가져오는데 실패했습니다.")]}
            
            # 전체 source 데이터 사용
            json_text = json.dumps(data[0]["_source"], ensure_ascii=False)
            
            # JSON 청킹
            chunks = split_json_into_chunks(json_text)
            if not chunks:
                return {"messages": [AIMessage(content=f"특허 데이터(ID: {doc_id}) 처리 중 오류가 발생했습니다.")]}
            
            # 벡터 DB 생성
            vectorstore, _ = create_vectordb(chunks)
            if not vectorstore:
                return {"messages": [AIMessage(content="벡터 데이터베이스 생성 중 오류가 발생했습니다.")]}
            
            # 결과 캐시
            _cache[doc_id] = (vectorstore, chunks)
        
        # 캐시된 데이터 사용
        vectorstore, chunks = _cache[doc_id]
        
        # 하이브리드 검색 수행
        print("\n=== 검색 수행 ===")
        results = hybrid_search(current_message.content, vectorstore, chunks, k=3)
        
        # 검색 결과로 컨텍스트 구성
        context_parts = []
        for i, (text, hybrid_score, vector_score, bm25_score, tokens, is_keyword_match) in enumerate(results, 1):
            context_parts.append(f"[검색 결과 {i}]\n{text}")
            print(f"\n결과 {i}:")
            print(f"키워드 매칭: {'예' if is_keyword_match else '아니오'}")
            print(f"하이브리드 점수: {hybrid_score:.4f}")
            print(f"추출된 키워드: {', '.join(tokens[:10])}...")
        
        context = "\n\n".join(context_parts)
        
        print("\n[검색된 컨텍스트]")
        print(context[:200] + "..." if len(context) > 200 else context)
        
        # LLM 프롬프트 구성
        prompt = DATABASE_PROMPT.format(
            context=context,
            question=current_message.content
        )
        
        # LLM 응답 생성
        model = get_model(model_name)  # model_name 전달
        response = model.invoke(prompt)
        
        print(f"\n[LLM 응답]")
        print(response.content)
        print("="*50)
        
        return {"messages": [response]}
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        return {"messages": [AIMessage(content=f"처리 중 오류가 발생했습니다: {str(e)}")]}


generate_default_response = generate_database_response
# def generate_default_response(state):
#     current_message = state["messages"][-1]
#     model_name = state.get("model_name", "exaone3.5:7.8b")  # 추가
#     print("\n" + "="*50)
#     print(f"[기본 생성기] 입력 메시지: {current_message.content}")
    
#     prompt = DEFAULT_PROMPT.format(question=current_message.content)
#     model = get_model(model_name)
#     response = model.invoke(prompt)
    
#     print(f"[기본 생성기] 생성된 응답: {response.content}")
#     print("="*50)
#     return {"messages": [response]}

# ... existing code ...

from langchain_core.messages import AIMessage, HumanMessage  # 추가

from .tools import tavily_search  # tavily_search 함수 직접 import

def generate_search_response(state):
    current_message = state["messages"][-1]
    doc_id = state.get("doc_id", "kr20240172646b1")
    model_name = state.get("model_name", "exaone3.5:7.8b")

    print("\n" + "="*50)
    print(f"[검색 생성기] 입력 메시지: {current_message.content}")
    print(f"[특허 ID]: {doc_id}")
    
    try:
        # 특허의 발명 명칭 가져오기
        parsed_data = DataParser.parse_for_summary(doc_id)
        if "error" in parsed_data:
            print(f"[경고] 특허 데이터 파싱 실패: {parsed_data['error']}")
            invention_title = ""
        else:
            invention_title = parsed_data.get("발명의 명칭", "")
            print(f"[발명의 명칭]: {invention_title}")

        # 검색 쿼리 구성
        search_query = f"{invention_title} {current_message.content}" if invention_title else current_message.content
        print(f"[최종 검색 쿼리]: {search_query}")
        
        # Tavily 검색 수행
        search_results = tavily_search(search_query)
        print(f"[검색 생성기] 검색 결과: {search_results}")
        
        if not search_results or "검색 중 오류 발생" in str(search_results):
            response_content = "죄송합니다. 검색 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
            return {"messages": [AIMessage(content=response_content)]}
        
        # 검색 결과를 바탕으로 응답 생성
        prompt = f"""
        다음 검색 결과를 바탕으로 사용자의 질문에 대해 상세하게 답변해주세요.
        
        특허 제목: {invention_title}
        사용자 질문: {current_message.content}
        
        검색 결과:
        {search_results}
        
        위 특허와 관련된 정보를 중심으로 답변해주세요.
        너무 길게 적지 말고 핵심 부분만 요약해서 적어주세요.
        """
        
        model = get_model(model_name)
        response = model.invoke(prompt)
        
        print(f"[검색 생성기] 생성된 응답: {response.content}")
        print("="*50)
        return {"messages": [response]}
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        return {"messages": [AIMessage(content=f"처리 중 오류가 발생했습니다: {str(e)}")]}

# if __name__ == "__main__":
#     from langchain_core.messages import HumanMessage
    
#     print("\n=== 특허 본문 분석 테스트 ===")
    
#     test_queries = [
#         "이 특허의 목적은 무엇인가?",
#         "주요 기술적 특징은 무엇인가?",
#         "어떤 효과가 있는가?"
#     ]
    
#     for query in test_queries:
#         print(f"\n[테스트 질문] {query}")
        
#         test_state = {
#             "messages": [HumanMessage(content=query)]
#         }
        
#         try:
#             generate_body_response(test_state)  # 함수 내부에서 이미 응답을 출력하므로 여기서는 따로 출력하지 않음
#         except Exception as e:
#             print(f"\n[오류 발생] {str(e)}")
        
#         print("\n" + "="*50)

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    
    print("\n=== 데이터베이스 검색 테스트 ===")
    
    test_queries = [
        "이 특허의 청구항은 몇 개야?",
        "이 특허의 출원일이 언제야?",
        "이 특허의 출원인이 누구야?",
        "청구항 1번이 뭐야?"
    ]
    
    for query in test_queries:
        print(f"\n[테스트 질문] {query}")
        test_state = {
            "messages": [HumanMessage(content=query)]
        }
        
        try:
            generate_database_response(test_state)
        except Exception as e:
            print(f"\n[오류 발생] {str(e)}")
        
        print("\n" + "="*50)