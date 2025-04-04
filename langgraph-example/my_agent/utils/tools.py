# from langchain_community.tools.tavily_search import TavilySearchResults

# tools = [TavilySearchResults(max_results=1)]


# ... existing code ...

from langchain.tools import Tool
from tavily import TavilyClient
from typing import Optional
import os
from dotenv import load_dotenv
from pathlib import Path

# .env 파일 경로 수정
env_path = Path(__file__).parents[3] / 'langgraph-example' / '.env'  # 경로 수정
load_dotenv(env_path)

def tavily_search(query: str) -> Optional[str]:
    """
    Tavily API를 사용하여 웹 검색을 수행합니다.
    """
    try:
        # 환경 변수에서 API 키 가져오기
        api_key = os.getenv("TAVILY_API_KEY")
        client = TavilyClient(api_key=api_key)
        print(f"[Tavily 검색] 쿼리: {query}")
        search_result = client.search(query=query, search_depth="advanced")
        print(f"[Tavily 검색] 결과: {search_result}")
        
        # 검색 결과 포맷팅
        formatted_results = []
        for result in search_result.get('results', []):
            formatted_results.append(f"제목: {result.get('title')}\n내용: {result.get('content')}\n")
        
        return "\n".join(formatted_results) if formatted_results else "검색 결과가 없습니다."
        
    except Exception as e:
        print(f"Tavily 검색 중 오류 발생: {str(e)}")
        return f"검색 중 오류 발생: {str(e)}"

# 도구 목록 정의
tools = [
    Tool(
        name="web_search",
        description="웹에서 정보를 검색할 때 사용합니다.",
        func=tavily_search
    )
]

# 테스트 코드
if __name__ == "__main__":
    # 테스트 검색 실행
    test_query = "인공지능이란 무엇인가?"
    result = tavily_search(test_query)
    print("\n=== 테스트 검색 결과 ===")
    print(result)