from elasticsearch import Elasticsearch
from typing import Optional, List, Dict, Any

def get_elasticsearch_data(doc_id: str) -> Optional[List[Dict[str, Any]]]:
    """
    Elasticsearch에서 문서 ID로 데이터를 검색하여 반환합니다.
    
    Args:
        doc_id (str): 검색할 문서 ID
        
    Returns:
        Optional[List[Dict[str, Any]]]: 검색 결과. 실패 시 None 반환
    """
    try:
        # Elasticsearch 연결
        es = Elasticsearch([{"host": "112.175.148.53", "port": 49200}])
        
        # ids 쿼리 구성
        query = {
            "query": {
                "ids": {
                    "values": [doc_id]  # 검색하고자 하는 문서 ID
                }
            }
        }
        
        # 검색 실행
        response = es.search(index="kipo", body=query)
        hits = response["hits"]["hits"]
        
        return hits if hits else None
        
    except Exception as e:
        print(f"Elasticsearch 검색 중 오류 발생: {str(e)}")
        return None

# 테스트용 코드
if __name__ == "__main__":
    test_id = "kr20250012784a"
    result = get_elasticsearch_data(test_id)
    if result:
        print(f"검색 결과: {result}")
    else:
        print("검색 결과가 없거나 오류가 발생했습니다.")