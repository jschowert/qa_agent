from rank_bm25 import BM25Okapi
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import numpy as np
import os
import gc
import shutil
import time
import MeCab
import json

# 필드 매핑 딕셔너리
detailed_fields_dict = {
    "agentInfo": "대리인",
    "applicantInfo": "출원인",
    "applicationDate": "출원일",
    "applicationNumber": "출원 번호",
    "citationDoc": "인용",
    "claimCount": "청구항 수",
    "claims": "청구항",
    "dartipInfo": "세계 소송 정보",
    "description": "상세설명",
    "drawings": "도면 정보",
    "extendedFamilyId": "확장된 패밀리 ID",
    "firstApplicantInfo": "최초 출원인",
    "inpadocFamilyId": "inpadoc 패밀리 id",
    "internationalApplicationDate": "국제 출원일",
    "internationalApplicationNumber": "국제 출원 번호",
    "internationalPublicationDate": "국제 공개일",
    "internationalPublicationNumber": "국제 공개 번호",
    "inventionTitle": "발명의 명칭",
    "inventorInfo": "발명자",
    "ipcVersion": "IPC 버전",
    "documentId": "키워트 고유 id",
    "patentDate": "특허일",
    "patentNumber": "특허 번호",
    "patentType": "특허 타입",
    "publicationDate": "공개일",
    "publicationNumber": "공개 번호",
    "registerDate": "등록일",
    "registerNumber": "등록 번호",
    "sepInfo": "표준 특허",
    "subCpc": "서브 CPC",
    "subIpc": "서브 IPC",
    "summary": "요약",
    "designatedStates": "지정국",
    "currentRightHolderInfo": "현재 권리자",
    "firstRightHolderInfo": "최초 권리자",
    "openDate": "공고일"
}

def read_json_file(file_path):
    """JSON 파일 읽기"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='cp949') as f:
            return f.read()

def convert_fields(data):
    """JSON 데이터의 필드명을 한글로 변환"""
    if isinstance(data, dict):
        return {
            detailed_fields_dict.get(k, k): convert_fields(v)
            for k, v in data.items()
        }
    elif isinstance(data, list):
        return [convert_fields(item) for item in data]
    else:
        return data


def split_json_into_chunks(json_text, chunk_size=500, batch_size=1000):
    """JSON 데이터를 의미 단위로 청킹"""
    try:
        # JSON 파싱
        data = json.loads(json_text)
        
        # 필드명 한글 변환
        data = convert_fields(data)
        
        chunks = []
        current_chunk = {}
        current_length = 0
        batch = []

        # JSON 데이터를 순회하며 청킹
        for key, value in data.items():
            # 큰 텍스트 필드 처리 (예: '상세설명')
            if isinstance(value, list) and len(str(value)) > chunk_size:
                # 현재까지의 청크가 있다면 저장
                if current_chunk:
                    batch.append(json.dumps(current_chunk, ensure_ascii=False))
                    current_chunk = {}
                    current_length = 0
                
                # 텍스트 내용을 청크 크기로 분할
                text = str(value)
                start = 0
                while start < len(text):
                    end = start + chunk_size
                    if end < len(text):
                        # 마지막 공백이나 구두점에서 자르기
                        while end > start and text[end] not in [' ', '.', ',', '\n', '>', '}']: 
                            end -= 1
                        if end == start:  # 적절한 구분점을 찾지 못한 경우
                            end = start + chunk_size  # 강제로 자르기
                    
                    chunk_text = text[start:end]
                    chunk_data = {key: chunk_text}
                    batch.append(json.dumps(chunk_data, ensure_ascii=False))
                    
                    if len(batch) >= batch_size:
                        chunks.extend(batch)
                        print(f"배치 처리 완료: {len(chunks)}개 청크")
                        batch = []
                        gc.collect()
                    
                    start = end
            else:
                # 일반 필드 처리
                field_json = json.dumps({key: value}, ensure_ascii=False)
                field_length = len(field_json)
                
                if current_length + field_length <= chunk_size:
                    current_chunk[key] = value
                    current_length += field_length
                else:
                    if current_chunk:
                        batch.append(json.dumps(current_chunk, ensure_ascii=False))
                        if len(batch) >= batch_size:
                            chunks.extend(batch)
                            print(f"배치 처리 완료: {len(chunks)}개 청크")
                            batch = []
                            gc.collect()
                    
                    current_chunk = {key: value}
                    current_length = field_length
        
        # 마지막 청크와 배치 처리
        if current_chunk:
            batch.append(json.dumps(current_chunk, ensure_ascii=False))
        
        if batch:
            chunks.extend(batch)
        
        # 중복 제거
        chunks = list(dict.fromkeys(chunks))
        print(f"총 생성된 청크 수: {len(chunks)}")
        
        return chunks
    
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {str(e)}")
        return []
    
# FAISS 인덱스가 저장되는 디렉토리
VECTOR_DB_DIR = "vector_db_cache"

def create_vectordb(chunks):
    """청크로부터 벡터 DB 생성"""
    try:
        print("\n벡터 DB 생성 시작...")
        start_time = time.time()
        
        # OpenAI 임베딩 설정
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            chunk_size=1000
        )
        
        # 벡터 DB 생성
        vectorstore = FAISS.from_texts(chunks, embeddings)
        
        # 캐시 디렉토리에 저장
        vectorstore.save_local(VECTOR_DB_DIR)
        
        elapsed = time.time() - start_time
        print(f"벡터 DB 생성 완료: {elapsed:.2f}초")
        
        return vectorstore, embeddings
        
    except Exception as e:
        print(f"벡터 DB 생성 중 오류 발생: {str(e)}")
        return None, None

# MeCab 토크나이저 초기화
try:
    mecab = MeCab.Tagger('-r /opt/homebrew/etc/mecabrc -d /opt/homebrew/lib/mecab/dic/mecab-ko-dic')
except Exception as e:
    print(f"MeCab 초기화 실패: {str(e)}")
    try:
        mecab = MeCab.Tagger()
    except Exception as e:
        print(f"기본 설정으로도 MeCab 초기화 실패: {str(e)}")
        mecab = MeCab.Tagger('')

def tokenize_korean(text):
    """MeCab을 사용한 한국어 명사 추출"""
    try:
        parsed = mecab.parse(text)
        words = []
        for line in parsed.split('\n'):
            if line == 'EOS' or line == '':
                continue
            word, feature = line.split('\t')
            pos = feature.split(',')[0]
            if pos in ['NNG', 'NNP']:
                words.append(word)
        return words
    except Exception as e:
        print(f"토크나이징 실패: {str(e)}")
        return [w for w in text.split() if len(w) > 1]


def get_field_keywords():
    """detailed_fields_dict의 value들을 토크나이징하여 키워드 사전 생성"""
    field_keywords = {}
    for eng, kor in detailed_fields_dict.items():
        tokens = tokenize_korean(kor)
        for token in tokens:
            if token not in field_keywords:
                field_keywords[token] = []
            field_keywords[token].append(kor)
    return field_keywords

def find_matching_chunks(query_tokens, chunks):
    """쿼리 토큰과 매칭되는 필드를 포함하는 청크 찾기"""
    field_keywords = get_field_keywords()
    matching_fields = set()
    
    # 쿼리 토큰과 매칭되는 필드 찾기
    for token in query_tokens:
        if token in field_keywords:
            matching_fields.update(field_keywords[token])
    
    # 매칭되는 필드를 포함하는 청크 찾기
    matching_chunks = []
    for chunk in chunks:
        for field in matching_fields:
            if f'"{field}":' in chunk:
                matching_chunks.append(chunk)
                break
    
    return matching_chunks

def hybrid_search(query, vectorstore, chunks, alpha=0.5, k=5):
    try:
        # 쿼리 토크나이징
        query_tokens = tokenize_korean(query)
        
        # 키워드 매칭으로 관련 청크 찾기
        keyword_matches = find_matching_chunks(query_tokens, chunks)
        
        # 검색된 문서들의 인덱스 찾기
        search_indices = []
        matched_indices = set()
        
        # 키워드 매칭된 청크 모두 포함
        if keyword_matches:
            print(f"키워드 매칭된 청크 수: {len(keyword_matches)}")
            for chunk in keyword_matches:
                try:
                    idx = chunks.index(chunk)
                    if idx not in matched_indices:
                        search_indices.append(idx)
                        matched_indices.add(idx)
                except ValueError:
                    continue
        
        # 키워드 매칭 결과가 k개 미만이면 벡터 검색으로 추가
        remaining_k = max(0, k - len(search_indices))
        vector_results = []
        
        if remaining_k > 0:
            # 벡터 검색 수행 - 더 많은 결과를 가져와서 필터링
            vector_results = vectorstore.similarity_search_with_score(query, k=remaining_k * 2)
            
            # 벡터 검색 결과 처리
            for doc, score in vector_results:
                if len(search_indices) >= k:
                    break
                    
                doc_text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                try:
                    idx = chunks.index(doc_text)
                    if idx not in matched_indices:
                        search_indices.append(idx)
                        matched_indices.add(idx)
                except ValueError:
                    for i, chunk in enumerate(chunks):
                        if (doc_text in chunk or chunk in doc_text) and i not in matched_indices:
                            search_indices.append(i)
                            matched_indices.add(i)
                            break
        
        # BM25 검색
        tokenized_docs = [tokenize_korean(doc) for doc in chunks]
        bm25 = BM25Okapi(tokenized_docs)
        bm25_scores = bm25.get_scores(query_tokens)
        
        # 점수 계산 및 정규화
        vector_scores = []
        for idx in search_indices:
            if chunks[idx] in keyword_matches:
                vector_scores.append(1.0)
            else:
                # 벡터 검색 결과에서 점수 찾기
                matched_score = 0.0
                for doc, score in vector_results:
                    doc_text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                    if chunks[idx] == doc_text or chunks[idx] in doc_text or doc_text in chunks[idx]:
                        matched_score = float(1 - score)
                        break
                vector_scores.append(matched_score)
        
        vector_scores = np.array(vector_scores)
        if len(vector_scores) > 1:
            vector_scores = (vector_scores - vector_scores.min()) / (vector_scores.max() - vector_scores.min() + 1e-10)
        
        bm25_scores = np.array([bm25_scores[idx] for idx in search_indices])
        if len(bm25_scores) > 1:
            bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-10)
        
        # 하이브리드 점수 계산
        hybrid_scores = alpha * vector_scores + (1 - alpha) * bm25_scores
        
        # 결과 생성
        results = []
        for i, idx in enumerate(search_indices):
            results.append((
                chunks[idx],
                hybrid_scores[i],
                vector_scores[i],
                bm25_scores[i],
                tokenized_docs[idx],
                chunks[idx] in keyword_matches
            ))
        
        # 키워드 매칭 결과를 우선하여 정렬
        results.sort(key=lambda x: (not x[5], -x[1]))
        
        print(f"총 반환 결과 수: {len(results)} (키워드 매칭: {len(keyword_matches)}, 벡터 검색: {len(results) - len(keyword_matches)})")
        return results
            
    except Exception as e:
        print(f"하이브리드 검색 중 오류 발생: {str(e)}")
        return [(doc.page_content if hasattr(doc, 'page_content') else str(doc), 0, 0, 0, [], False) for doc, _ in vector_results[:k]]



def save_chunks_to_file(chunks, output_file="../data/processed_chunks.txt"):
    """청크를 하나의 텍스트 파일로 저장"""
    try:
        # 출력 디렉토리 생성
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        print(f"\n=== 청크 저장 시작 ({output_file}) ===")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks, 1):
                f.write(f"청크 {i}:\n")
                f.write(chunk)
                f.write("\n\n")
                
        print(f"청크 저장 완료: 총 {len(chunks)}개 청크")
        print(f"저장 위치: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"청크 저장 중 오류 발생: {str(e)}")
        return False

# 테스트 코드 수정
if __name__ == "__main__":
    # 파일 경로 설정
    input_file = "../data/kr20240172646b1_full.txt"
    output_file = "../data/processed_chunks.txt"
    
    try:
        # 1. JSON 파일 읽기
        print("\n=== JSON 파일 읽기 ===")
        json_text = read_json_file(input_file)
        if not json_text:
            print("파일 읽기 실패")
            exit(1)
            
        # 2. JSON 청킹
        print("\n=== JSON 청킹 시작 ===")
        chunks = split_json_into_chunks(json_text)
        if not chunks:
            print("청킹 실패")
            exit(1)
            
        print("\n[청크 샘플]")
        for i, chunk in enumerate(chunks, 1):
            print(f"\n청크 {i}:")
            print(f"{chunk[:200]}...")
            
        # 3. 청크 저장
        if not save_chunks_to_file(chunks, output_file):
            print("청크 저장 실패")
            exit(1)
            
        # 4. 벡터 DB 생성
        vectorstore, embeddings = create_vectordb(chunks)
        if not vectorstore or not embeddings:
            print("벡터 DB 생성 실패")
            exit(1)
                    
        # 5. 테스트 검색
        test_queries = [
            "이 특허의 목적은 무엇인가?",
            "주요 기술적 특징은 무엇인가?",
            "이 기술에 청구항은 몇개야?",
            "이 기술에 청구항좀 알려줘"
        ]
        
        print("\n=== 검색 테스트 시작 ===")
        for query in test_queries:
            print(f"\n[검색 쿼리] {query}")
            results = hybrid_search(query, vectorstore, chunks, k=5)
            
            print("\n[검색 결과]")
            for i, (text, hybrid_score, vector_score, bm25_score, tokens, is_keyword_match) in enumerate(results, 1):
                print(f"\n결과 {i}:")
                print(f"키워드 매칭: {'예' if is_keyword_match else '아니오'}")
                print(f"텍스트: {text[:200]}...")
                print(f"하이브리드 점수: {hybrid_score:.4f}")
                print(f"벡터 점수: {vector_score:.4f}")
                print(f"BM25 점수: {bm25_score:.4f}")
                print(f"추출된 키워드: {', '.join(tokens[:10])}...")
            
            print("\n" + "="*50)
        
        print("\n검색 테스트 완료!")
        
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")