# from langchain_openai import ChatOpenAI
# from functools import lru_cache

# @lru_cache(maxsize=1)
# def get_model():
#     return ChatOpenAI(
#         model_name="gpt-3.5-turbo",
#         temperature=0
#     )

import requests
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from typing import Any, List, Optional
from functools import lru_cache

class ExaOneChat(BaseChatModel):
    url: str = "http://211.54.28.164:11434/api/chat"
    model_name: str = "exaone3.5:7.8b"
    
    def __init__(self, model_name: str = "exaone3.5:7.8b"):
        super().__init__()
        self.model_name = model_name
        print(f"\n[ExaOneChat] 모델 초기화: {model_name}")
    
    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any
    ) -> ChatResult:
        last_message = messages[-1].content if messages else ""
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": last_message}
            ],
            "stream": False
        }
        
        print(f"\n[API 요청] 사용 모델: {self.model_name}")
        print(f"[API 요청] 메시지: {last_message[:100]}...")
        
        response = requests.post(self.url, json=payload)
        if response.status_code == 200:
            content = response.json()["message"]["content"]
            print(f"[API 응답] 상태: 성공")
            print(f"[API 응답] 응답 내용: {content[:100]}...")
            message = AIMessage(content=content)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
        else:
            error_msg = f"API Error {response.status_code}: {response.text}"
            print(f"[API 응답] 상태: 실패\n{error_msg}")
            raise Exception(error_msg)
    
    @property
    def _llm_type(self) -> str:
        return "exaone"

_model_instance = None

def get_model(model_name: str = "exaone3.5:7.8b"):
    global _model_instance
    if _model_instance is None or _model_instance.model_name != model_name:
        print(f"\n[get_model] 새 모델 인스턴스 생성: {model_name}")
        _model_instance = ExaOneChat(model_name=model_name)
    else:
        print(f"\n[get_model] 기존 모델 인스턴스 재사용: {model_name}")
    return _model_instance