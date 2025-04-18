
DEFAULT_PROMPT = """
당신은 도움이 되는 답변을 제공하는 어시스턴트입니다.
질문: {question}
친절하고 상세한 답변을 제공해주세요.
너무 길게 적지 말고 핵심 부분만 짧게 적어주세요.
"""

BODY_PROMPT = """
다음은 특허 문서의 주요 내용입니다:

{context}

사용자 질문: {question}

위의 특허 문서 내용을 바탕으로 사용자의 질문에 상세하게 답변해주세요.
가능한 한 구체적인 정보를 포함시켜 주시고, 특허 문서의 내용에 충실하게 답변해주세요.
너무 길게 적지 말고 핵심 부분만 짧게 적어주세요.
"""

DATABASE_PROMPT = """
다음은 특허 문서에서 검색된 관련 정보입니다:

{context}

사용자 질문: {question}

위의 검색 결과를 바탕으로 사용자의 질문에 정확하게 답변해주세요.
검색된 정보에서 찾을 수 있는 내용만을 포함하여 답변해주세요.
너무 길게 적지 말고 핵심 부분만 짧게 적어주세요.
"""

SEARCH_PROMPT = """
당신은 도움이 되는 답변을 제공하는 어시스턴트입니다.
질문: {question}
친절하고 상세한 답변을 제공해주세요.
너무 길게 적지 말고 핵심 부분만 짧게 적어주세요.
"""