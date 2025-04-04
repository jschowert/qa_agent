import streamlit as st
from langgraph_example.my_agent.for_studio import graph
from langgraph_example.my_agent.utils.state import AgentState
from langchain_core.messages import HumanMessage

def main():
    # 페이지 설정
    st.set_page_config(
        page_title="특허 분석 AI 어시스턴트",
        page_icon="🤖",
        layout="wide"
    )

    # CSS 스타일 추가
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E88E5;
            margin-bottom: 2rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #424242;
            margin-bottom: 1rem;
        }
        .sample-question {
            padding: 0.8rem;
            background-color: #E3F2FD;
            border: 1px solid #90CAF9;
            border-radius: 0.5rem;
            margin-bottom: 0.8rem;
            cursor: pointer;
            color: #1565C0;
            transition: all 0.3s ease;
        }
        .sample-question:hover {
            background-color: #90CAF9;
            color: #FFFFFF;
            transform: translateX(5px);
        }
        </style>
    """, unsafe_allow_html=True)

    # 메인 헤더
    st.markdown("<h1 class='main-header'>🤖 특허 분석 AI 어시스턴트</h1>", unsafe_allow_html=True)
    
    # 사이드바 설정
    with st.sidebar:
        st.markdown("### ⚙️ 설정")
        
        # 모델 선택
        model_options = {
            "exaone3.5:7.8b": "ExaOne 3.5 (7.8B)",
            "phi4:latest": "Phi-4",
            "qwq:latest": "QWQ",
            "gemma3:27b": "Gemma-3 (27B)",
            "deepseek-r1:32b": "DeepSeek (32B)",
            "deepseek-r1:14b": "DeepSeek (14B)"
        }
        selected_model = st.selectbox(
            "모델 선택",
            list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=0
        )
        
        # doc_id 입력
        st.markdown("### 📄 특허 정보")
        doc_id = st.text_input(
            "특허 ID",
            value="kr20240172646b1",
            help="분석할 특허의 ID를 입력하세요"
        )
        
        # 구분선
        st.divider()
        
        # 샘플 질문
        st.markdown("### 💡 샘플 질문")
        sample_questions = [
            "이 특허의 청구항은 뭐야?",
            "이 특허의 목적이 뭐야?",
            "이 특허와 비슷한 특허 찾아줘",
            "이 특허의 발명자가 누구야?",
            "이 특허의 기술적 효과는 뭐야?",
            "이 특허의 해결하고자 하는 과제는?"
        ]
        
        for q in sample_questions:
            st.markdown(f"""
                <div class='sample-question' onclick='
                    document.querySelector("textarea").value = "{q}";
                    document.querySelector("textarea").focus();
                '>
                    {q}
                </div>
            """, unsafe_allow_html=True)
    
    # 메인 화면 채팅 인터페이스
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 채팅 메시지 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])
    
    # 새 메시지 입력
    if prompt := st.chat_input("질문을 입력하세요..."):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)
        
        # 에이전트 응답 생성
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🤔 생각하는 중..."):
                state = {
                    "messages": [HumanMessage(content=prompt)],
                    "doc_id": doc_id,
                    "model_name": selected_model
                }
                
                response = graph.invoke(state)
                result = response["messages"][-1].content
                st.markdown(result)
                
                st.session_state.messages.append({"role": "assistant", "content": result})

if __name__ == "__main__":
    main()