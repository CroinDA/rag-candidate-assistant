# UI (Streamlit)

import streamlit as st
from rag_pipeline import ask_question

st.set_page_config(page_title="Candidate CV RAG", page_icon="🖥️")

st.title("Candidate CV RAG Assistant")
st.write("지원자의 이력서를 기반으로 질문에 답하는 RAG 챗봇입니다.")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"]=[]
    
# 사이드바: 간단 설명
with st.sidebar:
    st.header("설정")
    
    # 1) 대상 선택 셀렉트박스
    target = st.selectbox(
        "질문 대상 선택",
        options=["이력서(CV)", "도메인 문서(준비중)"],
        index=0,
    )
    
    st.markdown(
        """
        - 데이터: 현재는 지원자 이력서 1개만 사용
        - LLM: Ollama 로컬 모델 (예: llama3, gemma2 등)
        - 검색: Chroma + HuggingFace 임베딩
        """
    )
    
# 내부적으로 사용할 target 코드값으로 변환
if target.startswith("이력서"):
    target_code = "cv"
else:
    target_code = "domain"  # 나중에 구현할 용도
    
# 이전 대화 렌더링
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
# 사용자 입력
user_input = st.chat_input("무엇이든 물어보세요.")

if user_input:
    # 유저 메시지 저장/표시
    st.session_state["messages"].append({"role":"user","content":user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    # RAG 호출(선택된 target 코드 전달)
    with st.chat_message("assistant"):
        with st.spinner("생각중..."):
            answer = ask_question(user_input)
            st.markdown(answer)
            
    # 답변 저장
    st.session_state["messages"].append({"role": "assistant", "content": answer})
