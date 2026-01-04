# UI (Streamlit)

import streamlit as st
import os
import tempfile
from pathlib import Path
from rag_pipeline import ask_question, check_vectorstore_exists, get_available_candidates
from ingest import process_uploaded_documents

st.set_page_config(page_title="Candidate CV RAG", page_icon="🖥️", layout="wide")

st.title("📄 Candidate CV RAG Assistant")
st.write("지원자의 이력서를 업로드하고 질문에 답하는 RAG 챗봇입니다.")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "vectorstore_ready" not in st.session_state:
    st.session_state["vectorstore_ready"] = check_vectorstore_exists()
if "uploaded_files_list" not in st.session_state:
    st.session_state["uploaded_files_list"] = []

# 사이드바: 문서 업로드 및 설정
with st.sidebar:
    st.header("⚙️ 설정")
    
    # 문서 업로드 섹션
    st.subheader("📤 문서 업로드")
    
    uploaded_files = st.file_uploader(
        "PDF 파일을 업로드하세요",
        type=["pdf"],
        accept_multiple_files=True,
        help="여러 개의 PDF 파일을 동시에 업로드할 수 있습니다."
    )
    
    if uploaded_files:
        if st.button("🚀 업로드한 지원자 문서 처리", type="primary"):
            with st.spinner("문서 처리중..."):
                try:
                    # 임시 디렉토리에 업로드된 파일 저장
                    temp_dir = tempfile.mkdtemp()
                    file_paths = []
                    original_filenames = []
                    
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        file_paths.append(file_path)
                        original_filenames.append(uploaded_file.name)
                    
                    # 문서 처리 및 벡터DB 생성
                    result = process_uploaded_documents(file_paths, original_filenames=original_filenames)
                    
                    # 세션 상태 업데이트
                    st.session_state["vectorstore_ready"] = True
                    st.session_state["uploaded_files_list"].extend([f.name for f in uploaded_files])
                    
                    st.success(f"✅ {result['num_chunks']}개의 청크가 생성되어 벡터 DB에 저장되었습니다!")
                    st.info(f"📁 처리된 파일: {', '.join([f.name for f in uploaded_files])}")
                    
                    # 임시 파일 정리
                    import shutil
                    shutil.rmtree(temp_dir)
                    
                except Exception as e:
                    st.error(f"❌ 문서 처리 중 오류가 발생했습니다: {str(e)}")
    
    # 업로드된 지원자 목록 표시
    st.divider()
    st.subheader("👥 등록된 지원자")
    candidates = get_available_candidates()
    if candidates:
        for idx, candidate in enumerate(candidates, 1):
            st.text(f"{idx}. {candidate}")
    else:
        st.info("등록된 지원자가 없습니다.")
    
    # 벡터 DB 상태 표시
    st.divider()
    st.subheader("🗄️ 벡터 DB 상태")
    if st.session_state["vectorstore_ready"]:
        st.success("✅ 벡터 DB 준비 완료")
    else:
        st.warning("⚠️ 벡터 DB가 없습니다. 문서를 업로드해주세요.")
    
    # 시스템 정보
    st.divider()
    st.subheader("ℹ️ 시스템 정보")
    st.markdown(
        """
        - **LLM**: Ollama (qwen2.5:14b)
        - **임베딩**: HuggingFace MiniLM-L6-v2
        - **벡터 DB**: ChromaDB
        - **청크 크기**: 800 (오버랩: 200)
        - **검색 결과**: Top-3
        - **GPU**: CUDA 12.1 지원
        """
    )
    
    # 벡터 DB 초기화 버튼
    if st.button("🗑️ 모든 벡터 DB 초기화"):
        import shutil
        vector_base_dir = "vector_store"
        if os.path.exists(vector_base_dir):
            shutil.rmtree(vector_base_dir)
            os.makedirs(vector_base_dir, exist_ok=True)
            st.session_state["vectorstore_ready"] = False
            st.session_state["uploaded_files_list"] = []
            st.session_state["messages"] = []
            st.success("✅ 모든 벡터 DB가 초기화되었습니다.")
            st.rerun()

# 메인 채팅 영역
if not st.session_state["vectorstore_ready"]:
    st.info("👈 왼쪽 사이드바에서 PDF 문서를 업로드하고 처리해주세요.\n\n💡 파일명은 `{지원자명}_CV.pdf` 형식으로 업로드해주세요. (예: Kyle_CV.pdf, 박광진_CV.pdf)")
else:
    # 등록된 지원자 안내
    candidates = get_available_candidates()
    if candidates:
        st.info(f"💬 질문 시 지원자 이름을 포함해주세요.\n\n등록된 지원자: {', '.join(candidates)}\n\n예시: '{candidates[0]}의 경력을 알려줘'")
    # 이전 대화 렌더링
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # 사용자 입력
    user_input = st.chat_input("무엇이든 물어보세요...")
    
    if user_input:
        # 유저 메시지 저장/표시
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # RAG 호출
        with st.chat_message("assistant"):
            with st.spinner("🤔 생각하는 중..."):
                try:
                    answer = ask_question(user_input)
                    st.markdown(answer)
                    # 답변 저장
                    st.session_state["messages"].append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_msg = f"❌ 오류가 발생했습니다: {str(e)}"
                    st.error(error_msg)
                    st.session_state["messages"].append({"role": "assistant", "content": error_msg})
