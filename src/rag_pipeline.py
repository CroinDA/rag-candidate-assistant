# Retriever + RAG 체인 정의

import os
import re
from typing import List
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

from langchain_ollama import ChatOllama

# ingest.py에서 공통 함수 import
from ingest import (
    CHROMA_BASE_DIR,
    normalize_filename,
    get_chroma_dir,
    get_embeddings
)

load_dotenv()

# ============================================
# 1. 유틸리티 함수들
# ============================================

def get_available_candidates() -> List[str]:
    """
    저장된 지원자 목록 반환 (정규화 적용)
    """
    if not os.path.exists(CHROMA_BASE_DIR):
        return []
    
    candidates = []
    try:
        items = os.listdir(CHROMA_BASE_DIR)
    except Exception as e:
        print(f"  ⚠️ 디렉토리 읽기 실패: {e}")
        return []
    
    for item in items:
        candidate_dir = os.path.join(CHROMA_BASE_DIR, item)
        chroma_db = os.path.join(candidate_dir, "chroma_db")
        
        if os.path.isdir(candidate_dir) and os.path.exists(chroma_db):
            item_normalized = normalize_filename(item)
            candidates.append(item_normalized)
    
    return candidates


def extract_candidate_from_query(query: str) -> str:
    """
    질문에서 지원자 이름 추출 (정규화 적용)
    한글/영문 모두 정확히 매칭되도록 개선
    """
    candidates = get_available_candidates()
    
    if not candidates:
        return None
    
    # 질문 정규화 (NFC로 통일)
    query_normalized = normalize_filename(query.strip())
    
    # 각 지원자 이름으로 매칭 시도 (긴 이름부터 우선순위)
    candidates_sorted = sorted(candidates, key=len, reverse=True)
    
    for candidate in candidates_sorted:
        # 지원자 이름도 정규화 (이미 정규화되어 있지만 확실히)
        candidate_normalized = normalize_filename(candidate.strip())
        
        # 방법 1: 정확한 문자열 포함 체크
        if candidate_normalized in query_normalized:
            return candidate
        
        # 방법 2: 구두점 제거 후 매칭 (예: "박광진의" -> "박광진" 매칭)
        query_clean = re.sub(r'[^\w가-힣]', '', query_normalized)
        candidate_clean = re.sub(r'[^\w가-힣]', '', candidate_normalized)
        
        if candidate_clean in query_clean:
            return candidate
        
        # 방법 3: 단어 단위로 매칭 (한글은 공백 없이도 단어로 인식)
        # "박광진의" -> ["박광진", "의"] -> "박광진" 매칭
        query_words = re.findall(r'[\w가-힣]+', query_normalized)
        for word in query_words:
            word_normalized = normalize_filename(word)
            if candidate_clean in word_normalized:
                # 정확한 매칭 확인 (부분 문자열이 아닌)
                if len(candidate_clean) <= len(word_normalized) and candidate_clean in word_normalized:
                    return candidate
    
    return None


def check_vectorstore_exists() -> bool:
    """
    벡터스토어가 존재하는지 확인합니다.
    
    Returns:
        벡터스토어 존재 여부
    """
    return len(get_available_candidates()) > 0


# ============================================
# 2. 벡터 스토어 및 Retriever 함수들
# ============================================

def load_vectorstore(candidate_name: str):
    """
    지원자별 Chroma DB를 로드합니다.
    
    Args:
        candidate_name: 지원자 이름
        
    Returns:
        Chroma 벡터스토어 인스턴스
    """
    chroma_dir = get_chroma_dir(candidate_name)
    
    if not os.path.exists(chroma_dir):
        raise FileNotFoundError(
            f"'{candidate_name}' 지원자의 벡터 스토어를 찾을 수 없습니다.\n"
            f"경로: {chroma_dir}"
        )
    
    embeddings = get_embeddings()
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=chroma_dir,
    )
    return vectorstore


def get_retriever(candidate_name: str, k: int = 7):
    """
    지원자별 벡터스토어에서 retriever를 생성합니다.
    
    Args:
        candidate_name: 지원자 이름
        k: 검색할 문서 개수 (기본값: 7, 5페이지 CV 최적화)
        
    Returns:
        Retriever 인스턴스
    """
    vs = load_vectorstore(candidate_name)
    return vs.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )


# ============================================
# 3. RAG 체인 구성 함수들
# ============================================

def format_docs(docs: List[Document]) -> str:
    """
    검색된 문서들을 포맷팅합니다.
    
    Args:
        docs: 검색된 문서 리스트
        
    Returns:
        포맷팅된 문서 문자열
    """
    return "\n\n".join(doc.page_content.strip() for doc in docs)


def build_rag_chain(candidate_name: str):
    """
    지원자별 RAG 체인을 구성합니다.
    
    Args:
        candidate_name: 지원자 이름
        
    Returns:
        RAG 체인
    """
    retriever = get_retriever(candidate_name, k=7)

    system_prompt = """당신은 지원자의 이력서 정보를 기반으로 답변하는 전문 어시스턴트입니다.
    이력서는 한국어와 영어가 혼용되어 있을 수 있으며, 두 언어 모두를 완벽하게 이해하고 처리할 수 있습니다.

    **역할 및 지침:**
    1. 주어진 '지원자 이력서 정보'를 최대한 활용하여 질문에 구체적이고 상세하게 답변하세요.
    2. 질문이 한국어로 되어 있으면 한국어로, 영어로 되어 있으면 영어로 답변하세요. 이력서에 한국어와 영어가 혼용되어 있어도 정확하게 이해하고 답변하세요.
    3. 이력서의 구조(교육, 경력, 프로젝트, 기술스택 등)를 이해하고, 질문에 맞는 섹션의 정보를 찾아 답변하세요.
    4. 정보가 명확하지 않거나 문서에 없는 내용은 추측하지 말고 솔직하게 "해당 정보는 이력서에 없습니다"라고 답변하세요. 단, 질문의 의도를 파악하여 유사한 정보가 있다면 그것을 언급하세요.
    5. 여러 문서에서 관련 정보를 찾았다면, 시간순서나 중요도에 따라 종합하여 답변하세요.
    6. 날짜, 기간, 회사명, 학교명, 프로젝트명 등 구체적인 정보는 정확하게 언급하세요.
    7. 가능한 경우 출처(문서명, 페이지)를 언급하세요.

    **지원자 이력서 정보:**
    {context}
    """

    user_prompt = "질문: {question}"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", user_prompt),
        ]
    )

    # Ollama LLM 설정 (qwen2.5:14b - 더 강력한 모델)
    llm = ChatOllama(
        model="qwen2.5:14b",
        temperature=0.2,
        num_ctx=8192,  # 컨텍스트 윈도우 크기 (14B 모델에 맞게 증가)
    )

    # RAG 체인 구성
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ============================================
# 4. 메인 파이프라인
# ============================================

def ask_question(question: str) -> str:
    """
    질문에 대한 답변을 생성합니다.
    질문에서 지원자 이름을 자동으로 추출하여 해당 지원자의 벡터 DB를 사용합니다.
    
    Args:
        question: 사용자 질문
        
    Returns:
        생성된 답변
    """
    try:
        # 사용 가능한 지원자 목록 가져오기
        available = get_available_candidates()
        
        if not available:
            return "⚠️ 저장된 지원자 정보가 없습니다. 먼저 문서를 업로드해주세요."
        
        # 질문에서 지원자 이름 추출
        candidate_name = extract_candidate_from_query(question)
        
        if not candidate_name:
            candidates_list = ", ".join(available)
            return f"⚠️ 질문에 지원자 이름을 포함해주세요.\n\n사용 가능한 지원자: {candidates_list}\n\n예시: '{available[0]}의 경력을 알려줘'"
        
        # 해당 지원자의 RAG 체인 구성 및 답변 생성
        chain = build_rag_chain(candidate_name)
        answer = chain.invoke(question)
        return answer
        
    except FileNotFoundError as e:
        return f"⚠️ {str(e)}"
    except Exception as e:
        return f"❌ 답변 생성 중 오류가 발생했습니다: {str(e)}"
