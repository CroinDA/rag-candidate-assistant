# 문서 로드 + 청크 + 임베딩 + 벡터DB 저장

import os
import unicodedata
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb

load_dotenv()

# ============================================
# 상수 설정
# ============================================
CHROMA_BASE_DIR = "vector_store"

# GPU/CPU 자동 감지
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

EMBEDDING_MODEL_KWARGS = {
    'device': DEVICE
}

EMBEDDING_ENCODE_KWARGS = {
    'normalize_embeddings': True
}


# ============================================
# 1. 유틸리티 함수들
# ============================================

def normalize_filename(filename: str) -> str:
    """
    파일명을 NFC로 정규화 (Linux/Docker 표준)
    macOS의 NFD 형태도 NFC로 통일하여 처리
    """
    return unicodedata.normalize('NFC', filename)


def extract_candidate_name(filename: str) -> str:
    """
    파일명에서 지원자 이름 추출 (정규화 적용)
    예: "박광진_CV.pdf" -> "박광진"
    """
    basename = os.path.basename(filename)
    # 정규화 적용
    basename = normalize_filename(basename)
    
    if '_CV.pdf' in basename:
        candidate = basename.replace('_CV.pdf', '')
    else:
        # 규칙에 맞지 않으면 .pdf 제거
        candidate = basename.replace('.pdf', '')
    
    # 반환 시에도 정규화
    return normalize_filename(candidate)


def get_chroma_dir() -> str:
    """
    단일 벡터 DB 경로 반환
    예: "vector_store/chroma_db"
    """
    return os.path.join(CHROMA_BASE_DIR, "chroma_db")


def get_candidates_dir() -> str:
    """
    지원자 인덱스용 ChromaDB 경로 반환
    예: "vector_store/candidates_index"
    """
    return os.path.join(CHROMA_BASE_DIR, "candidates_index")


def get_candidates_collection():
    """
    지원자 목록을 관리하는 ChromaDB 컬렉션을 반환합니다.
    (지원자를 구별하기 위한 가벼운 인덱스)
    """
    candidates_dir = get_candidates_dir()
    os.makedirs(candidates_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=candidates_dir)
    collection = client.get_or_create_collection("candidates_index")
    return collection


# ============================================
# 2. 문서 처리 함수들
# ============================================

def load_documents_from_paths(file_paths: List[str], original_filenames: List[str] = None) -> Dict[str, List[Any]]:
    """
    여러 PDF 파일 경로를 받아서 지원자별로 문서를 로드합니다.
    
    Args:
        file_paths: PDF 파일 경로 리스트
        original_filenames: 원본 파일명 리스트 (파일명 깨짐 방지용)
        
    Returns:
        지원자별 문서 딕셔너리 {지원자명: [문서들]}
    """
    docs_by_candidate = {}
    
    for idx, file_path in enumerate(file_paths):
        if not os.path.exists(file_path):
            print(f"⚠️ 경고: {file_path} 파일을 찾을 수 없습니다. 건너뜁니다.")
            continue
            
        try:
            # 원본 파일명 사용 (파일명 깨짐 방지)
            if original_filenames and idx < len(original_filenames):
                original_name = normalize_filename(original_filenames[idx])
            else:
                original_name = normalize_filename(os.path.basename(file_path))
            
            # 지원자 이름 추출
            candidate_name = extract_candidate_name(original_name)
            
            loader = PyMuPDFLoader(file_path)
            docs = loader.load()
            
            # 메타데이터에 원본 파일명과 지원자명 추가
            for doc in docs:
                doc.metadata['source_file'] = original_name
                doc.metadata['candidate'] = candidate_name
            
            # 지원자별로 분류
            if candidate_name not in docs_by_candidate:
                docs_by_candidate[candidate_name] = []
            docs_by_candidate[candidate_name].extend(docs)
            
            print(f"✅ {file_path} 로드 완료 ({len(docs)} 페이지) - 지원자: {candidate_name}")
            
        except Exception as e:
            print(f"❌ {file_path} 로드 실패: {str(e)}")
            continue
    
    return docs_by_candidate


def split_documents(documents: List[Any]) -> List[Any]:
    """
    문서를 chunk 단위로 분할합니다.
    
    Args:
        documents: 분할할 문서 리스트
        
    Returns:
        청크로 분할된 문서 리스트
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,  # 800 → 1200 (섹션 단위 보존)
        chunk_overlap=300,  # 200 → 300 (경계 정보 보존)
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def get_embeddings():
    """
    임베딩 모델을 생성합니다 (GPU 지원, 한글/영어 지원)
    
    Returns:
        HuggingFaceEmbeddings 인스턴스
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 다국어 지원 (한/영)
        model_kwargs=EMBEDDING_MODEL_KWARGS,
        encode_kwargs=EMBEDDING_ENCODE_KWARGS
    )
    return embeddings


def create_vectorstore(chunks: List[Any], persist: bool = True) -> Chroma:
    """
    청크들을 임베딩하고 단일 Chroma 벡터DB에 저장합니다.
    
    Args:
        chunks: 임베딩할 청크 리스트 (메타데이터에 candidate 정보 포함)
        persist: 벡터 DB를 디스크에 저장할지 여부
        
    Returns:
        Chroma 벡터스토어 인스턴스
    """
    embeddings = get_embeddings()
    
    chroma_dir = get_chroma_dir()
    os.makedirs(chroma_dir, exist_ok=True)
    
    # 폴더 생성 확인
    if not os.path.exists(chroma_dir):
        raise RuntimeError(f"벡터 DB 디렉토리 생성 실패: {chroma_dir}")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_dir,
    )
    
    if persist:
        vectorstore.persist()
    
    # 저장 후 다시 확인
    if not os.path.exists(chroma_dir):
        raise RuntimeError(f"벡터 DB 저장 후 디렉토리 확인 실패: {chroma_dir}")
    
    return vectorstore


def add_to_existing_vectorstore(chunks: List[Any]) -> Chroma:
    """
    기존 벡터스토어에 새로운 청크를 추가합니다.
    
    Args:
        chunks: 추가할 청크 리스트 (메타데이터에 candidate 정보 포함)
        
    Returns:
        업데이트된 Chroma 벡터스토어 인스턴스
    """
    embeddings = get_embeddings()
    chroma_dir = get_chroma_dir()
    
    # 기존 벡터스토어 로드
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=chroma_dir,
    )
    
    # 새 문서 추가
    vectorstore.add_documents(chunks)
    vectorstore.persist()
    
    return vectorstore


# ============================================
# 3. 메인 파이프라인
# ============================================

def process_uploaded_documents(file_paths: List[str], original_filenames: List[str] = None) -> Dict[str, Any]:
    """
    업로드된 문서들을 지원자별로 처리하여 벡터 DB에 저장합니다.
    
    Args:
        file_paths: 처리할 PDF 파일 경로 리스트
        original_filenames: 원본 파일명 리스트 (파일명 깨짐 방지용)
        
    Returns:
        처리 결과 정보 딕셔너리
    """
    print("=" * 60)
    print("📄 문서 처리 시작")
    print("=" * 60)
    
    # 1. 지원자별 문서 로드
    print("\n1️⃣ 문서 로드 중...")
    docs_by_candidate = load_documents_from_paths(file_paths, original_filenames)
    
    if not docs_by_candidate:
        raise ValueError("로드된 문서가 없습니다.")
    
    total_docs = sum(len(docs) for docs in docs_by_candidate.values())
    print(f"✅ 총 {total_docs} 페이지 로드 완료 ({len(docs_by_candidate)}명의 지원자)\n")
    
    # 2. 모든 문서를 하나로 합쳐서 처리
    total_chunks = 0
    processed_candidates = []
    all_chunks = []
    
    for candidate_name, docs in docs_by_candidate.items():
        print(f"\n🔹 지원자: {candidate_name}")
        
        # 문서 분할
        print("  2️⃣ 문서 분할 중...")
        chunks = split_documents(docs)
        print(f"  ✅ {len(chunks)}개의 청크 생성 완료")
        
        all_chunks.extend(chunks)
        total_chunks += len(chunks)
        processed_candidates.append(candidate_name)
    
    # 3. 벡터 스토어 생성 또는 업데이트 (단일 DB)
    print("\n3️⃣ 벡터 스토어 처리 중...")
    chroma_dir = get_chroma_dir()
    
    if os.path.exists(chroma_dir) and os.listdir(chroma_dir):
        print(f"  기존 벡터 DB에 추가합니다...")
        vectorstore = add_to_existing_vectorstore(all_chunks)
    else:
        print(f"  새로운 벡터 DB를 생성합니다...")
        vectorstore = create_vectorstore(all_chunks)
    
    print(f"  ✅ 벡터 DB 저장 완료: {chroma_dir}")

    # 4. 지원자 인덱스 업데이트 (메타데이터 기반)
    print("\n4️⃣ 지원자 인덱스 업데이트 중...")
    try:
        candidates_collection = get_candidates_collection()
        unique_candidates = sorted(set(processed_candidates))
        if unique_candidates:
            candidates_collection.upsert(
                ids=unique_candidates,
                metadatas=[{"candidate": name} for name in unique_candidates],
                documents=["" for _ in unique_candidates],
            )
        print("  ✅ 지원자 인덱스 업데이트 완료")
    except Exception as e:
        print(f"  ⚠️ 지원자 인덱스 업데이트 실패: {e}")
    
    print("\n" + "=" * 60)
    print("✨ 문서 처리 완료!")
    print(f"처리된 지원자: {', '.join(processed_candidates)}")
    print("=" * 60)
    
    return {
        "num_docs": total_docs,
        "num_chunks": total_chunks,
        "candidates": processed_candidates,
        "vectorstore_path": CHROMA_BASE_DIR
    }
