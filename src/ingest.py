# 문서 로드 + 청크 + 임베딩 + 벡터DB 저장

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_core import vectorstores
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

DATA_PATH = "data/cv.pdf"
CHROMA_DIR = "vector_store/chroma_db"

def load_documents():
    # PDF에서 문서를 로드
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} 파일을 찾을 수 없습니다.")
    loader = PyPDFLoader(DATA_PATH)
    docs = loader.load()
    return docs

def split_documents(documents):
    # 문서를 chunk 단위로 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vectorstore(chunks):
    # chunk들을 임베딩하고, Chroma 벡터DB에 저장
    # HuggingFace의 무료 임베딩 모델 사용 (API 키 불필요)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    os.makedirs(CHROMA_DIR, exist_ok=True)
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    vectorstore.persist()
    return vectorstore

def main():
    print("1) 문서 로드 중...")
    docs = load_documents()
    print(f"- 생성된 Chunk 수: {len(docs)}")
    
    print("2) 문서 분할 중...")
    chunks = split_documents(docs)
    print(f"- 생성된 Chunk 수: {len(chunks)}")
    
    print("3) 벡터 스토어 생성 및 저장중...")
    create_vectorstore(chunks)
    print(f"- Chroma DB가 {CHROMA_DIR}에 저장되었습니다.")
    
if __name__ == "__main__":
    main()