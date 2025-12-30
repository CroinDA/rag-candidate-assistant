# Retriever + RAG 체인 정의

# src/rag_pipeline.py

import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_ollama import ChatOllama           # Ollama용 LLM
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

CHROMA_DIR = "vector_store/chroma_db"


def load_vectorstore():
    """이미 ingest.py에서 생성된 Chroma DB를 로드."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vectorstore


def get_retriever():
    vs = load_vectorstore()
    return vs.as_retriever(search_kwargs={"k": 3})


def build_rag_chain(target: str = "cv"):
    """
    target: 'cv' 또는 'domain' 같은 문자열.
    지금은 cv만 쓰지만, 나중에 target 값에 따라 다른 벡터DB를 로드하도록 확장할 수 있음.
    """
    retriever = get_retriever()     # 지금은 target을 무시하고 동일 retriever 사용

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    system_prompt = """
당신은 한 구직자의 이력서 정보를 기반으로 답변하는 어시스턴트입니다.
주어진 '지원자 이력서 정보'를 최대한 활용해 질문에 한국어 또는 영어로 구체적으로 답변하세요.
모르는 내용은 추측하지 말고 모른다고 말하세요.

지원자 이력서 정보:
{context}
"""

    user_prompt = "질문: {question}"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", user_prompt),
        ]
    )

    llm = ChatOllama(
        model="gemma2:2b",      # 로컬에서 돌리기 가벼운 모델(추가적인 성능 필요시 llama 3.1 추천)
        temperature=0.2,
    )

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


def ask_question(question: str) -> str:
    chain = build_rag_chain()
    return chain.invoke(question)


if __name__ == "__main__":
    q = "이 지원자가 해온 주요 프로젝트와 역할을 요약해줘."
    print("Q:", q)
    print("A:", ask_question(q))
