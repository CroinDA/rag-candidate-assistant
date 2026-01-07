# 지원자 이력서 기반 RAG 챗봇 (Candidate CV RAG Assistant)

이 프로젝트는 **Streamlit**, **LangChain**, **Ollama**, **ChromaDB**를 기반으로 한 RAG(Retrieval-Augmented Generation) 챗봇(MVP)입니다. 여러 지원자의 이력서(CV)를 업로드하면, 단일 벡터 데이터베이스에 메타데이터로 지원자를 구분하여 저장하고, 질문에 포함된 지원자 이름을 자동으로 인식하여 메타데이터 필터링으로 해당 지원자의 정보만을 기반으로 답변을 제공합니다.

## ✨ 주요 기능

- **지원자별 메타데이터 자동 태깅**: 업로드된 CV 파일명(`{지원자명}_CV.pdf`)을 기반으로 각 청크의 메타데이터에 지원자 정보를 자동으로 포함하여 단일 벡터 데이터베이스에 저장합니다.
- **지원자 자동 인식**: 질문에 포함된 지원자 이름을 자동으로 추출하여 메타데이터 필터링으로 해당 지원자의 정보만 검색할 수 있도록 구현하였습니다.
- **다중 파일 동시 처리**: 여러 지원자의 CV를 동시에 업로드하여 단일 벡터 데이터베이스에 한 번에 저장할 수 있습니다.
- **한글 파일명 지원**: Unicode 정규화(NFC)를 통해 macOS와 Linux/Docker 환경 간 한글 파일명 호환성을 향상시킵니다.
- **실시간 웹 인터페이스**: Streamlit을 사용한 직관적인 웹 UI로 문서 업로드 및 챗봇 상호작용을 제공합니다.
- **로컬 LLM 연동**: Ollama를 통해 로컬 환경에서 `qwen2.5:14b` 모델을 사용하여 데이터가 외부 서버로 전송되지 않아 프라이버시를 보호합니다.
- **GPU 가속 지원**: NVIDIA CUDA 12.1을 활용한 GPU 가속으로 임베딩 생성 속도를 향상시킵니다.
- **컨테이너 기반 실행**: Docker와 Docker Compose를 사용하여 의존성 문제를 최소화하여 일관된 환경에서 실행할 수 있도록 하였습니다.

## 🛠️ 기술 스택

### 핵심 프레임워크 및 라이브러리

- **언어**: Python 3.10
- **웹 프레임워크**: 
  - **Streamlit** (>=1.38.0): 대화형 웹 UI 구축
- **RAG 파이프라인**:
  - **LangChain** (>=0.3.0): RAG 체인 구성 및 문서 처리
    - `langchain-core`: 핵심 체인 및 런타임
    - `langchain-community`: 커뮤니티 통합 (ChromaDB, Ollama 등)
    - `langchain-ollama`: Ollama LLM 통합
    - `langchain-text-splitters`: 문서 청킹 전략
- **LLM/AI**:
  - **Ollama**: 로컬 LLM 구동 (qwen2.5:14b 모델)
  - **Sentence-Transformers** (>=3.0.0): 텍스트 임베딩 생성
    - 모델: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384차원 다국어 벡터, 한/영 지원)
  - **PyTorch** (>=2.5.0): 딥러닝 프레임워크 및 GPU 가속
    - CUDA 12.1 호환
- **벡터 데이터베이스**:
  - **ChromaDB** (>=0.5.0): 벡터 스토어 및 유사도 검색
    - **이중 컬렉션 구조**: 성능 최적화를 위해 본문 컬렉션과 인덱스 컬렉션을 분리
      - 본문 컬렉션(`chroma_db`): 모든 지원자의 청크를 저장하고, 메타데이터 필터링(`filter={"candidate": "지원자명"}`)으로 검색
      - 인덱스 컬렉션(`candidates_index`): 지원자 목록만 저장하는 경량 인덱스로, 빠른 목록 조회를 제공
    - ChromaDB의 메타데이터 필터링 기본 기능을 검색 단계에서 적극 활용하여 유연한 쿼리를 제공합니다
- **문서 처리**:
  - **PyMuPDF** (>=1.23.0): PDF 파싱 및 텍스트 추출
    - 한글 지원 및 고성능 처리
- **Docker 기반 환경**:
  - **Docker**: 컨테이너 기반 실행 환경
  - **Docker Compose**: 다중 서비스 오케스트레이션
- **GPU 지원**:
  - **NVIDIA CUDA 12.1**: GPU 가속 (NVIDIA GPU가 있는 Linux 환경에서 사용 가능)
  - **CPU 모드**: GPU가 없는 환경에서도 자동으로 CPU 모드로 실행됩니다
  - **Mac 호환성**: Dockerfile은 CPU 전용 베이스 이미지를 사용하여 Mac Apple Silicon에서도 빌드 및 실행이 가능합니다 (CPU 모드로 실행)

### 시스템 의존성

- **Ubuntu 22.04** (Docker 컨테이너 내부)
- **한글 로케일**: `ko_KR.UTF-8` 지원

## 🚀 시작하기

### 사전 준비 (Host Machine)

이 프로젝트는 Docker 컨테이너 환경에서 실행됩니다. 호스트 머신에서 다음 사전 준비가 필요합니다.

1. **Docker 및 Docker Compose 설치**
   - [Docker 공식 홈페이지](https://docs.docker.com/get-docker/)를 참고하여 Docker를 설치합니다.
   - Docker Compose v2+가 포함되어 있는지 확인합니다.
   - 설치 확인:
     ```bash
     docker --version
     docker compose version
     ```

2. **Ollama 설치 및 모델 다운로드**
   - [Ollama 웹사이트](https://ollama.com/)의 안내에 따라 호스트 머신에 Ollama를 설치합니다.
   - 터미널에서 다음 명령어를 실행하여 사용할 모델을 미리 다운로드합니다:
     ```bash
     ollama pull qwen2.5:14b
     ```
   - Ollama 서버가 실행 중인지 확인합니다:
     ```bash
     ollama serve
     ```
   - **중요**: Ollama가 외부 애플리케이션(Docker 컨테이너)의 요청을 받을 수 있도록 설정되어 있어야 합니다. 기본적으로 `localhost:11434`에서 실행됩니다.

3. **GPU 사용 시 (선택사항, NVIDIA GPU만 지원)**
   - **NVIDIA GPU**: NVIDIA GPU가 있는 Linux 환경에서 GPU 가속을 사용하려면:
     - NVIDIA Container Toolkit 설치 필요
     - `docker-compose.yml`의 GPU 환경 변수 및 deploy 설정 주석 해제
     - Dockerfile을 NVIDIA CUDA 베이스 이미지로 변경 필요 (현재는 CPU 전용)
   - **Mac (Apple Silicon)**: Docker 컨테이너에서는 CPU 모드로만 실행됩니다. Mac GPU 가속을 사용하려면 Docker 없이 로컬 Python 환경에서 실행하세요.

### 설치 및 실행

1. **프로젝트 클론**
   ```bash
   git clone https://github.com/your-username/rag-candidate-assistant.git
   cd rag-candidate-assistant
   ```

2. **Docker Compose로 애플리케이션 실행**
   프로젝트 루트 디렉토리에서 다음 명령어를 실행합니다:
   ```bash
   docker-compose up --build
   ```
   
   또는 백그라운드에서 실행하려면:
   ```bash
   docker-compose up -d --build
   ```

3. **웹 브라우저에서 접속**
   웹 브라우저를 열고 `http://localhost:8501` 주소로 접속하면 챗봇 애플리케이션을 사용할 수 있습니다.

4. **애플리케이션 중지**
   ```bash
   docker-compose down
   ```

## 📖 사용 방법

### 1. 지원자 CV 업로드

![웹 UI 스크린샷](docs/images/ui_screenshot.png)

1. 웹 UI의 왼쪽 사이드바에서 "문서 업로드" 섹션으로 이동합니다.
2. PDF 파일을 업로드합니다. **파일명 형식**: `{지원자명}_CV.pdf`
   - 예시: `박광진_CV.pdf`, `Oliver_CV.pdf`
3. "🚀 업로드한 지원자 문서 처리" 버튼을 클릭합니다.
4. 시스템이 자동으로:
   - 파일명에서 지원자 이름을 추출
   - 문서를 청크로 분할 (청크 크기: 1200, 오버랩: 300)
   - 각 청크의 메타데이터에 지원자 정보 포함
   - 임베딩 생성 및 단일 벡터 데이터베이스에 저장

### 2. 질문하기

1. 메인 채팅 영역에서 질문을 입력합니다.
2. **질문에 지원자 이름을 포함**해야 합니다.
   - 예시: "박광진의 출신 대학교는?", "Oliver의 경력을 알려줘."
3. 시스템이 자동으로:
   - 질문에서 지원자 이름 추출
   - 메타데이터 필터링으로 해당 지원자의 정보만 검색 (상위 7개 청크)
   - LLM을 통해 답변 생성

### 3. 벡터DB 관리

- **등록된 지원자 확인**: 왼쪽 사이드바의 "등록된 지원자" 섹션에서 확인 가능


## 🏗️ 프로젝트 구조

```
rag-candidate-assistant/
├── src/
│   ├── interface_streamlit.py  # Streamlit 웹 UI
│   ├── ingest.py               # 문서 로드, 청킹, 벡터DB 저장
│   └── rag_pipeline.py         # RAG 체인 구성 및 질문 처리
├── vector_store/               # 벡터DB 저장 위치 (이중 컬렉션 구조)
│   ├── chroma_db/              # 본문 컬렉션: 모든 지원자의 청크 저장 (메타데이터 필터링)
│   └── candidates_index/       # 인덱스 컬렉션: 지원자 목록만 저장 (빠른 조회용)
├── cache/                      # HuggingFace 모델 캐시
├── docker-compose.yml          # Docker Compose 설정
├── Dockerfile                  # Docker 이미지 빌드 설정
├── requirements.txt            # Python 의존성
└── README.md                   # 프로젝트 문서
```

## ⚙️ 주요 설정

### 환경 변수 (docker-compose.yml)

- `OLLAMA_HOST`: Ollama 서버 주소 (기본: `http://host.docker.internal:11434`)
- `OLLAMA_MODEL`: 사용할 LLM 모델 (기본: `qwen2.5:14b`)
- `CHROMA_SERVER_HOST`: ChromaDB 로컬 모드 강제 (기본: `localhost`)
- GPU 관련 환경 변수: CUDA 설정

### RAG 파이프라인 설정

- **청크 크기**: 1200자 (5페이지 이내 CV 섹션 단위 보존)
- **청크 오버랩**: 300자 (섹션 경계 정보 보존)
- **검색 결과 수 (k)**: 7개 (정확도 향상을 위한 Top-7 검색)
- **임베딩 모델**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (한글/영어 다국어 지원)
- **LLM 모델**: `qwen2.5:14b` (Ollama)

## 🔧 기술적 특징

### 메타데이터 기반 지원자 구분 아키텍처 (이중 컬렉션 설계)

이 프로젝트는 **ChromaDB의 기본 기능(메타데이터 필터링)을 활용**하면서도 **성능 최적화를 위한 이중 컬렉션 구조**를 채택했습니다.

#### 설계 배경 및 기술적 고민

**문제 인식:** 지원자 목록 조회를 위해 매번 전체 컬렉션의 메타데이터를 스캔하는 것은 비효율적이며, 지원자 수가 증가할수록 지연이 발생합니다.

**해결 방안:** 
- **본문 컬렉션 (`chroma_db`)**: 모든 청크를 저장하고 메타데이터 필터링(`filter={"candidate": "지원자명"}`)으로 검색
- **인덱스 컬렉션 (`candidates_index`)**: 지원자 목록만 저장하는 경량 인덱스로 빠른 조회 제공

**장점:** 지원자 목록 조회는 O(지원자 수)으로 빠르고, 단일 본문 컬렉션 구조로 향후 복합 필터링 기능으로 확장하기 용이합니다.

#### 아키텍처 상세

- **저장 구조**:
  - 본문 컬렉션: `vector_store/chroma_db` - 모든 청크 저장 (메타데이터: `{"candidate": "지원자명", ...}`)
  - 인덱스 컬렉션: `vector_store/candidates_index` - 지원자 목록만 저장 (문서 수 = 지원자 수)
- **검색 흐름**:
  1. 지원자 목록 조회: 인덱스 컬렉션에서 빠르게 조회 (O(지원자 수))
  2. 질문 처리: 본문 컬렉션에서 `filter={"candidate": "지원자명"}` 필터링으로 Top-k(7개) 검색
- **업로드 흐름**:
  1. 문서 청크를 본문 컬렉션에 저장 (메타데이터에 `candidate` 포함)
  2. 지원자 이름을 인덱스 컬렉션에 업데이트 (`upsert`로 중복 방지)

### 한글 파일명 처리

- Unicode 정규화(NFC)를 통해 macOS와 Linux/Docker 환경 간 호환성을 향상시킵니다.
- 한글 지원자 이름과 파일명을 정확하게 처리합니다.

### 임베딩 & 검색 파이프라인

**저장 단계:**
- `PyMuPDF`로 PDF에서 텍스트를 추출합니다.
- `RecursiveCharacterTextSplitter`로 약 1200자 단위의 청크로 분할하고 300자 오버랩을 적용합니다.
- 다국어 임베딩 모델(`paraphrase-multilingual-MiniLM-L12-v2`)로 각 청크를 임베딩합니다.
- 본문 컬렉션(`chroma_db`)에 청크를 저장하며, 각 청크의 메타데이터에 `candidate` 필드를 포함합니다.
- 동시에 인덱스 컬렉션(`candidates_index`)에 지원자 이름을 업데이트하여 빠른 목록 조회를 지원합니다.

**검색 단계:**
- 지원자 목록 조회: 인덱스 컬렉션에서 O(지원자 수) 시간으로 빠르게 조회합니다.
- 질문 임베딩 생성 후, 본문 컬렉션에서 `filter={"candidate": "지원자명"}` 메타데이터 필터링을 적용하여 코사인 유사도 기반 Top-7 청크를 검색합니다.
- 검색된 청크들을 LangChain RAG 체인(`ChatPromptTemplate` + `ChatOllama` + `StrOutputParser`)에 컨텍스트로 전달하여 최종 답변을 생성합니다.

### GPU 가속

- PyTorch와 CUDA 12.1을 활용한 GPU 가속으로 임베딩 생성 속도를 향상시킵니다.
- CPU 모드도 지원하여 다양한 환경에서 실행 가능합니다.

## 📝 향후 개선할 점

아래와 같은 개선을 통해, 더욱 완성도 높은 챗봇으로 발전시키고자 합니다.

- **응답 속도 개선**: LLM 응답을 토큰 단위로 스트리밍하여 첫 응답까지의 대기 시간을 단축하고, RAG 파이프라인 최적화를 통해 질문-답변 처리 속도를 향상시킵니다.
- **프롬프트 엔지니어링 고도화**: 프롬프트 추가 최적화를 통해 RAG의 성능을 향상시킵니다.
- **임베딩 모델 선택 기능**: UI에서 직접 임베딩 모델을 선택할 수 있는 기능을 추가해 사용자가 원하는 성능에 따라 모델을 선택할 수 있도록 합니다.
- **Mac GPU 가속 지원**: Mac Apple Silicon의 MPS(Metal Performance Shaders)를 활용한 GPU 가속을 지원하도록 합니다. (Docker 없이 로컬 Python 환경에서 실행 시)
- **OCR 기반 PDF 처리**: 현재는 텍스트 기반 PDF만 처리하지만, Poppler-utils와 Tesseract OCR을 활용하여 이미지나 스캔된 PDF도 처리할 수 있도록 확장합니다.
- **대화 기록 저장**: 채팅 기록을 파일이나 DB에 저장하는 기능을 만들어 이전에 검색한 내용을 쉽게 다시 찾아볼 수 있도록 합니다.
- **벡터DB 백업/복원**: 벡터DB 데이터의 백업 및 복원 기능을 추가해 편의를 도모합니다.
- **성능 모니터링**: 검색 성능 및 응답 시간 모니터링 대시보드를 만들어 원활한 성능 확인이 가능하도록 합니다.

## 🤝 기여하기

이 프로젝트에 기여하고 싶으시거나 궁금증이 있으시다면, 언제든 이슈를 남겨주세요!

## 📄 라이선스

이 프로젝트는 [LICENSE](LICENSE) 파일에 명시된 라이선스를 따릅니다.

---
