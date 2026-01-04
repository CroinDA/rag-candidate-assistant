# NVIDIA CUDA 12.1 + Ubuntu 22.04 (GPU 가속 지원)
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HOME=/app

# 작업 디렉토리 설정
WORKDIR /app

# Python 및 시스템 의존성 설치 (한글 로케일 포함)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-kor \
    tesseract-ocr-eng \
    build-essential \
    g++ \
    libopenblas-dev \
    curl \
    wget \
    git \
    locales \
    && rm -rf /var/lib/apt/lists/*

# 한글 로케일 생성 및 설정
RUN locale-gen ko_KR.UTF-8 && \
    update-locale LANG=ko_KR.UTF-8 LC_ALL=ko_KR.UTF-8

# Python 심볼릭 링크 설정
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Python 의존성 설치
COPY requirements.txt .

RUN python3 -m pip install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY src/ ./src/

# 벡터 스토어 및 캐시 디렉토리 생성
RUN mkdir -p vector_store .cache .streamlit && \
    chmod -R 777 vector_store .cache .streamlit

# 한글 로케일 환경 변수 설정
ENV LANG=ko_KR.UTF-8
ENV LC_ALL=ko_KR.UTF-8

# GPU 환경 변수 설정
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# PyTorch CUDA 설정
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0"

# 캐시 디렉토리 설정 (HuggingFace 모델 캐시)
ENV HF_HOME=/app/.cache
ENV TRANSFORMERS_CACHE=/app/.cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache

# Streamlit 설정
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Streamlit 포트 노출
EXPOSE 8501

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Streamlit 실행
CMD ["streamlit", "run", "src/interface_streamlit.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]

