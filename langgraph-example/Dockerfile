FROM langchain/langgraph-api:3.11

# lxml 빌드에 필요한 의존성 설치
RUN apt-get update && apt-get install -y \
    python3-dev \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# lxml 설치
RUN pip install lxml==5.3.1