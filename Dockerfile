# Dockerfile — Sales RAG Assistant
# Author: Suresh D R | AI Product Developer & Technology Mentor

FROM python:3.11-slim AS builder
WORKDIR /build

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip

# Install all libraries with exact pinned versions
# openai and httpx are pinned together to avoid the proxies conflict
RUN pip install --no-cache-dir --user \
    streamlit==1.32.0 \
    openai==2.38.0 \
    httpx==0.27.0 \
    chromadb==0.4.22 \
    python-dotenv==1.0.0

FROM python:3.11-slim AS runtime
WORKDIR /app

# Copy installed libraries from builder
COPY --from=builder /root/.local /root/.local

# Copy all project files
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY data/ ./data/

ENV PATH=/root/.local/bin:$PATH

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "frontend/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
