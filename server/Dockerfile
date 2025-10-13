# server/Dockerfile
FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install FastAPI and httpx
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "fastapi[standard]>=0.116.1" httpx

COPY server/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY ./server /app/server

# Copy LLM folder
COPY ./llm /app/llm

# Set PYTHONPATH so server can import llm
ENV PYTHONPATH=/app

EXPOSE 9000

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "9000"]