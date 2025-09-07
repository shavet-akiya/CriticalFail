# syntax=docker/dockerfile:1

FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install runtime dependencies (from pyproject: fastapi[standard])
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "fastapi[standard]>=0.116.1" httpx

# Copy application source
COPY ./server /app/server

EXPOSE 8000

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]