FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    gcc \
    g++ \
    python3-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir git+https://github.com/m-bain/whisperx.git

RUN mkdir -p models/whisperx models/whisperx_align models/speechbrain transcripts recordings

# Skip model download during build - will download on first run
# This avoids the ctranslate2 executable stack issue

COPY . .

# Add security flags to allow ctranslate2 to work
ENV LD_PRELOAD=""
ENV MALLOC_CONF="dirty_decay_ms:0"

EXPOSE 8001

CMD ["uvicorn", "speech_server:app", "--host", "0.0.0.0", "--port", "8001"]