FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    gcc \
    g++ \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch 2.3.1 WITHOUT torchvision (not needed for audio)
RUN pip install --no-cache-dir \
    torch==2.3.1 \
    torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir git+https://github.com/m-bain/whisperx.git

RUN mkdir -p models/whisperx models/whisperx_align models/speechbrain transcripts recordings

COPY . .

EXPOSE 8001

CMD ["python", "-m", "uvicorn", "speech_server:app", "--host", "0.0.0.0", "--port", "8001"]