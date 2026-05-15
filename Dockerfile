FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# Pre-download Silero VAD model so first room join is instant
RUN python -c "from livekit.plugins.silero import VAD; VAD.load()" || true

COPY agent.py .

CMD ["python", "agent.py", "start"]
