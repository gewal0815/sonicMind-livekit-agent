FROM python:3.11-slim

WORKDIR /app

# System deps for audio processing (silero VAD)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model weights at build time so cold starts are fast
RUN python -c "from livekit.plugins import silero; silero.VAD.load()" || true

COPY agent.py .

CMD ["python", "agent.py", "start"]
