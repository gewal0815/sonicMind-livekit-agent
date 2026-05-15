FROM python:3.11-slim

WORKDIR /app

# System deps: ffmpeg for audio decoding, libsndfile1 for soundfile I/O
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps — torch is installed first because it's the heaviest
# and pip resolves it before the livekit-agents extras
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY agent.py .

# The livekit-agents CLI entry point; "start" connects to LiveKit and waits for room dispatches.
CMD ["python", "agent.py", "start"]
