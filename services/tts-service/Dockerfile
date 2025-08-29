FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including audio processing tools
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libpq-dev \
    python3-dev \
    ffmpeg \
    libavcodec-extra \
    && rm -rf /var/lib/apt/lists/*

# Create audio directory
RUN mkdir -p /tmp/audio

# Copy shared utilities
COPY shared/ /app/shared/

# Copy service-specific files
COPY services/tts-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/tts-service/ .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app /tmp/audio
USER appuser

EXPOSE 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]