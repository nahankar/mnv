FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libpq-dev \
    python3-dev \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    libgtk-3-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy shared utilities
COPY shared/ /app/shared/

# Copy service-specific files
COPY services/image-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/image-service/ .

EXPOSE 8003

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]