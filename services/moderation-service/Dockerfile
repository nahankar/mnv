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
    libmagic1 \
    tesseract-ocr \
    libtesseract-dev \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Create storage directory
RUN mkdir -p /app/storage/moderation

# Copy shared utilities
COPY shared/ /app/shared/

# Copy service-specific files
COPY services/moderation-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/moderation-service/ .

# Set permissions for storage directory
RUN chmod 755 /app/storage/moderation

EXPOSE 8006

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8006"]