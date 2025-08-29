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
    && rm -rf /var/lib/apt/lists/*

# Copy shared utilities
COPY shared/ /app/shared/

# Copy service-specific files
COPY services/story-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/story-service/ .

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]