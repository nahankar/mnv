FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libpq-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy shared
COPY shared/ /app/shared/

# Install requirements
COPY services/analytics-service/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY services/analytics-service/ /app/

EXPOSE 8008

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8008"]
