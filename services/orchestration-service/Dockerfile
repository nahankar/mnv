FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libpq-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY shared/ /app/shared/
COPY services/orchestration-service/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY services/orchestration-service/ /app/

EXPOSE 8010

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010"]
