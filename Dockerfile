FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY config ./config
COPY data ./data

EXPOSE 8501

# Copy startup script
COPY start.sh .
RUN chmod +x start.sh

# Use PORT env var if available (Railway), otherwise default to 8501
CMD ["./start.sh"]



