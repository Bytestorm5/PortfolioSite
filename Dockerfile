FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./        # you said you'll create this file
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

ENV PORT=80 HOST=0.0.0.0
EXPOSE 80

CMD ["python", "app.py"]
