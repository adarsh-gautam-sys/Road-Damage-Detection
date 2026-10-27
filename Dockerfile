FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install --no-deps -e .

EXPOSE 8000

CMD ["uvicorn", "rdai.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
