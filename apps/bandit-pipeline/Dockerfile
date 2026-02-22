FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline/ ./pipeline/

ENV DAGSTER_HOME=/opt/dagster
RUN mkdir -p /opt/dagster && chmod 777 /opt/dagster
COPY dagster.yaml /opt/dagster/dagster.yaml

RUN useradd -m pipeline
USER pipeline

CMD ["dagster", "api", "grpc", "-m", "pipeline.repository", "-h", "0.0.0.0", "-p", "4000"]
