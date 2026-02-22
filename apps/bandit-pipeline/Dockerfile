FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Pre-resolved lock (avoids pip resolution-too-deep). obp excluded — its imports are try/except
# with synthetic fallback. Regenerate from repo root with:
#   uv pip compile apps/bandit-pipeline/requirements-docker.txt \
#     --python-version 3.11 --python-platform x86_64-manylinux2014 \
#     -o apps/bandit-pipeline/requirements-lock.txt
COPY requirements-lock.txt ./
RUN pip install --no-cache-dir -r requirements-lock.txt

COPY pipeline/ ./pipeline/

ENV DAGSTER_HOME=/opt/dagster
RUN mkdir -p /opt/dagster && chmod 777 /opt/dagster
COPY dagster.yaml /opt/dagster/dagster.yaml

RUN useradd -m pipeline
USER pipeline

CMD ["dagster", "api", "grpc", "-m", "pipeline.repository", "-h", "0.0.0.0", "-p", "4000"]
