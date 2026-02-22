FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Pre-resolved lock (avoids pip resolution-too-deep). Regenerate from repo root with:
#   uv pip compile apps/bandit-pipeline/requirements.txt -c apps/bandit-pipeline/constraints.txt \
#     --python-version 3.11 --python-platform x86_64-manylinux2014 --torch-backend cpu \
#     -o apps/bandit-pipeline/requirements-lock.txt
COPY requirements-lock.txt ./
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-lock.txt

COPY pipeline/ ./pipeline/

ENV DAGSTER_HOME=/opt/dagster
RUN mkdir -p /opt/dagster && chmod 777 /opt/dagster
COPY dagster.yaml /opt/dagster/dagster.yaml

RUN useradd -m pipeline
USER pipeline

CMD ["dagster", "api", "grpc", "-m", "pipeline.repository", "-h", "0.0.0.0", "-p", "4000"]
