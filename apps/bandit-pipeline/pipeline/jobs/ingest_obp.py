"""Dagster job: ingest_obp

Downloads the Open Bandit Dataset zip from ZOZO Research and uploads the
extracted CSVs to MinIO at bandit/raw/obp/.

Dataset: https://research.zozo.com/data.html
~26M interactions across 6 CSV files (bts/random × all/men/women) plus item_context.csv.
"""

from __future__ import annotations

import io
import logging
import os
import zipfile

import boto3
import httpx
from dagster import job, op

logger = logging.getLogger(__name__)

DATASET_URL = "https://research.zozo.com/data_release/open_bandit_dataset.zip"
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio-secondary.storage.svc.cluster.local:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "")
BUCKET = "bandit"
OBP_PREFIX = "raw/obp/"


def _s3_client() -> boto3.client:
    return boto3.client(
        "s3",
        endpoint_url=f"http://{MINIO_ENDPOINT}",
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
    )


@op
def download_obp_dataset(context) -> bytes:
    """Stream the OBP zip from ZOZO Research into memory."""
    context.log.info(f"Downloading OBP dataset from {DATASET_URL}")
    with httpx.Client(follow_redirects=True, timeout=300) as client:
        response = client.get(DATASET_URL)
        response.raise_for_status()
    size_mb = len(response.content) / 1024 / 1024
    context.log.info(f"Downloaded {size_mb:.1f} MB")
    return response.content


@op
def upload_to_minio(context, zip_bytes: bytes) -> None:
    """Extract the zip and upload every CSV to MinIO at bandit/raw/obp/."""
    s3 = _s3_client()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_files = [name for name in zf.namelist() if name.endswith(".csv")]
        context.log.info(f"Found {len(csv_files)} CSV files in zip: {csv_files}")

        for name in csv_files:
            # Flatten nested paths: e.g. open_bandit_dataset/bts/all.csv → bts/all.csv
            relative = name.split("/", 1)[-1] if "/" in name else name
            key = OBP_PREFIX + relative

            with zf.open(name) as f:
                data = f.read()

            s3.upload_fileobj(io.BytesIO(data), BUCKET, key)
            context.log.info(f"Uploaded {key} ({len(data) / 1024 / 1024:.1f} MB)")

    context.log.info("OBP dataset upload complete")


@job(name="ingest_obp")
def ingest_obp_job() -> None:
    upload_to_minio(download_obp_dataset())
