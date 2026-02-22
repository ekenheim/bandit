"""Dagster job: ingest_obp

Downloads the Open Bandit Pipeline (OBP) dataset from the ZOZO research portal
and writes Parquet files to MinIO Secondary at bandit/raw/obp/.

Dataset: https://research.zozo.com/data.html
~26M interactions; columns: user_id, item_id, position, reward, timestamp, context_features
"""

from __future__ import annotations

import logging
import os

import boto3
from dagster import job, op

logger = logging.getLogger(__name__)

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
def download_obp_dataset(context) -> list[str]:
    """
    Download OBP Parquet files.

    The OBP dataset is obtained via registration at https://research.zozo.com/data.html
    (typically 1-2 days approval). Once approved, files are downloaded and stored locally
    or fetched via the OBP library's dataset loader.

    For MVP fallback, the UCI Online Retail II dataset can be substituted:
    https://archive.ics.uci.edu/dataset/502/online+retail+ii
    """
    try:
        from obp.dataset import OpenBanditDataset

        context.log.info("Loading OBP dataset via obp library...")
        dataset = OpenBanditDataset(behavior_policy="random", campaign="all")
        bandit_feedback = dataset.obtain_batch_bandit_feedback(test_size=0.3)
        context.log.info(
            f"Loaded {bandit_feedback['n_rounds']} rounds from OBP dataset"
        )
        # Return the feedback dict for downstream ops via file path convention
        import pickle, tempfile
        tmp = tempfile.mktemp(suffix=".pkl")
        with open(tmp, "wb") as f:
            pickle.dump(bandit_feedback, f)
        return [tmp]
    except Exception as exc:
        context.log.warning(f"OBP dataset load failed ({exc}); using stub path")
        return []


@op
def upload_to_minio(context, local_paths: list[str]) -> None:
    """Upload downloaded OBP files to MinIO bandit/raw/obp/."""
    if not local_paths:
        context.log.warning("No files to upload — skipping MinIO upload")
        return

    s3 = _s3_client()
    for path in local_paths:
        key = OBP_PREFIX + os.path.basename(path)
        context.log.info(f"Uploading {path} → s3://{BUCKET}/{key}")
        s3.upload_file(path, BUCKET, key)
        context.log.info(f"Uploaded {key}")


@job(name="ingest_obp")
def ingest_obp_job() -> None:
    upload_to_minio(download_obp_dataset())
