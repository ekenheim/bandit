"""Dagster job: snapshot_posteriors

Reads current alpha/beta values from Dragonfly for all active experiments
and writes a snapshot row to the posterior_snapshots Postgres table.
Triggered every 100 events by a Dagster sensor (or run manually).

The snapshot table drives the Grafana posterior distribution dashboard.
"""

import logging
import os
from datetime import datetime, timezone

import redis
import sqlalchemy as sa
from dagster import Config, job, op

logger = logging.getLogger(__name__)

DRAGONFLY_HOST = os.getenv("DRAGONFLY_HOST", "dragonfly.database.svc.cluster.local")
DRAGONFLY_PORT = int(os.getenv("DRAGONFLY_PORT", "6379"))
DRAGONFLY_DB = int(os.getenv("DRAGONFLY_DB", "2"))
PG_DSN = os.getenv("PG_CONNECTION_STRING", "")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS posterior_snapshots (
    snapshot_at    TIMESTAMPTZ NOT NULL,
    experiment_id  TEXT NOT NULL,
    arm_id         INTEGER NOT NULL,
    alpha          INTEGER NOT NULL,
    beta           INTEGER NOT NULL,
    primary_prob   FLOAT,
    PRIMARY KEY (snapshot_at, experiment_id, arm_id)
);
"""


class SnapshotConfig(Config):
    experiment_ids: list[str] = []


def _dragonfly() -> redis.Redis:
    return redis.Redis(
        host=DRAGONFLY_HOST, port=DRAGONFLY_PORT, db=DRAGONFLY_DB, decode_responses=True
    )


@op
def read_posteriors_from_dragonfly(context, config: SnapshotConfig) -> list[dict]:
    """Read alpha/beta for all arms of each experiment via pipelined MGET."""
    df = _dragonfly()
    rows = []
    now = datetime.now(timezone.utc)

    for exp_id in config.experiment_ids:
        n_arms_raw = df.get(f"experiment:{exp_id}:n_arms")
        if n_arms_raw is None:
            context.log.warning(f"Experiment {exp_id} not found in Dragonfly — skipping")
            continue
        n_arms = int(n_arms_raw)

        keys = []
        for k in range(n_arms):
            keys.extend([
                f"experiment:{exp_id}:arm:{k}:alpha",
                f"experiment:{exp_id}:arm:{k}:beta",
            ])
        values = df.mget(keys)

        for k in range(n_arms):
            alpha = int(values[2 * k] or 1)
            beta = int(values[2 * k + 1] or 1)
            rows.append({
                "snapshot_at": now,
                "experiment_id": exp_id,
                "arm_id": k,
                "alpha": alpha,
                "beta": beta,
            })

    context.log.info(
        f"Collected {len(rows)} posterior snapshots across "
        f"{len(config.experiment_ids)} experiments"
    )
    return rows


@op
def write_snapshots_to_postgres(context, rows: list[dict]) -> None:
    """Write snapshot rows to posterior_snapshots table."""
    if not rows:
        context.log.info("No rows to write")
        return

    if not PG_DSN:
        context.log.warning("PG_CONNECTION_STRING not set — skipping Postgres write")
        return

    engine = sa.create_engine(PG_DSN)
    with engine.begin() as conn:
        conn.execute(sa.text(CREATE_TABLE_SQL))
        conn.execute(
            sa.text(
                """
                INSERT INTO posterior_snapshots
                    (snapshot_at, experiment_id, arm_id, alpha, beta)
                VALUES
                    (:snapshot_at, :experiment_id, :arm_id, :alpha, :beta)
                ON CONFLICT DO NOTHING
                """
            ),
            rows,
        )
    context.log.info(f"Wrote {len(rows)} snapshot rows to Postgres")


@job(name="snapshot_posteriors")
def snapshot_posteriors_job() -> None:
    rows = read_posteriors_from_dragonfly()
    write_snapshots_to_postgres(rows)
