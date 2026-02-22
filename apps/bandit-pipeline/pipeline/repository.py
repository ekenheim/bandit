"""Dagster code location entry point.

Registers all bandit jobs with the Dagster instance.
The gRPC server exposes this module on port 4000 (CMD in Dockerfile).
"""

from dagster import Definitions, ScheduleDefinition

from .jobs.conclude_experiments import conclude_experiments_job
from .jobs.compute_regret import compute_regret_job
from .jobs.evaluate_ope import evaluate_ope_job
from .jobs.ingest_obp import ingest_obp_job
from .jobs.obp_replay import obp_replay_job
from .jobs.snapshot_posteriors import snapshot_posteriors_job

# Periodic schedule: check stopping rule every 30 minutes
conclude_schedule = ScheduleDefinition(
    job=conclude_experiments_job,
    cron_schedule="*/30 * * * *",
    name="conclude_experiments_schedule",
)

defs = Definitions(
    jobs=[
        ingest_obp_job,
        obp_replay_job,
        snapshot_posteriors_job,
        compute_regret_job,
        evaluate_ope_job,
        conclude_experiments_job,
    ],
    schedules=[conclude_schedule],
)
