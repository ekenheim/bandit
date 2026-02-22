"""Dagster job: obp_replay_simulator

Replays logged OBP events through the live bandit-service as if they were
real-time traffic. This lets you observe posterior updates, allocation shifts,
and convergence behaviour using real interaction data from a production system.

The replay pumps events to:
  POST http://bandit-service.development.svc.cluster.local:8000/select
  POST http://bandit-service.development.svc.cluster.local:8000/reward
"""

import logging
import os
import time
from typing import Any

import httpx
import numpy as np
from dagster import Config, job, op

logger = logging.getLogger(__name__)

BANDIT_SERVICE_URL = os.getenv(
    "BANDIT_SERVICE_URL",
    "http://bandit-service.development.svc.cluster.local:8000",
)


class ReplayConfig(Config):
    experiment_id: str = "obp-replay-001"
    n_events: int = 10_000
    n_arms: int = 2
    requests_per_second: float = 100.0


@op
def initialise_experiment(context, config: ReplayConfig) -> str:
    """Create experiment state in Dragonfly via bandit-service /experiments."""
    resp = httpx.post(
        f"{BANDIT_SERVICE_URL}/experiments",
        json={"experiment_id": config.experiment_id, "n_arms": config.n_arms},
    )
    resp.raise_for_status()
    context.log.info(f"Initialised experiment {config.experiment_id} with {config.n_arms} arms")
    return config.experiment_id


@op
def load_obp_feedback(context, config: ReplayConfig) -> dict[str, Any]:
    """Load OBP bandit feedback for replay."""
    try:
        from obp.dataset import OpenBanditDataset

        context.log.info("Loading OBP dataset (random policy, for unbiased OPE)...")
        dataset = OpenBanditDataset(behavior_policy="random", campaign="all")
        feedback = dataset.obtain_batch_bandit_feedback(test_size=0.3)
        context.log.info(f"Loaded {feedback['n_rounds']} rounds")
        return feedback
    except Exception as exc:
        context.log.warning(f"OBP not available ({exc}); generating synthetic feedback")
        rng = np.random.default_rng(42)
        n = config.n_events
        arm_probs = [0.05, 0.08, 0.12, 0.15][: config.n_arms]
        actions = rng.integers(0, config.n_arms, size=n)
        rewards = np.array([rng.binomial(1, arm_probs[a]) for a in actions], dtype=float)
        return {
            "n_rounds": n,
            "action": actions,
            "reward": rewards,
            "context": rng.standard_normal((n, 5)),
        }


@op
def replay_events(
    context,
    config: ReplayConfig,
    experiment_id: str,
    feedback: dict[str, Any],
) -> dict[str, int]:
    """
    Pump logged OBP events through bandit-service.

    For each event:
      1. Call /select to let the bandit pick an arm (updates its own state)
      2. Look up the observed reward for the logged action
      3. Call /reward to feed the reward back

    This is not true importance-weighted replay — it uses the bandit's chosen arm
    to update posteriors, demonstrating convergence. For rigorous OPE use evaluate_ope.
    """
    n = min(config.n_events, feedback["n_rounds"])
    sleep_s = 1.0 / config.requests_per_second
    counts: dict[str, int] = {f"arm_{k}": 0 for k in range(config.n_arms)}

    context.log.info(f"Replaying {n} events at {config.requests_per_second} req/s...")

    with httpx.Client(timeout=5.0) as client:
        for i in range(n):
            select_resp = client.post(
                f"{BANDIT_SERVICE_URL}/select",
                json={"experiment_id": experiment_id},
            )
            select_resp.raise_for_status()
            arm_id = select_resp.json()["arm_id"]
            counts[f"arm_{arm_id}"] += 1

            # Use the reward associated with the logged action at this position
            reward = float(feedback["reward"][i])
            client.post(
                f"{BANDIT_SERVICE_URL}/reward",
                json={"experiment_id": experiment_id, "arm_id": arm_id, "reward": reward},
            ).raise_for_status()

            if i % 1000 == 0 and i > 0:
                context.log.info(f"Replayed {i}/{n} events — allocation: {counts}")

            time.sleep(sleep_s)

    context.log.info(f"Replay complete. Final allocation: {counts}")
    return counts


@job(name="obp_replay_simulator")
def obp_replay_job() -> None:
    exp_id = initialise_experiment()
    feedback = load_obp_feedback()
    replay_events(exp_id, feedback)
