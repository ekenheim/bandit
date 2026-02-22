"""Dagster job: compute_regret

Computes cumulative regret for the Thompson Sampling policy versus a uniform
random allocation baseline, using the OBP replay event log.

Regret = sum over t of [ r*(t) - r_chosen(t) ]
       where r*(t) is the reward of the best arm and r_chosen(t) is the
       reward of the arm actually chosen.

Results (regret curve, allocation ratios) are logged to MLflow at milestones:
  1000, 5000, 10000 events.

Target: < 40% cumulative regret of uniform allocation after 5000 events.
"""

import logging
import os

import mlflow
import numpy as np
from dagster import Config, job, op

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow.datasci.svc.cluster.local:5000")


class RegretConfig(Config):
    experiment_id: str = "obp-replay-001"
    n_arms: int = 2
    n_events: int = 10_000
    mlflow_experiment_name: str = "bandit-regret"


@op
def load_replay_rewards(context, config: RegretConfig) -> dict[str, np.ndarray]:
    """
    Load the arm selection and reward sequence from the OBP replay.

    In production this reads from the bandit_events Postgres table.
    For MVP, generates a synthetic sequence using known arm reward rates
    to demonstrate regret convergence behaviour.
    """
    rng = np.random.default_rng(42)
    n = config.n_events

    # Synthetic: arm 0 has 5% CTR, arm 1 has 8% CTR (arm 1 is the better arm)
    true_rates = [0.05, 0.08][: config.n_arms]
    context.log.info(f"True arm rates: {true_rates}")

    # Simulate Thompson Sampling allocation (biased toward better arm over time)
    alphas = np.ones(config.n_arms)
    betas = np.ones(config.n_arms)
    ts_arms = np.zeros(n, dtype=int)
    ts_rewards = np.zeros(n)

    for i in range(n):
        samples = np.array([rng.beta(a, b) for a, b in zip(alphas, betas)])
        arm = int(np.argmax(samples))
        reward = rng.binomial(1, true_rates[arm])
        ts_arms[i] = arm
        ts_rewards[i] = reward
        if reward:
            alphas[arm] += 1
        else:
            betas[arm] += 1

    # Uniform allocation baseline
    uniform_arms = rng.integers(0, config.n_arms, size=n)
    uniform_rewards = np.array([rng.binomial(1, true_rates[a]) for a in uniform_arms])

    context.log.info(
        f"Thompson Sampling total reward: {ts_rewards.sum():.0f} "
        f"vs Uniform: {uniform_rewards.sum():.0f}"
    )

    return {
        "ts_arms": ts_arms,
        "ts_rewards": ts_rewards,
        "uniform_rewards": uniform_rewards,
        "true_rates": np.array(true_rates),
    }


@op
def log_regret_to_mlflow(
    context, config: RegretConfig, data: dict[str, np.ndarray]
) -> None:
    """Compute cumulative regret curves and log to MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.mlflow_experiment_name)

    ts_rewards = data["ts_rewards"]
    uniform_rewards = data["uniform_rewards"]
    true_rates = data["true_rates"]
    best_rate = float(true_rates.max())

    # Cumulative regret vs optimal (always picking the best arm)
    optimal_rewards = np.random.binomial(1, best_rate, size=len(ts_rewards))
    ts_regret = np.cumsum(optimal_rewards - ts_rewards)
    uniform_regret = np.cumsum(optimal_rewards - uniform_rewards)

    milestones = [1000, 5000, 10_000]

    with mlflow.start_run(run_name=f"regret-{config.experiment_id}"):
        mlflow.log_param("experiment_id", config.experiment_id)
        mlflow.log_param("n_arms", config.n_arms)
        mlflow.log_param("true_rates", str(true_rates.tolist()))

        for m in milestones:
            if m > len(ts_regret):
                continue
            mlflow.log_metric("ts_cumulative_regret", float(ts_regret[m - 1]), step=m)
            mlflow.log_metric("uniform_cumulative_regret", float(uniform_regret[m - 1]), step=m)
            ratio = ts_regret[m - 1] / (uniform_regret[m - 1] + 1e-9)
            mlflow.log_metric("regret_ratio_vs_uniform", float(ratio), step=m)
            context.log.info(
                f"t={m}: TS regret={ts_regret[m-1]:.1f}, "
                f"Uniform regret={uniform_regret[m-1]:.1f}, "
                f"ratio={ratio:.2%}"
            )

        # Allocation share at final step
        ts_arms = data["ts_arms"]
        for k in range(config.n_arms):
            share = float((ts_arms == k).mean())
            mlflow.log_metric(f"arm_{k}_allocation_share", share)
            context.log.info(f"Arm {k} final allocation share: {share:.2%}")


@job(name="compute_regret")
def compute_regret_job() -> None:
    data = load_replay_rewards()
    log_regret_to_mlflow(data)
