"""Thompson Sampling core logic — reads/writes Dragonfly state."""

from __future__ import annotations

import os

import numpy as np
import redis

_dragonfly: redis.Redis | None = None


def get_dragonfly() -> redis.Redis:
    global _dragonfly
    if _dragonfly is None:
        _dragonfly = redis.Redis(
            host=os.getenv("DRAGONFLY_HOST", "dragonfly.database.svc.cluster.local"),
            port=int(os.getenv("DRAGONFLY_PORT", "6379")),
            db=int(os.getenv("DRAGONFLY_DB", "2")),
            decode_responses=True,
        )
    return _dragonfly


def _exp_key(experiment_id: str) -> str:
    return f"experiment:{experiment_id}"


def get_n_arms(experiment_id: str) -> int:
    df = get_dragonfly()
    value = df.get(f"{_exp_key(experiment_id)}:n_arms")
    if value is None:
        raise KeyError(f"Experiment {experiment_id!r} not found in Dragonfly")
    return int(value)


def read_posteriors(experiment_id: str, n_arms: int) -> tuple[list[int], list[int]]:
    """Read alpha and beta for all arms via pipeline (single round-trip)."""
    df = get_dragonfly()
    exp_key = _exp_key(experiment_id)
    keys = []
    for k in range(n_arms):
        keys.append(f"{exp_key}:arm:{k}:alpha")
        keys.append(f"{exp_key}:arm:{k}:beta")
    values = df.mget(keys)
    alphas = [int(values[2 * k] or 1) for k in range(n_arms)]
    betas = [int(values[2 * k + 1] or 1) for k in range(n_arms)]
    return alphas, betas


def thompson_sample(alphas: list[int], betas: list[int]) -> tuple[int, float]:
    """
    Sample once from each Beta posterior and return the winning arm index
    plus P(arm is best) estimated via 1000-sample Monte Carlo.

    Keeps /select hot path to: read → sample → argmax.
    p_best computation is cheap at 1000 samples; move to snapshot cadence
    if QPS becomes a concern.
    """
    samples = np.array([np.random.beta(a, b) for a, b in zip(alphas, betas)])
    arm_id = int(np.argmax(samples))

    mc = np.array([np.random.beta(a, b, size=1000) for a, b in zip(alphas, betas)])
    p_best = float((mc.argmax(axis=0) == arm_id).mean())

    return arm_id, p_best


def record_reward(experiment_id: str, arm_id: int, reward: float) -> None:
    """Atomically update Beta posterior and increment total_draws."""
    df = get_dragonfly()
    exp_key = _exp_key(experiment_id)
    pipe = df.pipeline(transaction=False)
    if reward > 0:
        pipe.incr(f"{exp_key}:arm:{arm_id}:alpha")
    else:
        pipe.incr(f"{exp_key}:arm:{arm_id}:beta")
    pipe.incr(f"{exp_key}:total_draws")
    pipe.execute()


def init_experiment(experiment_id: str, n_arms: int) -> None:
    """Seed Dragonfly state for a new experiment (uniform Beta(1,1) priors)."""
    df = get_dragonfly()
    exp_key = _exp_key(experiment_id)
    pipe = df.pipeline(transaction=False)
    pipe.set(f"{exp_key}:n_arms", n_arms)
    pipe.set(f"{exp_key}:total_draws", 0)
    for k in range(n_arms):
        pipe.set(f"{exp_key}:arm:{k}:alpha", 1)
        pipe.set(f"{exp_key}:arm:{k}:beta", 1)
    pipe.execute()
