"""Posterior probability stopping rule — no p-values."""

from __future__ import annotations

import numpy as np

from .bandit import get_n_arms, read_posteriors


def should_conclude(
    experiment_id: str,
    threshold: float = 0.95,
    n_samples: int = 10_000,
) -> tuple[bool, int | None]:
    """
    Returns (should_stop, winning_arm_id).

    The experiment concludes when P(θ_{k*} = max_k θ_k) > threshold.
    Estimated via Monte Carlo over each Beta posterior.

    This is valid at any sample size — unlike fixed-horizon tests, there is
    no multiple-comparisons penalty for checking continuously.
    """
    n_arms = get_n_arms(experiment_id)
    alphas, betas = read_posteriors(experiment_id, n_arms)

    samples = np.array(
        [np.random.beta(a, b, size=n_samples) for a, b in zip(alphas, betas)]
    )
    # p_best[k] = fraction of MC draws where arm k had the highest sample
    p_best = (samples.argmax(axis=0)[:, None] == np.arange(n_arms)).mean(axis=0)

    winner = int(p_best.argmax())
    return bool(p_best[winner] >= threshold), winner


def p_best_all_arms(
    experiment_id: str,
    n_samples: int = 10_000,
) -> list[float]:
    """Return P(arm k is best) for every arm — used by Grafana dashboard."""
    n_arms = get_n_arms(experiment_id)
    alphas, betas = read_posteriors(experiment_id, n_arms)

    samples = np.array(
        [np.random.beta(a, b, size=n_samples) for a, b in zip(alphas, betas)]
    )
    p_best = (samples.argmax(axis=0)[:, None] == np.arange(n_arms)).mean(axis=0)
    return p_best.tolist()
