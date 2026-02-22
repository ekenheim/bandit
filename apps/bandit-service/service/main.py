"""FastAPI bandit inference service.

Endpoints:
  POST /select  — Thompson Sampling arm selection
  POST /reward  — Record reward, update Beta posterior
  POST /experiments  — Create / seed a new experiment
  GET  /health  — Liveness probe
  GET  /metrics — Prometheus metrics (via instrumentator)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from .bandit import (
    get_n_arms,
    init_experiment,
    read_posteriors,
    record_reward,
    thompson_sample,
)
from .stopping import p_best_all_arms, should_conclude

app = FastAPI(title="Bandit Service", version="0.1.0")

Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class BanditRequest(BaseModel):
    experiment_id: str
    user_id: str | None = None
    context: dict | None = None  # reserved for Phase 2 contextual bandit


class BanditResponse(BaseModel):
    arm_id: int
    arm_name: str
    p_best: float  # P(this arm is the best) — from posterior simulation


class RewardRequest(BaseModel):
    experiment_id: str
    arm_id: int
    reward: float  # 1.0 = success, 0.0 = failure


class ExperimentCreateRequest(BaseModel):
    experiment_id: str
    n_arms: int


class ConcludeResponse(BaseModel):
    should_conclude: bool
    winner_arm_id: int | None
    checked_at: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/select", response_model=BanditResponse)
def select_arm(req: BanditRequest) -> BanditResponse:
    """Thompson Sampling arm selection. Target latency: < 2ms."""
    try:
        n_arms = get_n_arms(req.experiment_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Experiment {req.experiment_id!r} not found")

    alphas, betas = read_posteriors(req.experiment_id, n_arms)
    arm_id, p_best = thompson_sample(alphas, betas)

    return BanditResponse(
        arm_id=arm_id,
        arm_name=f"arm_{arm_id}",
        p_best=p_best,
    )


@app.post("/reward", status_code=204)
def reward(req: RewardRequest) -> None:
    """Record observed reward; atomically update Beta posterior in Dragonfly."""
    try:
        get_n_arms(req.experiment_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Experiment {req.experiment_id!r} not found")

    if req.reward not in (0.0, 1.0) and not (0.0 <= req.reward <= 1.0):
        raise HTTPException(status_code=422, detail="reward must be in [0, 1]")

    record_reward(req.experiment_id, req.arm_id, req.reward)


@app.post("/experiments", status_code=201)
def create_experiment(req: ExperimentCreateRequest) -> dict:
    """Seed Dragonfly state for a new experiment with uniform Beta(1,1) priors."""
    if req.n_arms < 2:
        raise HTTPException(status_code=422, detail="n_arms must be >= 2")
    init_experiment(req.experiment_id, req.n_arms)
    return {"experiment_id": req.experiment_id, "n_arms": req.n_arms, "status": "initialised"}


@app.get("/experiments/{experiment_id}/conclude", response_model=ConcludeResponse)
def check_conclude(experiment_id: str, threshold: float = 0.95) -> ConcludeResponse:
    """Check whether the stopping rule has been met for an experiment."""
    try:
        get_n_arms(experiment_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id!r} not found")

    stop, winner = should_conclude(experiment_id, threshold=threshold)
    return ConcludeResponse(
        should_conclude=stop,
        winner_arm_id=winner if stop else None,
        checked_at=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/experiments/{experiment_id}/p_best")
def get_p_best(experiment_id: str) -> dict:
    """Return P(arm k is best) for every arm — powers the Grafana gauge panel."""
    try:
        get_n_arms(experiment_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id!r} not found")

    return {"experiment_id": experiment_id, "p_best": p_best_all_arms(experiment_id)}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
