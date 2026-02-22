# Architecture

## System Overview

The bandit system has two deployable components plus a Seldon routing layer:

- **bandit-service** — stateless FastAPI inference service. Handles arm selection (Thompson Sampling) and reward recording. All state lives in Dragonfly.
- **bandit-pipeline** — Dagster code location. Handles data ingestion, OBP replay simulation, posterior snapshotting, regret computation, off-policy evaluation, and experiment conclusion.
- **SeldonDeployment** — Kubernetes CR that configures Seldon Core to use `bandit-service` as the ROUTER, routing each prediction request to one of N variant model pods.

## Data Flow

```
MinIO (bandit/raw/obp/)
    │
    ├── ingest_obp [Dagster]
    │     Download OBP Parquet → MinIO
    │
    └── obp_replay_simulator [Dagster]
          Pump logged events → POST /select → POST /reward
                │
                ▼
         bandit-service (FastAPI :8000)
                │
      ┌─────────┴──────────┐
      ▼                    ▼
  Dragonfly            Postgres
  (alpha/beta)         (bandit_events)
      │
      ▼
  snapshot_posteriors [Dagster every 100 events]
      │
      ▼
  Postgres (posterior_snapshots) → Grafana
      │
  compute_regret [Dagster]
      │
      ▼
  MLflow (regret curves, OPE results)
      │
  evaluate_ope [Dagster]
      │
  conclude_experiments [Dagster]
      │
      ▼
  Postgres (experiments.status = 'concluded', winner_arm)
  + Grafana annotation
```

## bandit-service

**Image**: `ghcr.io/ekenheim/bandit-service`  
**Port**: 8000 (HTTP), 9090 (Prometheus metrics)  
**State**: stateless — all bandit state is in Dragonfly

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/select` | Thompson Sampling arm selection |
| `POST` | `/reward` | Record observed reward, update posterior |
| `GET` | `/health` | Liveness probe |
| `GET` | `/metrics` | Prometheus metrics |

### Thompson Sampling (Phase 1 — Beta-Bernoulli)

For each arm `k`, maintain `Beta(α_k, β_k)` posterior over conversion rate `θ_k`.

On each `/select` request:
1. Read `alpha_k`, `beta_k` from Dragonfly for all arms
2. Sample `θ̃_k ~ Beta(α_k, β_k)` for each arm
3. Serve arm `k* = argmax_k θ̃_k`
4. Compute `p_best` via 1000-sample Monte Carlo

On each `/reward` request:
- If `reward > 0`: `INCR experiment:{id}:arm:{k}:alpha`
- Otherwise: `INCR experiment:{id}:arm:{k}:beta`
- Dragonfly `INCR` is atomic — safe for concurrent updates

### Contextual Bandit (Phase 2 — LinUCB)

`θ̂_k = (A_k⁻¹ b_k)ᵀ x + α √(xᵀ A_k⁻¹ x)`

`A` and `b` matrices serialised as NumPy arrays and stored in Dragonfly via `SET`/`GET`. Use rank-1 inverse updates (Sherman-Morrison) for high-throughput production.

## bandit-pipeline

**Image**: `ghcr.io/ekenheim/bandit-pipeline`  
**Port**: 4000 (Dagster gRPC)

### Jobs

| Job | Schedule | Description |
|---|---|---|
| `ingest_obp` | Manual / on-demand | Download OBP Parquet → MinIO `bandit/raw/obp/` |
| `obp_replay_simulator` | Manual / on-demand | Replay logged events through bandit-service |
| `snapshot_posteriors` | Sensor (every 100 events) | Write `alpha`/`beta` to `posterior_snapshots` table |
| `compute_regret` | After replay | Thompson Sampling vs uniform regret curves → MLflow |
| `evaluate_ope` | After replay | IPS/DM estimators on held-out OBP data → MLflow |
| `conclude_experiments` | Periodic sensor | Check P(best) > 0.95; mark winner; Grafana annotation |

## Dragonfly State Schema

```
experiment:{exp_id}:arm:{k}:alpha  → INTEGER  (successes + 1, init=1)
experiment:{exp_id}:arm:{k}:beta   → INTEGER  (failures + 1, init=1)
experiment:{exp_id}:n_arms         → INTEGER
experiment:{exp_id}:total_draws    → INTEGER
```

**DB**: 2 (`DRAGONFLY_DB=2`)  
**Host**: `dragonfly.database.svc.cluster.local:6379`

## Postgres Schema

```sql
experiments          -- experiment definitions + status + winner
experiment_arms      -- variant definitions (name, seldon_model)
bandit_events        -- every arm selection + reward
posterior_snapshots  -- alpha/beta snapshots every 100 events (for Grafana)
```

## Stopping Rule

No p-values. Experiment concludes when:

`P(θ_{k*} = max_k θ_k) > 0.95`

Estimated via 10,000 Monte Carlo samples from each Beta posterior. Valid at any sample size (sequential), calibrated (95% posterior probability), and decision-theoretic (maps to expected loss from wrong decision).

## Seldon Routing

Seldon Core v1.19.0 in `datasci` namespace. The `bandit-router` SeldonDeployment uses type `ROUTER` — Seldon calls `bandit-service` to get the arm ID, then forwards the request to the corresponding child predictor (`arm-0`, `arm-1`, ...).

```
Client → Seldon Gateway → thompson-router (bandit-service)
                                │
                         returns arm_id
                                │
              ┌─────────────────┴──────────────┐
              ▼                                ▼
           arm-0 (model pod)              arm-1 (model pod)
```

## Kubernetes Layout

```
k8s/datasci/bandit/
├── ks.yaml                    # Flux Kustomization
└── app/
    ├── kustomization.yaml
    ├── externalsecret.yaml    # bandit-minio + bandit-postgres
    ├── db/postgres.yaml       # db.movetokube.com CRDs
    ├── pipeline/              # Dagster code location Deployment + Service
    ├── service/               # FastAPI Deployment + Service + Ingress
    └── seldon/                # SeldonDeployment bandit-router
```

## CI/CD

Path-scoped GitHub Actions workflows build and push images to GHCR on merge to `main` or on version tags (`v*.*.*`). Renovate tracks `ghcr.io` image tags and opens PRs to bump `deployment.yaml` image references automatically.
