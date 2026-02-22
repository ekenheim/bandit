# Bayesian Bandit

Thompson Sampling for Adaptive Experimentation — Beyond Fixed-Allocation A/B Tests

A platform for defining multi-armed bandit experiments, serving variants via Seldon, updating Beta posteriors in real-time with Dragonfly, and declaring winners using a posterior probability stopping rule (no p-values).

## Architecture

```
[OBP Dataset / Simulated Traffic]
         │
         ▼
[Dagster: ingest_obp]
(parse logged bandit data → MinIO bandit/raw/obp/)
         │
         ▼
[Seldon SeldonDeployment: bandit-router]
(Thompson Sampling router → one of N variant pods)
         │
         ├──▶ arm-0  (control)
         ├──▶ arm-1  (treatment_a)
         └──▶ arm-N  (treatment_n)
         │
         ▼
[Reward feedback loop]
User click/no-click → bandit-service /reward
         │
         ▼
[Dragonfly: Beta posterior state]
experiment:{exp_id}:arm:{k}:alpha / :beta
         │
         ▼
[Thompson Sampling: /select]
reads alpha/beta → samples Beta → argmax → arm_id
         │
         ├──▶ [Postgres: bandit_events + posterior_snapshots]
         └──▶ [Grafana: live posterior distribution dashboard]
```

See [docs/architecture.md](docs/architecture.md) for full system design.

## Environment (uv)

The repo uses [uv](https://docs.astral.sh/uv/) for the Python environment. From the repo root:

```bash
# Create .venv and install dependencies (bandit-service + shared)
uv sync

# Run bandit-service (Dragonfly must be reachable)
cd apps/bandit-service && uv run uvicorn service.main:app --reload --port 8000

# Run bandit-pipeline (Dagster)
cd apps/bandit-pipeline && uv run dagster dev -m pipeline.repository
```

To install the full stack including the pipeline (Dagster, OBP, MLflow, etc.), use the optional group. Note: the `pipeline` group pulls in older transitive deps that may not have Windows wheels; it is best used on Linux or in CI.

```bash
uv sync --extra pipeline
```

## Repo Structure

```
bandit/
├── apps/
│   ├── bandit-service/       # FastAPI inference service (/select, /reward)
│   └── bandit-pipeline/      # Dagster code location (ingest, replay, OPE, conclude)
├── .github/workflows/        # CI: build + push to ghcr.io per component
└── docs/                     # Architecture and design docs
```

Kubernetes manifests live in the cluster’s k8s repo (Flux-reconciled).

## Infrastructure Dependencies

All services are pre-provisioned in the cluster. No setup required before starting.

| Service | Namespace | Purpose |
|---|---|---|
| Dragonfly | `database` | Bandit state — `alpha`/`beta` per arm |
| Seldon Core v1.19.0 | `datasci` | Multi-armed bandit router |
| Dagster | `datasci` | Pipeline orchestration |
| MLflow | `datasci` | Experiment tracking & regret curves |
| Crunchy Postgres | `database` | Bandit events & posterior snapshots |
| MinIO Secondary | `storage` | OBP dataset (Bronze/Silver) |
| Grafana | `observability` | Live posterior distribution dashboard |

## Local Development

After `uv sync` (see [Environment (uv)](#environment-uv)):

### bandit-service

```bash
cd apps/bandit-service
# Requires Dragonfly at localhost:6379 (or set DRAGONFLY_HOST / DRAGONFLY_PORT / DRAGONFLY_DB)
uv run uvicorn service.main:app --reload --port 8000
```

### bandit-pipeline

```bash
cd apps/bandit-pipeline
uv run dagster dev -m pipeline.repository
```

If port 3000 is in use, pass a different port: `uv run dagster dev -m pipeline.repository -p 3001`

## Deployment

1. Push to `master` with changes in `apps/bandit-service/**` or `apps/bandit-pipeline/**`.
2. GitHub Actions builds the corresponding image and pushes to `ghcr.io/ekenheim/bandit-service` or `ghcr.io/ekenheim/bandit-pipeline`.
3. Update the image tag in the bandit app manifests in your **k8s repo** (e.g. `app/service/deployment.yaml`, `app/pipeline/deployment.yaml`, `app/seldon/seldondeployment.yaml`).
4. FluxCD reconciles and rolls out the new image.

### Register Dagster code location

Add `bandit-pipeline` to the Dagster HelmRelease workspace servers alongside `hempriser-pipeline`:

```yaml
dagsterWebserver:
  workspace:
    servers:
      - host: hempriser-pipeline.development.svc.cluster.local
        port: 4000
        name: hempriser-pipeline
      - host: bandit-pipeline.datasci.svc.cluster.local
        port: 4000
        name: bandit-pipeline
```

## Dataset

Primary: **Open Bandit Pipeline (OBP)** — ZOZO Research  
~26 million real fashion e-commerce interactions. Enables rigorous off-policy evaluation via importance-weighted estimators.

Fallback (MVP): UCI Online Retail II — simulate CTR from product category purchase rates.

## License

Private project.
