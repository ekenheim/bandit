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

## Repo Structure

```
bandit/
├── apps/
│   ├── bandit-service/       # FastAPI inference service (/select, /reward)
│   └── bandit-pipeline/      # Dagster code location (ingest, replay, OPE, conclude)
├── k8s/
│   └── datasci/bandit/       # Flux-reconciled Kubernetes manifests
├── .github/workflows/        # CI: build + push to ghcr.io per component
└── docs/                     # Architecture and design docs
```

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

### bandit-service

```bash
cd apps/bandit-service
pip install -r requirements.txt

# Requires Dragonfly accessible at localhost:6379 (or set DRAGONFLY_HOST)
export DRAGONFLY_HOST=localhost
export DRAGONFLY_PORT=6379
export DRAGONFLY_DB=2

uvicorn service.main:app --reload --port 8000
```

### bandit-pipeline

```bash
cd apps/bandit-pipeline
pip install -r requirements.txt

dagster dev -m pipeline.repository
```

## Deployment

1. Push to `main` with changes in `apps/bandit-service/**` or `apps/bandit-pipeline/**`.
2. GitHub Actions builds the corresponding image and pushes to `ghcr.io/ekenheim/bandit-service` or `ghcr.io/ekenheim/bandit-pipeline`.
3. Update the image tag in the relevant `k8s/datasci/bandit/app/*/deployment.yaml`.
4. FluxCD reconciles the updated manifest and rolls out the new image.

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
