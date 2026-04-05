# Unified Monitoring And Deployment Plan

## Decisions

- Repo model: multi-repo for the product apps
- Shared monitoring: create first as a separate shared repo/package
- App deployment: Google Cloud Run
- Database now: serverless Postgres outside GCP, preferably Neon
- Database later if needed: Cloud SQL Postgres
- Dashboard auth: IAP
- Monitoring API auth: private Cloud Run IAM
- ETL: scheduled Cloud Run Job, not streaming
- Telemetry durability: best-effort; occasional monitoring loss is acceptable

## Why This Shape Fits The Current Scale

- The apps have low to moderate operational volume: 5-10 users and hundreds of events per day.
- Cloud Run keeps ops simple and cost low while still allowing bursty on-demand workloads.
- A shared Postgres database solves concurrent write limits that the current shared SQLite approach cannot handle safely across multiple services.
- Neon keeps monthly cost close to zero while preserving a clean migration path to Cloud SQL later.
- Creating the shared monitoring repo first is reasonable as long as its initial scope stays narrow: schema, client package, and monitoring services only.

## Target Architecture

### Repositories

1. `portfolio-builder-agent`
2. `spectrum-news-agent`
3. `agent-monitoring`

### Responsibilities

- `portfolio-builder-agent`: product UI, agent execution, writes telemetry to shared Postgres
- `spectrum-news-agent`: product UI, agent execution, writes telemetry to shared Postgres
- `agent-monitoring`: shared telemetry package, schema migrations, monitoring API, monitoring dashboard, ETL job

### Runtime Topology

1. `portfolio-builder` Cloud Run service
2. `spectrum-news` Cloud Run service
3. `monitoring-api` Cloud Run service, private, IAM-protected
4. `monitoring-ui` Cloud Run service, public only through IAP
5. `etl-job` Cloud Run Job, scheduled by Cloud Scheduler
6. `Neon Postgres` shared across all services

## Shared Data Model

### Common Tables

- `events`
- `llm_calls`
- `tool_calls`

### App-Specific Tables

- `agent_performance` for portfolio-builder
- `article_metadata` for spectrum-news

### Required Shared Columns

- `id`
- `app`
- `service`
- `environment`
- `session_id`
- `run_id`
- `timestamp`
- `schema_version`

### Data Rules

- Shared tables must include `app` so the dashboard and ETL can filter by application.
- Writes must be append-only except for ETL materialization tables that may upsert by `run_id`.
- Telemetry writes must never block the user request for long; use short timeouts and bounded retries.
- If the database is unavailable, log locally and drop telemetry after retry limits are reached.

## Phase 1: Create The Shared Monitoring Repo First

### Goal

Create a minimal shared repo that defines the telemetry contract before either app migrates.

### Deliverables

1. New repo: `agent-monitoring`
2. Python package inside the repo, for example `agent_monitoring`
3. Unified dataclasses or models for common and app-specific records
4. `EventStore` and `MonitoringStore` protocols
5. `PostgresEventStore` implementation
6. SQL migration scripts for the unified schema
7. Small integration test suite against Postgres in Docker
8. Versioning and release strategy for the package

### Scope Guardrails

- Do not move product logic into the shared repo.
- Do not create a queue, Pub/Sub topic, or separate ingest service in the first iteration.
- Keep the package focused on schema, storage, and monitoring service code.

### Package Contents

- `models.py`
- `store/base.py`
- `store/postgres.py`
- `migrations/`
- `settings.py`
- `auth/` helpers for Cloud Run service-to-service auth where needed

### Acceptance Criteria

- Both apps can install the package from git or an internal package source.
- The package can create and query the unified schema in a Postgres container.
- Integration tests prove concurrent writes from multiple threads/processes do not error under expected load.

## Phase 2: Build The Shared Monitoring Services In The Same Repo

### Goal

Move monitoring out of the portfolio app and make it a standalone shared platform for both apps.

### Deliverables

1. `monitoring-api` FastAPI service in `agent-monitoring`
2. `monitoring-ui` Streamlit service in `agent-monitoring`
3. API endpoints for shared and app-specific tables
4. App filter support in the UI and API
5. Auth model finalized

### Auth Design

- `monitoring-ui`: protected with IAP
- `monitoring-api`: private Cloud Run service using IAM
- `monitoring-ui` service account gets `roles/run.invoker` on `monitoring-api`
- Direct developer access to `monitoring-api` is allowed through authenticated `gcloud` identity tokens when needed

### Acceptance Criteria

- The UI no longer reads the database directly.
- All admin queries go through the API.
- The API is not publicly accessible without IAM-authenticated requests.

## Phase 3: Migrate Portfolio Builder To The Shared Package

### Goal

Replace the portfolio app's current SQLite-first monitoring path with the shared Postgres-backed package.

### Tasks

1. Add `agent-monitoring` package dependency
2. Replace local event store imports with shared package imports
3. Implement Postgres backend configuration using secret-provided DSN
4. Keep local SQLite only as an optional developer fallback if still useful
5. Remove assumptions in the app that monitoring storage is file-backed
6. Stop treating portfolio's local monitoring API and dashboard as the source of truth

### Acceptance Criteria

- Portfolio writes `events`, `llm_calls`, `tool_calls`, and `agent_performance` to shared Postgres.
- App behavior is unchanged if telemetry temporarily fails.
- Docker tests still pass.

## Phase 4: Migrate Spectrum News To The Shared Package

### Goal

Make spectrum-news emit telemetry to the same shared schema and infrastructure.

### Tasks

1. Add `agent-monitoring` package dependency
2. Replace the local SQLite-only event store path
3. Map spectrum-specific writes into shared common tables plus `article_metadata`
4. Add app identity fields so monitoring can distinguish sources cleanly

### Acceptance Criteria

- Spectrum writes `events`, `llm_calls`, `tool_calls`, and `article_metadata` to shared Postgres.
- Docker tests still pass.

## Phase 5: Provision Shared Infrastructure

### Neon Setup

1. Create Neon project and database
2. Create application user with least-privilege permissions
3. Apply SQL migrations from `agent-monitoring`
4. Store DSN in Google Secret Manager

### GCP Setup

1. Create separate Cloud Run services for both product apps
2. Create Cloud Run services for `monitoring-api` and `monitoring-ui`
3. Configure service accounts per service
4. Grant `monitoring-ui` permission to invoke `monitoring-api`
5. Configure IAP for `monitoring-ui`
6. Configure Secret Manager access for DSN and application secrets

### Acceptance Criteria

- All services start with secret-provided configuration only.
- No service depends on shared filesystem state.

## Phase 6: ETL And Aggregation

### Goal

Materialize higher-level analytics without adding streaming complexity.

### Deliverables

1. `etl-job` in `agent-monitoring`
2. Cloud Scheduler trigger
3. Materialized or upserted summary tables

### Initial ETL Outputs

- Run summaries by app and run ID
- Tool usage counts by app and day
- LLM latency and token metrics by model and app
- Error rates by app and stage

### Scheduling

- Start with every 15 minutes or hourly depending on how fresh the dashboard needs to be
- Increase frequency only if the dashboard requires near-real-time rollups

## Phase 7: Deployment Rollout Order

1. Create `agent-monitoring` repo and ship package v0.1
2. Deploy `monitoring-api` and `monitoring-ui` against Neon
3. Migrate and deploy `portfolio-builder-agent`
4. Migrate and deploy `spectrum-news-agent`
5. Deploy ETL job and scheduler
6. Remove obsolete SQLite and GCS-FUSE-based monitoring paths when stable

## Local Development Plan

1. `agent-monitoring` repo provides a local Postgres compose profile
2. Both apps point to local Postgres through `EVENT_STORE_DSN`
3. Local development may still keep SQLite fallback temporarily, but Postgres should be the default integration path

## Risk Register

### Risk: Shared package churn slows both apps

Mitigation: keep v0.1 very small and focused on telemetry only.

### Risk: Cloud Run to Neon connection spikes

Mitigation: use pooled Neon connection string, short-lived connections or a small connection pool, and buffered writes.

### Risk: Monitoring API auth makes dashboard integration harder

Mitigation: use private Cloud Run IAM for the API and give only the dashboard service account invoke permission.

### Risk: Schema divergence between apps

Mitigation: shared repo owns migrations and canonical models from day one.

## Non-Goals For The First Iteration

- Kubernetes or GKE
- Pub/Sub or Kafka-based telemetry ingestion
- BigQuery analytics pipeline
- Cloud SQL until scale or governance requires it

## Success Criteria

1. Both apps run independently on Cloud Run.
2. Both apps write telemetry to one shared Postgres database.
3. Monitoring dashboard shows both apps with app-level filtering.
4. Monitoring API is private and authenticated.
5. Dashboard is IAP-protected.
6. ETL runs on a schedule and produces shared admin metrics.
7. No shared filesystem database remains in production.