# ADR-001: SQLite and Partition Manifests Over Live Kafka Brokers and Cloud Buckets

## Status
Accepted

## Context
Auditing point-in-time feature leakage across Kafka, Spark Streaming, S3, GCS, and Azure Blob normally requires Dockerized brokers, cloud credentials, and live buckets, preventing offline reproducibility in CI.

## Decision
Replay partitioned Kafka offsets (`event_ts`, `ingestion_ts`) and multi-cloud Hive-partitioned manifests (`s3://`, `gs://`, `az://`) deterministically into in-memory SQLite (`fact_events` and `dim_predictions`).

## Alternatives Considered
1. **Docker Compose Kafka + LocalStack S3 + Azurite:** Heavyweight, slow to boot in GitHub Actions, and prone to port/timing flakes unrelated to feature correctness.
2. **Pure Python loops without SQL:** Misses the exact SQL join predicate difference (`event_ts <= prediction_ts` vs adding `AND ingestion_ts <= prediction_ts`).

## Consequences
* `make verify` runs completely offline across all 4800 streaming events and 3 cloud providers with zero containers or cloud keys.
* Trade-off: Broker rebalance pauses and Spark executor OOM behavior are outside the offline boundary (recorded in ADR-005).
