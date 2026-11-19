# ADR-005: Offline Replay Boundary and What This Repository Cannot Measure

## Status
Accepted

## Context
Deterministic offline replay isolates bitemporal SQL leakage and watermark eviction math cleanly, but it cannot observe distributed infrastructure failure modes that occur only on live Kafka, Spark, and cloud storage clusters.

## Decision
State explicitly which properties transfer identically to production feature pipelines and which operational behaviors lie outside this repository's measurement boundary.

## What Survives Identically Against Production Pipelines
* The bitemporal join gap (`event_ts <= prediction_ts` vs adding `AND ingestion_ts <= prediction_ts`) and exact corrupted-row identification.
* Exact Mann-Whitney ROC-AUC inflation calculation when a batch-trained model is evaluated on online serving features.
* Spark Structured Streaming watermark eviction semantics over `(event_ts, ingestion_ts)` arrival order and multi-cloud manifest SHA-256 parity.

## What This Repository Deliberately Cannot Measure
1. **Kafka consumer rebalance pauses and broker replication lag:** Live partition reassignments create bursty ingestion delays that cannot be measured without real broker telemetry.
2. **Spark Structured Streaming executor OOMs and RocksDB state-store I/O:** Widening watermarks from 15m to 60m or 360m increases state memory linearly with active key cardinality, which requires cluster-level profiling.
3. **Cloud object-store listing latency and cross-region egress costs:** Manifest normalization verifies schema and digest parity across S3, GCS, and Azure Blob, not live network throughput or billing.
4. **Offline sandbox tooling boundary:** In offline environments without `pytest`, `ruff`, or Playwright, `Makefile` falls back to stdlib `unittest` + `trace` (verifying 74 tests at 90%+ line coverage), `tools/lint_check.py`, and `weasyprint`/`pdftoppm` + `PIL`/`ffmpeg`, while `.github/workflows/ci.yml` executes `ruff` and `pytest` across Python 3.10, 3.11, and 3.12.
