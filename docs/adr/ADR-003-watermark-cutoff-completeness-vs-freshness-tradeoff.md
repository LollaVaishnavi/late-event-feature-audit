# ADR-003: Spark Streaming Watermark Cutoff Trade-Off (Completeness vs Freshness)

## Status
Accepted

## Context
Spark Structured Streaming bounds state-store memory by dropping events whose `event_ts` falls behind `max_event_ts - watermark`. Choosing a watermark solely to shrink memory silently evicts valid late-arriving features.

## Decision
Model Spark Structured Streaming watermark eviction explicitly (`apply_spark_watermark`) and report the fraction of late events dropped alongside resulting online ROC-AUC across watermarks from 5 minutes to 360 minutes.

## Alternatives Considered
1. **Tight 15-minute watermark by default:** Keeps state small, but results in 59.4% of late events dropped in our replay (804 of 1353 late events).
2. **Unbounded state retention:** Preserves completeness, but causes unbounded memory growth in long-running Spark Streaming jobs.

## Consequences
* Surfaces the exact completeness-vs-freshness frontier before production deployment.
* Shows that widening the watermark from 5m (82.3% dropped, AUC 0.7408) to 60m (16.4% dropped, AUC 0.7632) recovers online model quality while bounding state growth.
