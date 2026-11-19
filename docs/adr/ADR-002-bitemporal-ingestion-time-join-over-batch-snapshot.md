# ADR-002: Bitemporal (`event_ts`, `ingestion_ts`) Joins Over Batch Snapshots

## Status
Accepted

## Context
Warehouse backfills typically query the latest table state using `event_ts <= prediction_ts`. When gateway risk alerts or mobile events arrive late, that batch query silently includes records that had not landed when online serving scored the transaction.

## Decision
Require an immutable `ingestion_ts` column on every streaming event and enforce the bitemporal predicate `event_ts <= prediction_ts AND ingestion_ts <= prediction_ts` for offline training feature extraction.

## Alternatives Considered
1. **Event-time-only batch snapshots:** Structurally lacks the arrival-time edge; produces 4.7 AUC points of phantom offline inflation at 2h p95 lag (and 8.4 points at 12h p95 lag) while corrupting 7.4% of training rows.
2. **Logging online serving features only:** Avoids leakage for future models, but cannot backfill new features over historical periods without a bitemporal event log.

## Consequences
* Offline ROC-AUC matches online serving ROC-AUC honestly.
* Impossible clock-skew events (`ingestion_ts < event_ts`) are rejected at the join boundary rather than silently distorting counts.
