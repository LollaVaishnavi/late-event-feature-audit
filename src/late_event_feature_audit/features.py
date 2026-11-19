"""Bitemporal SQLite feature extraction contrasting batch snapshots vs point-in-time joins.

Exists to demonstrate in real SQL why filtering only on event_ts <= prediction_ts
leaks late-arriving streaming events into offline training rows, while adding
the bitemporal predicate ingestion_ts <= prediction_ts restores online parity.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np

from late_event_feature_audit.stream import PredictionPoint, StreamEvent

FEATURE_NAMES: tuple[str, ...] = (
    "events_24h",
    "amount_sum_24h",
    "burst_count_24h",
    "risk_alerts_24h",
    "max_amount_24h",
)

SCHEMA_SQL = """
CREATE TABLE fact_events (
    event_id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL,
    partition INTEGER NOT NULL,
    kafka_offset INTEGER NOT NULL,
    event_ts_min REAL NOT NULL,
    ingestion_ts_min REAL NOT NULL,
    event_type TEXT NOT NULL,
    amount REAL NOT NULL,
    is_risk_signal INTEGER NOT NULL
);

CREATE TABLE dim_predictions (
    prediction_id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL,
    prediction_ts_min REAL NOT NULL,
    label INTEGER NOT NULL
);

CREATE INDEX idx_ev_entity_ts ON fact_events (entity_id, event_ts_min, ingestion_ts_min);
CREATE INDEX idx_pred_entity_ts ON dim_predictions (entity_id, prediction_ts_min);
"""

# Naive batch snapshot query: missing ingestion_ts <= prediction_ts edge!
BATCH_FEATURE_SQL = """
SELECT
    p.prediction_id,
    p.label,
    COUNT(e.event_id) AS events_24h,
    COALESCE(SUM(e.amount), 0.0) AS amount_sum_24h,
    COALESCE(SUM(CASE WHEN e.event_type = 'order_burst' THEN 1 ELSE 0 END), 0) AS burst_count_24h,
    COALESCE(SUM(e.is_risk_signal), 0) AS risk_alerts_24h,
    COALESCE(MAX(e.amount), 0.0) AS max_amount_24h
FROM dim_predictions AS p
LEFT JOIN fact_events AS e
    ON e.entity_id = p.entity_id
   AND e.event_ts_min <= p.prediction_ts_min
   AND e.event_ts_min > (p.prediction_ts_min - 1440.0)
GROUP BY p.prediction_id, p.label
ORDER BY p.prediction_id;
"""

# Strict point-in-time query: enforces BOTH event_ts <= prediction_ts AND ingestion_ts <= prediction_ts
PIT_FEATURE_SQL = """
SELECT
    p.prediction_id,
    p.label,
    COUNT(e.event_id) AS events_24h,
    COALESCE(SUM(e.amount), 0.0) AS amount_sum_24h,
    COALESCE(SUM(CASE WHEN e.event_type = 'order_burst' THEN 1 ELSE 0 END), 0) AS burst_count_24h,
    COALESCE(SUM(e.is_risk_signal), 0) AS risk_alerts_24h,
    COALESCE(MAX(e.amount), 0.0) AS max_amount_24h
FROM dim_predictions AS p
LEFT JOIN fact_events AS e
    ON e.entity_id = p.entity_id
   AND e.event_ts_min <= p.prediction_ts_min
   AND e.event_ts_min > (p.prediction_ts_min - 1440.0)
   AND e.ingestion_ts_min <= p.prediction_ts_min
GROUP BY p.prediction_id, p.label
ORDER BY p.prediction_id;
"""


@dataclass(frozen=True)
class FeatureComparisonMatrices:
    """Paired batch-snapshot vs point-in-time feature matrices for identical predictions."""

    prediction_ids: tuple[str, ...]
    labels: np.ndarray
    X_batch: np.ndarray
    X_pit: np.ndarray
    corrupted_row_count: int
    corrupted_row_pct: float
    risk_alert_leak_rows: int


def extract_batch_and_pit_features(
    events: tuple[StreamEvent, ...] | list[StreamEvent],
    predictions: tuple[PredictionPoint, ...] | list[PredictionPoint],
) -> FeatureComparisonMatrices:
    """Load events into in-memory SQLite and extract both batch and point-in-time features."""
    if not predictions:
        raise ValueError("predictions must not be empty")

    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(SCHEMA_SQL)
        conn.executemany(
            """
            INSERT INTO fact_events (
                event_id, entity_id, partition, kafka_offset,
                event_ts_min, ingestion_ts_min, event_type, amount, is_risk_signal
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    e.event_id,
                    e.entity_id,
                    e.partition,
                    e.kafka_offset,
                    e.event_ts_min,
                    e.ingestion_ts_min,
                    e.event_type,
                    e.amount,
                    e.is_risk_signal,
                )
                for e in events
            ],
        )
        conn.executemany(
            """
            INSERT INTO dim_predictions (prediction_id, entity_id, prediction_ts_min, label)
            VALUES (?, ?, ?, ?)
            """,
            [(p.prediction_id, p.entity_id, p.prediction_ts_min, p.label) for p in predictions],
        )
        conn.commit()
        batch_rows = conn.execute(BATCH_FEATURE_SQL).fetchall()
        pit_rows = conn.execute(PIT_FEATURE_SQL).fetchall()
    finally:
        conn.close()

    pred_ids = tuple(str(r[0]) for r in batch_rows)
    labels = np.asarray([int(r[1]) for r in batch_rows], dtype=int)
    X_batch = np.asarray([[float(v) for v in r[2:]] for r in batch_rows], dtype=float)
    X_pit = np.asarray([[float(v) for v in r[2:]] for r in pit_rows], dtype=float)

    row_diff = np.any(np.abs(X_batch - X_pit) > 1e-9, axis=1)
    corrupted_count = int(np.sum(row_diff))
    corrupted_pct = round(100.0 * corrupted_count / labels.size, 1)
    risk_leak_rows = int(np.sum(X_batch[:, 3] > X_pit[:, 3] + 1e-9))

    return FeatureComparisonMatrices(
        prediction_ids=pred_ids,
        labels=labels,
        X_batch=X_batch,
        X_pit=X_pit,
        corrupted_row_count=corrupted_count,
        corrupted_row_pct=corrupted_pct,
        risk_alert_leak_rows=risk_leak_rows,
    )
