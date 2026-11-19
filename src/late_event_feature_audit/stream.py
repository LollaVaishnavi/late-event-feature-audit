"""Deterministic Kafka partition/offset and Spark Streaming watermark replay engine.

Exists to simulate realistic bitemporal event streams where each event carries
both an event_ts (when the action occurred) and an ingestion_ts (when it landed
in Kafka / cloud storage), plus Spark Structured Streaming watermark eviction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from late_event_feature_audit.config import StreamConfig


@dataclass(frozen=True)
class StreamEvent:
    """Single Kafka event with explicit event-time and ingestion-time stamps."""

    event_id: str
    entity_id: str
    partition: int
    kafka_offset: int
    event_ts_min: float
    ingestion_ts_min: float
    event_type: str
    amount: float
    is_risk_signal: int


@dataclass(frozen=True)
class PredictionPoint:
    """Decision timestamp where an online model scores an entity."""

    prediction_id: str
    entity_id: str
    prediction_ts_min: float
    label: int


@dataclass(frozen=True)
class WatermarkReplayResult:
    """Outcome of applying a Spark Structured Streaming allowed-lateness watermark."""

    watermark_minutes: float
    total_events: int
    late_events_total: int
    accepted_events: tuple[StreamEvent, ...]
    dropped_events_count: int
    dropped_late_share_pct: float


@dataclass(frozen=True)
class StreamCorpus:
    """Complete seeded streaming dataset with Kafka events and prediction points."""

    events: tuple[StreamEvent, ...]
    predictions: tuple[PredictionPoint, ...]
    p50_lag_minutes: float
    p95_lag_minutes: float
    p99_lag_minutes: float


def generate_stream_corpus(config: StreamConfig | None = None) -> StreamCorpus:
    """Generate deterministic bitemporal Kafka events and prediction points.

    Risk-associated accounts experience genuine historical activity plus
    late-arriving payment/gateway risk alerts whose event_ts precedes
    prediction_ts while ingestion_ts lands after prediction_ts.
    """
    cfg = config or StreamConfig()
    rng = np.random.default_rng(cfg.seed)

    # Horizon in minutes: 14 days = 20,160 minutes
    horizon_min = 20160.0
    entity_ids = [f"ACCT-{i + 1:04d}" for i in range(cfg.entity_count)]
    entity_base_risk = {
        eid: float(rng.beta(1.5, 5.5)) for eid in entity_ids
    }

    # Log-normal lag calibrated so 95th percentile of late events hits cfg.p95_lag_minutes
    sigma = 1.0
    mu = float(np.log(max(1.0, cfg.p95_lag_minutes)) - 1.64485 * sigma)

    raw_events: list[dict] = []
    predictions: list[PredictionPoint] = []
    lags: list[float] = []

    # 1. Generate prediction points across the evaluation window [2880, 19440] min
    preds_per_entity = max(1, cfg.prediction_count // cfg.entity_count)
    pred_idx = 0
    for eid in entity_ids:
        base_r = entity_base_risk[eid]
        for _ in range(preds_per_entity):
            if pred_idx >= cfg.prediction_count:
                break
            p_ts = float(rng.uniform(2880.0, horizon_min - 720.0))
            # True risk probability driven by entity propensity + local burst
            burst = 1 if rng.random() < (0.12 + 0.55 * base_r) else 0
            label = 1 if rng.random() < (0.06 + 0.62 * burst + 0.20 * base_r) else 0
            predictions.append(
                PredictionPoint(
                    prediction_id=f"PRED-{pred_idx + 1:05d}",
                    entity_id=eid,
                    prediction_ts_min=round(p_ts, 2),
                    label=label,
                )
            )
            pred_idx += 1

            # Normal prior signal (arrives on time, visible to both batch and online)
            if burst:
                ev_ts = p_ts - float(rng.uniform(30.0, 720.0))
                lag = float(rng.uniform(0.2, 2.0))
                raw_events.append(
                    {
                        "entity_id": eid,
                        "event_ts": ev_ts,
                        "ingestion_ts": ev_ts + lag,
                        "event_type": "order_burst",
                        "amount": round(float(rng.uniform(90.0, 420.0)), 2),
                        "is_risk_signal": 0,
                    }
                )
                lags.append(lag)

            # Late-arriving gateway risk alert linked to positive outcome:
            # event_ts is just before prediction_ts, but ingestion_ts arrives late!
            if label == 1 and rng.random() < (0.55 if cfg.late_event_rate > 0 else 0.0):
                ev_ts = p_ts - float(rng.uniform(5.0, 90.0))
                lag = float(rng.lognormal(mean=mu, sigma=sigma))
                raw_events.append(
                    {
                        "entity_id": eid,
                        "event_ts": ev_ts,
                        "ingestion_ts": ev_ts + lag,
                        "event_type": "gateway_risk_alert",
                        "amount": round(float(rng.uniform(150.0, 650.0)), 2),
                        "is_risk_signal": 1,
                    }
                )
                lags.append(lag)

    # 2. Fill remaining background streaming events across all entities
    target_events = cfg.entity_count * cfg.events_per_entity
    while len(raw_events) < target_events:
        eid = entity_ids[int(rng.integers(0, cfg.entity_count))]
        ev_ts = float(rng.uniform(0.0, horizon_min - 360.0))
        is_late = rng.random() < cfg.late_event_rate
        lag = float(rng.lognormal(mean=mu, sigma=sigma)) if is_late else float(rng.uniform(0.1, 2.0))
        is_risk = 1 if rng.random() < 0.04 else 0
        raw_events.append(
            {
                "entity_id": eid,
                "event_ts": ev_ts,
                "ingestion_ts": ev_ts + lag,
                "event_type": "gateway_risk_alert" if is_risk else "page_checkout",
                "amount": round(float(rng.uniform(15.0, 220.0)), 2),
                "is_risk_signal": is_risk,
            }
        )
        lags.append(lag)

    # Sort by ingestion_ts to assign realistic per-partition Kafka offsets
    raw_events.sort(key=lambda r: (r["ingestion_ts"], r["entity_id"], r["event_ts"]))
    partition_offsets = [0] * cfg.partitions
    events: list[StreamEvent] = []
    for idx, row in enumerate(raw_events):
        part = sum(ord(ch) for ch in row["entity_id"]) % cfg.partitions
        offset = partition_offsets[part]
        partition_offsets[part] += 1
        events.append(
            StreamEvent(
                event_id=f"EV-{idx + 1:06d}",
                entity_id=row["entity_id"],
                partition=part,
                kafka_offset=offset,
                event_ts_min=round(row["event_ts"], 2),
                ingestion_ts_min=round(row["ingestion_ts"], 2),
                event_type=row["event_type"],
                amount=row["amount"],
                is_risk_signal=row["is_risk_signal"],
            )
        )

    lag_arr = np.asarray(lags, dtype=float)
    return StreamCorpus(
        events=tuple(events),
        predictions=tuple(predictions),
        p50_lag_minutes=round(float(np.quantile(lag_arr, 0.50)), 2),
        p95_lag_minutes=round(float(np.quantile(lag_arr, 0.95)), 2),
        p99_lag_minutes=round(float(np.quantile(lag_arr, 0.99)), 2),
    )


def apply_spark_watermark(
    events: tuple[StreamEvent, ...] | list[StreamEvent],
    watermark_minutes: float,
) -> WatermarkReplayResult:
    """Replay Kafka events in ingestion order and evict events older than max_event_ts - watermark."""
    if watermark_minutes < 0.0:
        raise ValueError(f"watermark_minutes must be non-negative, got {watermark_minutes}")

    ordered = sorted(events, key=lambda e: (e.ingestion_ts_min, e.partition, e.kafka_offset))
    max_event_ts = -1e18
    accepted: list[StreamEvent] = []
    dropped = 0
    late_total = 0

    for ev in ordered:
        lag = ev.ingestion_ts_min - ev.event_ts_min
        if lag > 5.0:
            late_total += 1
        if max_event_ts > -1e17 and ev.event_ts_min < (max_event_ts - watermark_minutes):
            dropped += 1
            continue
        if ev.event_ts_min > max_event_ts:
            max_event_ts = ev.event_ts_min
        accepted.append(ev)

    drop_late_pct = 100.0 * dropped / late_total if late_total > 0 else 0.0
    return WatermarkReplayResult(
        watermark_minutes=watermark_minutes,
        total_events=len(ordered),
        late_events_total=late_total,
        accepted_events=tuple(accepted),
        dropped_events_count=dropped,
        dropped_late_share_pct=round(min(100.0, drop_late_pct), 1),
    )
