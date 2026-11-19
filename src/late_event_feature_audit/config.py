"""Validated configuration dataclasses for streaming lag and leakage audits.

Exists so event-time vs ingestion-time lag distributions, Kafka partition
counts, Spark Streaming watermarks, and leakage gate policies live in typed,
validated objects rather than scattered constants.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


class StreamConfigError(ValueError):
    """Raised when stream replay, watermark, or policy settings are invalid."""


@dataclass(frozen=True)
class StreamConfig:
    """Deterministic configuration for the offline Kafka/Spark event-log replay.

    Attributes:
        seed: Random seed controlling entities, events, and lag draws.
        entity_count: Number of distinct customer/account entities.
        prediction_count: Total prediction decision points evaluated.
        events_per_entity: Average streaming events per entity.
        partitions: Number of Kafka topic partitions.
        late_event_rate: Fraction of events arriving with non-trivial lag.
        p95_lag_minutes: Target 95th-percentile arrival lag (minutes) for late events.
        watermark_minutes: Spark Structured Streaming allowed lateness watermark.
    """

    seed: int = 20260619
    entity_count: int = 120
    prediction_count: int = 1200
    events_per_entity: int = 40
    partitions: int = 4
    late_event_rate: float = 0.28
    p95_lag_minutes: float = 120.0
    watermark_minutes: float = 60.0

    def __post_init__(self) -> None:
        if self.entity_count < 10:
            raise StreamConfigError(f"entity_count must be >= 10, got {self.entity_count}")
        if self.prediction_count < 50:
            raise StreamConfigError(
                f"prediction_count must be >= 50, got {self.prediction_count}"
            )
        if self.events_per_entity < 5:
            raise StreamConfigError(
                f"events_per_entity must be >= 5, got {self.events_per_entity}"
            )
        if self.partitions < 1:
            raise StreamConfigError(f"partitions must be >= 1, got {self.partitions}")
        if not (0.0 <= self.late_event_rate <= 1.0):
            raise StreamConfigError("late_event_rate must be in [0, 1]")
        if self.p95_lag_minutes <= 0.0:
            raise StreamConfigError("p95_lag_minutes must be positive")
        if self.watermark_minutes < 0.0:
            raise StreamConfigError("watermark_minutes must be non-negative")


@dataclass(frozen=True)
class LeakagePolicy:
    """CI gate thresholds for offline-to-online point-in-time leakage audits."""

    max_auc_inflation: float = 0.02
    max_corrupted_row_pct: float = 5.0
    max_watermark_drop_pct: float = 10.0

    def __post_init__(self) -> None:
        if self.max_auc_inflation < 0.0:
            raise StreamConfigError("max_auc_inflation must be non-negative")
        if not (0.0 <= self.max_corrupted_row_pct <= 100.0):
            raise StreamConfigError("max_corrupted_row_pct must be in [0, 100]")
        if not (0.0 <= self.max_watermark_drop_pct <= 100.0):
            raise StreamConfigError("max_watermark_drop_pct must be in [0, 100]")

    @classmethod
    def from_json(cls, path: str | Path) -> LeakagePolicy:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)
