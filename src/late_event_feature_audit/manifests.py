"""Unified multi-cloud partition manifest contract for AWS S3, GCS, and Azure Blob.

Exists to normalize Hive-partitioned event-log manifests across s3://, gs://,
and az:// (or abfss://) storage layouts into a single verified schema offline,
ensuring point-in-time feature extraction behaves identically across clouds.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from late_event_feature_audit.stream import StreamEvent

SUPPORTED_SCHEMES: dict[str, str] = {
    "s3://": "aws_s3",
    "gs://": "gcp_gcs",
    "az://": "azure_blob",
    "abfss://": "azure_adls",
}


class ManifestError(ValueError):
    """Raised when a multi-cloud URI or partition manifest violates contract rules."""


@dataclass(frozen=True)
class CloudPartitionEntry:
    """Single object-store partition file in a multi-cloud event manifest."""

    uri: str
    provider: str
    partition_id: int
    event_count: int
    min_event_ts: float
    max_ingestion_ts: float
    sha256: str


@dataclass(frozen=True)
class CloudManifest:
    """Normalized multi-cloud manifest covering all partitions of an event stream."""

    provider: str
    scheme: str
    bucket_root: str
    total_events: int
    entries: tuple[CloudPartitionEntry, ...]
    manifest_sha256: str


def detect_cloud_provider(uri: str) -> str:
    """Identify the cloud storage provider from a URI scheme."""
    for prefix, provider in SUPPORTED_SCHEMES.items():
        if uri.startswith(prefix):
            return provider
    raise ManifestError(
        f"Unsupported cloud storage URI '{uri}'. Expected one of {sorted(SUPPORTED_SCHEMES)}"
    )


def _canonical_events_digest(events: list[StreamEvent]) -> str:
    payload = [
        [
            e.event_id,
            e.entity_id,
            e.partition,
            e.kafka_offset,
            e.event_ts_min,
            e.ingestion_ts_min,
            e.event_type,
            e.amount,
            e.is_risk_signal,
        ]
        for e in sorted(events, key=lambda item: (item.partition, item.kafka_offset))
    ]
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_cloud_manifest(
    events: tuple[StreamEvent, ...] | list[StreamEvent],
    bucket_root: str,
    date_partition: str = "2026-06-19",
) -> CloudManifest:
    """Build a normalized multi-cloud manifest from partitioned stream events."""
    provider = detect_cloud_provider(bucket_root)
    scheme = next(prefix for prefix in SUPPORTED_SCHEMES if bucket_root.startswith(prefix))
    clean_root = bucket_root.rstrip("/")

    by_part: dict[int, list[StreamEvent]] = {}
    for ev in events:
        by_part.setdefault(ev.partition, []).append(ev)

    if not by_part:
        raise ManifestError("Cannot build a cloud manifest from an empty event stream")

    entries: list[CloudPartitionEntry] = []
    for part_id in sorted(by_part):
        part_events = by_part[part_id]
        uri = f"{clean_root}/dt={date_partition}/partition={part_id}/part-{part_id:04d}.jsonl"
        entries.append(
            CloudPartitionEntry(
                uri=uri,
                provider=provider,
                partition_id=part_id,
                event_count=len(part_events),
                min_event_ts=min(e.event_ts_min for e in part_events),
                max_ingestion_ts=max(e.ingestion_ts_min for e in part_events),
                sha256=_canonical_events_digest(part_events),
            )
        )

    manifest_digest = hashlib.sha256(
        json.dumps([e.sha256 for e in entries], separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return CloudManifest(
        provider=provider,
        scheme=scheme,
        bucket_root=clean_root,
        total_events=sum(e.event_count for e in entries),
        entries=tuple(entries),
        manifest_sha256=manifest_digest,
    )
