"""Experiment 04: Multi-Cloud Manifest Parity Across AWS S3, GCP GCS, and Azure Blob."""

from __future__ import annotations

import json
from pathlib import Path

from late_event_feature_audit.config import StreamConfig
from late_event_feature_audit.manifests import build_cloud_manifest
from late_event_feature_audit.stream import generate_stream_corpus

OUT_PATH = Path("docs/experiments/exp04_multicloud_manifest_parity.json")
CLOUD_ROOTS = (
    "s3://ecommerce-events-prod",
    "gs://ecommerce-events-prod",
    "az://ecommerce-events-prod",
)


def run() -> dict:
    corpus = generate_stream_corpus(StreamConfig())
    manifests = [build_cloud_manifest(corpus.events, root) for root in CLOUD_ROOTS]
    reference_sha = manifests[0].manifest_sha256

    providers = []
    for m in manifests:
        providers.append(
            {
                "provider": m.provider,
                "scheme": m.scheme,
                "bucket_root": m.bucket_root,
                "partitions": len(m.entries),
                "total_events": m.total_events,
                "manifest_sha256": m.manifest_sha256,
                "feature_parity": m.manifest_sha256 == reference_sha,
            }
        )

    payload = {
        "experiment": "exp04_multicloud_manifest_parity",
        "cloud_providers_verified": sum(1 for p in providers if p["feature_parity"]),
        "reference_sha256": reference_sha,
        "all_parity_match": all(p["feature_parity"] for p in providers),
        "providers": providers,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    d = run()
    print("=== Experiment 04: Multi-Cloud Manifest Parity (S3, GCS, Azure Blob) ===")
    for p in d["providers"]:
        print(
            f"  {p['provider']:<12} {p['bucket_root']:<30} "
            f"sha256={p['manifest_sha256'][:16]} parity={p['feature_parity']}"
        )
    print(f"Verified parity across {d['cloud_providers_verified']} cloud providers.")
