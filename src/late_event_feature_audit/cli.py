"""Command-line interface for late-event-feature-audit with semantic exit codes.

Exit codes:
    0: Audit passed policy (no point-in-time leakage breach).
    2: Point-in-time leakage or AUC inflation breached policy gate.
    1: Invalid input, unsupported cloud URI, or runtime error.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from late_event_feature_audit.config import LeakagePolicy, StreamConfig, StreamConfigError
from late_event_feature_audit.features import extract_batch_and_pit_features
from late_event_feature_audit.manifests import ManifestError, build_cloud_manifest
from late_event_feature_audit.model import ModelEvalError, evaluate_leakage_gap
from late_event_feature_audit.stream import apply_spark_watermark, generate_stream_corpus


def _cmd_audit(args: argparse.Namespace) -> int:
    cfg = StreamConfig(
        seed=args.seed,
        entity_count=args.entities,
        prediction_count=args.predictions,
        p95_lag_minutes=args.p95_lag,
        late_event_rate=args.late_rate,
    )
    policy = LeakagePolicy.from_json(args.policy) if args.policy else LeakagePolicy()
    corpus = generate_stream_corpus(cfg)
    mats = extract_batch_and_pit_features(corpus.events, corpus.predictions)
    res = evaluate_leakage_gap(
        X_batch=mats.X_batch,
        X_pit=mats.X_pit,
        labels=mats.labels,
        corrupted_row_count=mats.corrupted_row_count,
        corrupted_row_pct=mats.corrupted_row_pct,
        risk_alert_leak_rows=mats.risk_alert_leak_rows,
        max_auc_inflation=policy.max_auc_inflation,
        max_corrupted_row_pct=policy.max_corrupted_row_pct,
    )

    payload = {
        "event_count": len(corpus.events),
        "prediction_count": len(corpus.predictions),
        "p95_lag_minutes": corpus.p95_lag_minutes,
        "batch_offline_auc": res.batch_offline_auc,
        "batch_model_on_online_auc": res.batch_model_on_online_auc,
        "pit_honest_auc": res.pit_honest_auc,
        "auc_inflation_points": res.auc_inflation_points,
        "corrupted_row_count": res.corrupted_row_count,
        "corrupted_row_pct": res.corrupted_row_pct,
        "risk_alert_leak_rows": res.risk_alert_leak_rows,
        "gate_passed": res.gate_passed,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"late-event-feature-audit | events={payload['event_count']} | "
            f"predictions={payload['prediction_count']} | p95_lag={corpus.p95_lag_minutes:.1f}m"
        )
        print(
            f"  batch offline AUC:  {res.batch_offline_auc:.4f} -> "
            f"online serving AUC: {res.batch_model_on_online_auc:.4f} "
            f"(+{res.auc_inflation_points:.1f} AUC points phantom inflation)"
        )
        print(
            f"  corrupted rows:     {res.corrupted_row_count}/{payload['prediction_count']} "
            f"({res.corrupted_row_pct:.1f}% of training rows; "
            f"{res.risk_alert_leak_rows} leaked risk alerts)"
        )
        verdict = "PASS" if res.gate_passed else "REJECT (POINT-IN-TIME LEAKAGE)"
        print(f"  verdict:            {verdict}")
    return 0 if res.gate_passed else 2


def _cmd_watermark(args: argparse.Namespace) -> int:
    cfg = StreamConfig(
        seed=args.seed,
        p95_lag_minutes=args.p95_lag,
        watermark_minutes=args.watermark,
    )
    corpus = generate_stream_corpus(cfg)
    wm = apply_spark_watermark(corpus.events, watermark_minutes=args.watermark)
    payload = {
        "watermark_minutes": wm.watermark_minutes,
        "total_events": wm.total_events,
        "late_events_total": wm.late_events_total,
        "dropped_events_count": wm.dropped_events_count,
        "dropped_late_share_pct": wm.dropped_late_share_pct,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"Spark Watermark Replay (watermark={wm.watermark_minutes:.0f}m): "
            f"dropped {wm.dropped_events_count}/{wm.late_events_total} late events "
            f"({wm.dropped_late_share_pct:.1f}% of late events dropped)"
        )
    return 0


def _cmd_manifest(args: argparse.Namespace) -> int:
    corpus = generate_stream_corpus(StreamConfig(seed=args.seed))
    manifest = build_cloud_manifest(corpus.events, bucket_root=args.bucket_uri)
    payload = {
        "provider": manifest.provider,
        "scheme": manifest.scheme,
        "bucket_root": manifest.bucket_root,
        "partitions": len(manifest.entries),
        "total_events": manifest.total_events,
        "manifest_sha256": manifest.manifest_sha256,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"Cloud Manifest [{manifest.provider}] {manifest.bucket_root} | "
            f"partitions={len(manifest.entries)} | events={manifest.total_events} | "
            f"sha256={manifest.manifest_sha256[:16]}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="late-event-feature-audit",
        description="Audit point-in-time feature leakage from late-arriving streaming events.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_aud = sub.add_parser("audit", help="Compare batch snapshot vs point-in-time serving AUC.")
    p_aud.add_argument("--seed", type=int, default=20260619)
    p_aud.add_argument("--entities", type=int, default=120)
    p_aud.add_argument("--predictions", type=int, default=1200)
    p_aud.add_argument("--p95-lag", type=float, default=120.0)
    p_aud.add_argument("--late-rate", type=float, default=0.28)
    p_aud.add_argument("--policy", type=str, default=None)
    p_aud.add_argument("--json", action="store_true")

    p_wm = sub.add_parser("watermark", help="Simulate Spark Streaming watermark eviction.")
    p_wm.add_argument("--seed", type=int, default=20260619)
    p_wm.add_argument("--p95-lag", type=float, default=120.0)
    p_wm.add_argument("--watermark", type=float, default=15.0)
    p_wm.add_argument("--json", action="store_true")

    p_man = sub.add_parser("manifest", help="Build a normalized S3, GCS, or Azure Blob manifest.")
    p_man.add_argument("--bucket-uri", type=str, required=True)
    p_man.add_argument("--seed", type=int, default=20260619)
    p_man.add_argument("--json", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "audit":
            return _cmd_audit(args)
        if args.command == "watermark":
            return _cmd_watermark(args)
        if args.command == "manifest":
            return _cmd_manifest(args)
        return 1
    except (StreamConfigError, ManifestError, ModelEvalError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
