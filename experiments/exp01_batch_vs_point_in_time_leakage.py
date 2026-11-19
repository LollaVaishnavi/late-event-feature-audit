"""Experiment 01: Batch Snapshot vs Strict Point-in-Time Leakage Audit."""

from __future__ import annotations

import json
from pathlib import Path

from late_event_feature_audit.config import LeakagePolicy, StreamConfig
from late_event_feature_audit.features import extract_batch_and_pit_features
from late_event_feature_audit.model import evaluate_leakage_gap
from late_event_feature_audit.stream import generate_stream_corpus

OUT_PATH = Path("docs/experiments/exp01_batch_vs_point_in_time_leakage.json")


def run() -> dict:
    cfg = StreamConfig()
    policy = LeakagePolicy()
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
        "experiment": "exp01_batch_vs_point_in_time_leakage",
        "event_count": len(corpus.events),
        "prediction_count": len(corpus.predictions),
        "p50_lag_minutes": corpus.p50_lag_minutes,
        "p95_lag_minutes": corpus.p95_lag_minutes,
        "p99_lag_minutes": corpus.p99_lag_minutes,
        "batch_offline_auc": res.batch_offline_auc,
        "batch_model_on_online_auc": res.batch_model_on_online_auc,
        "pit_honest_auc": res.pit_honest_auc,
        "auc_inflation_points": res.auc_inflation_points,
        "corrupted_row_count": res.corrupted_row_count,
        "corrupted_row_pct": res.corrupted_row_pct,
        "risk_alert_leak_rows": res.risk_alert_leak_rows,
        "gate_passed": res.gate_passed,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    d = run()
    print("=== Experiment 01: Batch Snapshot vs Point-in-Time Leakage ===")
    print(
        f"Events={d['event_count']} | Predictions={d['prediction_count']} | "
        f"p95 lag={d['p95_lag_minutes']:.1f}m"
    )
    print(
        f"Batch Offline AUC={d['batch_offline_auc']:.4f} -> "
        f"Online Serving AUC={d['batch_model_on_online_auc']:.4f} "
        f"(+{d['auc_inflation_points']:.1f} AUC points inflation)"
    )
    print(
        f"Corrupted rows={d['corrupted_row_count']}/{d['prediction_count']} "
        f"({d['corrupted_row_pct']:.1f}% of training rows; "
        f"{d['risk_alert_leak_rows']} leaked risk alerts)"
    )
