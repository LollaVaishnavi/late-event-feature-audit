"""Experiment 02: Arrival Lag Tail Regimes vs Phantom Offline AUC Inflation."""

from __future__ import annotations

import json
from pathlib import Path

from late_event_feature_audit.config import StreamConfig
from late_event_feature_audit.features import extract_batch_and_pit_features
from late_event_feature_audit.model import evaluate_leakage_gap
from late_event_feature_audit.stream import generate_stream_corpus

OUT_PATH = Path("docs/experiments/exp02_lateness_tail_vs_auc_inflation.json")
LAG_REGIMES = (
    ("5m_tight", 5.0),
    ("30m_moderate", 30.0),
    ("2h_default", 120.0),
    ("12h_heavy_tail", 720.0),
)


def run() -> dict:
    rows = []
    for label, p95_min in LAG_REGIMES:
        cfg = StreamConfig(p95_lag_minutes=p95_min)
        corpus = generate_stream_corpus(cfg)
        mats = extract_batch_and_pit_features(corpus.events, corpus.predictions)
        res = evaluate_leakage_gap(
            mats.X_batch,
            mats.X_pit,
            mats.labels,
            mats.corrupted_row_count,
            mats.corrupted_row_pct,
            mats.risk_alert_leak_rows,
        )
        rows.append(
            {
                "regime": label,
                "p95_lag_minutes": p95_min,
                "corrupted_row_count": res.corrupted_row_count,
                "corrupted_row_pct": res.corrupted_row_pct,
                "batch_offline_auc": res.batch_offline_auc,
                "batch_model_on_online_auc": res.batch_model_on_online_auc,
                "auc_inflation_points": res.auc_inflation_points,
            }
        )

    heavy = next(r for r in rows if r["regime"] == "12h_heavy_tail")
    payload = {
        "experiment": "exp02_lateness_tail_vs_auc_inflation",
        "heavy_tail_auc_inflation": heavy["auc_inflation_points"],
        "heavy_tail_corrupted_pct": heavy["corrupted_row_pct"],
        "lag_regimes": rows,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    d = run()
    print("=== Experiment 02: Arrival Lag Tail vs Phantom Offline AUC Inflation ===")
    hdr = (
        f"{'Regime':<16} {'p95 Lag':>9} {'Corrupted':>10} "
        f"{'Batch AUC':>10} {'Online AUC':>11} {'Inflation':>10}"
    )
    print(hdr)
    for r in d["lag_regimes"]:
        print(
            f"{r['regime']:<16} {r['p95_lag_minutes']:>7.0f}m {r['corrupted_row_pct']:>9.1f}% "
            f"{r['batch_offline_auc']:>10.4f} {r['batch_model_on_online_auc']:>11.4f} "
            f"{r['auc_inflation_points']:>+9.1f}"
        )
