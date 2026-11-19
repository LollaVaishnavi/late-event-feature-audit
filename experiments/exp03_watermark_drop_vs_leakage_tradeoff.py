"""Experiment 03: Spark Streaming Watermark Eviction vs Online Model Quality."""

from __future__ import annotations

import json
from pathlib import Path

from late_event_feature_audit.config import StreamConfig
from late_event_feature_audit.features import extract_batch_and_pit_features
from late_event_feature_audit.model import LogisticRiskModel, roc_auc_score
from late_event_feature_audit.stream import apply_spark_watermark, generate_stream_corpus

OUT_PATH = Path("docs/experiments/exp03_watermark_drop_vs_leakage_tradeoff.json")
WATERMARKS = (5.0, 15.0, 60.0, 360.0)


def run() -> dict:
    corpus = generate_stream_corpus(StreamConfig())
    rows = []
    for wm_min in WATERMARKS:
        wm = apply_spark_watermark(corpus.events, watermark_minutes=wm_min)
        mats = extract_batch_and_pit_features(wm.accepted_events, corpus.predictions)
        split_idx = int(mats.labels.size * 0.65)
        model = LogisticRiskModel(l2_reg=1.0).fit(mats.X_pit[:split_idx], mats.labels[:split_idx])
        online_auc = roc_auc_score(
            mats.labels[split_idx:], model.predict_proba(mats.X_pit[split_idx:])
        )
        rows.append(
            {
                "watermark_minutes": wm_min,
                "dropped_events_count": wm.dropped_events_count,
                "late_events_total": wm.late_events_total,
                "dropped_late_share_pct": wm.dropped_late_share_pct,
                "online_auc": round(online_auc, 4),
            }
        )

    wm15 = next(r for r in rows if r["watermark_minutes"] == 15.0)
    payload = {
        "experiment": "exp03_watermark_drop_vs_leakage_tradeoff",
        "wm_15m_dropped_late_pct": wm15["dropped_late_share_pct"],
        "wm_15m_dropped_count": wm15["dropped_events_count"],
        "watermarks": rows,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    d = run()
    print("=== Experiment 03: Spark Streaming Watermark Eviction Trade-Off ===")
    print(f"{'Watermark':>10} {'Dropped':>9} {'Late Dropped %':>15} {'Online AUC':>11}")
    for r in d["watermarks"]:
        print(
            f"{r['watermark_minutes']:>8.0f}m {r['dropped_events_count']:>9} "
            f"{r['dropped_late_share_pct']:>14.1f}% {r['online_auc']:>11.4f}"
        )
