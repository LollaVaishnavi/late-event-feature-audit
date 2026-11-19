"""Self-contained HTML audit report renderer for late-event-feature-audit."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

REPORT_CSS = """
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background-color: #ffffff;
    color: #111827;
    margin: 0;
    padding: 32px;
    line-height: 1.5;
}
.report-shell {
    max-width: 920px;
    margin: 0 auto;
}
.header-block {
    border-bottom: 2px solid #e5e7eb;
    padding-bottom: 16px;
    margin-bottom: 24px;
}
.report-title {
    font-size: 26px;
    margin: 0 0 8px 0;
    color: #0f172a;
}
.report-subtitle {
    font-size: 14px;
    color: #475569;
    margin: 0 0 12px 0;
}
.badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 6px;
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 0.02em;
}
.badge-fail {
    background-color: #991b1b;
    color: #ffffff;
}
.badge-pass {
    background-color: #166534;
    color: #ffffff;
}
.section-heading {
    font-size: 19px;
    color: #0f172a;
    margin-top: 26px;
    margin-bottom: 10px;
    border-left: 4px solid #1d4ed8;
    padding-left: 10px;
}
.kpi-grid {
    display: flex;
    gap: 14px;
    margin: 14px 0 20px 0;
}
.kpi-card {
    flex: 1;
    background-color: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 12px 14px;
}
.kpi-label {
    font-size: 12px;
    color: #475569;
    text-transform: uppercase;
}
.kpi-value {
    font-size: 21px;
    font-weight: 700;
    color: #0f172a;
    margin-top: 4px;
}
.data-table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0 20px 0;
    font-size: 13px;
}
.data-table th {
    background-color: #f1f5f9;
    color: #0f172a;
    text-align: left;
    padding: 8px 10px;
    border: 1px solid #cbd5e1;
}
.data-table td {
    padding: 8px 10px;
    border: 1px solid #cbd5e1;
}
.num-cell {
    text-align: right;
    font-variant-numeric: tabular-nums;
}
.alert-cell {
    color: #991b1b;
    font-weight: 700;
}
.ok-cell {
    color: #166534;
    font-weight: 700;
}
.footnote {
    font-size: 12px;
    color: #475569;
    margin-top: 24px;
    border-top: 1px solid #e5e7eb;
    padding-top: 10px;
}
"""


def render_html_report(
    exp01: dict[str, Any],
    exp02: dict[str, Any],
    exp03: dict[str, Any],
    exp04: dict[str, Any],
) -> str:
    """Render all four streaming leakage experiment summaries into standalone HTML."""
    passed = bool(exp01.get("gate_passed", False))
    badge_cls = "badge badge-pass" if passed else "badge badge-fail"
    badge_txt = (
        "GATE PASSED: POINT-IN-TIME PARITY VERIFIED"
        if passed
        else "GATE REJECTED: POINT-IN-TIME LEAKAGE DETECTED"
    )

    lag_rows = []
    for r in exp02["lag_regimes"]:
        cls = "alert-cell" if r["auc_inflation_points"] >= 2.0 else "ok-cell"
        lag_rows.append(
            "<tr>"
            f"<td>{html.escape(r['regime'])}</td>"
            f"<td class=\"num-cell\">{r['p95_lag_minutes']:.1f} min</td>"
            f"<td class=\"num-cell\">{r['corrupted_row_pct']:.1f}%</td>"
            f"<td class=\"num-cell\">{r['batch_offline_auc']:.4f}</td>"
            f"<td class=\"num-cell\">{r['batch_model_on_online_auc']:.4f}</td>"
            f"<td class=\"num-cell {cls}\">+{r['auc_inflation_points']:.1f} pts</td>"
            "</tr>"
        )

    wm_rows = []
    for r in exp03["watermarks"]:
        cls = "alert-cell" if r["dropped_late_share_pct"] >= 30.0 else "ok-cell"
        wm_rows.append(
            "<tr>"
            f"<td class=\"num-cell\">{r['watermark_minutes']:.0f} min</td>"
            f"<td class=\"num-cell\">{r['dropped_events_count']}</td>"
            f"<td class=\"num-cell {cls}\">{r['dropped_late_share_pct']:.1f}%</td>"
            f"<td class=\"num-cell\">{r['online_auc']:.4f}</td>"
            "</tr>"
        )

    cloud_rows = []
    for r in exp04["providers"]:
        cloud_rows.append(
            "<tr>"
            f"<td>{html.escape(r['provider'])}</td>"
            f"<td>{html.escape(r['bucket_root'])}</td>"
            f"<td class=\"num-cell\">{r['total_events']}</td>"
            f"<td>{html.escape(r['manifest_sha256'][:16])}...</td>"
            f"<td class=\"ok-cell\">{'MATCH' if r['feature_parity'] else 'DRIFT'}</td>"
            "</tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>late-event-feature-audit Report</title>
<style>{REPORT_CSS}</style>
</head>
<body>
<div class="report-shell">
  <div class="header-block">
    <h1 class="report-title">late-event-feature-audit: Point-in-Time Leakage Report</h1>
    <p class="report-subtitle">Offline Kafka/Spark Streaming replay across {exp01['event_count']} streaming events and {exp01['prediction_count']} decision points.</p>
    <span class="{badge_cls}">{badge_txt}</span>
  </div>

  <h2 class="section-heading" id="leakage-summary">1. Batch Snapshot vs Point-in-Time Leakage</h2>
  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-label">Phantom Offline AUC Inflation</div>
      <div class="kpi-value">{exp01['auc_inflation_points']:.1f} AUC points</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Corrupted Training Rows</div>
      <div class="kpi-value">{exp01['corrupted_row_pct']:.1f}% of training rows</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">15m Watermark Eviction Loss</div>
      <div class="kpi-value">{exp03['wm_15m_dropped_late_pct']:.1f}% of late events dropped</div>
    </div>
  </div>

  <h2 class="section-heading" id="lag-sweep">2. Arrival Lag Tail vs Offline AUC Inflation</h2>
  <table class="data-table">
    <thead>
      <tr>
        <th>Lag Regime</th>
        <th class="num-cell">Target p95 Lag</th>
        <th class="num-cell">Corrupted Rows</th>
        <th class="num-cell">Batch Offline AUC</th>
        <th class="num-cell">Online Serving AUC</th>
        <th class="num-cell">Phantom Inflation</th>
      </tr>
    </thead>
    <tbody>
      {''.join(lag_rows)}
    </tbody>
  </table>

  <h2 class="section-heading" id="watermark-tradeoff">3. Spark Streaming Watermark Eviction Trade-Off</h2>
  <table class="data-table">
    <thead>
      <tr>
        <th class="num-cell">Watermark</th>
        <th class="num-cell">Dropped Events</th>
        <th class="num-cell">Late Events Dropped</th>
        <th class="num-cell">Resulting Online AUC</th>
      </tr>
    </thead>
    <tbody>
      {''.join(wm_rows)}
    </tbody>
  </table>

  <h2 class="section-heading" id="multicloud-parity">4. Multi-Cloud Manifest Parity (S3, GCS, Azure Blob)</h2>
  <table class="data-table">
    <thead>
      <tr>
        <th>Cloud Provider</th>
        <th>Bucket / Container Root</th>
        <th class="num-cell">Events</th>
        <th>Manifest SHA-256</th>
        <th>Feature Parity</th>
      </tr>
    </thead>
    <tbody>
      {''.join(cloud_rows)}
    </tbody>
  </table>

  <p class="footnote">Generated offline by late-event-feature-audit. Every figure traces to docs/experiments/*.json and docs/metrics.json.</p>
</div>
</body>
</html>
"""


def write_html_report(
    output_path: str | Path,
    exp01: dict[str, Any],
    exp02: dict[str, Any],
    exp03: dict[str, Any],
    exp04: dict[str, Any],
) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_html_report(exp01, exp02, exp03, exp04), encoding="utf-8")
    return target
