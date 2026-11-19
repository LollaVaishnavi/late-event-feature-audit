"""Property tests for CLI exit codes, multi-cloud manifest commands, and HTML report CSS."""

from __future__ import annotations

import io
import json
import re
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from late_event_feature_audit.cli import main
from late_event_feature_audit.report import REPORT_CSS, render_html_report, write_html_report


def _sample_payloads() -> tuple[dict, dict, dict, dict]:
    e1 = {
        "event_count": 4800,
        "prediction_count": 1200,
        "auc_inflation_points": 4.7,
        "corrupted_row_pct": 7.4,
        "gate_passed": False,
    }
    e2 = {
        "lag_regimes": [
            {
                "regime": "p95=120m",
                "p95_lag_minutes": 120.0,
                "corrupted_row_pct": 7.4,
                "batch_offline_auc": 0.8076,
                "batch_model_on_online_auc": 0.7609,
                "auc_inflation_points": 4.7,
            }
        ]
    }
    e3 = {
        "wm_15m_dropped_late_pct": 59.4,
        "watermarks": [
            {
                "watermark_minutes": 15.0,
                "dropped_events_count": 804,
                "dropped_late_share_pct": 59.4,
                "online_auc": 0.75,
            }
        ],
    }
    e4 = {
        "providers": [
            {
                "provider": "aws_s3",
                "bucket_root": "s3://ecommerce-events-prod",
                "total_events": 4800,
                "manifest_sha256": "17aa35aa3de9406e1234",
                "feature_parity": True,
            }
        ]
    }
    return e1, e2, e3, e4


class TestCLIAndReport(unittest.TestCase):
    def test_cli_audit_returns_exit_2_on_point_in_time_leakage_and_0_when_late_rate_zero(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            code_leak = main(["audit", "--json"])
        self.assertEqual(code_leak, 2)
        payload = json.loads(buf.getvalue())
        self.assertFalse(payload["gate_passed"])

        buf_ok = io.StringIO()
        with redirect_stdout(buf_ok):
            code_ok = main(["audit", "--late-rate", "0.0"])
        self.assertEqual(code_ok, 0)
        self.assertIn("PASS", buf_ok.getvalue())

    def test_cli_audit_loads_custom_policy_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "policy.json"
            p.write_text(
                json.dumps({
                    "max_auc_inflation": 0.50,
                    "max_corrupted_row_pct": 100.0,
                    "max_watermark_drop_pct": 100.0,
                }),
                encoding="utf-8",
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = main(["audit", "--policy", str(p)])
        self.assertEqual(code, 0)

    def test_cli_watermark_and_manifest_subcommands_emit_text_and_json(self) -> None:
        for cmd in [
            ["watermark", "--watermark", "15", "--json"],
            ["watermark", "--watermark", "15"],
            ["manifest", "--bucket-uri", "s3://prod-events", "--json"],
            ["manifest", "--bucket-uri", "gs://prod-events"],
            ["manifest", "--bucket-uri", "az://prod-events"],
        ]:
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = main(cmd)
            self.assertEqual(code, 0)
            self.assertGreater(len(buf.getvalue()), 10)

    def test_cli_returns_exit_1_on_invalid_cloud_uri_or_bad_args(self) -> None:
        err = io.StringIO()
        with redirect_stderr(err):
            code = main(["manifest", "--bucket-uri", "ftp://bad-bucket"])
        self.assertEqual(code, 1)
        self.assertIn("error:", err.getvalue())

        err2 = io.StringIO()
        with redirect_stderr(err2):
            code2 = main(["audit", "--p95-lag", "-10"])
        self.assertEqual(code2, 1)

    def test_every_css_class_in_html_report_has_a_defined_stylesheet_rule(self) -> None:
        e1, e2, e3, e4 = _sample_payloads()
        for passed in (False, True):
            markup = render_html_report({**e1, "gate_passed": passed}, e2, e3, e4)
            used: set[str] = set()
            for m in re.findall(r'class="([^"]+)"', markup):
                used.update(m.split())
            defined = set(re.findall(r"\.([a-zA-Z0-9_-]+)\s*\{", REPORT_CSS))
            self.assertEqual(used - defined, set())

    def test_write_html_report_creates_nested_file_with_all_four_headings(self) -> None:
        e1, e2, e3, e4 = _sample_payloads()
        with tempfile.TemporaryDirectory() as tmp:
            out = write_html_report(Path(tmp) / "sub" / "report.html", e1, e2, e3, e4)
            text = out.read_text(encoding="utf-8")
        for hid in ("leakage-summary", "lag-sweep", "watermark-tradeoff", "multicloud-parity"):
            self.assertIn(f'id="{hid}"', text)


if __name__ == "__main__":
    unittest.main()
