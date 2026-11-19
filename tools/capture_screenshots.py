"""Build HTML report, verify badge contrast, and capture PNG screenshots."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from late_event_feature_audit.report import REPORT_CSS, write_html_report  # noqa: E402

EXP_DIR = ROOT / "docs" / "experiments"
REPORT_HTML = ROOT / "reports" / "audit_report.html"
REPORT_PDF = ROOT / "reports" / "audit_report.pdf"
EVIDENCE_DIR = ROOT / "docs" / "evidence"


def _hex_luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))

    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def _contrast_ratio(fg: str, bg: str) -> float:
    l1, l2 = _hex_luminance(fg), _hex_luminance(bg)
    hi, lo = max(l1, l2), min(l1, l2)
    return round((hi + 0.05) / (lo + 0.05), 2)


def _verify_badge_and_headings(html_text: str) -> dict:
    for hid in ("leakage-summary", "lag-sweep", "watermark-tradeoff", "multicloud-parity"):
        if f'id="{hid}"' not in html_text:
            raise RuntimeError(f"Required report heading id='{hid}' missing")

    fail_match = re.search(
        r"\.badge-fail\s*\{[^}]*background-color:\s*(#[0-9a-fA-F]{6});"
        r"[^}]*color:\s*(#[0-9a-fA-F]{6});",
        REPORT_CSS,
    )
    if not fail_match:
        raise RuntimeError("Could not parse .badge-fail colors from REPORT_CSS")
    bg, fg = fail_match.group(1), fail_match.group(2)
    ratio = _contrast_ratio(fg, bg)
    if ratio < 4.5:
        raise RuntimeError(f"Badge contrast ratio {ratio} below WCAG AA 4.5")
    return {"badge_class": "badge-fail", "fg": fg, "bg": bg, "contrast_ratio": ratio}


def main() -> int:
    e1 = json.loads(
        (EXP_DIR / "exp01_batch_vs_point_in_time_leakage.json").read_text(encoding="utf-8")
    )
    e2 = json.loads(
        (EXP_DIR / "exp02_lateness_tail_vs_auc_inflation.json").read_text(encoding="utf-8")
    )
    e3 = json.loads(
        (EXP_DIR / "exp03_watermark_drop_vs_leakage_tradeoff.json").read_text(encoding="utf-8")
    )
    e4 = json.loads(
        (EXP_DIR / "exp04_multicloud_manifest_parity.json").read_text(encoding="utf-8")
    )

    write_html_report(REPORT_HTML, e1, e2, e3, e4)
    badge_info = _verify_badge_and_headings(REPORT_HTML.read_text(encoding="utf-8"))

    import weasyprint  # type: ignore

    weasyprint.HTML(filename=str(REPORT_HTML)).write_pdf(str(REPORT_PDF))
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    prefix = EVIDENCE_DIR / "_page"
    subprocess.run(["pdftoppm", "-png", "-r", "150", str(REPORT_PDF), str(prefix)], check=True)

    pages = sorted(EVIDENCE_DIR.glob("_page-*.png"))
    if not pages:
        raise RuntimeError("pdftoppm produced no PNG pages")
    summary_png = EVIDENCE_DIR / "report_summary.png"
    pages[0].replace(summary_png)
    captured = [
        {
            "image": str(summary_png.relative_to(ROOT)),
            "report": str(REPORT_HTML.relative_to(ROOT)),
            "headings": ["leakage-summary", "lag-sweep"],
        }
    ]
    if len(pages) > 1:
        extra_png = EVIDENCE_DIR / "report_multicloud.png"
        pages[1].replace(extra_png)
        captured.append(
            {
                "image": str(extra_png.relative_to(ROOT)),
                "report": str(REPORT_HTML.relative_to(ROOT)),
                "headings": ["watermark-tradeoff", "multicloud-parity"],
            }
        )
    for extra in pages[2:]:
        extra.unlink()

    (EVIDENCE_DIR / "manifest.json").write_text(
        json.dumps(
            {
                "report_html": str(REPORT_HTML.relative_to(ROOT)),
                "badge_verification": badge_info,
                "screenshots": captured,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    ratio = badge_info["contrast_ratio"]
    print(f"capture_screenshots: captured {len(captured)} PNG(s), badge contrast={ratio}:1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
