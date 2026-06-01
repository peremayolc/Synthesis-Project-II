import argparse
import csv
import json
from collections import Counter, defaultdict
from html import escape
from pathlib import Path
from typing import Dict, Iterable, List

from paths import OUTPUTS_DIR, UNITRON_PROCESSED_DIR, ensure_dirs
from pipeline import TranslationPipeline
from translation_file_parser import write_unitron_jsonl


def load_jsonl(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "document", "segment_id", "variant", "decision", "needs_human_review",
        "issue_count", "high_issue_count", "issue_types", "source_en", "reference_es",
        "mt_es", "corrected_es", "correction_explanation",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            issues = row.get("issues") or []
            writer.writerow({
                "document": row.get("document"),
                "segment_id": row.get("segment_id"),
                "variant": row.get("variant"),
                "decision": row.get("decision"),
                "needs_human_review": row.get("needs_human_review"),
                "issue_count": len(issues),
                "high_issue_count": sum(1 for i in issues if i.get("severity") == "high"),
                "issue_types": "; ".join(i.get("type", "") for i in issues),
                "source_en": row.get("source_en"),
                "reference_es": row.get("reference_es"),
                "mt_es": row.get("mt_es"),
                "corrected_es": row.get("corrected_es"),
                "correction_explanation": json.dumps(row.get("correction_explanation"), ensure_ascii=False),
            })


def summarize(rows: List[Dict[str, object]]) -> Dict[str, object]:
    by_variant = defaultdict(lambda: Counter(total=0, human_review=0, accepted=0, auto_corrected_or_warned=0))
    issue_counts = Counter()
    doc_counts = defaultdict(lambda: Counter(total=0, human_review=0))

    for row in rows:
        variant = str(row.get("variant"))
        doc = str(row.get("document"))
        decision = str(row.get("decision"))
        by_variant[variant]["total"] += 1
        by_variant[variant][decision] += 1
        if row.get("needs_human_review"):
            doc_counts[doc]["human_review"] += 1
        doc_counts[doc]["total"] += 1
        for issue in row.get("issues") or []:
            issue_counts[str(issue.get("type"))] += 1

    return {
        "total_checked_translations": len(rows),
        "by_variant": {k: dict(v) for k, v in sorted(by_variant.items())},
        "by_document": {k: dict(v) for k, v in sorted(doc_counts.items())},
        "issue_counts": dict(issue_counts.most_common()),
    }


def write_html(path: Path, rows: List[Dict[str, object]], summary: Dict[str, object], limit: int = 400) -> None:
    cards = []
    for row in rows[:limit]:
        issues = row.get("issues") or []
        issue_html = "".join(
            f"<li><b>{escape(str(i.get('type')))}</b> "
            f"<span class='sev {escape(str(i.get('severity')))}'>{escape(str(i.get('severity')))}</span><br>"
            f"{escape(str(i.get('explanation')))}</li>"
            for i in issues
        ) or "<li>No issues detected</li>"
        cls = "review" if row.get("needs_human_review") else "ok"
        cards.append(f"""
        <section class="card {cls}">
          <div class="meta">
            <b>{escape(str(row.get('document')))} #{escape(str(row.get('segment_id')))} / {escape(str(row.get('variant')))}</b>
            <span>{escape(str(row.get('decision')))}</span>
          </div>
          <div class="grid">
            <div><h3>EN source</h3><p>{escape(str(row.get('source_en')))}</p></div>
            <div><h3>Human reference ES</h3><p>{escape(str(row.get('reference_es') or ''))}</p></div>
            <div><h3>ML translation ES</h3><p>{escape(str(row.get('mt_es')))}</p></div>
            <div><h3>Rule-corrected ES</h3><p>{escape(str(row.get('corrected_es')))}</p></div>
          </div>
          <h3>Detected issues</h3>
          <ul>{issue_html}</ul>
        </section>
        """)

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Unitron Translation QA Report</title>
<style>
body {{ font-family: Arial, sans-serif; background:#f5f7fb; color:#172033; padding:32px; }}
h1 {{ margin-bottom:0; }}
pre {{ background:#111827; color:#f9fafb; padding:16px; border-radius:12px; overflow:auto; }}
.card {{ background:white; border-radius:14px; padding:20px; margin:18px 0; box-shadow:0 4px 14px rgba(0,0,0,.07); border-left:8px solid #16a34a; }}
.card.review {{ border-left-color:#dc2626; }}
.meta {{ display:flex; justify-content:space-between; gap:16px; margin-bottom:14px; }}
.grid {{ display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:14px; }}
.grid div {{ background:#f9fafb; border-radius:10px; padding:12px; }}
h3 {{ margin:0 0 8px; font-size:15px; }}
p {{ line-height:1.45; }}
.sev {{ color:white; border-radius:999px; padding:2px 8px; font-size:12px; }}
.sev.high {{ background:#dc2626; }} .sev.medium {{ background:#d97706; }} .sev.low {{ background:#2563eb; }}
</style>
</head>
<body>
<h1>Unitron Translation QA Report</h1>
<p>Generated by the local deterministic verification pipeline.</p>
<pre>{escape(json.dumps(summary, indent=2, ensure_ascii=False))}</pre>
{''.join(cards)}
</body>
</html>"""
    path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Unitron ML translations with the QA pipeline.")
    parser.add_argument("--input", type=Path, default=None, help="Parsed Unitron JSONL input. If missing, raw txt files are parsed first.")
    parser.add_argument("--output-prefix", default="unitron_verification", help="Output filename prefix inside outputs/.")
    args = parser.parse_args()

    ensure_dirs()
    input_path = args.input or (UNITRON_PROCESSED_DIR / "unitron_translation_pairs.jsonl")
    if not input_path.exists():
        input_path = write_unitron_jsonl(input_path)

    pipeline = TranslationPipeline()
    results: List[Dict[str, object]] = []
    for rec in load_jsonl(input_path):
        for variant in ("base", "ft"):
            mt = rec.get(variant) or ""
            if not mt:
                continue
            results.append(
                pipeline.process(
                    source_en=rec.get("en", ""),
                    reference_es=rec.get("ref", ""),
                    mt_es=mt,
                    document=rec.get("document"),
                    segment_id=rec.get("segment_id"),
                    variant=variant.upper(),
                )
            )

    prefix = OUTPUTS_DIR / args.output_prefix
    summary = summarize(results)
    write_jsonl(prefix.with_suffix(".jsonl"), results)
    write_csv(prefix.with_suffix(".csv"), results)
    write_html(prefix.with_suffix(".html"), results, summary)
    (prefix.parent / f"{args.output_prefix}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved: {prefix.with_suffix('.jsonl')}")
    print(f"Saved: {prefix.with_suffix('.csv')}")
    print(f"Saved: {prefix.with_suffix('.html')}")


if __name__ == "__main__":
    main()
