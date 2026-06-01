import json
from pathlib import Path
from html import escape

input_path = Path("../outputs/pipeline_results.jsonl")
output_path = Path("../outputs/demo_report.html")

rows = []

with open(input_path, encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)

        status_class = "review" if data["needs_human_review"] else "accepted"
        status_text = "Human review" if data["needs_human_review"] else "Accepted"

        issues_html = ""
        if data["issues"]:
            for issue in data["issues"]:
                issues_html += f"""
                <li>
                    <b>{escape(issue["type"])}</b>
                    <span class="severity {escape(issue["severity"])}">{escape(issue["severity"])}</span>
                    <br>
                    {escape(issue["explanation"])}
                </li>
                """
        else:
            issues_html = "<li>No issues detected</li>"

        rows.append(f"""
        <div class="card {status_class}">
            <div class="topline">
                <span class="badge {status_class}">{status_text}</span>
                <span class="small">Issues: {len(data["issues"])}</span>
            </div>

            <div class="grid">
                <div>
                    <h3>English Source</h3>
                    <p>{escape(data["source_en"])}</p>
                </div>

                <div>
                    <h3>Machine Translation</h3>
                    <p>{escape(data["mt_es"])}</p>
                </div>

                <div>
                    <h3>Corrected Spanish</h3>
                    <p>{escape(data["corrected_es"])}</p>
                </div>
            </div>

            <h3>Detected Issues</h3>
            <ul>{issues_html}</ul>
        </div>
        """)

html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Translation QA Demo Report</title>
<style>
body {{
    font-family: Arial, sans-serif;
    background: #f4f6f8;
    margin: 0;
    padding: 32px;
    color: #1f2937;
}}

h1 {{
    margin-bottom: 4px;
}}

.subtitle {{
    color: #6b7280;
    margin-bottom: 28px;
}}

.card {{
    background: white;
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 22px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    border-left: 8px solid #9ca3af;
}}

.card.accepted {{
    border-left-color: #16a34a;
}}

.card.review {{
    border-left-color: #dc2626;
}}

.topline {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 18px;
}}

.badge {{
    padding: 7px 12px;
    border-radius: 999px;
    color: white;
    font-weight: bold;
    font-size: 14px;
}}

.badge.accepted {{
    background: #16a34a;
}}

.badge.review {{
    background: #dc2626;
}}

.small {{
    color: #6b7280;
    font-size: 14px;
}}

.grid {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 18px;
}}

.grid div {{
    background: #f9fafb;
    border-radius: 10px;
    padding: 14px;
}}

h3 {{
    margin-top: 0;
    font-size: 15px;
    color: #374151;
}}

p {{
    line-height: 1.45;
}}

li {{
    margin-bottom: 10px;
}}

.severity {{
    font-size: 12px;
    padding: 3px 7px;
    border-radius: 6px;
    margin-left: 8px;
    color: white;
}}

.severity.low {{
    background: #2563eb;
}}

.severity.medium {{
    background: #f59e0b;
}}

.severity.high {{
    background: #dc2626;
}}
</style>
</head>

<body>
<h1>Translation QA Demo Report</h1>
<div class="subtitle">
Heuristics + terminology correction + human-review routing
</div>

{''.join(rows[:100])}

</body>
</html>
"""

output_path.write_text(html, encoding="utf-8")

print("Saved HTML report to:", output_path)