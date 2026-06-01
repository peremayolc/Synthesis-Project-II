import json
import pandas as pd

input_path = "../outputs/pipeline_results.jsonl"
output_path = "../outputs/demo_report.csv"

rows = []

with open(input_path, encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)

        rows.append({
            "id": data.get("id"),
            "source_en": data["source_en"],
            "machine_translation_es": data["mt_es"],
            "corrected_es": data["corrected_es"],
            "needs_human_review": data["needs_human_review"],
            "issues": json.dumps(data["issues"], ensure_ascii=False),
            "correction_explanation": json.dumps(data["correction_explanation"], ensure_ascii=False),
            "reference_es": data.get("reference_es"),
            "gold_error_labels": data.get("gold_error_labels"),
        })

df = pd.DataFrame(rows)
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print("Saved CSV report to:", output_path)