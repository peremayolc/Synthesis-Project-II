import json
from collections import Counter

from paths import OUTPUTS_DIR

path = OUTPUTS_DIR / "unitron_verification.jsonl"

total = 0
flagged = 0
accepted = 0
issue_counter = Counter()

with path.open(encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        total += 1
        if row["needs_human_review"]:
            flagged += 1
        else:
            accepted += 1
        for issue in row["issues"]:
            issue_counter[issue["type"]] += 1

print("===== UNITRON EVALUATION SUMMARY =====")
print("Total translations:", total)
print("Accepted / warned:", accepted)
print("Flagged for human review:", flagged)
if total:
    print("Human-review rate:", round(flagged / total, 3))
print("\nIssue counts:")
for issue_type, count in issue_counter.most_common():
    print(f"- {issue_type}: {count}")
