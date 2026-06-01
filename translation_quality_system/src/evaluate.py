import json

path = "../outputs/pipeline_results.jsonl"

total = 0
flagged = 0
auto_fixed = 0

with open(path, encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)

        total += 1

        if data["needs_human_review"]:
            flagged += 1
        else:
            auto_fixed += 1

print("Total:", total)
print("Flagged for review:", flagged)
print("Auto-fixed:", auto_fixed)
print("Auto-fix rate:", auto_fixed / total)