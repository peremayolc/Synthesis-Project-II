import json
from heuristics import run_heuristics

with open("../data/synthetic/synthetic_translation_qa.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f):
        data = json.loads(line)

        issues = run_heuristics(data["source_en"], data["mt_es"])

        print("----")
        print("EN:", data["source_en"])
        print("ES:", data["mt_es"])
        print("Detected:", issues)

        if i == 5:
            break