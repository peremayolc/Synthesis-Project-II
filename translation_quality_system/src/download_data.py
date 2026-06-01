from datasets import load_dataset
import json
from pathlib import Path
from tqdm import tqdm

OUT = Path("data/raw/opus100_en_es.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

def main():
    print("Downloading EN-ES dataset...")

    dataset = load_dataset(
        "Helsinki-NLP/opus-100",
        "en-es",
        split="train[:2000]"
    )

    with OUT.open("w", encoding="utf-8") as f:
        for i, row in enumerate(tqdm(dataset)):
            en = row["translation"]["en"]
            es = row["translation"]["es"]

            record = {
                "id": f"opus_{i:06d}",
                "source_en": en,
                "reference_es": es,
                "domain": "general"
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Saved to:", OUT)

if __name__ == "__main__":
    main()