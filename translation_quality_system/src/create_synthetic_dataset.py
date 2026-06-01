import json
import random
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

RAW_PATH = Path("data/raw/opus100_en_es.jsonl")
GLOSSARY_PATH = Path("data/glossary/technical_glossary.csv")
OUT_PATH = Path("data/synthetic/synthetic_translation_qa.jsonl")

random.seed(42)


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def inject_number_error(text):
    numbers = re.findall(r"\b\d+\b", text)

    if not numbers:
        return text, None

    old = random.choice(numbers)
    new = str(int(old) + random.choice([1, 2, 5, 10]))

    return text.replace(old, new, 1), {
        "type": "number",
        "before": old,
        "after": new
    }


def inject_punctuation_error(text):
    if "¿" in text:
        return text.replace("¿", "", 1), {
            "type": "punctuation",
            "before": "¿",
            "after": ""
        }

    if "¡" in text:
        return text.replace("¡", "", 1), {
            "type": "punctuation",
            "before": "¡",
            "after": ""
        }

    return text + "?", {
        "type": "punctuation",
        "before": "",
        "after": "?"
    }


def inject_terminology_error(text, glossary):
    rows = glossary.sample(frac=1, random_state=random.randint(0, 999999)).to_dict("records")

    for row in rows:
        good = str(row["approved_es"])
        bad = str(row["forbidden_es"])

        if good and bad and good.lower() in text.lower():
            pattern = re.compile(re.escape(good), re.IGNORECASE)
            new_text = pattern.sub(bad, text, count=1)

            return new_text, {
                "type": "terminology",
                "before": good,
                "after": bad
            }

    return text, None


def inject_omission_error(text):
    words = text.split()

    if len(words) < 10:
        return text, None

    start = random.randint(2, max(2, len(words) - 5))
    removed = words[start:start + 3]
    new_words = words[:start] + words[start + 3:]

    return " ".join(new_words), {
        "type": "omission",
        "before": " ".join(removed),
        "after": ""
    }


def inject_placeholder_error(text):
    text_with_placeholder = text + " {user_id}"
    corrupted = text_with_placeholder.replace("{user_id}", "{usuario_id}")

    return corrupted, {
        "type": "placeholder",
        "before": "{user_id}",
        "after": "{usuario_id}"
    }


def main():
    glossary = pd.read_csv(GLOSSARY_PATH)
    records = list(load_jsonl(RAW_PATH))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    error_types = [
        "number",
        "punctuation",
        "terminology",
        "omission",
        "placeholder"
    ]

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for rec in tqdm(records):
            source_en = rec["source_en"]
            reference_es = rec["reference_es"]

            error_type = random.choice(error_types)

            mt_es = reference_es
            error = None

            if error_type == "number":
                mt_es, error = inject_number_error(reference_es)

            elif error_type == "punctuation":
                mt_es, error = inject_punctuation_error(reference_es)

            elif error_type == "terminology":
                mt_es, error = inject_terminology_error(reference_es, glossary)

            elif error_type == "omission":
                mt_es, error = inject_omission_error(reference_es)

            elif error_type == "placeholder":
                mt_es, error = inject_placeholder_error(reference_es)

            if error is None:
                error = {
                    "type": "clean",
                    "before": None,
                    "after": None
                }

            out = {
                "id": rec["id"],
                "domain": rec["domain"],
                "source_en": source_en,
                "reference_es": reference_es,
                "mt_es": mt_es,
                "corrected_es": reference_es,
                "style_guide_id": "technical_es_v1",
                "error_labels": [error["type"]],
                "synthetic_error": error,
                "needs_human_review": error["type"] in ["omission"]
            }

            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print("Saved synthetic dataset to:", OUT_PATH)


if __name__ == "__main__":
    main()