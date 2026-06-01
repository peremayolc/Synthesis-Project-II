import json
from pathlib import Path

from paths import DATA_DIR, OUTPUTS_DIR, ensure_dirs
from pipeline import TranslationPipeline


def main() -> None:
    ensure_dirs()
    pipeline = TranslationPipeline()
    input_path = DATA_DIR / "synthetic" / "synthetic_translation_qa.jsonl"
    output_path = OUTPUTS_DIR / "pipeline_results.jsonl"

    count = 0
    with input_path.open(encoding="utf-8") as f_in, output_path.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            data = json.loads(line)
            result = pipeline.process(
                source_en=data["source_en"],
                mt_es=data["mt_es"],
                reference_es=data.get("reference_es"),
                document="synthetic",
                segment_id=data.get("id"),
                variant="MT",
            )
            result["gold_corrected_es"] = data.get("corrected_es")
            result["gold_error_labels"] = data.get("error_labels")
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            count += 1

    print(f"Processed {count} examples")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
