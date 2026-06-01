import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

from paths import UNITRON_RAW_DIR, UNITRON_PROCESSED_DIR, ensure_dirs

LABELS = ("EN", "REF", "BASE", "FT")


def _clean(value: str) -> str:
    return " ".join((value or "").strip().split())


def parse_translation_file(path: Path, document_name: str | None = None) -> List[Dict[str, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n(?=\[\d+\]\n)", text)
    records: List[Dict[str, str]] = []
    document_name = document_name or path.stem.replace("translations_", "")

    for block in blocks:
        header = re.match(r"\[(\d+)\]\n", block)
        if not header:
            continue

        record: Dict[str, str] = {
            "document": document_name,
            "segment_id": header.group(1),
        }
        for label in LABELS:
            match = re.search(
                rf"(?ms)^  {label}\s+:\s*(.*?)(?=\n  (?:EN|REF|BASE|FT)\s+:|\Z)",
                block,
            )
            record[label.lower()] = _clean(match.group(1)) if match else ""

        if record.get("en"):
            records.append(record)

    return records


def parse_all_unitron_files(raw_dir: Path = UNITRON_RAW_DIR) -> List[Dict[str, str]]:
    all_records: List[Dict[str, str]] = []
    for path in sorted(raw_dir.glob("translations_unitron_*.txt")):
        all_records.extend(parse_translation_file(path))
    return all_records


def write_unitron_jsonl(output_path: Path | None = None) -> Path:
    ensure_dirs()
    output_path = output_path or (UNITRON_PROCESSED_DIR / "unitron_translation_pairs.jsonl")
    records = parse_all_unitron_files()
    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return output_path


if __name__ == "__main__":
    out = write_unitron_jsonl()
    print(f"Wrote parsed Unitron pairs to {out}")
