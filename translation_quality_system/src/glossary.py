import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from paths import GLOSSARY_PATH


@dataclass(frozen=True)
class GlossaryEntry:
    source_term: str
    approved_es: str
    forbidden_es: List[str]
    domain: str = ""


def _split_forbidden(value: str) -> List[str]:
    if not value:
        return []
    value = str(value).strip()
    if not value or value.lower() == "nan":
        return []
    return [part.strip() for part in re.split(r"\s*\|\s*", value) if part.strip()]


def load_glossary(path: Optional[Path] = None) -> List[GlossaryEntry]:
    path = path or GLOSSARY_PATH
    if not path.exists():
        return []

    entries: List[GlossaryEntry] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source = (row.get("source_term") or "").strip()
            approved = (row.get("approved_es") or "").strip()
            if not source or not approved:
                continue
            entries.append(
                GlossaryEntry(
                    source_term=source,
                    approved_es=approved,
                    forbidden_es=_split_forbidden(row.get("forbidden_es") or ""),
                    domain=(row.get("domain") or "").strip(),
                )
            )
    return entries


def contains_phrase(text: str, phrase: str) -> bool:
    if not text or not phrase:
        return False
    # Word boundary for alphanumeric terms, flexible whitespace for phrases.
    escaped = re.escape(phrase).replace(r"\ ", r"\s+")
    if re.search(r"\w", phrase):
        pattern = rf"(?<!\w){escaped}(?!\w)"
    else:
        pattern = escaped
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def replace_phrase(text: str, phrase: str, replacement: str) -> str:
    escaped = re.escape(phrase).replace(r"\ ", r"\s+")
    if re.search(r"\w", phrase):
        pattern = rf"(?<!\w){escaped}(?!\w)"
    else:
        pattern = escaped
    return re.sub(pattern, replacement, text, flags=re.IGNORECASE)
