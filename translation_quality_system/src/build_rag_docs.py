import json
from pathlib import Path

from glossary import load_glossary
from paths import DATA_DIR, GLOSSARY_PATH, STYLE_GUIDE_PATH

OUT_PATH = DATA_DIR / "rag_docs" / "rag_documents.jsonl"


def main() -> None:
    docs = []
    for row in load_glossary(GLOSSARY_PATH):
        forbidden = ", ".join(row.forbidden_es) if row.forbidden_es else "None"
        text = (
            f"Terminology rule. English: {row.source_term}. "
            f"Approved Spanish: {row.approved_es}. "
            f"Forbidden or non-preferred Spanish: {forbidden}. "
            f"Domain: {row.domain}."
        )
        docs.append({"id": f"glossary_{row.source_term}", "type": "glossary", "text": text})

    if STYLE_GUIDE_PATH.exists():
        docs.append({"id": "style_guide", "type": "style_guide", "text": STYLE_GUIDE_PATH.read_text(encoding="utf-8")})

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print("Saved RAG docs to:", OUT_PATH)


if __name__ == "__main__":
    main()
