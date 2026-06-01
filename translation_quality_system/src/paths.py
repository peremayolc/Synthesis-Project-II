from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
GLOSSARY_PATH = DATA_DIR / "glossary" / "technical_glossary.csv"
STYLE_GUIDE_PATH = DATA_DIR / "style_guides" / "technical_es_v1.md"
UNITRON_RAW_DIR = DATA_DIR / "unitron" / "raw"
UNITRON_PROCESSED_DIR = DATA_DIR / "unitron" / "processed"


def ensure_dirs() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    UNITRON_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
