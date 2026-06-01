# Translation Quality System — Unitron QA Version

This project verifies English-to-Spanish machine translations for Unitron hearing-aid and fitting-software documentation.

Focuses on:

- deterministic quality checks that run locally;
- Unitron/audiology-specific terminology;
- comparison against the provided human reference translations when available;
- human-review routing for risky translations;
- reproducible CSV, JSONL, and HTML reports.

## Important security note

The uploaded ZIP contained a real `.env` file. It has been removed from this cleaned project and replaced with `.env.example`. Do not commit real API keys.

## Main files

```text
data/glossary/technical_glossary.csv              # active glossary used by the QA system
data/glossary/unitron_synthetic_glossary.csv      # same glossary kept as explicit deliverable
data/style_guides/technical_es_v1.md              # Unitron-specific style guide
data/unitron/raw/*.txt                            # original translated files you attached
data/unitron/processed/unitron_translation_pairs.jsonl
outputs/unitron_verification.jsonl                # detailed machine-readable results
outputs/unitron_verification.csv                  # spreadsheet-friendly results
outputs/unitron_verification.html                 # visual report
src/translation_file_parser.py                    # parser for EN/REF/BASE/FT txt files
src/heuristics.py                                 # QA checks
src/rule_corrector.py                             # deterministic corrections
src/pipeline.py                                   # decision logic
src/run_unitron_verification.py                   # end-to-end verification runner
run_pipeline.py                                   # root-level launcher
```

## How to run on Windows PowerShell

From the project folder:

```powershell
cd "C:\Users\suzan\Desktop\translation_quality_system"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_pipeline.py
```

The outputs will be regenerated in:

```text
outputs/unitron_verification.jsonl
outputs/unitron_verification.csv
outputs/unitron_verification.html
outputs/unitron_verification_summary.json
```

Open the HTML report with:

```powershell
start outputs\unitron_verification.html
```

## Current verification summary

The attached Unitron files produced 307 aligned source/reference segments. The system checks both ML variants, `BASE` and `FT`, so it evaluates 614 translations.

Current run:

```json
{
  "total_checked_translations": 614,
  "by_variant": {
    "BASE": {"total": 307, "human_review": 180, "accepted": 105, "auto_corrected_or_warned": 22},
    "FT": {"total": 307, "human_review": 202, "accepted": 81, "auto_corrected_or_warned": 24}
  }
}
```



## Checks implemented

- number/version mismatch detection;
- placeholder/tag preservation;
- protected product/brand/acronym preservation;
- glossary violations and forbidden terminology;
- possible untranslated English leakage;
- omission and hallucination/repetition risk using source/reference length and similarity;
- repeated phrase loops;
- missing negation in safety or regulatory contexts.

## Human-review policy

The pipeline sends a segment to human review if it finds any high-risk issue, including:

- changed numbers or versions;
- missing protected technical terms;
- forbidden terminology in a safety/medical context;
- omission, hallucination, or repetition;
- missing negation/prohibition;
- very low similarity with the human reference.

The deterministic corrector only performs safe glossary replacements and punctuation fixes. It does not pretend to fully rewrite semantically broken segments.
