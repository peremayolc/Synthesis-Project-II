# Synthesis-Project-II

Project development for the Synthesis Project II subject.

This repository contains the work for an English-to-Spanish technical translation
and quality-assurance system, together with the model fine-tuning pipeline and a
demo web interface used to present the results.

## Repository structure

### `finetuning/`

Scripts and data for fine-tuning Llama-based translation models with LoRA adapters
(via Unsloth) on domain-specific document pairs. It covers three domains —
automotive, legal, and medical — each with its own model-loading and dataset-preparation
module (`automotive.py`, `legal.py`, `medical.py`). Key files:

- `finetune.py` — main training entry point that builds the datasets and runs the SFT trainer.
- `compare-models.py` — side-by-side evaluation of a fine-tuned checkpoint against its base model using BLEU and chrF on aligned bilingual document pairs.
- `test-finetune.py` — checks that a fine-tuned model is present and complete on the Hugging Face Hub.
- `imports.py` — shared imports for the fine-tuning code.
- `slurm.sh` — SLURM batch script for launching training on a GPU cluster.
- `documents-testing/` — bilingual (EN/ES) source documents used for testing and evaluation, grouped into `automotive/`, `legal/`, and `medical/` subfolders.

### `translation_quality_system/`

A self-contained quality-assurance system that verifies English-to-Spanish machine
translations for technical documentation. It runs deterministic, glossary-aware checks,
routes risky segments to human review, and produces reproducible reports. See the
folder's own [README](translation_quality_system/README.md) for full details. Layout:

- `src/` — the pipeline source code: file parsing, heuristic QA checks, rule-based and LLM correctors, RAG retrieval/indexing, evaluation, report export, and their tests.
- `data/` — input and reference data, including the technical glossary, style guides, RAG documents and FAISS index, synthetic datasets, and the Unitron translation files.
- `outputs/` — generated verification results and summaries.
- `run_pipeline.py` — root-level launcher for the end-to-end verification pipeline.

### `UI/`

The front-end used to demonstrate the system. It contains `LEXOR-Demo-Website-Builder/`,
a pnpm workspace (TypeScript + Vite) whose `artifacts/lexor/` directory holds the LEXOR
demo web application that presents the translation and quality-assurance results.
