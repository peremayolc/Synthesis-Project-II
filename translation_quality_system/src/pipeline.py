from typing import Dict, Optional

from heuristics import run_heuristics
from rule_corrector import RuleCorrector

HIGH_RISK_TYPES = {
    "number_mismatch",
    "placeholder_missing",
    "protected_term_missing",
    "possible_omission",
    "possible_hallucination_or_repetition",
    "reference_divergence",
    "repetition_loop",
    "negation_missing",
    "safety_content_too_short",
}


class TranslationPipeline:
    def __init__(self):
        self.corrector = RuleCorrector()

    def process(
        self,
        source_en: str,
        mt_es: str,
        reference_es: Optional[str] = None,
        document: Optional[str] = None,
        segment_id: Optional[str] = None,
        variant: Optional[str] = None,
    ) -> Dict[str, object]:
        issues = run_heuristics(source_en, mt_es, reference_es=reference_es)
        correction = self.corrector.correct(source_en, mt_es)
        corrected_es = correction["corrected_es"]

        high = [i for i in issues if i.get("severity") == "high"]
        medium = [i for i in issues if i.get("severity") == "medium"]
        has_high_risk = any(i.get("type") in HIGH_RISK_TYPES for i in high)

        needs_human = bool(has_high_risk or len(high) >= 1 or len(medium) >= 4)
        if needs_human:
            decision = "human_review"
        elif correction["explanation"] or issues:
            decision = "auto_corrected_or_warned"
        else:
            decision = "accepted"

        return {
            "document": document,
            "segment_id": segment_id,
            "variant": variant,
            "source_en": source_en,
            "reference_es": reference_es,
            "mt_es": mt_es,
            "corrected_es": corrected_es,
            "issues": issues,
            "needs_human_review": needs_human,
            "decision": decision,
            "correction_explanation": correction["explanation"],
        }
