from typing import Dict, List

from glossary import load_glossary, replace_phrase, contains_phrase


class RuleCorrector:
    """Deterministic terminology and punctuation corrector.

    This is deliberately conservative: it fixes known glossary violations and Spanish opening
    punctuation, but it does not rewrite whole sentences. Segments with semantic, numerical,
    omission, repetition, or safety issues are routed to human review by the pipeline.
    """

    def __init__(self):
        self.glossary = load_glossary()

    def correct_terminology(self, text: str):
        corrected = text or ""
        fixes: List[Dict[str, str]] = []

        for entry in self.glossary:
            for bad in entry.forbidden_es:
                if contains_phrase(corrected, bad):
                    corrected = replace_phrase(corrected, bad, entry.approved_es)
                    fixes.append({"type": "terminology", "before": bad, "after": entry.approved_es})

        return corrected, fixes

    def correct_punctuation(self, text: str):
        corrected = text or ""
        fixes: List[Dict[str, str]] = []

        stripped = corrected.lstrip()
        if "?" in corrected and not stripped.startswith("¿"):
            corrected = corrected.replace("?", "?")
            corrected = "¿" + corrected
            fixes.append({"type": "punctuation", "before": "?", "after": "¿...?"})

        stripped = corrected.lstrip()
        if "!" in corrected and not stripped.startswith("¡"):
            corrected = "¡" + corrected
            fixes.append({"type": "punctuation", "before": "!", "after": "¡...!"})

        return corrected, fixes

    def correct(self, source_en: str, mt_es: str):
        corrected = mt_es or ""
        all_fixes: List[Dict[str, str]] = []

        corrected, fixes = self.correct_terminology(corrected)
        all_fixes.extend(fixes)

        corrected, fixes = self.correct_punctuation(corrected)
        all_fixes.extend(fixes)

        return {"corrected_es": corrected, "explanation": all_fixes}
