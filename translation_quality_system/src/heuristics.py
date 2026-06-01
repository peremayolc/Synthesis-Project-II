import difflib
import re
from collections import Counter
from typing import Dict, List, Optional

from glossary import contains_phrase, load_glossary

NUMBER_PATTERN = r"\b\d+(?:[.,]\d+)?\b"
PLACEHOLDER_PATTERN = r"(\{[^}]+\}|%s|%d|<[^>]+>|\[[A-Z_]+\])"
PROTECTED_TERMS = [
    "Unitron", "Sonova", "Bluetooth", "Bluetooth®", "GMS", "BT-LE", "iOS", "Android",
    "Remote Plus", "Remote Control", "TV Connector", "PartnerMic", "Roger", "Noahlink",
    "iCube", "FLEX:TRIAL", "SlimTip", "cShell", "IntelliVent", "Vivante",
    "RECD", "REUG", "WDRC", "NAL-NL1", "NAL-NL2", "DSL v5", "GHz", "dB", "Hz",
    "Moxi", "TrueFit", "Sonova AG", "Bluetooth SIG",
]
ENGLISH_STOPWORDS = {
    "the", "and", "or", "to", "of", "in", "on", "for", "with", "your", "you", "if", "is", "are", "be",
    "this", "that", "these", "those", "can", "will", "may", "not", "use", "used", "using", "a", "an",
    "it", "its", "as", "by", "from", "at", "when", "while", "into", "than", "then", "after", "before",
}
SPANISH_NEGATION = {"no", "nunca", "jamás", "sin", "ningún", "ninguna", "debe", "deben", "evite"}
SOURCE_NEGATION = {"not", "never", "without", "avoid", "do not", "should never", "must not"}
SAFETY_WORDS = {
    "warning", "required", "must", "never", "avoid", "caution", "flight mode", "battery", "batteries",
    "chlorinated", "salt water", "medical", "directive", "regulation", "compliance", "tinnitus",
}


def _extract(pattern: str, text: str) -> List[str]:
    return re.findall(pattern, text or "")


def _norm_number(n: str) -> str:
    return n.replace(",", ".").strip()


def _issue(issue_type: str, severity: str, explanation: str, **extra) -> Dict[str, object]:
    data: Dict[str, object] = {"type": issue_type, "severity": severity, "explanation": explanation}
    data.update(extra)
    return data


def check_numbers(source_en: str, mt_es: str) -> List[Dict[str, object]]:
    src = [_norm_number(x) for x in _extract(NUMBER_PATTERN, source_en)]
    tgt = [_norm_number(x) for x in _extract(NUMBER_PATTERN, mt_es)]
    if sorted(src) != sorted(tgt):
        return [_issue("number_mismatch", "high", f"Numbers differ between source and translation: EN={src}, ES={tgt}")]
    return []


def check_placeholders(source_en: str, mt_es: str) -> List[Dict[str, object]]:
    issues: List[Dict[str, object]] = []
    src = set(_extract(PLACEHOLDER_PATTERN, source_en))
    tgt = set(_extract(PLACEHOLDER_PATTERN, mt_es))
    for missing in sorted(src - tgt):
        issues.append(_issue("placeholder_missing", "high", f"Missing placeholder/tag: {missing}"))
    for extra in sorted(tgt - src):
        issues.append(_issue("placeholder_extra", "medium", f"Unexpected placeholder/tag: {extra}"))
    return issues


def check_protected_terms(source_en: str, mt_es: str) -> List[Dict[str, object]]:
    issues: List[Dict[str, object]] = []
    for term in PROTECTED_TERMS:
        if contains_phrase(source_en, term) and not contains_phrase(mt_es, term):
            issues.append(_issue("protected_term_missing", "high", f"Protected brand/technical term missing or changed: {term}"))
    return issues


def check_glossary(source_en: str, mt_es: str) -> List[Dict[str, object]]:
    issues: List[Dict[str, object]] = []
    for entry in load_glossary():
        if contains_phrase(source_en, entry.source_term) and not contains_phrase(mt_es, entry.approved_es):
            # If the approved term is a protected unchanged term with no forbidden variants, do not overflag.
            if entry.forbidden_es:
                issues.append(
                    _issue(
                        "terminology_expected",
                        "medium",
                        f"Expected approved Spanish term '{entry.approved_es}' for source term '{entry.source_term}'.",
                    )
                )
        for bad in entry.forbidden_es:
            if contains_phrase(mt_es, bad):
                issues.append(
                    _issue(
                        "terminology_forbidden",
                        "high",
                        f"Forbidden or non-preferred term '{bad}' found; use '{entry.approved_es}'.",
                        before=bad,
                        after=entry.approved_es,
                    )
                )
    return issues


def _english_words(text: str) -> List[str]:
    return [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z-]{3,}", text or "")]


def check_untranslated_english(source_en: str, mt_es: str) -> List[Dict[str, object]]:
    source_words = set(_english_words(source_en)) - ENGLISH_STOPWORDS
    target_words = set(_english_words(mt_es)) - ENGLISH_STOPWORDS

    protected_lower = set()
    for term in PROTECTED_TERMS:
        protected_lower.update(_english_words(term))

    leaked = sorted(w for w in (source_words & target_words) if w not in protected_lower)
    # Do not flag isolated UI acronyms/product-like leftovers; flag repeated/raw English content.
    if len(leaked) >= 3:
        severity = "high" if len(leaked) >= 8 else "medium"
        return [_issue("untranslated_english", severity, f"Possible untranslated English words remain: {', '.join(leaked[:20])}")]
    return []


def check_length_and_repetition(source_en: str, mt_es: str, reference_es: Optional[str] = None) -> List[Dict[str, object]]:
    issues: List[Dict[str, object]] = []
    target_words = re.findall(r"\w+", mt_es or "", flags=re.UNICODE)
    source_words = re.findall(r"\w+", source_en or "", flags=re.UNICODE)
    ref_words = re.findall(r"\w+", reference_es or "", flags=re.UNICODE)

    if source_words and len(target_words) < max(4, 0.25 * len(source_words)):
        issues.append(_issue("possible_omission", "high", "Translation is much shorter than the source; possible missing content."))

    if ref_words:
        if len(target_words) < max(4, 0.45 * len(ref_words)):
            issues.append(_issue("possible_omission", "high", "Translation is much shorter than the reference translation."))
        if len(target_words) > 2.2 * max(len(ref_words), 1) and len(target_words) > 25:
            issues.append(_issue("possible_hallucination_or_repetition", "high", "Translation is much longer than the reference; possible hallucination or repeated content."))

        ratio = difflib.SequenceMatcher(None, (reference_es or "").lower(), (mt_es or "").lower()).ratio()
        if ratio < 0.18 and len(ref_words) >= 8 and len(target_words) >= 8:
            issues.append(_issue("reference_divergence", "high", f"Very low similarity with the available human reference (ratio={ratio:.2f})."))

    # Repeated 3-gram detector catches loops such as repeated model/product names.
    grams = [tuple(w.lower() for w in target_words[i:i + 3]) for i in range(max(0, len(target_words) - 2))]
    repeated = [gram for gram, count in Counter(grams).items() if count >= 4 and any(len(x) > 2 for x in gram)]
    if repeated:
        sample = " ".join(repeated[0])
        issues.append(_issue("repetition_loop", "high", f"Repeated phrase detected: '{sample}...'"))

    return issues


def check_negation_and_safety(source_en: str, mt_es: str) -> List[Dict[str, object]]:
    issues: List[Dict[str, object]] = []
    src_l = (source_en or "").lower()
    tgt_l = (mt_es or "").lower()

    has_source_neg = any(term in src_l for term in SOURCE_NEGATION)
    has_spanish_neg = any(re.search(rf"\b{re.escape(term)}\b", tgt_l) for term in SPANISH_NEGATION)
    safety_context = any(term in src_l for term in SAFETY_WORDS)

    if has_source_neg and not has_spanish_neg:
        severity = "high" if safety_context else "medium"
        issues.append(_issue("negation_missing", severity, "A source negation/prohibition may be missing in the Spanish translation."))

    if safety_context and len(re.findall(r"\w+", mt_es or "")) < 6:
        issues.append(_issue("safety_content_too_short", "high", "Safety/regulatory content is translated too briefly to be reliable."))

    return issues


def run_heuristics(source_en: str, mt_es: str, reference_es: Optional[str] = None) -> List[Dict[str, object]]:
    issues: List[Dict[str, object]] = []
    issues.extend(check_numbers(source_en, mt_es))
    issues.extend(check_placeholders(source_en, mt_es))
    issues.extend(check_protected_terms(source_en, mt_es))
    issues.extend(check_glossary(source_en, mt_es))
    issues.extend(check_untranslated_english(source_en, mt_es))
    issues.extend(check_length_and_repetition(source_en, mt_es, reference_es))
    issues.extend(check_negation_and_safety(source_en, mt_es))

    # De-duplicate repeated glossary/protected messages while preserving order.
    seen = set()
    deduped: List[Dict[str, object]] = []
    for issue in issues:
        key = (issue.get("type"), issue.get("explanation"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(issue)
    return deduped
