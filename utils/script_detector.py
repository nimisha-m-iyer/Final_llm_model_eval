"""
Detects likely script (proxy for language) via Unicode code-point ranges.
Fast, deterministic, no model download — used as the language fallback.
"""

from collections import Counter

SCRIPT_RANGES = {
    "Devanagari (Hindi/Marathi/Nepali/Sanskrit)": [(0x0900, 0x097F)],
    "Bengali/Assamese": [(0x0980, 0x09FF)],
    "Gurmukhi (Punjabi)": [(0x0A00, 0x0A7F)],
    "Gujarati": [(0x0A80, 0x0AFF)],
    "Odia": [(0x0B00, 0x0B7F)],
    "Tamil": [(0x0B80, 0x0BFF)],
    "Telugu": [(0x0C00, 0x0C7F)],
    "Kannada": [(0x0C80, 0x0CFF)],
    "Malayalam": [(0x0D00, 0x0D7F)],
    "Sinhala": [(0x0D80, 0x0DFF)],
    "Urdu/Arabic": [(0x0600, 0x06FF), (0x0750, 0x077F)],
    "Latin (English/Romanized)": [(0x0041, 0x005A), (0x0061, 0x007A)],
}


def detect_script(text: str) -> str:
    if not text:
        return "unknown"
    counts = Counter()
    for ch in text:
        cp = ord(ch)
        for script, ranges in SCRIPT_RANGES.items():
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[script] += 1
                break
    return counts.most_common(1)[0][0] if counts else "unknown"
