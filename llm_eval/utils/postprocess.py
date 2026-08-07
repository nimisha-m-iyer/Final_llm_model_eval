import re


def normalize_label(raw: object) -> str:
    text = str(raw).strip().lower()
    if text in ("1", "true", "profane", "not_safe", "notsafe"):
        return "not safe"
    if text in ("0", "false", "not_profane", "notprofane"):
        return "safe"
    if "not safe" in text or "unsafe" in text or "not-safe" in text or "profane" in text:
        return "not safe"
    if "safe" in text:
        return "safe"
    return "unknown"


def parse_label_and_reason(raw_text: str):
    label_match = re.search(r"label\s*:\s*(.+)", raw_text, re.IGNORECASE)
    reason_match = re.search(r"reason\s*:\s*(.+)", raw_text, re.IGNORECASE)

    label = normalize_label(label_match.group(1)) if label_match else normalize_label(raw_text)
    reason = reason_match.group(1).strip() if reason_match else ""
    return label, reason
