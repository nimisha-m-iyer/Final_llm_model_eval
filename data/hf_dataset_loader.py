"""
Streams the dataset and normalizes rows to {"text", "label", "language"}.

FALLBACK LOGIC (built in, not a separate step):
  - text/label columns: auto-detected from common names unless CONFIG
    specifies them explicitly.
  - language column: if the dataset has one, use it directly. If not,
    "language" is left as None here — the evaluator fills it in via
    script detection at run time. This IS the fallback: check first,
    detect only if missing.
"""

from typing import Any, Dict, Optional, Tuple

from datasets import load_dataset

TEXT_COLUMN_CANDIDATES = ("text", "transcript", "sentence", "content", "comment", "utterance")
LABEL_COLUMN_CANDIDATES = ("label", "labels", "is_profane", "profane", "class", "target", "profanity")
LANGUAGE_COLUMN_CANDIDATES = ("language", "lang", "lang_code", "language_code", "locale")


def _detect_columns(dataset, ds_cfg: Dict[str, Any]) -> Tuple[str, Optional[str], Optional[str]]:
    text_col = ds_cfg.get("text_column")
    label_col = ds_cfg.get("label_column")
    lang_col = ds_cfg.get("language_column")

    first = next(iter(dataset))

    if text_col is None:
        text_col = next((c for c in TEXT_COLUMN_CANDIDATES if c in first), None)
    if label_col is None:
        label_col = next((c for c in LABEL_COLUMN_CANDIDATES if c in first), None)
    if lang_col is None:
        lang_col = next((c for c in LANGUAGE_COLUMN_CANDIDATES if c in first), None)

    if text_col is None:
        raise ValueError(
            f"Could not auto-detect a text column. Available columns: {list(first.keys())}. "
            "Set dataset.text_column explicitly in CONFIG."
        )
    return text_col, label_col, lang_col


def load_streaming_dataset(config: Dict[str, Any]):
    ds_cfg = config["dataset"]

    dataset = load_dataset(
        ds_cfg["name"],
        split=ds_cfg.get("split", "train"),
        streaming=ds_cfg.get("streaming", True),
    )

    text_col, label_col, lang_col = _detect_columns(dataset, ds_cfg)

    if lang_col:
        print(f"[data_loader] text='{text_col}' label='{label_col}' language='{lang_col}' (from dataset)")
    else:
        print(f"[data_loader] text='{text_col}' label='{label_col}'. "
              "No language column — will auto-detect via script per row (fallback).")

    def _preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "text": example.get(text_col, ""),
            "label": example.get(label_col) if label_col else None,
            "language": example.get(lang_col) if lang_col else None,
        }

    return dataset.map(_preprocess)
