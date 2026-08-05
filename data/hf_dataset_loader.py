"""
Loads evaluation data from either a Hugging Face dataset (streamed) or a
local/uploaded CSV file, selected via config["dataset"]["source"].

Both paths normalize rows to the same {"text", "label", "language"} shape
and support .skip(n), so evaluator.py's checkpoint/resume logic works
identically regardless of the data source.
"""

import csv
from typing import Any, Dict, Iterable, Optional, Tuple

from datasets import load_dataset

TEXT_COLUMN_CANDIDATES = ("text", "transcript", "sentence", "content", "comment", "utterance")
LABEL_COLUMN_CANDIDATES = ("label", "labels", "is_profane", "profane", "class", "target", "profanity")
LANGUAGE_COLUMN_CANDIDATES = ("language", "lang", "lang_code", "language_code", "locale")


class _SkippableRows:
    """
    Minimal wrapper so a plain in-memory list of CSV rows supports the
    same .skip(n) + iteration interface that a streaming HF IterableDataset
    provides — this is what lets the evaluator's checkpoint/resume logic
    work identically for both CSV and Hugging Face sources.
    """

    def __init__(self, rows, skip_n: int = 0):
        self._rows = rows
        self._skip_n = skip_n

    def skip(self, n: int) -> "_SkippableRows":
        return _SkippableRows(self._rows, skip_n=self._skip_n + n)

    def __iter__(self):
        for row in self._rows[self._skip_n:]:
            yield row


def _detect_columns_from_sample(sample: Dict[str, Any], ds_cfg: Dict[str, Any]) -> Tuple[str, Optional[str], Optional[str]]:
    text_col = ds_cfg.get("text_column")
    label_col = ds_cfg.get("label_column")
    lang_col = ds_cfg.get("language_column")

    if text_col is None:
        text_col = next((c for c in TEXT_COLUMN_CANDIDATES if c in sample), None)
    if label_col is None:
        label_col = next((c for c in LABEL_COLUMN_CANDIDATES if c in sample), None)
    if lang_col is None:
        lang_col = next((c for c in LANGUAGE_COLUMN_CANDIDATES if c in sample), None)

    if text_col is None:
        raise ValueError(
            f"Could not auto-detect a text column. Available columns: {list(sample.keys())}. "
            "Set dataset.text_column explicitly in CONFIG."
        )
    return text_col, label_col, lang_col


def _load_huggingface_dataset(ds_cfg: Dict[str, Any]):
    dataset = load_dataset(
        ds_cfg["name"],
        split=ds_cfg.get("split", "train"),
        streaming=ds_cfg.get("streaming", True),
    )

    first = next(iter(dataset))
    text_col, label_col, lang_col = _detect_columns_from_sample(first, ds_cfg)

    if lang_col:
        print(f"[data_loader] (HuggingFace) text='{text_col}' label='{label_col}' language='{lang_col}' (from dataset)")
    else:
        print(f"[data_loader] (HuggingFace) text='{text_col}' label='{label_col}'. "
              "No language column — will auto-detect via script per row (fallback).")

    def _preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "text": example.get(text_col, ""),
            "label": example.get(label_col) if label_col else None,
            "language": example.get(lang_col) if lang_col else None,
        }

    return dataset.map(_preprocess)


def _load_csv_dataset(ds_cfg: Dict[str, Any]):
    csv_path = ds_cfg.get("path")
    if not csv_path:
        raise ValueError(
            "dataset.source is 'csv' but dataset.path is not set. "
            "Point it at a CSV file, e.g. 'outputs/my_test_set.csv' or "
            "'/kaggle/input/my-dataset/test.csv'."
        )

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV file '{csv_path}' has no rows.")

    text_col, label_col, lang_col = _detect_columns_from_sample(rows[0], ds_cfg)

    if lang_col:
        print(f"[data_loader] (CSV: {csv_path}) text='{text_col}' label='{label_col}' language='{lang_col}' (from CSV)")
    else:
        print(f"[data_loader] (CSV: {csv_path}) text='{text_col}' label='{label_col}'. "
              "No language column — will auto-detect via script per row (fallback).")

    normalized_rows = [
        {
            "text": row.get(text_col, ""),
            "label": row.get(label_col) if label_col else None,
            "language": row.get(lang_col) if lang_col else None,
        }
        for row in rows
    ]

    print(f"[data_loader] loaded {len(normalized_rows)} rows from CSV.")
    return _SkippableRows(normalized_rows)


def load_dataset_for_eval(config: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    Single entry point used by run_pipeline.py. Dispatches to Hugging Face
    or CSV loading based on config["dataset"]["source"] (default: "huggingface").
    """
    ds_cfg = config["dataset"]
    source = ds_cfg.get("source", "huggingface")

    if source == "huggingface":
        return _load_huggingface_dataset(ds_cfg)
    elif source == "csv":
        return _load_csv_dataset(ds_cfg)
    else:
        raise ValueError(f"Unknown dataset.source '{source}'. Use 'huggingface' or 'csv'.")
