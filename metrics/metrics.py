"""
Computes overall AND per-language classification metrics in a single pass:
accuracy, per-class precision/recall/F1, macro/weighted averages, confusion matrix.
"""

from collections import defaultdict
from typing import Any, Dict, List

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

LABELS = ["safe", "not safe"]


def _compute_block(gold: List[str], pred: List[str]) -> Dict[str, Any]:
    pairs = [(g, p) for g, p in zip(gold, pred) if g in LABELS]
    if not pairs:
        return {"num_labeled": 0}

    gold_f = [g for g, _ in pairs]
    pred_f = [p for _, p in pairs]

    acc = accuracy_score(gold_f, pred_f)
    precision, recall, f1, support = precision_recall_fscore_support(gold_f, pred_f, labels=LABELS, zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(gold_f, pred_f, labels=LABELS, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(gold_f, pred_f, labels=LABELS, average="weighted", zero_division=0)
    cm = confusion_matrix(gold_f, pred_f, labels=LABELS).tolist()

    per_class = {}
    for i, label in enumerate(LABELS):
        per_class[label] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }

    return {
        "num_labeled": len(pairs),
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_precision": round(float(weighted_p), 4),
        "weighted_recall": round(float(weighted_r), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": per_class,
        "confusion_matrix": {"labels": LABELS, "matrix": cm},
    }


def compute_metrics_with_language(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    gold_all = [r["gold_label_norm"] for r in rows]
    pred_all = [r["predicted_label"] for r in rows]

    overall = _compute_block(gold_all, pred_all)
    overall["num_samples"] = len(rows)

    by_lang = defaultdict(list)
    for r in rows:
        by_lang[r.get("language") or "unknown"].append(r)

    per_language = {}
    for lang, lang_rows in sorted(by_lang.items()):
        gold = [r["gold_label_norm"] for r in lang_rows]
        pred = [r["predicted_label"] for r in lang_rows]
        block = _compute_block(gold, pred)
        block["num_samples"] = len(lang_rows)
        per_language[lang] = block

    return {"overall": overall, "per_language": per_language}
