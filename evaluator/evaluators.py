"""
The full evaluation loop. Language detection fallback lives INLINE here —
if a row already has a language (from the dataset), it's used as-is; if
not, it's detected via Unicode script on the spot. No separate phase, no
separate file. Checkpointing means a re-run resumes instead of restarting.
"""

import csv
import json
import os
from typing import Any, Dict

from metrics.metrics import compute_metrics_with_language
from prompts.prompt_builder import build_prompt
from utils.postprocess import normalize_label
from utils.script_detector import detect_script

CSV_FIELDS = ["text", "language", "gold_label", "gold_label_norm", "raw_model_output", "predicted_label"]


def _checkpoint_path(pred_path: str) -> str:
    return pred_path + ".checkpoint"


def _read_checkpoint(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        c = f.read().strip()
    return int(c) if c else 0


def _write_checkpoint(path: str, n: int) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(n))


def run_evaluation(config: Dict[str, Any], model, dataset, verbose_first_n: int = 3) -> Dict[str, Any]:
    gen_cfg = config.get("generation", {})
    output_cfg = config["output"]
    max_samples = config.get("evaluation", {}).get("max_samples")

    pred_path = os.path.abspath(output_cfg["prediction_file"])
    metric_path = os.path.abspath(output_cfg["metric_file"])
    os.makedirs(os.path.dirname(pred_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(metric_path) or ".", exist_ok=True)

    checkpoint_path = _checkpoint_path(pred_path)
    already = _read_checkpoint(checkpoint_path)
    resuming = already > 0 and os.path.exists(pred_path)

    if resuming:
        print(f"[evaluator] RESUMING from checkpoint: {already} samples already done")
        mode, write_header = "a", False
        dataset = dataset.skip(already)
    else:
        already = 0
        mode, write_header = "w", True

    f = open(pred_path, mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()
        f.flush()

    i = already
    try:
        for example in dataset:
            if max_samples is not None and i >= max_samples:
                break

            text = example["text"]
            language = example.get("language") or detect_script(text)  # <- FALLBACK
            gold_raw = example.get("label")
            gold_norm = normalize_label(gold_raw) if gold_raw is not None else "unknown"

            messages = build_prompt(text, config)
            raw_output = model.generate(messages, gen_cfg)
            pred_label = normalize_label(raw_output)

            if i - already < verbose_first_n:
                print(f"\n----- SAMPLE {i} -----")
                print("TEXT:      ", text)
                print("LANGUAGE:  ", language)
                print("RAW OUTPUT:", repr(raw_output))
                print("PREDICTED: ", pred_label, "  | GOLD:", gold_norm)

            writer.writerow({
                "text": text, "language": language,
                "gold_label": gold_raw, "gold_label_norm": gold_norm,
                "raw_model_output": raw_output, "predicted_label": pred_label,
            })

            i += 1
            if i % 20 == 0:
                f.flush()
                _write_checkpoint(checkpoint_path, i)
                print(f"[evaluator] processed {i} samples...")
    finally:
        f.flush()
        f.close()
        _write_checkpoint(checkpoint_path, i)

    print(f"[evaluator] total processed: {i}")

    rows = []
    with open(pred_path, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)

    metrics = compute_metrics_with_language(rows)
    with open(metric_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)

    print(f"[evaluator] saved predictions -> {pred_path}")
    print(f"[evaluator] saved metrics -> {metric_path}")
    _print_summary(metrics)
    return metrics


def _print_summary(metrics: Dict[str, Any]) -> None:
    ov = metrics["overall"]
    print("\n===================== OVERALL =====================")
    print(f"samples: {ov.get('num_samples')}  labeled: {ov.get('num_labeled')}")
    if ov.get("num_labeled", 0) == 0:
        print("No labeled samples.")
        return
    print(f"accuracy: {ov.get('accuracy')}  macro_f1: {ov.get('macro_f1')}  weighted_f1: {ov.get('weighted_f1')}")
    for label, pc in ov.get("per_class", {}).items():
        print(f"  [{label:9s}] precision={pc['precision']} recall={pc['recall']} f1={pc['f1']} support={pc['support']}")

    print("\n===================== PER LANGUAGE =====================")
    for lang, block in metrics["per_language"].items():
        if block.get("num_labeled", 0) == 0:
            continue
        print(f"{lang:45s} n={block['num_samples']:5d} acc={block.get('accuracy')} macro_f1={block.get('macro_f1')}")
