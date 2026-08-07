"""
The public entry point: evaluate(). Accepts a plain list of dicts,
returns a plain list of dicts. No dataset streaming, no file handling --
this is what makes it a genuine, portable library rather than a script.
"""

from typing import Any, Dict, List, Optional

import torch

from .models.model_factory import load_model
from .prompts.prompt_builder import build_prompt
from .utils.postprocess import parse_label_and_reason
from .utils.script_detector import detect_script

DEFAULT_GENERATION_CONFIG = {"max_new_tokens": 60, "do_sample": False}


def _normalize_record(record: Dict[str, Any], index: int) -> Dict[str, Any]:
    text = record.get("text", record.get("transcript"))
    if text is None:
        raise ValueError(
            f"Record at index {index} must have a 'text' or 'transcript' key. Got: {list(record.keys())}"
        )
    return {
        "id": record.get("id", str(index)),
        "text": text,
        "language": record.get("language"),
        "label": record.get("label"),
    }


def _run_batch_with_backoff(model, batch_records, prompt_config, generation_config, batch_size):
    if batch_size <= 1 or len(batch_records) <= 1:
        outputs = []
        for r in batch_records:
            messages = build_prompt(r["text"], prompt_config)
            outputs.append(model.generate(messages, generation_config))
        return outputs

    try:
        messages_list = [build_prompt(r["text"], prompt_config) for r in batch_records]
        return model.generate_batch(messages_list, generation_config)
    except torch.cuda.OutOfMemoryError:
        print(f"[llm_eval] OOM at batch_size={batch_size}, retrying at {batch_size // 2}")
        torch.cuda.empty_cache()
        half = max(1, batch_size // 2)
        results = []
        for i in range(0, len(batch_records), half):
            chunk = batch_records[i:i + half]
            results.extend(_run_batch_with_backoff(model, chunk, prompt_config, generation_config, half))
        return results


def evaluate(
    records: List[Dict[str, Any]],
    model_config: Dict[str, Any],
    prompt_config: Optional[Dict[str, Any]] = None,
    generation_config: Optional[Dict[str, Any]] = None,
    mode: str = "sequence",
    batch_size: int = 8,
    generation_timeout_seconds: int = 60,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    records: list of dicts. Each needs "text" or "transcript".
             Optional: "id", "language", "label" (gold label, for scoring).
    model_config: e.g. {"name": "google/gemma-3-4b-it", "torch_dtype": "bfloat16", "device_map": "auto"}
    mode: "sequence" or "batch".
    batch_size: only used when mode="batch"; auto-reduced on GPU OOM.
    Returns: one dict per record — id, text, language, gold_label,
             raw_model_output, predicted_label, reason.
    """
    if not records:
        return []

    prompt_config = prompt_config or {}
    generation_config = {**DEFAULT_GENERATION_CONFIG, **(generation_config or {})}
    full_config = {"model": model_config, "generation_timeout_seconds": generation_timeout_seconds}

    if verbose:
        print(f"[llm_eval] loading model: {model_config['name']}")
    model = load_model(full_config)

    normalized = [_normalize_record(r, i) for i, r in enumerate(records)]
    for r in normalized:
        r["language"] = r["language"] or detect_script(r["text"])

    results: List[Dict[str, Any]] = []

    if mode == "sequence":
        for i, r in enumerate(normalized):
            messages = build_prompt(r["text"], prompt_config)
            raw_output = model.generate(messages, generation_config)
            label, reason = parse_label_and_reason(raw_output)
            results.append({
                "id": r["id"], "text": r["text"], "language": r["language"],
                "gold_label": r["label"], "raw_model_output": raw_output,
                "predicted_label": label, "reason": reason,
            })
            if verbose and (i + 1) % 20 == 0:
                print(f"[llm_eval] processed {i + 1}/{len(normalized)}")

    elif mode == "batch":
        for start in range(0, len(normalized), batch_size):
            chunk = normalized[start:start + batch_size]
            raw_outputs = _run_batch_with_backoff(model, chunk, prompt_config, generation_config, batch_size)
            for r, raw_output in zip(chunk, raw_outputs):
                label, reason = parse_label_and_reason(raw_output)
                results.append({
                    "id": r["id"], "text": r["text"], "language": r["language"],
                    "gold_label": r["label"], "raw_model_output": raw_output,
                    "predicted_label": label, "reason": reason,
                })
            if verbose:
                print(f"[llm_eval] processed {min(start + batch_size, len(normalized))}/{len(normalized)}")
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'sequence' or 'batch'.")

    return results
