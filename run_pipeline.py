"""
============================================================
 SINGLE CONTROL PANEL — edit CONFIG below, then run this file.
 Everything else in the repo is internal plumbing you should not
 need to touch for a normal run.
============================================================
"""

import os
import sys
import json
import traceback

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

# ======================================================================
# 1. EDIT THIS CONFIG BLOCK — everything you need to change lives here
# ======================================================================
CONFIG = {
    "experiment_name": "gemma_profanity_demo",

    # ---- MODE ----
    # "demo"      -> type ONE sentence, see the full prompt + output right now
    # "full_eval" -> run the whole dataset, save CSV+JSON, push to GitHub
    "mode": "demo",

    # only used when mode == "demo"
    "demo_text": "nee bayangara mosham manushyan aan",

    # ---- MODEL ----
    # Swap models here. Examples that already work:
    #   "google/gemma-3-4b-it"
    #   "CohereLabs/aya-expanse-8b"
    #   "Qwen/Qwen3-4B-Instruct-2507"
    #   "gpt-4o-mini"                <- routes to OpenAI API automatically
    "model": {
        "name": "google/gemma-3-4b-it",
        "torch_dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": False,
        "load_in_4bit": False,
    },

    # per-sample generation timeout (seconds) — stops one bad sample
    # from hanging the whole run
    "generation_timeout_seconds": 60,

    # ---- DATASET (only used when mode == "full_eval") ----
    "dataset": {
        "name": "mangalathkedar/multilingual-indic-profane",
        "split": "train",
        "streaming": True,
        "text_column": None,       # None = auto-detect
        "label_column": None,      # None = auto-detect
        "language_column": None,   # None = auto-detect; if dataset has no
                                    # language column, script detection
                                    # kicks in automatically (fallback)
    },

    # ---- PROMPT ----
    "prompt": {
        "system_prompt": (
            "You are an expert multilingual profanity detection system. "
            "You judge text written in any language, including Indic "
            "languages and code-mixed text."
        ),
        "user_template": (
            "Classify the following text.\n"
            "Return ONLY one of these two labels and nothing else:\n"
            "safe\n"
            "not safe\n\n"
            "Text: {text}\n\n"
            "Answer:"
        ),
    },

    # ---- GENERATION ----
    "generation": {
        "max_new_tokens": 8,
        "do_sample": False,
    },

    # ---- EVALUATION (only used when mode == "full_eval") ----
    "evaluation": {
        "max_samples": None,   # None = full dataset, or an int to cap it
    },

    # ---- OUTPUT (only used when mode == "full_eval") ----
    "output": {
        "prediction_file": "outputs/gemma_predictions.csv",
        "metric_file": "outputs/gemma_metrics.json",
    },

    # ---- AUTO-PUSH TO GITHUB (only used when mode == "full_eval") ----
    "auto_push_to_github": True,
}
# ======================================================================
# END OF EDITABLE CONFIG — nothing below this line needs to change
# ======================================================================


def run_demo(config):
    from models.model_factory import load_model
    from prompts.prompt_builder import build_prompt
    from utils.postprocess import normalize_label
    from utils.script_detector import detect_script

    print("=" * 70)
    print("LIVE DEMO MODE")
    print("=" * 70)
    print(f"Model: {config['model']['name']}")

    text = config["demo_text"]
    messages = build_prompt(text, config)
    language = detect_script(text)

    print("\n--- INPUT TEXT ---")
    print(text)
    print("\n--- DETECTED LANGUAGE/SCRIPT ---")
    print(language)
    print("\n--- FULL PROMPT SENT TO MODEL ---")
    for m in messages:
        print(f"[{m['role']}] {m['content']}")

    print("\nLoading model (first load can take a minute)...")
    model = load_model(config)

    print("\nGenerating...")
    raw_output = model.generate(messages, config["generation"])
    predicted = normalize_label(raw_output)

    print("\n--- RAW MODEL OUTPUT ---")
    print(repr(raw_output))
    print("\n--- FINAL PREDICTED LABEL ---")
    print(predicted)
    print("=" * 70)

    demo_result = {
        "model": config["model"]["name"],
        "text": text,
        "language": language,
        "prompt": messages,
        "generation_config": config["generation"],
        "raw_model_output": raw_output,
        "predicted_label": predicted,
    }
    os.makedirs("outputs", exist_ok=True)
    demo_path = "outputs/demo_result.json"
    with open(demo_path, "w", encoding="utf-8") as f:
        json.dump(demo_result, f, indent=2, ensure_ascii=False)
    print(f"\nDemo result saved -> {demo_path}")

    return demo_result


def run_full_eval(config):
    from data.hf_dataset_loader import load_streaming_dataset
    from models.model_factory import load_model
    from evaluator.evaluators import run_evaluation
    from utils.git_push import push_outputs_to_github

    print("=" * 70)
    print("FULL EVALUATION MODE")
    print(f"Experiment: {config['experiment_name']}")
    print(f"Model: {config['model']['name']}")
    print(f"Dataset: {config['dataset']['name']}")
    print("=" * 70)

    dataset = load_streaming_dataset(config)
    model = load_model(config)
    metrics = run_evaluation(config, model, dataset)

    if config.get("auto_push_to_github", False):
        pred_path = os.path.abspath(config["output"]["prediction_file"])
        metric_path = os.path.abspath(config["output"]["metric_file"])
        push_outputs_to_github(
            repo_root=REPO_ROOT,
            file_paths=[pred_path, metric_path],
            commit_message=f"Auto: add {config['experiment_name']} evaluation results",
        )

    return metrics


def main():
    mode = CONFIG.get("mode", "demo")
    try:
        if mode == "demo":
            run_demo(CONFIG)
        elif mode == "full_eval":
            run_full_eval(CONFIG)
        else:
            raise ValueError(f"Unknown mode '{mode}' — use 'demo' or 'full_eval'.")
    except Exception as e:
        print("\n" + "=" * 70)
        print("PIPELINE ERROR")
        print("=" * 70)
        print(f"{type(e).__name__}: {e}")
        traceback.print_exc()
        print("\nCommon fixes:")
        print(" - Gated model (Gemma/Aya): accept the license on huggingface.co, then re-login.")
        print(" - Missing HF_TOKEN: run the Hugging Face login cell before this script.")
        print(" - Missing GH_TOKEN/GH_REPO: auto-push is skipped, not fatal.")
        print(" - GPT model: make sure OPENAI_API_KEY is set and 'openai' package is installed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
