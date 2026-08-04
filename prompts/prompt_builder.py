"""
Builds the chat "messages" list directly from CONFIG's prompt block, so
the prompt is editable from run_pipeline.py without touching this file.
Falls back to sensible defaults if CONFIG doesn't specify them.
"""

from typing import Any, Dict, List

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert multilingual profanity detection system. You judge "
    "text written in any language, including Indic languages and code-mixed text."
)

DEFAULT_USER_TEMPLATE = (
    "Classify the following text.\n"
    "Return ONLY one of these two labels and nothing else:\n"
    "safe\n"
    "not safe\n\n"
    "Text: {text}\n\n"
    "Answer:"
)


def build_prompt(text: str, config: Dict[str, Any]) -> List[Dict[str, str]]:
    prompt_cfg = config.get("prompt", {})
    system_prompt = prompt_cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    user_template = prompt_cfg.get("user_template", DEFAULT_USER_TEMPLATE)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_template.format(text=text)},
    ]
