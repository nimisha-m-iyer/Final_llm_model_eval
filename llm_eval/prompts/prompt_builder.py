from typing import Any, Dict, List

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert multilingual profanity detection system. You judge "
    "text written in any language, including Indic languages and code-mixed text."
)

DEFAULT_USER_TEMPLATE = (
    "Classify the following text.\n"
    "Respond in EXACTLY this two-line format and nothing else:\n"
    "Label: <safe or not safe>\n"
    "Reason: <one short sentence explaining why>\n\n"
    "Text: {text}"
)


def build_prompt(text: str, prompt_config: Dict[str, Any] = None) -> List[Dict[str, str]]:
    prompt_config = prompt_config or {}
    system_prompt = prompt_config.get("system_prompt") or DEFAULT_SYSTEM_PROMPT
    user_template = prompt_config.get("user_template") or DEFAULT_USER_TEMPLATE
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_template.format(text=text)},
    ]
