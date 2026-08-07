import os
from typing import Any, Dict, List

from .base_model import BaseModel


class OpenAIModel(BaseModel):
    def load(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("Install with: pip install openai") from e

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=api_key)

    def generate(self, messages: List[Dict[str, str]], generation_config: Dict[str, Any]) -> str:
        max_tokens = generation_config.get("max_new_tokens", 60)
        do_sample = generation_config.get("do_sample", False)
        temperature = generation_config.get("temperature", 0.0) if do_sample else 0.0

        is_reasoning_model = any(tag in self.model_name.lower() for tag in ("o1", "o3", "thinking", "reasoning"))
        if is_reasoning_model:
            response = self.client.chat.completions.create(
                model=self.model_name, messages=messages,
                max_completion_tokens=max(max_tokens, 512),
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model_name, messages=messages,
                max_tokens=max_tokens, temperature=temperature,
            )
        return response.choices[0].message.content.strip()
