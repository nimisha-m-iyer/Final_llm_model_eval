"""
Wrapper for OpenAI GPT models, used via the API rather than local weights.
Implements the exact same BaseModel interface as every local model, so
run_pipeline.py and the evaluator never need to know the difference.

Requires OPENAI_API_KEY to be set as an environment variable before
load() runs (e.g. via a Kaggle Secret).
"""

import os
from typing import Any, Dict, List

from models.base_model import BaseModel


class OpenAIModel(BaseModel):
    def load(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "The 'openai' package is required for GPT models. "
                "Install it with: pip install openai"
            ) from e

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Set it (e.g. via a Kaggle Secret) "
                "before running a GPT model."
            )
        self.client = OpenAI(api_key=api_key)

    def generate(self, messages: List[Dict[str, str]], generation_config: Dict[str, Any]) -> str:
        max_tokens = generation_config.get("max_new_tokens", 8)
        do_sample = generation_config.get("do_sample", False)
        temperature = generation_config.get("temperature", 0.0) if do_sample else 0.0

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
