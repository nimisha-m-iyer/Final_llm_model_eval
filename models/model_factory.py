"""
Selects a model wrapper class purely from config["model"]["name"].

To add a new model family: if it needs no special handling beyond the
generic chat pattern, it already works via the HFChatModel fallback. If
it needs a quirk, add a small subclass in models/<name>.py and register
it below. evaluator.py and run_pipeline.py never need to change either way.
"""

from typing import Any, Dict

from models.aya import AyaModel
from models.base_model import BaseModel
from models.gemma import GemmaModel
from models.hf_chat_model import HFChatModel
from models.llama import LlamaModel
from models.openai_model import OpenAIModel
from models.qwen import QwenModel

_REGISTRY = {
    "gemma": GemmaModel,
    "aya": AyaModel,
    "qwen": QwenModel,
    "llama": LlamaModel,
    "gpt": OpenAIModel,
    # "sarvam": SarvamModel,
}


def load_model(config: Dict[str, Any]) -> BaseModel:
    model_name = config["model"]["name"].lower()
    model_cls = HFChatModel
    for key, cls in _REGISTRY.items():
        if key in model_name:
            model_cls = cls
            break
    model = model_cls(config)
    model.load()
    return model
