"""
Routes config["model"]["name"] to the correct wrapper. Add a new model
family by adding one subclass + one line here. Anything unregistered
falls back to the generic local HFChatModel.
"""

from typing import Any, Dict

from models.aya import AyaModel
from models.base_model import BaseModel
from models.gemma import GemmaModel
from models.hf_chat_model import HFChatModel
from models.openai_model import OpenAIModel
from models.qwen import QwenModel

_REGISTRY = {
    "gemma": GemmaModel,
    "aya": AyaModel,
    "qwen": QwenModel,
    "gpt": OpenAIModel,
    # "llama": LlamaModel,
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
