from typing import Any, Dict

from .aya import AyaModel
from .base_model import BaseModel
from .gemma import GemmaModel
from .hf_chat_model import HFChatModel
from .llama import LlamaModel
from .openai_model import OpenAIModel
from .qwen import QwenModel

_REGISTRY = {
    "gemma": GemmaModel, "aya": AyaModel, "qwen": QwenModel,
    "llama": LlamaModel, "gpt": OpenAIModel,
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
