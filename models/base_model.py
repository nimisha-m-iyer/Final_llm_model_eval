"""
Every model wrapper — local (Gemma/Aya/Qwen) or API-based (GPT) —
implements exactly this interface: load() then generate(messages, config).
Nothing else in the pipeline ever needs to know which kind it's talking to.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseModel(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config["model"]["name"]
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], generation_config: Dict[str, Any]) -> str:
        raise NotImplementedError
