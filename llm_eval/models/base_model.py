"""
Every model wrapper implements load(), generate(), and (optionally)
generate_batch(). generate_batch() has a safe default: if a subclass
doesn't override it, records are processed one at a time through
generate(). This means every model works correctly in batch mode from
day one, even before true batched generation is implemented for it.
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

    def generate_batch(self, list_of_messages: List[List[Dict[str, str]]], generation_config: Dict[str, Any]) -> List[str]:
        return [self.generate(m, generation_config) for m in list_of_messages]
