# gemma.py
from .hf_chat_model import HFChatModel
class GemmaModel(HFChatModel):
    supports_system_role = False
