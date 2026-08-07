# llama.py
from .hf_chat_model import HFChatModel
class LlamaModel(HFChatModel):
    supports_system_role = True
