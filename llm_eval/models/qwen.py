# qwen.py
from .hf_chat_model import HFChatModel
class QwenModel(HFChatModel):
    supports_system_role = True
