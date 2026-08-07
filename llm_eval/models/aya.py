# aya.py
from .hf_chat_model import HFChatModel
class AyaModel(HFChatModel):
    supports_system_role = True
