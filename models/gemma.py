# models/gemma.py
from models.hf_chat_model import HFChatModel

class GemmaModel(HFChatModel):
    supports_system_role = False   # Gemma's chat template rejects a system role
