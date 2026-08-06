"""
Llama 3.1 (Meta) instruct family — e.g. meta-llama/Llama-3.1-8B-Instruct.

Source: verified directly against Hugging Face's model card for
meta-llama/Llama-3.1-8B-Instruct, which documents standard usage via
AutoModelForCausalLM + AutoTokenizer + apply_chat_template(), identical
in shape to the pattern already used by HFChatModel for every other
model in this framework.

Llama 3.1's chat template natively supports a system role (confirmed by
the model card's own example messages and its raw special-token format:
<|start_header_id|>system<|end_header_id|> ... <|start_header_id|>user
<|end_header_id|> ...), so no role-merging override is needed, unlike
Gemma. No "thinking mode" exists for this model family either, unlike
Qwen3 — this is a standard, direct-answer instruct model.

NOTE: meta-llama/Llama-3.1-8B-Instruct is a GATED repository. You must
visit the model page on huggingface.co while logged in and accept
Meta's license/community agreement before your HF_TOKEN can download it.
"""

from models.hf_chat_model import HFChatModel


class LlamaModel(HFChatModel):
    supports_system_role = True
