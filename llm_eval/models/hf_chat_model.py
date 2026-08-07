"""
Generic wrapper for local Hugging Face chat models. Implements true
batched generation: multiple prompts are tokenized together with
left-padding, generated in a single model.generate() call, then each
output is sliced starting from the shared post-padding prompt length.
"""

import signal
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .base_model import BaseModel


class GenerationTimeout(Exception):
    pass


class HFChatModel(BaseModel):
    supports_system_role: bool = True

    def load(self) -> None:
        model_cfg = self.config["model"]
        dtype_str = model_cfg.get("torch_dtype", "bfloat16")
        dtype = getattr(torch, dtype_str)
        trust_remote_code = model_cfg.get("trust_remote_code", False)
        load_in_4bit = model_cfg.get("load_in_4bit", False)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=trust_remote_code)

        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=None if load_in_4bit else dtype,
            quantization_config=quantization_config,
            device_map=model_cfg.get("device_map", "auto"),
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.generation_timeout_seconds = self.config.get("generation_timeout_seconds", 60)

    def _build_inputs(self, messages: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        if not self.supports_system_role:
            messages = self._merge_system_into_user(messages)
        try:
            encoded = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            )
        except Exception:
            list_format = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in messages]
            encoded = self.tokenizer.apply_chat_template(
                list_format, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            )
        return {k: v.to(self.model.device) for k, v in encoded.items()}

    @staticmethod
    def _merge_system_into_user(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        system_chunks = [m["content"] for m in messages if m["role"] == "system"]
        rest = [m for m in messages if m["role"] != "system"]
        if not system_chunks or not rest:
            return rest or messages
        if rest[0]["role"] == "user":
            rest[0] = {"role": "user", "content": "\n\n".join(system_chunks + [rest[0]["content"]])}
        return rest

    def _generate_with_timeout(self, encoded, gen_kwargs):
        def _handler(signum, frame):
            raise GenerationTimeout()
        has_alarm = hasattr(signal, "SIGALRM")
        if has_alarm:
            old_handler = signal.signal(signal.SIGALRM, _handler)
            signal.alarm(self.generation_timeout_seconds)
        try:
            return self.model.generate(**encoded, **gen_kwargs)
        except GenerationTimeout:
            print(f"[hf_chat_model] WARNING: generation exceeded {self.generation_timeout_seconds}s — skipped.")
            return None
        finally:
            if has_alarm:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

    def _gen_kwargs(self, generation_config):
        gen_kwargs = dict(generation_config)
        gen_kwargs.pop("max_length", None)
        gen_kwargs.setdefault("max_new_tokens", 60)
        gen_kwargs.setdefault("do_sample", False)
        gen_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        return gen_kwargs

    @torch.inference_mode()
    def generate(self, messages: List[Dict[str, str]], generation_config: Dict[str, Any]) -> str:
        encoded = self._build_inputs(messages)
        prompt_len = encoded["input_ids"].shape[-1]
        gen_kwargs = self._gen_kwargs(generation_config)

        output_ids = self._generate_with_timeout(encoded, gen_kwargs)
        if output_ids is None:
            return ""
        new_tokens = output_ids[0][prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    @torch.inference_mode()
    def generate_batch(self, list_of_messages: List[List[Dict[str, str]]], generation_config: Dict[str, Any]) -> List[str]:
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        try:
            prompt_texts = []
            for messages in list_of_messages:
                if not self.supports_system_role:
                    messages = self._merge_system_into_user(messages)
                text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                prompt_texts.append(text)

            encoded = self.tokenizer(prompt_texts, return_tensors="pt", padding=True).to(self.model.device)
            prompt_len = encoded["input_ids"].shape[1]
            gen_kwargs = self._gen_kwargs(generation_config)

            output_ids = self._generate_with_timeout(encoded, gen_kwargs)
            if output_ids is None:
                return ["" for _ in list_of_messages]

            results = []
            for i in range(len(prompt_texts)):
                new_tokens = output_ids[i][prompt_len:]
                results.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
            return results
        finally:
            self.tokenizer.padding_side = original_padding_side
