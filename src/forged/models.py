"""Model loading shared by every script: dtype handling across transformers 4 and 5, LoRA adapters, 4-bit."""

from __future__ import annotations

import torch


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(
    model_id: str,
    dtype: str = "float16",
    device_map: str | None = "auto",
    adapter: str | None = None,
    load_4bit: bool = False,
):
    """Load a causal LM, optionally with a LoRA adapter or in 4-bit (bitsandbytes NF4)."""
    from transformers import AutoModelForCausalLM

    torch_dtype = getattr(torch, dtype)
    kwargs: dict = {"device_map": device_map}
    if load_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch_dtype
        )
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype, **kwargs)
    except TypeError:  # transformers < 4.56 only knows torch_dtype
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, **kwargs)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    return model.eval()
