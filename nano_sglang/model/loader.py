import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nano_sglang.config import EngineConfig


def get_torch_dtype(dtype: str):
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def load_hf_model_and_tokenizer(config: EngineConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        cache_dir=config.cache_dir,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=get_torch_dtype(config.dtype),
        device_map=config.device,
        cache_dir=config.cache_dir,
    )
    model.eval()
    return model, tokenizer
