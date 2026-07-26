from dataclasses import dataclass


@dataclass
class EngineConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B"
    device: str = "auto"
    dtype: str = "float16"
    cache_dir: str | None = None
    max_new_tokens: int = 32
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    seed: int | None = None
    max_batch_size: int = 8
    eos_token_id: int | None = None
    pad_token_id: int | None = None
