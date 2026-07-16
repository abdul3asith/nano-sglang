from dataclasses import dataclass


@dataclass
class EngineConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B"
    device: str = "cuda"
    dtype: str = "float16"
    cache_dir: str | None = None
    max_new_tokens: int = 32
    temperature: float = 0.0
    top_p: float = 1.0
