from __future__ import annotations

import importlib

from nano_sglang.config import EngineConfig
from nano_sglang.engine import NanoSGLangEngine


def create_app(config: EngineConfig | None = None):
    """Create a tiny FastAPI app if FastAPI is installed.

    FastAPI is intentionally optional so the core learning runtime only needs
    torch and transformers.
    """

    fastapi = importlib.import_module("fastapi")
    pydantic = importlib.import_module("pydantic")

    class GenerateBody(pydantic.BaseModel):
        prompt: str
        max_new_tokens: int | None = None

    app = fastapi.FastAPI(title="Nano-SGLang")
    engine = NanoSGLangEngine(config or EngineConfig())

    @app.post("/generate")
    def generate(body: GenerateBody):
        return engine.generate_with_metrics(
            body.prompt,
            max_new_tokens=body.max_new_tokens,
        )

    return app
