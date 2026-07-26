from nano_sglang.engine import NanoSGLangEngine


def gen(prompt: str, engine: NanoSGLangEngine, max_new_tokens: int | None = None) -> str:
    """Small placeholder frontend helper for later DSL work."""

    return engine.generate(prompt, max_new_tokens=max_new_tokens)


__all__ = ["gen"]
