from __future__ import annotations

import argparse

from nano_sglang import EngineConfig, NanoSGLangEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Nano-SGLang locally.")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    engine = NanoSGLangEngine(
        EngineConfig(
            model_name=args.model,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
        )
    )
    print(engine.generate(args.prompt, max_new_tokens=args.max_new_tokens))


if __name__ == "__main__":
    main()
