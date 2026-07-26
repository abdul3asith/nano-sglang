from __future__ import annotations

import argparse
from time import perf_counter

from nano_sglang import EngineConfig, NanoSGLangEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Naive throughput benchmark.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    prompts = [f"Write one short fact about number {i}:" for i in range(args.num_prompts)]
    engine = NanoSGLangEngine(
        EngineConfig(
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
        )
    )

    start = perf_counter()
    results = engine.generate_many(prompts, max_new_tokens=args.max_new_tokens)
    elapsed = perf_counter() - start
    tokens = sum(len(result["generated_token_ids"]) for result in results)

    print(f"prompts: {len(prompts)}")
    print(f"generated tokens: {tokens}")
    print(f"seconds: {elapsed:.3f}")
    print(f"tokens/sec: {tokens / elapsed if elapsed else 0:.2f}")


if __name__ == "__main__":
    main()
