from nano_sglang import EngineConfig, NanoSGLangEngine


def main() -> None:
    engine = NanoSGLangEngine(EngineConfig())
    prompt = "The capital of France is"
    output = engine.generate(prompt, max_new_tokens=32)
    print(f"Prompt: {prompt}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
