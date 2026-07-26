from nano_sglang import EngineConfig, NanoSGLangEngine


def main() -> None:
    engine = NanoSGLangEngine(EngineConfig())
    print("Nano-SGLang chat. Ctrl-D to exit.")
    while True:
        try:
            prompt = input("\nuser> ")
        except EOFError:
            print()
            break

        if not prompt.strip():
            continue

        output = engine.generate(prompt, max_new_tokens=64)
        print(f"assistant> {output}")


if __name__ == "__main__":
    main()
