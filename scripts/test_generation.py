from nano_sglang import EngineConfig, NanoSGLangEngine


def main() -> None:
    engine = NanoSGLangEngine(
        EngineConfig(
            model_name="Qwen/Qwen2.5-0.5B",
            device="auto",
            dtype="float16",
            max_new_tokens=8,
        )
    )
    result = engine.generate_with_metrics("The capital of France is", max_new_tokens=8)
    assert isinstance(result["text"], str)
    assert result["finish_reason"] in {"eos", "length"}
    print(result)


if __name__ == "__main__":
    main()
