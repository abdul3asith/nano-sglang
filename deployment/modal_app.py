import modal

from deployment.modal_image import hf_cache, image


app = modal.App("nano-sglang")


@app.function(
    image=image,
    gpu="T4",
    timeout=600,
    volumes={"/cache": hf_cache},
)
def generate_remote(prompt: str, max_new_tokens: int = 32) -> dict[str, object]:
    from nano_sglang import EngineConfig, NanoSGLangEngine

    engine = NanoSGLangEngine(
        EngineConfig(
            model_name="Qwen/Qwen2.5-0.5B",
            device="cuda",
            dtype="float16",
            cache_dir="/cache",
            max_new_tokens=max_new_tokens,
        )
    )
    result = engine.generate_with_metrics(prompt, max_new_tokens=max_new_tokens)
    hf_cache.commit()
    return result


@app.local_entrypoint()
def main(prompt: str = "The capital of France is", max_new_tokens: int = 32):
    result = generate_remote.remote(prompt, max_new_tokens=max_new_tokens)
    print(result["text"])
