# stage 1 - loading a simple model and running simple prompt through it in the most naive way possible.
# here i am using modal for my GPUs

import modal

image = modal.Image.debian_slim().pip_install("torch", "transformers", "accelerate")

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

app = modal.App("nano-sglang")


@app.function(
    image=image,
    gpu="T4",
    volumes={"/cache": hf_cache},
    timeout=600,
)
def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/cache")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="cuda",
        cache_dir="/cache",
    )
    model.eval()

    inputs = tokenizer("The capital of France is", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inputs)

    print(tokenizer.decode(out.logits[0, -1].argmax()))
    hf_cache.commit()


@app.local_entrypoint()
def main():
    load_model.remote()
