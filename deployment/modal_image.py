import modal


image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers", "safetensors")
    .add_local_python_source("nano_sglang")
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
