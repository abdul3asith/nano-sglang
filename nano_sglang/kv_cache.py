# modal_app.py
import modal

image = (
    modal.Image.debian_slim()
    .pip_install("torch")
    .add_local_python_source("kv_cache")  # mounts your local kv_cache.py
)

app = modal.App("nano-sglang-stage2")


@app.function(image=image, gpu="T4", timeout=300)
def run_paged_cache_demo():
    import torch
    from kv_cache import PagedKVCache

    num_layers, num_heads, head_dim = 4, 8, 64
    block_size, num_blocks = 16, 64

    cache = PagedKVCache(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
        num_blocks=num_blocks,
        dtype=torch.float16,
        device="cuda",
    )

    # simulate 3 sequences generating tokens, interleaved like real decode would
    for seq_id in ["req_a", "req_b", "req_c"]:
        cache.new_sequence(seq_id)

    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated() / 1e6

    lengths = {"req_a": 40, "req_b": 12, "req_c": 65}
    for seq_id, length in lengths.items():
        for _ in range(length):
            for layer in range(num_layers):
                k = torch.randn(num_heads, head_dim, dtype=torch.float16, device="cuda")
                v = torch.randn(num_heads, head_dim, dtype=torch.float16, device="cuda")
                cache.append(seq_id, layer, k, v)

    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated() / 1e6

    # correctness check: gathered length should match what we wrote
    for seq_id, length in lengths.items():
        k, v = cache.gather(seq_id, layer=0)
        assert k.shape[0] == length, f"{seq_id}: expected {length}, got {k.shape[0]}"

    blocks_used = num_blocks - len(cache.free_blocks)
    print(f"GPU memory for cache pool:   {mem_after:.2f} MB (allocated once, fixed)")
    print(f"blocks in use:               {blocks_used} / {num_blocks}")
    print(f"tokens written per sequence: {lengths}")
    print("gather() correctness: OK, shapes match seq_lens")

    # free one sequence, confirm blocks return to the pool
    freed = len(cache.block_tables["req_b"])
    cache.free("req_b")
    print(
        f"freed req_b: {freed} blocks returned, free list now {len(cache.free_blocks)}"
    )


@app.local_entrypoint()
def main():
    run_paged_cache_demo.remote()
