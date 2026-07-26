import torch

from nano_sglang.memory import ContiguousKVCache, PagedKVCache


def test_contiguous_kv_cache_append_gather_free():
    cache = ContiguousKVCache()
    key = torch.ones(2, 4)
    value = torch.zeros(2, 4)

    cache.append("req", 0, key, value)
    gathered_key, gathered_value = cache.gather("req", 0)

    assert gathered_key.shape == (1, 2, 4)
    assert gathered_value.shape == (1, 2, 4)

    cache.free("req")
    assert cache.sequence_length("req") == 0


def test_paged_kv_cache_reuses_blocks():
    cache = PagedKVCache(
        num_layers=1,
        num_heads=2,
        head_dim=4,
        block_size=2,
        num_blocks=4,
        dtype=torch.float32,
        device="cpu",
    )

    cache.append("req", 0, torch.ones(2, 4), torch.zeros(2, 4))
    cache.append("req", 0, torch.ones(2, 4), torch.zeros(2, 4))
    cache.append("req", 0, torch.ones(2, 4), torch.zeros(2, 4))

    key, value = cache.gather("req", 0)

    assert key.shape == (3, 2, 4)
    assert value.shape == (3, 2, 4)
    assert cache.block_manager.used_blocks == 2

    cache.free("req")
    assert cache.block_manager.used_blocks == 0
