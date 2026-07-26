import torch

from nano_sglang.attention import paged_attention, reference_attention


def test_paged_attention_matches_reference_after_gather():
    query = torch.randn(1, 2, 1, 4)
    key_blocks = torch.randn(2, 2, 2, 4)
    value_blocks = torch.randn(2, 2, 2, 4)
    block_ids = [0, 1]
    seq_len = 3

    key = key_blocks[block_ids].reshape(-1, 2, 4)[:seq_len].transpose(0, 1).unsqueeze(0)
    value = value_blocks[block_ids].reshape(-1, 2, 4)[:seq_len].transpose(0, 1).unsqueeze(0)

    expected = reference_attention(query, key, value, causal=False)
    actual = paged_attention(query, key_blocks, value_blocks, block_ids, seq_len, causal=False)

    torch.testing.assert_close(actual, expected)
