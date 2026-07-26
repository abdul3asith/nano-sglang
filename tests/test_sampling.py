import torch

from nano_sglang.sampling import Sampler


def test_greedy_sampler_picks_argmax():
    logits = torch.tensor([[[0.1, 2.0, 0.3]]])
    sampler = Sampler(temperature=0.0)
    assert sampler.sample(logits) == 1
