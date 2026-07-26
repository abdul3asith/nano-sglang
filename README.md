# nano-sglang

A tiny LLM inference engine, built from scratch to understand how the fast ones actually work.

This is not a production serving stack and it is not trying to be. It is a from-scratch reimplementation of the ideas behind [SGLang](https://github.com/sgl-project/sglang), stripped down to the parts that teach you something: the scheduler, the KV cache, prefix reuse, and the memory tricks that make LLM serving fast.

I am building it in the open, one piece at a time, and writing down what I learn (including the parts that break). If you have ever used vLLM or SGLang and wondered what is happening between "send prompt" and "get tokens," this repo is my attempt to find out by building it.

> Following the build log on X: [@basithtwts](https://x.com/basithtwts)

## Why this exists

Reading a paper and nodding along is easy. Making it run, and match the numbers, is where the understanding actually happens. The goal here is not a new framework. It is to earn the intuition for how a modern inference engine is put together, so that contributing to the real projects stops feeling like magic.

If that is your goal too, star it and follow along. Every stage ships with an explanation and real numbers.

## What you will learn from the code

- Why prefill is compute bound and decode is memory bandwidth bound, and why that one fact drives every optimization that follows
- How paged memory kills KV cache fragmentation (the PagedAttention idea)
- How a radix tree lets you reuse KV cache across requests (the RadixAttention idea)
- How continuous batching keeps the GPU busy instead of waiting on the slowest request
- How a small frontend language turns ordinary Python into requests the runtime can optimize

## The build, stage by stage

Each stage is a working checkpoint with its own example and benchmark. This is the roadmap and the current progress.

- [ ] Stage 1: single request forward pass, naive KV cache, get tokens out
- [ ] Stage 2: paged KV cache, block tables, measure the fragmentation win
- [ ] Stage 3: continuous batching, iteration level scheduling
- [ ] Stage 4: RadixAttention, radix tree prefix reuse, cache aware scheduling
- [ ] Stage 5: a small frontend DSL (gen, select, fork) on top

Watch this list fill in. Each checkbox links to the commit and the write up when it lands.

## Quickstart

```bash
git clone https://github.com/abdul3asith/nano-sglang.git
cd nano-sglang
pip install -e .

# run the first example against a small model
python examples/single.py
```

You need a GPU. A single small one (T4, L4, or A10) is plenty for a 0.5B to 1B model. If you do not have one, `modal_app.py` runs the examples and benchmarks on serverless GPU so you only pay while they execute.

```bash
modal run modal_app.py
```

## How it is put together

```
nano_sglang/
  engine.py        ties the runtime together
  config.py        runtime configuration
  tokenizer/       tokenization and detokenization
  scheduler/       request lifecycle and naive scheduling
  model/           Hugging Face loading and forward execution
  memory/          contiguous and paged KV-cache learning structures
  attention/       reference attention and naive paged attention
  sampling/        logits processing and token sampling
  server/          optional API wrapper
  utils/           logging and metrics helpers
deployment/        Modal image and remote GPU entry point
scripts/           local runs and benchmarks
examples/          small example programs
tests/             unit tests for runtime pieces
```

The current request path is intentionally naive: tokenize the prompt, run prefill once through a Hugging Face model, decode one token at a time with Hugging Face `past_key_values`, sample the next token, and detokenize the result. The scheduler and paged-cache code are present as learning components, but optimized paged attention is not wired into model execution yet.

## Benchmarks

Numbers land as each stage does. The harness in `bench/` tracks throughput (tokens per second), time to first token, inter token latency, and cache hit rate, so every change is backed by a measurement rather than a vibe.

| stage | throughput | notes |
|-------|-----------|-------|
| 1 naive | coming | baseline, on purpose slow |
| 2 paged | coming | |
| 3 batching | coming | |
| 4 radix | coming | |

## Following along

I post the build logs, benchmarks, diagrams, and the bugs on X: [@basithtwts](https://x.com/basithtwts). If you want to see the whole thing come together in real time, that is the place.

Questions, corrections, and "you got this wrong" issues are all welcome. This is a learning project, so being told where I am wrong is the point.

## References

Standing on the shoulders of the papers that make this possible.

- SGLang: Efficient Execution of Structured Language Model Programs (Zheng et al., NeurIPS 2024)
- Efficient Memory Management for Large Language Model Serving with PagedAttention (Kwon et al.)
- Orca: A Distributed Serving System for Transformer-Based Generative Models (Yu et al.)
- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al.)

## License

MIT. Learn from it, fork it, build your own.
