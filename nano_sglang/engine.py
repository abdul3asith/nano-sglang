from __future__ import annotations

import torch

from nano_sglang.config import EngineConfig
from nano_sglang.model.loader import load_hf_model_and_tokenizer
from nano_sglang.model.model_runner import ModelRunner
from nano_sglang.sampling.sampler import Sampler
from nano_sglang.scheduler.request import GenerationRequest, RequestStatus
from nano_sglang.scheduler.scheduler import Scheduler
from nano_sglang.tokenizer.tokenizer_manager import TokenizerManager
from nano_sglang.utils.metrics import GenerationMetrics, Timer


class NanoSGLangEngine:
    def __init__(self, config: EngineConfig | None = None):
        self.config = config or EngineConfig()
        model, tokenizer, device = load_hf_model_and_tokenizer(self.config)
        self.device = device
        self.model_runner = ModelRunner(model)
        self.tokenizer_manager = TokenizerManager(tokenizer, device)
        self.sampler = Sampler(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            seed=self.config.seed,
        )

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        result = self.generate_with_metrics(prompt, max_new_tokens=max_new_tokens)
        return result["text"]

    def generate_with_metrics(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
    ) -> dict[str, object]:
        request = GenerationRequest(
            prompt=prompt,
            max_new_tokens=max_new_tokens or self.config.max_new_tokens,
        )
        metrics = self._run_single_request(request)
        return {
            "request_id": request.request_id,
            "prompt": prompt,
            "text": self.tokenizer_manager.decode(request.generated_token_ids),
            "generated_token_ids": request.generated_token_ids,
            "finish_reason": request.finish_reason,
            "metrics": metrics,
        }

    def generate_many(
        self,
        prompts: list[str],
        max_new_tokens: int | None = None,
    ) -> list[dict[str, object]]:
        scheduler = Scheduler(max_batch_size=self.config.max_batch_size)
        requests = [
            GenerationRequest(
                prompt=prompt,
                max_new_tokens=max_new_tokens or self.config.max_new_tokens,
            )
            for prompt in prompts
        ]
        for request in requests:
            scheduler.add_request(request)

        metrics_by_id: dict[str, GenerationMetrics] = {}

        while scheduler.has_work():
            prefill_batch = scheduler.admit()
            if prefill_batch is not None:
                for request in prefill_batch.requests:
                    metrics_by_id[request.request_id] = self._prefill_request(request)

            decode_batch = scheduler.decode_batch()
            if decode_batch is not None:
                for request in decode_batch.requests:
                    self._decode_request_once(request, metrics_by_id[request.request_id])

            scheduler.retire_finished()

        return [
            {
                "request_id": request.request_id,
                "prompt": request.prompt,
                "text": self.tokenizer_manager.decode(request.generated_token_ids),
                "generated_token_ids": request.generated_token_ids,
                "finish_reason": request.finish_reason,
                "metrics": metrics_by_id[request.request_id],
            }
            for request in requests
        ]

    def _run_single_request(self, request: GenerationRequest) -> GenerationMetrics:
        metrics = self._prefill_request(request)
        while not request.is_finished:
            self._decode_request_once(request, metrics)
        metrics.total_seconds = metrics.total_seconds or metrics.time_to_first_token or 0.0
        return metrics

    def _prefill_request(self, request: GenerationRequest) -> GenerationMetrics:
        timer = Timer()
        encoded = self.tokenizer_manager.encode(request.prompt)
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")

        request.prompt_token_ids = input_ids[0].tolist()
        logits, request.past_key_values = self.model_runner.prefill(input_ids, attention_mask)
        request.status = RequestStatus.DECODING

        next_token_id = self.sampler.sample(logits)
        metrics = GenerationMetrics(prompt_tokens=input_ids.shape[-1])
        metrics.time_to_first_token = timer.elapsed()
        self._accept_or_finish(request, next_token_id, metrics)
        metrics.total_seconds = timer.elapsed()
        return metrics

    def _decode_request_once(
        self,
        request: GenerationRequest,
        metrics: GenerationMetrics,
    ) -> None:
        if request.is_finished:
            return

        timer = Timer()
        last_token_id = request.generated_token_ids[-1]
        token = torch.tensor([[last_token_id]], device=self.device, dtype=torch.long)
        logits, request.past_key_values = self.model_runner.decode_one_token(
            token,
            request.past_key_values,
        )
        next_token_id = self.sampler.sample(logits)
        self._accept_or_finish(request, next_token_id, metrics)
        metrics.total_seconds += timer.elapsed()

    def _accept_or_finish(
        self,
        request: GenerationRequest,
        token_id: int,
        metrics: GenerationMetrics,
    ) -> None:
        eos_token_id = self.config.eos_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer_manager.eos_token_id

        if eos_token_id is not None and token_id == eos_token_id:
            request.mark_finished("eos")
            return

        request.append_token(token_id)
        metrics.generated_tokens += 1

        if request.generated_tokens >= request.max_new_tokens:
            request.mark_finished("length")
