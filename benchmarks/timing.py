# SPDX-License-Identifier: Apache-2.0
"""Shared inputs, serial decoding and CUDA memory measurements."""

import hashlib
import time
from contextlib import nullcontext
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from reskv._io import git_commit, source_sha256, write_json

BENCHMARK_GPU = "NVIDIA RTX PRO 6000 Blackwell Server Edition"


def validate_gpu(parser):
    """Require the benchmark GPU before loading models or launching runs."""
    if not torch.cuda.is_available():
        parser.error(f"performance benchmarks require {BENCHMARK_GPU}; CUDA is unavailable")
    name = torch.cuda.get_device_name()
    if " ".join(name.split()).casefold() != BENCHMARK_GPU.casefold():
        parser.error(f"performance benchmarks require {BENCHMARK_GPU}; detected {name}")


def add_arguments(parser, *, batch_sizes=False):
    """Arguments shared by the single-run, Mustafar and comparison commands."""
    parser.add_argument("--model", required=True)
    if batch_sizes:
        parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    else:
        parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-length", type=int, default=8192)
    parser.add_argument("--new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)


def add_model_arguments(parser):
    parser.add_argument(
        "--sparsity", type=float, choices=[0.5, 0.75], default=0.75, help="Fraction of residuals discarded"
    )
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")


def validate_arguments(parser, args):
    sizes = getattr(args, "batch_sizes", None) or [args.batch_size]
    if min(*sizes, args.input_length, args.new_tokens, args.repeats) < 1 or args.warmup < 0:
        parser.error("batch size, lengths and repeats must be positive; warmup must be nonnegative")


def make_inputs(vocab_size, batch_size, input_length, seed, device="cuda"):
    """Generate identical token IDs across interpreters and GPU devices."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    low = min(128, vocab_size - 1)
    ids = torch.randint(low, vocab_size, (batch_size, input_length), generator=generator)
    digest = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
    return ids.to(device), digest


def cache_bytes(cache):
    """Count unique backing storages in layer and tuple caches."""
    storages = {}

    def visit(value):
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            storages[(value.device, storage.data_ptr())] = storage.nbytes()
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        elif is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                visit(getattr(value, field.name))
        elif hasattr(value, "layers"):
            visit(value.layers)
        elif value is not None:
            for name in ("keys", "values", "raw_keys", "compressed"):
                visit(getattr(value, name, None))

    visit(cache)
    return sum(storages.values())


def _attention_phase(model, implementation):
    """Select the dense model's attention for one phase; return the SDPA context."""
    if implementation == "eager":
        model.set_attn_implementation("eager")
        return nullcontext()
    model.set_attn_implementation("sdpa")
    return sdpa_kernel(SDPBackend.FLASH_ATTENTION)


def run(model, ids, new_tokens, *, dense=False, logits_to_keep=True):
    """One prefill and N-1 greedy continuation forwards, timed at phase boundaries."""
    if new_tokens < 1:
        raise ValueError("new_tokens must be positive")
    options = {"use_cache": True, **({"logits_to_keep": 1} if logits_to_keep else {})}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with _attention_phase(model, "flash") if dense else nullcontext():
        result = model(ids, **options)
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - start) * 1000
    prefill_peak = torch.cuda.max_memory_allocated()
    prefill_reserved = torch.cuda.max_memory_reserved()
    if result.logits.shape[-2] != 1:
        raise RuntimeError("benchmark requires final-position logits; check the model compatibility patch")
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with _attention_phase(model, "eager") if dense else nullcontext():
        for _ in range(new_tokens - 1):
            token = result.logits[:, -1].argmax(-1).unsqueeze(-1)
            result = model(token, past_key_values=result.past_key_values, **options)
    torch.cuda.synchronize()
    decode_ms = (time.perf_counter() - start) * 1000 if new_tokens > 1 else 0.0
    decode_peak = torch.cuda.max_memory_allocated()
    reserved = max(prefill_reserved, torch.cuda.max_memory_reserved())
    total_ms = prefill_ms + decode_ms
    return {
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "total_ms": total_ms,
        "tpot_ms": decode_ms / (new_tokens - 1) if new_tokens > 1 else None,
        "throughput_tps": ids.shape[0] * new_tokens / (total_ms / 1000),
        "decode_steps": new_tokens - 1,
        "prefill_peak_allocated_bytes": prefill_peak,
        "decode_peak_allocated_bytes": decode_peak,
        "peak_allocated_bytes": max(prefill_peak, decode_peak),
        "peak_reserved_bytes": reserved,
        "cache_bytes": cache_bytes(result.past_key_values),
        "finite_last_logits": bool(result.logits.isfinite().all()),
    }


def measure(model, ids, args, **options):
    """Warm up the measured shape, releasing caches between independent runs."""
    records = []
    with torch.inference_mode():
        for _ in range(args.warmup):
            run(model, ids, min(args.new_tokens, 8), **options)
        for _ in range(args.repeats):
            torch.cuda.empty_cache()
            record = run(model, ids, args.new_tokens, **options)
            if not record["finite_last_logits"]:
                raise RuntimeError("nonfinite logits in benchmark")
            records.append(record)
    return records


def report_metadata(args, ids_hash, transformers_version):
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "batch_size": args.batch_size,
        "input_length": args.input_length,
        "new_tokens": args.new_tokens,
        "decode_steps": args.new_tokens - 1,
        "seed": args.seed,
        "input_ids_sha256": ids_hash,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers_version,
        "git_commit": git_commit(),
        "source_sha256": source_sha256(),
    }


def write_report(path, report):
    print(write_json(path, report))
