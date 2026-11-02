# Performance benchmarks

Use the [documented environment](../README.md#installation) and a local
Llama-3.1-8B-Instruct checkpoint. Performance commands require NVIDIA RTX PRO
6000 Blackwell Server Edition. Select the device with `CUDA_VISIBLE_DEVICES`
and run without other GPU workloads.

## Compare methods

The [README](../README.md#performance-experiments) runs ResKV, Mustafar and dense
at both 50% and 75% sparsity. Mustafar requires its [separate setup](mustafar.md).
To compare only ResKV and dense:

```bash
python -m benchmarks.compare --model /models/llama-3.1-8b-instruct \
  --methods reskv dense --batch-sizes 1 4 8 16 32 \
  --input-length 8192 --new-tokens 2048 --sparsity 0.75 \
  --warmup 1 --repeats 3 --output results/performance/sparse75
```

Use `--sparsity 0.5` for the other sparsity setting. Performance commands keep
FP16 residuals; residual quantization is selected in the accuracy evaluator.

## Run one method

```bash
python -m benchmarks.inference --model /models/llama-3.1-8b-instruct \
  --method reskv --batch-size 16 --input-length 8192 --new-tokens 2048 \
  --sparsity 0.75 --output results/performance/reskv75_b16.json

python -m benchmarks.inference --model /models/llama-3.1-8b-instruct \
  --method dense --batch-size 16 --input-length 8192 --new-tokens 2048 \
  --output results/performance/dense_b16.json
```

The dense model keeps a full KV cache. ResKV and Mustafar use FP16; dense also
accepts `--dtype bf16`.

## Timing and memory

All methods use `benchmarks/timing.py`, the same CPU input generator and seed,
one prefill and N−1 decode forwards for N output tokens. EOS does not shorten
the requested output. Each model returns only the last position's logits.

| Metric | Definition |
| --- | --- |
| `prefill_ms` | Prefill elapsed time, including ResKV clustering and compression |
| `decode_ms` | Elapsed time for N−1 greedy decode forwards, including GPU argmax |
| `tpot_ms` | `decode_ms / (N - 1)`; null when N=1 |
| `throughput_tps` | `batch_size * N / ((prefill_ms + decode_ms) / 1000)` |
| `peak_allocated_bytes` | Maximum PyTorch allocated memory across prefill and decode |
| `cache_bytes` | Unique backing storage of the final KV cache |

GPU synchronization surrounds each timed phase. Warmup uses the requested input
shape and up to eight output tokens. Independent timed repeats use fresh caches
and release unused allocator blocks. Memory includes weights and temporary
buffers; divide bytes by `1024**3` for GiB. Reserved memory is recorded separately.

ResKV and patched Mustafar reuse the MLP gate buffer during inference. For large
batches, set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` consistently for
all methods to reduce allocator fragmentation.

The launcher writes one JSON and log per method/batch plus `summary.json`.
Summaries use median latency/throughput and maximum allocated memory over the
repeats. Input hashes, dtype, environment, revision and decode counts must agree.
Failed runs remain in the error list and return a nonzero exit code.

## Kernel measurements

```bash
python -m benchmarks.kernels --batch-sizes 1 16 32 --seq-len 8192 \
  --sparsities 0.5 0.75 --output results/performance/kernels.json
```

This measures the same sparse FP16 implementations as the model interface.
Compression uses precomputed clusters, so this measurement
excludes K-means. Each case first checks attention against the independent FP32
reference, then reports compression/decode latency and compressed storage.
