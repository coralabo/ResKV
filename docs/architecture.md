# API guide

## Configuration

```python
from reskv import ResKVConfig, ResKVKernelConfig

accuracy = ResKVConfig(sparsity=0.75, bits=4)
kernel = ResKVKernelConfig(sparsity=0.75)
```

| Interface | Configuration | Model argument |
| --- | --- | --- |
| PyTorch accuracy evaluation | `ResKVConfig` | `reskv_config` |
| CUDA inference | `ResKVKernelConfig` | `kernel_config` |

Both configurations use `sparsity` for the fraction of residual coordinates
discarded, `num_clusters`, `recent_size`, `group_size` and `kmeans_iterations`.
Accuracy additionally protects `sink_tokens`. Accuracy defaults to 15 K-means
iterations; CUDA defaults to 10. Both use 32 clusters and a recent window of 32.
Accuracy residuals use `bits=16` for sparsity only, or `4`/`2` for quantization.

The accuracy path reconstructs dense KV in PyTorch. The CUDA path stores
compressed payloads and uses fused attention. `resolve(batch_size)` selects its
kernel plan automatically; benchmark results include that plan.

## Models

```python
import torch
from reskv import ResKVForCausalLM, ResKVKernelConfig

model = ResKVForCausalLM.from_pretrained(
    "/models/llama-3.1-8b-instruct",
    kernel_config=ResKVKernelConfig(sparsity=0.75),
    dtype=torch.float16,
).cuda().eval()
```

Use `ResKVLlamaAccuracyForCausalLM` or `ResKVQwen3AccuracyForCausalLM` with
`reskv_config` for accuracy evaluation. Models require inference mode and
unpadded inputs. CUDA supports FP16 Llama with 128-dimensional heads and query/KV
head ratios 1, 4 or 8. Accuracy supports full-attention Llama/Qwen3 on CPU or CUDA.
Checkpoint configuration preserves the selected compression settings.

## Compression and attention

```python
import torch
from reskv import ResKVKernelConfig, attention, compress

k = torch.randn(1, 8, 1024, 128, device="cuda", dtype=torch.float16)
v = torch.randn_like(k)
q = torch.randn(1, 32, 1, 128, device="cuda", dtype=torch.float16)
cache = compress(k, v, ResKVKernelConfig(sparsity=0.75))
output = attention(q, cache, position=1024, rope_theta=500000.0)
```

K and Q are supplied before RoPE; V is unrotated. Tensors use
`[batch, heads, tokens, 128]`. `inv_freq` accepts the checkpoint's FP32 RoPE
frequencies on the same device. `recent=(rotated_q, rotated_k, v)` includes the
recent dense window. `compress(previous=cache)` appends a group while sharing
centroid tables. Treat cache tensors as read-only because native views may
share storage. `cache.nbytes` reports persistent compressed storage.

## Source layout

| Path | Responsibility |
| --- | --- |
| `reskv/accuracy.py`, `reskv/models/accuracy.py` | Accuracy reconstruction and Llama/Qwen3 integration |
| `reskv/compression.py`, `reskv/tensors.py` | CUDA compression and compressed tensors |
| `reskv/_attention.py`, `reskv/cache.py` | Fused decode and incremental cache |
| `reskv/models/llama.py` | CUDA Llama integration |
| `reskv/config.py`, `reskv/csrc/` | Configuration and native kernels |
| `evaluation/` | Task inputs, generation, scoring and Mustafar accuracy adapter |
| `benchmarks/` | Shared timing and ResKV/Mustafar/dense performance commands |
| `tests/` | Numerical and interface checks |

CUDA kernels compile on first use into `~/.cache/reskv`. `RESKV_CACHE_DIR` changes
the cache directory; `CUDA_HOME` selects the toolkit and `MAX_JOBS` limits build
concurrency.
