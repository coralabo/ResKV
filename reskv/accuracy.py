# SPDX-License-Identifier: Apache-2.0
"""ResKV compression-error simulation with dense KV reconstruction in PyTorch.

Quantizers are stage-specific. Prefill centroids are rounded to BF16 and retained
for decode. Attention uses reconstructed dense KV tensors.
"""

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache, CacheLayerMixin

from .config import ResKVConfig


def sparsify(residual, sparsity, block_size=None):
    """Keep the largest residual coordinates by magnitude, independently per token."""
    if sparsity == 0:
        return residual
    width = residual.shape[-1]
    block_size = block_size or width
    if width % block_size:
        complete = width // block_size * block_size
        return torch.cat(
            (
                sparsify(residual[..., :complete], sparsity, block_size),
                sparsify(residual[..., complete:], sparsity),
            ),
            dim=-1,
        )
    blocks = residual.reshape(*residual.shape[:-1], width // block_size, block_size)
    keep = max(1, int(block_size * (1 - sparsity)))
    indices = blocks.abs().topk(keep, dim=-1).indices
    # Multiplication preserves signed zeros in masked coordinates.
    mask = torch.zeros_like(blocks, dtype=torch.bool).scatter_(-1, indices, True)
    return (blocks * mask).reshape_as(residual)


def quantize_dequantize(residual, bits, stage):
    """Per-token affine prefill / symmetric decode quantization."""
    if bits == 16:
        return residual
    if bits not in (2, 4, 8) or stage not in ("prefill", "decode"):
        raise ValueError("invalid accuracy quantization bits or stage")
    lower, upper = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    if stage == "prefill":
        low = residual.amin(-1, keepdim=True)
        delta = residual.amax(-1, keepdim=True) - low
        valid = delta > 0
        scale = torch.where(valid, delta / (upper - lower), torch.ones_like(delta))
        zero = lower - (low / scale).round()
        restored = ((residual / scale + zero).round().clamp(lower, upper) - zero) * scale
    else:
        maximum = residual.abs().amax(-1, keepdim=True)
        valid = maximum > 1e-8
        scale = torch.where(valid, maximum / upper, torch.ones_like(maximum))
        restored = (residual / scale).round().clamp(lower, upper) * scale
    return torch.where(valid, restored, residual)


def _check_tensor(tensor):
    if tensor.ndim != 4 or tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("accuracy compression requires FP16/BF16/FP32 [batch, heads, tokens, dim] tensors")
    if tensor.shape[-1] < 1 or tensor.shape[0] < 1 or tensor.shape[1] < 1:
        raise ValueError("batch, heads and head dimension must be positive")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("accuracy compression input contains nonfinite values")


def _reconstruct(data, centroids, assignments, config, stage):
    if config.layout == "structured" and data.shape[-1] % 16:
        raise ValueError("4:16 sparsity requires head dimensions divisible by 16")
    nearest = centroids.gather(-2, assignments.unsqueeze(-1).expand(*assignments.shape, data.shape[-1]))
    residual = data - nearest
    block = 16 if config.layout == "structured" else data.shape[-1]
    sparse = sparsify(residual, config.sparsity, block)
    return quantize_dequantize(sparse, config.bits, stage) + nearest


def compress_prefill(tensor, config: ResKVConfig):
    """Global Lloyd clustering per KV head; return dense reconstruction and centroids."""
    _check_tensor(tensor)
    b, h, n, d = tensor.shape
    if n == 0:
        return tensor.clone(), tensor.new_empty(b, h, 0, d)
    data = tensor.reshape(b * h, n, d).float()
    # Initialize contiguous clusters on a sequence padded to a multiple of 32.
    padded = (n + 31) // 32 * 32
    data = F.pad(data, (0, 0, 0, padded - n))
    valid = (torch.arange(padded, device=tensor.device) < n).view(1, padded, 1)
    weights = valid.expand(b * h, -1, -1).to(data.dtype)
    count = min(config.num_clusters, n)
    initial = torch.arange(padded, device=tensor.device) * count // padded
    assignments = initial.unsqueeze(0).expand(b * h, -1)

    def update_centroids(ids, previous=None):
        one_hot = F.one_hot(ids, num_classes=count).to(data.dtype).transpose(1, 2)
        sums = one_hot @ (data * weights)
        counts = one_hot @ weights
        updated = sums / counts.clamp_min(torch.finfo(data.dtype).eps)
        return torch.where(counts > 0, updated, torch.zeros_like(updated) if previous is None else previous)

    centroids = update_centroids(assignments)
    for _ in range(config.kmeans_iterations):
        distances = torch.cdist(data, centroids, p=2)
        distances = distances.masked_fill(~valid, torch.finfo(data.dtype).max)
        assignments = distances.argmin(-1)
        centroids = update_centroids(assignments, centroids)
    centroids = centroids.to(torch.bfloat16).float()
    restored = _reconstruct(data, centroids, assignments, config, "prefill")[:, :n]
    if config.sparsity == 0 and config.bits == 16:
        restored = tensor.reshape(b * h, n, d).clone()
    return restored.reshape(b, h, n, d).to(tensor.dtype), centroids.reshape(b, h, count, d).to(torch.bfloat16)


def compress_decode(tensor, centroids, config: ResKVConfig):
    """Append one group mean, assign to the accumulated table and reconstruct once."""
    _check_tensor(tensor)
    if tensor.shape[-2] == 0:
        return tensor.clone(), centroids
    data = tensor.float()
    mean = data.mean(-2, keepdim=True).to(torch.bfloat16).float()
    updated = torch.cat((centroids.float(), mean), dim=-2) if centroids.numel() else mean
    assignments = torch.cdist(data, updated, p=2).argmin(-1)
    restored = _reconstruct(data, updated, assignments, config, "decode")
    if config.sparsity == 0 and config.bits == 16:
        restored = tensor.clone()
    return restored.to(tensor.dtype), updated.to(torch.bfloat16)


class ResKVAccuracyLayerCache(CacheLayerMixin):
    """Full-length pre-RoPE reconstructed KV with protected sink and recent tokens."""

    is_sliding = False

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.compressed_end = config.sink_tokens
        self.key_centroids = self.value_centroids = None

    def lazy_initialization(self, key_states):
        self.device = key_states.device
        self.is_initialized = True

    def update(self, key_states, value_states, cache_kwargs=None):
        if key_states.shape != value_states.shape:
            raise ValueError("key/value shapes must match")
        if key_states.dtype != value_states.dtype or key_states.device != value_states.device:
            raise ValueError("key/value dtype and device must match")
        prefill = self.get_seq_length() == 0
        if not prefill and key_states.shape[-2] != 1:
            raise ValueError("accuracy cache supports a single prefill followed by single-token decode")
        if prefill:
            self.lazy_initialization(key_states)
            self.keys, self.values = key_states.clone(), value_states.clone()
        else:
            self.keys = torch.cat((self.keys, key_states), dim=-2)
            self.values = torch.cat((self.values, value_states), dim=-2)
        cfg = self.config
        stop = self.get_seq_length() - cfg.recent_size
        if prefill:
            length = max(0, stop - cfg.sink_tokens) // cfg.group_size * cfg.group_size
            self.compressed_end = cfg.sink_tokens + length
            if length:
                region = slice(cfg.sink_tokens, self.compressed_end)
                self.keys[:, :, region], kc = compress_prefill(self.keys[:, :, region], cfg)
                self.values[:, :, region], vc = compress_prefill(self.values[:, :, region], cfg)
                self.key_centroids, self.value_centroids = kc, vc
            # Return unmodified prefill tensors, never aliases mutated by compression.
            return key_states, value_states
        while self.compressed_end + cfg.group_size <= stop:
            region = slice(self.compressed_end, self.compressed_end + cfg.group_size)
            if self.key_centroids is None:
                b, h, _, d = key_states.shape
                self.key_centroids = key_states.new_empty(b, h, 0, d)
                self.value_centroids = value_states.new_empty(b, h, 0, d)
            self.keys[:, :, region], self.key_centroids = compress_decode(
                self.keys[:, :, region], self.key_centroids, cfg
            )
            self.values[:, :, region], self.value_centroids = compress_decode(
                self.values[:, :, region], self.value_centroids, cfg
            )
            self.compressed_end += cfg.group_size
        return self.keys, self.values

    def get_seq_length(self):
        return 0 if self.keys is None else self.keys.shape[-2]

    def get_mask_sizes(self, cache_position):
        return self.get_seq_length() + cache_position.shape[0], 0

    def get_max_cache_shape(self):
        return -1

    def reset(self):
        self.keys = self.values = self.key_centroids = self.value_centroids = None
        self.compressed_end = self.config.sink_tokens
        self.is_initialized = False

    def reorder_cache(self, beam_idx):
        super().reorder_cache(beam_idx)
        for name in ("key_centroids", "value_centroids"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.index_select(0, beam_idx.to(value.device)))


class ResKVAccuracyCache(Cache):
    def __init__(self, config, num_layers):
        super().__init__(layers=[ResKVAccuracyLayerCache(config) for _ in range(num_layers)])

    def crop(self, max_length):
        raise NotImplementedError("accuracy compression does not support speculative cache rollback")

    def batch_repeat_interleave(self, repeats):
        for layer in self.layers:
            if layer.is_initialized:
                index = torch.arange(layer.keys.shape[0], device=layer.keys.device).repeat_interleave(repeats)
                layer.reorder_cache(index)

    def batch_select_indices(self, indices):
        self.reorder_cache(indices)
