# SPDX-License-Identifier: Apache-2.0
"""Transformers-compatible two-segment cache with bounded recent storage."""

import torch
from transformers.cache_utils import Cache, CacheLayerMixin

from .compression import compress
from .config import ResKVKernelConfig


class ResKVLayerCache(CacheLayerMixin):
    is_sliding = False

    def __init__(self, config: ResKVKernelConfig):
        super().__init__()
        self.config = config
        self.compressed = None
        self.raw_keys = None
        self.length = 0
        self.live = 0
        self.plan = None

    def lazy_initialization(self, key_states):
        b, h, _, d = key_states.shape
        shape = (b, h, self.config.recent_size + self.config.group_size, d)
        self.keys = torch.empty(shape, device=key_states.device, dtype=key_states.dtype)
        self.raw_keys = torch.empty_like(self.keys)
        self.values = torch.empty_like(self.keys)
        self.plan = self.config.resolve(b)
        self.is_initialized = True

    def update(self, key_states, value_states, cache_kwargs=None):
        raw = (cache_kwargs or {}).get("raw_keys")
        if raw is None:
            raise ValueError("ResKV cache update requires pre-RoPE raw_keys")
        q_len = key_states.shape[2]
        if not self.is_initialized:
            self.lazy_initialization(key_states)
        if self.length == 0:
            compressed_len = max(0, q_len - self.config.recent_size)
            compressed_len = compressed_len // self.config.group_size * self.config.group_size
            if compressed_len:
                self.compressed = compress(
                    raw[:, :, :compressed_len].contiguous(),
                    value_states[:, :, :compressed_len].contiguous(),
                    self.config,
                    plan=self.plan,
                )
            self.live = q_len - compressed_len
            self.keys[:, :, : self.live].copy_(key_states[:, :, compressed_len:])
            self.raw_keys[:, :, : self.live].copy_(raw[:, :, compressed_len:])
            self.values[:, :, : self.live].copy_(value_states[:, :, compressed_len:])
        else:
            if q_len != 1:
                raise ValueError(
                    "ResKV continuation supports one token per step; chunked prefill is unsupported"
                )
            self.keys[:, :, self.live : self.live + 1].copy_(key_states)
            self.raw_keys[:, :, self.live : self.live + 1].copy_(raw)
            self.values[:, :, self.live : self.live + 1].copy_(value_states)
            self.live += 1
            if self.live == self.keys.shape[2]:
                n = self.config.group_size
                self.compressed = compress(
                    self.raw_keys[:, :, :n].contiguous(),
                    self.values[:, :, :n].contiguous(),
                    self.config,
                    previous=self.compressed,
                    plan=self.plan,
                )
                remain = self.live - n
                for buf in (self.keys, self.raw_keys, self.values):
                    # clone avoids overlapping in-place copies.
                    buf[:, :, :remain].copy_(buf[:, :, n : self.live].clone())
                self.live = remain
        self.length += q_len
        return self.keys[:, :, : self.live], self.values[:, :, : self.live]

    def get_seq_length(self):
        return self.length

    def get_mask_sizes(self, cache_position):
        return self.length + cache_position.shape[0], 0

    def get_max_cache_shape(self):
        return -1

    def reset(self):
        self.compressed = None
        self.length = self.live = 0
        self.keys = self.values = self.raw_keys = None
        self.plan = None
        self.is_initialized = False

    def reorder_cache(self, beam_idx):
        if self.is_initialized:
            index = beam_idx.to(self.keys.device)
            self.keys = self.keys.index_select(0, index)
            self.raw_keys = self.raw_keys.index_select(0, index)
            self.values = self.values.index_select(0, index)
            if self.compressed is not None:
                self.compressed.key = self.compressed.key.index_select(index)
                self.compressed.value = self.compressed.value.index_select(index)


class ResKVCache(Cache):
    def __init__(self, config: ResKVKernelConfig, num_layers: int):
        super().__init__(layers=[ResKVLayerCache(config) for _ in range(num_layers)])

    def crop(self, max_length):
        raise NotImplementedError("ResKV does not support speculative cache rollback")

    def batch_repeat_interleave(self, repeats):
        for layer in self.layers:
            if layer.is_initialized:
                indices = torch.arange(layer.keys.shape[0], device=layer.keys.device).repeat_interleave(
                    repeats
                )
                layer.reorder_cache(indices)

    def batch_select_indices(self, indices):
        self.reorder_cache(indices)
