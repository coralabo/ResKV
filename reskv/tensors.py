# SPDX-License-Identifier: Apache-2.0
"""One cache representation for both K and V, with cached native views."""

from dataclasses import dataclass, field

import torch

from .config import KernelPlan


@dataclass
class CompressedTensor:
    """Compressed K or V cache of shape [B, H, T, 128] for one layer.

    Attributes:
        centroids: FP16 cluster centroids, [B, H, C, 128].
        cluster_ids: Centroid index of each token, [B, H, T] (stored as int8).
        payload: Retained residual values: FP16 [B, H, T, nnz] for ``bits=16``,
            otherwise int32 words of signed quantized residuals.
        metadata: Residual positions: two int64 bitmaps per token for unstructured
            FP16, int32 packed 4:16 indices for structured layouts, or ``None``
            for quantized unstructured residuals, whose payload packs all 128
            coordinates with the discarded ones quantized as zero.
        scale: FP16 per-token quantization scale [B, H, T], or ``None`` for FP16.
        plan: Kernel plan that produced this tensor.
    """

    centroids: torch.Tensor
    cluster_ids: torch.Tensor
    payload: torch.Tensor
    metadata: torch.Tensor | None
    scale: torch.Tensor | None
    plan: KernelPlan
    _native: tuple = field(default=(), init=False, repr=False)

    def __post_init__(self):
        if self.centroids.ndim != 4 or self.centroids.shape[-1] != 128:
            raise ValueError("centroids must have shape [B,H,C,128]")
        if self.cluster_ids.ndim != 3 or self.cluster_ids.shape[:2] != self.centroids.shape[:2]:
            raise ValueError("cluster IDs must have shape [B,H,T]")
        if not 1 <= self.centroids.shape[2] <= 64:
            raise ValueError("centroid count must be between 1 and 64")
        if self.payload.ndim != 4 or self.payload.shape[:3] != self.cluster_ids.shape:
            raise ValueError("payload must have shape [B,H,T,words_or_values]")
        tensors = [self.centroids, self.cluster_ids, self.payload, self.metadata, self.scale]
        if any(t is not None and t.device != self.centroids.device for t in tensors):
            raise ValueError("all cache tensors must share a device")
        if self.centroids.dtype != torch.float16:
            raise ValueError("centroids must be FP16")
        expected_dtype = torch.float16 if self.plan.bits == 16 else torch.int32
        if self.payload.dtype != expected_dtype:
            raise ValueError("payload dtype does not match the compression plan")
        b, h, t = self.cluster_ids.shape
        if min(b, h, t) < 1:
            raise ValueError("cache dimensions must be nonempty")
        if self.cluster_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("cluster IDs must be integer tensors")
        words = (
            self.plan.nnz
            if self.plan.bits == 16
            else 4
            if self.plan.layout == "structured"
            else 4 * self.plan.bits
        )
        if self.payload.shape[-1] != words:
            raise ValueError("payload width does not match the compression plan")
        if self.plan.bits == 16 or self.plan.layout == "structured":
            words = 2 if self.plan.layout == "unstructured" or self.plan.bits == 4 else 4
            dtype = torch.int64 if self.plan.layout == "unstructured" else torch.int32
            if (
                self.metadata is None
                or self.metadata.shape != (b, h, t, words)
                or self.metadata.dtype != dtype
            ):
                raise ValueError("sparse metadata shape/dtype does not match the compression plan")
        elif self.metadata is not None:
            raise ValueError("quantized unstructured payloads do not use sparse metadata")
        if self.plan.bits != 16:
            if self.scale is None or self.scale.shape != (b, h, t) or self.scale.dtype != torch.float16:
                raise ValueError("quantized scales must be FP16 [B,H,T]")
        elif self.scale is not None:
            raise ValueError("FP16 sparse-only caches do not use quantization scales")
        self.cluster_ids = self.cluster_ids.to(torch.int8).contiguous()
        self._native = (
            self.centroids.reshape(b * h, -1, 128).contiguous(),
            self.cluster_ids.view(b * h, t),
            self.payload.reshape(b * h, t, -1).contiguous(),
            self.metadata.reshape(b * h, t, -1).contiguous() if self.metadata is not None else None,
            self.scale.reshape(b * h, t).contiguous() if self.scale is not None else None,
        )

    @property
    def seq_len(self) -> int:
        return self.cluster_ids.shape[-1]

    @property
    def tensors(self) -> tuple:
        """All stored tensors, including the flattened views passed to CUDA."""
        return (self.centroids, self.cluster_ids, self.payload, self.metadata, self.scale, *self._native)

    @property
    def nbytes(self) -> int:
        return _storage_bytes(self.tensors)

    def append(self, other: "CompressedTensor") -> "CompressedTensor":
        if self.plan != other.plan:
            raise ValueError("cannot concatenate different cache plans")

        def cat(a, b):
            return torch.cat((a, b), dim=2) if a is not None else None

        return CompressedTensor(
            other.centroids,
            cat(self.cluster_ids, other.cluster_ids),
            cat(self.payload, other.payload),
            cat(self.metadata, other.metadata),
            cat(self.scale, other.scale),
            self.plan,
        )

    def index_select(self, index: torch.Tensor) -> "CompressedTensor":
        def take(t):
            return t.index_select(0, index.to(t.device)) if t is not None else None

        return CompressedTensor(
            take(self.centroids),
            take(self.cluster_ids),
            take(self.payload),
            take(self.metadata),
            take(self.scale),
            self.plan,
        )


@dataclass
class CompressedKV:
    key: CompressedTensor
    value: CompressedTensor

    def __post_init__(self):
        if self.key.plan != self.value.plan or self.key.cluster_ids.shape != self.value.cluster_ids.shape:
            raise ValueError("K/V compression plans and token dimensions must agree")
        if self.key.centroids.device != self.value.centroids.device:
            raise ValueError("K/V caches must share a device")

    @property
    def seq_len(self) -> int:
        return self.key.seq_len

    @property
    def nbytes(self) -> int:
        return _storage_bytes((*self.key.tensors, *self.value.tensors))


def _storage_bytes(tensors) -> int:
    """Bytes in unique backing storages; views of one storage count once."""
    unique = {}
    for t in tensors:
        if isinstance(t, torch.Tensor):
            storage = t.untyped_storage()
            unique[(t.device, storage.data_ptr())] = storage.nbytes()
    return sum(unique.values())
