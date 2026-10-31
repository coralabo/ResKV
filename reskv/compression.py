# SPDX-License-Identifier: Apache-2.0
"""Clustering and residual compression through one K/V API."""

import torch
import torch.nn.functional as F

from ._extension import load_extension
from .config import KernelPlan, ResKVKernelConfig
from .tensors import CompressedKV, CompressedTensor


def _validate(x: torch.Tensor):
    if x.ndim != 4 or x.shape[-1] != 128 or not x.is_cuda or x.dtype != torch.float16:
        raise ValueError("expected FP16 CUDA tensor [batch, kv_heads, tokens, 128]")
    if min(x.shape[:3]) < 1:
        raise ValueError("batch, head and token dimensions must be nonempty")


def cluster(x: torch.Tensor, count: int = 32, iterations: int = 10, *, value: bool = False):
    """Lloyd clustering with FP32 K and FP16 V.

    V uses tensor-core distance/aggregation and zeroes empty centroids; K
    retains empty centroids.
    """
    b, h, t, d = x.shape
    count = min(count, t) if value else count
    data = x.reshape(b * h, t, d).half() if value else x.reshape(b * h, t, d).float()
    if t < count:
        # Initialize additional centers for short inputs.
        indices = torch.randint(t, (count - t,), device=x.device)
        centers = torch.cat(
            (data.clone(), data[:, indices] + torch.randn_like(data[:, indices]) * 0.01), dim=1
        )
    else:
        indices = torch.linspace(0, t - 1, count, device=x.device).long()
        centers = data[:, indices].clone()
    for _ in range(iterations):
        mode = "use_mm_for_euclid_dist" if value else "use_mm_for_euclid_dist_if_necessary"
        ids = torch.cdist(data, centers, compute_mode=mode).argmin(-1)
        one_hot = F.one_hot(ids, count).to(data.dtype)
        counts = one_hot.sum(1, dtype=torch.float32).unsqueeze(-1)
        # Cluster sums may exceed FP16 range while their means remain representable.
        sums = (
            torch.bmm(one_hot.transpose(1, 2), data, out_dtype=torch.float32)
            if value
            else torch.bmm(one_hot.transpose(1, 2), data)
        )
        centers = (
            (sums / counts.clamp_min(1)).to(data.dtype)
            if value
            else torch.where(counts > 0, sums / counts.clamp_min(1), centers)
        )
    return centers.reshape(b, h, count, d).half(), ids.reshape(b, h, t).int()


def _incremental_cluster(x, centers, config, *, value=False):
    """Append a group mean until the centroid limit is reached."""
    b, h, t, d = x.shape
    if centers.shape[2] < config.max_clusters:
        centers = torch.cat((centers, x.mean(2, keepdim=True)), dim=2)
    if value:
        # Compute V assignments using the input norm precision.
        dist = (x**2).sum(-1, keepdim=True) + (centers**2).sum(-1).unsqueeze(2)
        dist = dist - 2 * torch.matmul(x.float(), centers.float().transpose(-2, -1))
    else:
        dist = torch.cdist(x.float().reshape(b * h, t, d), centers.float().reshape(b * h, -1, d)).reshape(
            b, h, t, -1
        )
    return centers, dist.argmin(-1).int()


def _encode(x, centers, ids, plan, *, value=False):
    ids = ids.int().contiguous()
    centers = centers.half().contiguous()
    if plan.bits == 16 and plan.layout == "unstructured":
        token_centers = torch.gather(centers, 2, ids.long().unsqueeze(-1).expand(-1, -1, -1, 128))
        residual = x - token_centers
        # K uses topk and V uses full sorting to resolve equal-magnitude entries.
        indices = (
            residual.abs().sort(-1, descending=True).indices[..., : plan.nnz]
            if value
            else residual.abs().topk(plan.nnz, dim=-1).indices
        )
        indices = indices.sort(-1).values
        payload = residual.gather(-1, indices).half().contiguous()
        mask = torch.zeros_like(residual, dtype=torch.bool).scatter_(-1, indices, True)
        weights = 1 << torch.arange(63, -1, -1, device=x.device, dtype=torch.int64)
        metadata = (mask.reshape(*ids.shape, 2, 64).long() * weights).sum(-1)
        return CompressedTensor(centers, ids, payload, metadata, None, plan)
    ext = load_extension("compression", device_index=x.device.index)
    if plan.bits == 16:
        payload, metadata = ext.compress_fp16(
            x.contiguous(), centers, ids, plan.nnz, plan.layout == "structured"
        )
        scale = None
    elif plan.layout == "structured":
        payload, metadata, scale = ext.compress_structured_int4(x.contiguous(), centers, ids)
    else:
        payload, scale = ext.compress_quantized(x.contiguous(), centers, ids, plan.nnz, plan.bits)
        metadata = None
    return CompressedTensor(centers, ids, payload, metadata, scale, plan)


@torch.no_grad()
def compress(
    key: torch.Tensor,
    value: torch.Tensor,
    config: ResKVKernelConfig | None = None,
    *,
    previous: CompressedKV | None = None,
    clustering: tuple | None = None,
    plan: KernelPlan | None = None,
) -> CompressedKV:
    """Compress pre-RoPE K and unrotated V; optionally append a decode group.

    ``clustering=((k_centers,k_ids),(v_centers,v_ids))`` reuses externally
    computed clustering for controlled benchmarks. ``previous`` reuses and
    extends the shared centroid tables during generation.
    """
    config = ResKVKernelConfig() if config is None else config
    if not isinstance(config, ResKVKernelConfig):
        raise TypeError("compress requires ResKVKernelConfig; ResKVConfig is for accuracy evaluation")
    _validate(key)
    _validate(value)
    if key.shape != value.shape or key.device != value.device:
        raise ValueError("K and V must have identical shapes and devices")
    plan = plan or (previous.key.plan if previous else config.resolve(key.shape[0]))
    if plan.bits != config.bits or plan.nnz != int(128 * config.keep_ratio):
        raise ValueError("compression plan and configuration disagree")
    if config.layout != "auto" and plan.layout != config.layout:
        raise ValueError("compression plan violates the requested layout")
    if config.centroid_fusion != "auto" and plan.centroid_fusion != config.centroid_fusion:
        raise ValueError("compression plan violates the requested centroid fusion")
    if plan.fixed_shift != (config.softmax_mode == "fixed_shift"):
        raise ValueError("compression plan and softmax configuration disagree")
    if previous is not None and (
        previous.key.cluster_ids.shape[:2] != key.shape[:2] or previous.key.centroids.device != key.device
    ):
        raise ValueError("previous cache has incompatible dimensions or device")
    if (
        previous is not None
        and max(previous.key.centroids.shape[2], previous.value.centroids.shape[2]) > config.max_clusters
    ):
        raise ValueError("previous cache exceeds the configured centroid limit")
    if clustering is not None and previous is not None:
        raise ValueError("provide clustering or previous, not both")
    if clustering is not None:
        (kc, ki), (vc, vi) = clustering
        for centers, ids in ((kc, ki), (vc, vi)):
            if centers.shape[:2] != key.shape[:2] or centers.ndim != 4 or centers.shape[-1] != 128:
                raise ValueError("external centers must have shape [B,H,C,128]")
            if not 1 <= centers.shape[2] <= config.max_clusters or ids.shape != key.shape[:3]:
                raise ValueError("external clustering has incompatible sizes")
            if centers.device != key.device or ids.device != key.device:
                raise ValueError("external clustering must share the K/V device")
            if ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
                raise ValueError("cluster IDs must be integer tensors")
            if bool(((ids < 0) | (ids >= centers.shape[2])).any()):
                raise ValueError("external cluster IDs are out of range")
    elif previous is not None:
        kc, ki = _incremental_cluster(key, previous.key.centroids, config)
        vc, vi = _incremental_cluster(value, previous.value.centroids, config, value=True)
    else:
        kc, ki = cluster(key, config.num_clusters, config.kmeans_iterations)
        vc, vi = cluster(value, config.num_clusters, config.kmeans_iterations, value=True)
    k = _encode(key, kc, ki, plan)
    v = _encode(value, vc, vi, plan, value=True)
    if previous is not None:
        k, v = previous.key.append(k), previous.value.append(v)
    return CompressedKV(k, v)
