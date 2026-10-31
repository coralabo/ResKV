# SPDX-License-Identifier: Apache-2.0
"""Independent FP32 reconstruction and attention for numerical validation.

This is a diagnostic reference, never an implicit runtime fallback.
"""

import torch

from .tensors import CompressedKV, CompressedTensor


def reconstruct(tensor: CompressedTensor) -> torch.Tensor:
    ids = tensor.cluster_ids.long()
    b, h, t = ids.shape
    p = tensor.plan
    residual = torch.zeros(b, h, t, 128, device=ids.device, dtype=torch.float32)
    if p.layout == "unstructured" and p.bits != 16:
        shifts = torch.arange(32 // p.bits, device=ids.device) * p.bits
        q = ((tensor.payload.long().unsqueeze(-1) >> shifts) & ((1 << p.bits) - 1)).reshape(b, h, t, 128)
        sign = 1 << (p.bits - 1)
        residual = ((q ^ sign) - sign).float() * tensor.scale.float().unsqueeze(-1)
    elif p.layout == "unstructured":
        dims = torch.arange(128, device=ids.device)
        selected = ((tensor.metadata[..., dims // 64] >> (63 - dims % 64)) & 1).bool()
        rank = selected.long().cumsum(-1) - 1
        residual = torch.where(selected, tensor.payload.gather(-1, rank.clamp_min(0)).float(), 0)
    elif p.bits == 4:
        # 1 retained dimension per 4D subgroup: 32 signed nibbles, 32 2-bit indices.
        qshift = torch.arange(8, device=ids.device) * 4
        q = ((tensor.payload.long().unsqueeze(-1) >> qshift) & 15).reshape(b, h, t, 32)
        q = ((q ^ 8) - 8).float() * tensor.scale.float().unsqueeze(-1)
        mshift = torch.arange(16, device=ids.device) * 2
        local = ((tensor.metadata.long().unsqueeze(-1) >> mshift) & 3).reshape(b, h, t, 32)
        dims = local + torch.arange(32, device=ids.device) * 4
        residual.scatter_(-1, dims, q)
    else:
        words = tensor.metadata.long().unsqueeze(-1)
        codes = ((words >> torch.tensor([0, 16], device=ids.device)) & 65535).reshape(b, h, t, 8)
        if p.layout == "structured":
            shifts = torch.arange(4, device=ids.device) * 4
            dims = ((codes.unsqueeze(-1) >> shifts) & 15) + torch.arange(8, device=ids.device).view(8, 1) * 16
            residual.scatter_(-1, dims.reshape(b, h, t, 32), tensor.payload.float())
        else:
            dims = torch.arange(16, device=ids.device)
            mask = ((codes.unsqueeze(-1) >> dims) & 1).bool().reshape(b, h, t, 128)
            rank = mask.long().cumsum(-1) - 1
            residual = torch.where(mask, tensor.payload.gather(-1, rank.clamp_min(0)).float(), 0)
    centers = tensor.centroids.float().gather(2, ids.unsqueeze(-1).expand(-1, -1, -1, 128))
    return centers + residual


def rotate(x: torch.Tensor, positions: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
    angles = positions.float().unsqueeze(-1) * frequencies.float()
    cos, sin = angles.cos(), angles.sin()
    a, b = x.float().chunk(2, dim=-1)
    return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)


def reference_attention(
    query: torch.Tensor,
    cache: CompressedKV,
    *,
    position: int,
    start_position: int = 0,
    rope_theta: float = 500000.0,
    inv_freq: torch.Tensor | None = None,
    scaling: float | None = None,
    recent: tuple | None = None,
):
    k, v = reconstruct(cache.key), reconstruct(cache.value)
    b, kh, t, _ = k.shape
    nh = query.shape[1]
    freq = (
        inv_freq
        if inv_freq is not None
        else rope_theta ** (-torch.arange(64, device=query.device).float() / 64)
    )
    q = rotate(query, torch.tensor([position], device=query.device), freq)
    k = rotate(k, torch.arange(start_position, start_position + t, device=query.device), freq)
    k, v = k.repeat_interleave(nh // kh, 1), v.repeat_interleave(nh // kh, 1)
    scale = 128**-0.5 if scaling is None else scaling
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if recent is not None:
        qr, kr, vr = recent
        # Honour the supplied rotated Q, including theta-derived RoPE mode's
        # different compressed/recent RoPE conventions.
        kr = kr.float().repeat_interleave(nh // kh, 1)
        recent_scores = torch.matmul(qr.float(), kr.transpose(-1, -2)) * scale
        scores = torch.cat((scores, recent_scores), dim=-1)
        v = torch.cat((v, vr.float().repeat_interleave(nh // kh, 1)), dim=2)
    return torch.matmul(scores.softmax(-1), v)
