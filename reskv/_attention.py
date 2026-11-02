# SPDX-License-Identifier: Apache-2.0
"""Unified fused decode with exact joint-softmax merging of recent tokens."""

import torch

from ._extension import load_extension
from .tensors import CompressedKV


@torch.no_grad()
def attention(
    query: torch.Tensor,
    cache: CompressedKV,
    *,
    position: int,
    start_position: int = 0,
    rope_theta: float = 500000.0,
    inv_freq: torch.Tensor | None = None,
    scaling: float | None = None,
    recent: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
):
    """Decode raw Q against compressed pre-RoPE K and V.

    Inputs use [B,H,1,128]. ``recent`` is (rotated Q, rotated K, V).
    ``inv_freq`` uses the model's 64 RoPE frequencies; omitting it selects
    theta-derived RoPE frequencies. Output has the same shape as Q.
    """
    if query.ndim != 4 or query.shape[2:] != (1, 128) or not query.is_cuda or query.dtype != torch.float16:
        raise ValueError("query must be FP16 CUDA [batch, query_heads, 1, 128]")
    b, nh = query.shape[:2]
    kh = cache.key.centroids.shape[1]
    if query.device != cache.key.payload.device or query.device != cache.value.payload.device:
        raise ValueError("query and cache must be on the same CUDA device")
    if b != cache.key.centroids.shape[0] or cache.key.seq_len != cache.value.seq_len:
        raise ValueError("query/cache batch or K/V sequence lengths differ")
    if cache.key.plan != cache.value.plan:
        raise ValueError("K and V plans differ")
    if nh % kh or nh // kh not in (1, 4, 8):
        raise ValueError("supported query/kv head ratios are 1, 4, and 8")
    if position < start_position + cache.seq_len - 1:
        raise ValueError("query position precedes compressed tokens")
    plan = cache.key.plan
    ext = load_extension(
        "attention" if plan.centroid_fusion == "per_token" else "cluster_mass", plan, query.device.index
    )
    q = query.reshape(b * nh, 128).contiguous()
    kc, ki, kp, km, ks = cache.key._native
    vc, vi, vp, vm, vs = cache.value._native
    scale = 128**-0.5 if scaling is None else float(scaling)
    if inv_freq is None:
        freq = torch.empty(0, dtype=torch.float32, device=query.device)
    else:
        if inv_freq.shape != (64,) or inv_freq.device != query.device:
            raise ValueError("inv_freq must be [64] on the query CUDA device")
        freq = inv_freq.float().contiguous()
    # FP16 residuals do not use quantization scales.
    empty = query.new_empty(0)
    suffix = (kh, nh, int(start_position), int(position), float(rope_theta), scale)
    if plan.centroid_fusion == "deferred":
        if plan.layout == "unstructured":
            out, m, normalizer = ext.decode_fp16(
                q,
                kc,
                ki,
                kp,
                km,
                vc,
                vi,
                vp,
                vm,
                *suffix,
                True,
                plan.splits,
                freq,
            )
        else:
            fn = ext.decode_structured_int4 if plan.bits == 4 else ext.decode_structured_fp16
            out, m, normalizer = fn(
                q,
                kc,
                ki,
                kp,
                km,
                ks if ks is not None else empty,
                vc,
                vi,
                vp,
                vm,
                vs if vs is not None else empty,
                *suffix,
                True,
                plan.splits,
                freq,
            )
    elif plan.layout == "unstructured":
        out, m, normalizer = ext.decode_quantized(
            q, kc, ki, kp, ks, vc, vi, vp, vs, *suffix, plan.bits, plan.bits, plan.splits, freq
        )
    else:
        fn = ext.decode_int4 if plan.bits == 4 else ext.decode_fp16
        out, m, normalizer = fn(
            q,
            kc,
            ki,
            kp,
            km,
            ks if ks is not None else empty,
            vc,
            vi,
            vp,
            vm,
            vs if vs is not None else empty,
            *suffix,
            plan.splits,
            freq,
        )
    if recent is not None and recent[1].shape[2] > 0:
        qr, kr, vr = recent
        if qr.shape != query.shape or kr.shape != vr.shape or kr.shape[:2] != (b, kh):
            raise ValueError("recent Q/K/V shapes do not match the compressed cache")
        if any(x.device != query.device or x.dtype != torch.float16 for x in recent):
            raise ValueError("recent tensors must be FP16 on the query device")
        if plan.centroid_fusion == "per_token":
            out = ext.merge_recent(
                out,
                m,
                normalizer,
                qr.reshape(b * nh, 128).contiguous(),
                kr.contiguous(),
                vr.contiguous(),
                kh,
                nh,
                scale,
            )
        else:
            # Keep the dot products in FP32 before applying the attention scale.
            scores = torch.bmm(
                qr.reshape(b * kh, nh // kh, 128),
                kr.reshape(b * kh, kr.shape[2], 128).transpose(1, 2),
                out_dtype=torch.float32,
            )
            scores = scores.mul_(scale).reshape(b, nh, -1).contiguous()
            out = ext.merge_recent(out, m, normalizer, scores, vr.contiguous(), kh, nh)
    return out.reshape(b, nh, 1, 128)
