# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from reskv import ResKVKernelConfig, attention, compress
from reskv.cache import ResKVLayerCache
from reskv.reference import reconstruct, reference_attention, rotate

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


CASES = [
    (16, 0.5, "tiled", "per_token"),
    (16, 0.5, "unstructured", "deferred"),
    (16, 0.75, "unstructured", "deferred"),
    (16, 0.75, "structured", "per_token"),
    (16, 0.75, "structured", "deferred"),
    (4, 0.5, "unstructured", "per_token"),
    (4, 0.75, "unstructured", "per_token"),
    (4, 0.75, "structured", "per_token"),
    (4, 0.75, "structured", "deferred"),
    (2, 0.5, "unstructured", "per_token"),
    (2, 0.75, "unstructured", "per_token"),
]


def inputs(b=1, t=256, heads=1):
    torch.manual_seed(123)
    k = torch.randn(b, heads, t, 128, device="cuda", dtype=torch.float16) * 0.5
    v = torch.randn_like(k) * 0.5
    q = torch.randn(b, heads * 4, 1, 128, device="cuda", dtype=torch.float16) * 0.5
    kc = torch.randn(b, heads, 32, 128, device="cuda", dtype=torch.float16) * 0.1
    vc = torch.randn_like(kc) * 0.1
    ki = torch.randint(32, (b, heads, t), device="cuda", dtype=torch.int32)
    vi = torch.randint(32, ki.shape, device="cuda", dtype=torch.int32)
    return k, v, q, ((kc, ki), (vc, vi))


@pytest.mark.parametrize("bits,sparsity,layout,fusion", CASES)
def test_fused_and_mixed_against_fp32(bits, sparsity, layout, fusion):
    k, v, q, clustering = inputs()
    cfg = ResKVKernelConfig(bits=bits, sparsity=sparsity, layout=layout, centroid_fusion=fusion)
    cache = compress(k, v, cfg, clustering=clustering)
    # Non-standard frequencies catch kernels accidentally hardcoding RoPE theta.
    freq = 500000.0 ** (-torch.arange(64, device="cuda").float() / 64)
    freq = freq / torch.linspace(1, 8, 64, device="cuda")
    out = attention(q, cache, position=288, inv_freq=freq)
    expected = reference_attention(q, cache, position=288, inv_freq=freq)
    torch.testing.assert_close(out.float(), expected, atol=3e-3, rtol=3e-2)
    kr = torch.randn(1, 1, 32, 128, device="cuda", dtype=torch.float16) * 0.5
    vr = torch.randn_like(kr) * 0.5
    qr = rotate(q, torch.tensor([288], device="cuda"), freq).half()
    mixed = attention(q, cache, position=288, inv_freq=freq, recent=(qr, kr, vr))
    ref = reference_attention(q, cache, position=288, inv_freq=freq, recent=(qr, kr, vr))
    torch.testing.assert_close(mixed.float(), ref, atol=3e-3, rtol=3e-2)


@pytest.mark.parametrize(
    "bits,layout",
    [(16, "unstructured"), (16, "structured"), (4, "unstructured"), (4, "structured"), (2, "unstructured")],
)
def test_current_stream(bits, layout):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        k, v, q, c = inputs(t=64)
        cache = compress(k, v, ResKVKernelConfig(bits=bits, sparsity=0.75, layout=layout), clustering=c)
        out = attention(q, cache, position=64)
        ref = reference_attention(q, cache, position=64)
        error = (out.float() - ref).abs().max()
    stream.synchronize()
    assert error.item() < 3e-3


def test_incremental_cache_no_tokens_lost_and_centroids_capped():
    cfg = ResKVKernelConfig(
        bits=16, layout="unstructured", num_clusters=4, max_clusters=6, group_size=16, recent_size=8
    )
    cache = ResKVLayerCache(cfg)
    k, v, _, _ = inputs(t=35)
    cache.update(k, v, {"raw_keys": k})
    assert cache.compressed.seq_len + cache.live == 35
    for _ in range(100):
        token = k[:, :, :1]
        cache.update(token, token, {"raw_keys": token})
        assert cache.compressed.seq_len + cache.live == cache.length
        assert cache.compressed.key.centroids.shape[2] <= 6
    assert cache.length == 135
    cache.reset()
    assert cache.get_seq_length() == 0 and cache.compressed is None


def test_layouts_have_different_selection_semantics():
    k, v, _, c = inputs(t=64)
    u = compress(k, v, ResKVKernelConfig(bits=16, sparsity=0.75, layout="unstructured"), clustering=c)
    s = compress(k, v, ResKVKernelConfig(bits=16, sparsity=0.75, layout="structured"), clustering=c)
    assert not torch.equal(reconstruct(u.key), reconstruct(s.key))
    assert u.key.payload.shape[-1] == s.key.payload.shape[-1] == 32


@pytest.mark.parametrize("ratio", [1, 8])
@pytest.mark.parametrize(
    "bits,layout", [(16, "unstructured"), (16, "structured"), (4, "structured"), (2, "unstructured")]
)
def test_gqa_and_64_centroids(ratio, bits, layout):
    k, v, q, _ = inputs(t=320)
    q = q[:, :1].repeat(1, ratio, 1, 1)
    centers = k[:, :, :64].contiguous()
    ids = torch.arange(320, device="cuda", dtype=torch.int32).remainder(64).view(1, 1, -1)
    cfg = ResKVKernelConfig(bits=bits, sparsity=0.75, layout=layout)
    cache = compress(k, v, cfg, clustering=((centers, ids), (centers, ids)))
    out = attention(q, cache, position=330, start_position=10)
    ref = reference_attention(q, cache, position=330, start_position=10)
    torch.testing.assert_close(out.float(), ref, atol=3e-3, rtol=3e-2)


def test_structured_int4_stable_softmax_large_logits():
    k, v, q, c = inputs(t=256)
    k, q = k * 12, q * 12
    cfg = ResKVKernelConfig(bits=4, sparsity=0.75, layout="structured", centroid_fusion="deferred")
    cache = compress(k, v, cfg, clustering=c)
    out = attention(q, cache, position=256)
    ref = reference_attention(q, cache, position=256)
    assert out.isfinite().all()
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=5e-2)


def test_invalid_external_cluster_ids_rejected():
    k, v, _, c = inputs(t=64)
    c[0][1][0, 0, 0] = 32
    with pytest.raises(ValueError, match="out of range"):
        compress(k, v, clustering=c)


def test_theta_rope_with_model_rotated_recent_tokens():
    k, v, q, c = inputs(t=256)
    cfg = ResKVKernelConfig(bits=4, sparsity=0.75, layout="unstructured", rope_mode="theta")
    cache = compress(k, v, cfg, clustering=c)
    model_freq = 500000.0 ** (-torch.arange(64, device="cuda").float() / 64) / 8
    qr = rotate(q, torch.tensor([288], device="cuda"), model_freq).half()
    recent = (qr, k[:, :, :32].contiguous(), v[:, :, :32].contiguous())
    out = attention(q, cache, position=288, recent=recent)
    ref = reference_attention(q, cache, position=288, recent=recent)
    torch.testing.assert_close(out.float(), ref, atol=3e-3, rtol=3e-2)


@pytest.mark.parametrize("sparsity", [0.5, 0.75])
def test_recent_attention_with_large_finite_scores(sparsity):
    data = torch.zeros(1, 1, 128, 128, device="cuda", dtype=torch.float16)
    centers = data[:, :, :1].contiguous()
    ids = torch.zeros(1, 1, 128, device="cuda", dtype=torch.int32)
    config = ResKVKernelConfig(sparsity=sparsity, centroid_fusion="deferred")
    cache = compress(data, data, config, clustering=((centers, ids), (centers, ids)))
    query = torch.full((1, 4, 1, 128), 100.0, device="cuda", dtype=torch.float16)
    recent = (query, torch.full_like(data[:, :, :32], 100.0), torch.full_like(data[:, :, :32], 2.0))
    output = attention(query, cache, position=159, recent=recent)
    expected = reference_attention(query, cache, position=159, recent=recent)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.float(), expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("count", [1, 32])
def test_value_cluster_mean_does_not_overflow(count):
    from reskv.compression import cluster

    data = torch.full((1, 1, 8192, 128), 20.0, device="cuda", dtype=torch.float16)
    centers, assignments = cluster(data, count=count, iterations=2, value=True)
    assert torch.isfinite(centers).all()
    restored = centers.gather(2, assignments.long().unsqueeze(-1).expand_as(data))
    torch.testing.assert_close(restored, data, atol=0, rtol=0)
