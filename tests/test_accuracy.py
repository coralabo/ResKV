# SPDX-License-Identifier: Apache-2.0
"""Numerical and cache invariants for independent accuracy compression."""

import subprocess
import sys
from dataclasses import replace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

from reskv.accuracy import (
    ResKVAccuracyCache,
    ResKVAccuracyLayerCache,
    ResKVConfig,
    compress_decode,
    compress_prefill,
    quantize_dequantize,
    sparsify,
)
from reskv.models.accuracy import (
    ResKVLlamaAccuracyAttention,
    ResKVLlamaAccuracyForCausalLM,
    ResKVQwen3AccuracyAttention,
    ResKVQwen3AccuracyForCausalLM,
)


@pytest.fixture(autouse=True)
def small_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("sparsity,bits", [(0.5, 16), (0.75, 16), (0.5, 4), (0.75, 4)])
def test_reconstruction_and_dense_regions(sparsity, bits):
    torch.manual_seed(13)
    cfg = ResKVConfig(
        sparsity=sparsity,
        bits=bits,
        num_clusters=3,
        recent_size=3,
        group_size=4,
        sink_tokens=2,
        kmeans_iterations=3,
    )
    layer = ResKVAccuracyLayerCache(cfg)
    keys, values = torch.randn(2, 2, 19, 32), torch.randn(2, 2, 19, 32)
    returned = layer.update(keys, values)
    assert torch.equal(returned[0], keys) and torch.equal(returned[1], values)
    assert layer.compressed_end == 14
    assert torch.equal(layer.keys[:, :, :2], keys[:, :, :2])
    assert torch.equal(layer.keys[:, :, 14:], keys[:, :, 14:])
    compressed = layer.keys[:, :, 2:14].clone()
    for _ in range(12):
        k, v = torch.randn(2, 2, 1, 32), torch.randn(2, 2, 1, 32)
        keys, values = torch.cat((keys, k), dim=2), torch.cat((values, v), dim=2)
        layer.update(k, v)
        assert torch.equal(layer.keys[:, :, :2], keys[:, :, :2])
        assert torch.equal(layer.keys[:, :, layer.compressed_end :], keys[:, :, layer.compressed_end :])
        assert torch.equal(layer.keys[:, :, 2:14], compressed)  # no recompression
        assert layer.get_seq_length() - layer.compressed_end >= cfg.recent_size
    assert layer.key_centroids.shape[-2] == 6
    assert layer.key_centroids.dtype == layer.value_centroids.dtype == torch.bfloat16
    layer.reorder_cache(torch.tensor([1, 0, 1]))
    assert layer.keys.shape[0] == layer.key_centroids.shape[0] == 3
    layer.reset()
    assert layer.get_seq_length() == 0 and layer.key_centroids is None


@pytest.mark.parametrize("bits", [2, 4, 8, 16])
def test_zero_constant_and_sparse_quantization(bits):
    zero = torch.zeros(1, 1, 7, 32)
    constant = torch.full_like(zero, -2.5)
    cfg = ResKVConfig(bits=bits, num_clusters=32, kmeans_iterations=2)
    for x in (zero, constant):
        restored, centers = compress_prefill(x, cfg)
        assert torch.equal(x, restored)
        decoded, _ = compress_decode(x, centers, cfg)
        assert torch.equal(x, decoded)
    sparse = torch.tensor([[[[-4.0, 0.0, 0.0, 2.0]]]])
    for stage in ("prefill", "decode"):
        quantized = quantize_dequantize(sparse, bits, stage)
        assert torch.isfinite(quantized).all()
        assert (quantized[..., 1:3] == 0).all()


def test_topk_is_residual_based_and_structured_mask():
    x = torch.arange(1, 33, dtype=torch.float32).reshape(1, 1, 1, 32)
    assert sparsify(x, 0.75).count_nonzero() == 8
    result = sparsify(x, 0.75, 16).reshape(2, 16)
    assert torch.equal(result.count_nonzero(-1), torch.tensor([4, 4]))
    assert (result[:, :12] == 0).all()


@pytest.mark.parametrize("bits", [4, 16])
def test_structured_reconstruction_preserves_block_mask(bits):
    token = torch.arange(1, 129, dtype=torch.float32).reshape(1, 1, 1, 128)
    values = torch.cat((token, -token), dim=2)
    cfg = ResKVConfig(layout="structured", num_clusters=1, bits=bits, kmeans_iterations=2)
    restored, centroids = compress_prefill(values, cfg)
    assert (centroids == 0).all() and centroids.dtype == torch.bfloat16
    counts = restored.reshape(1, 1, 2, 8, 16).count_nonzero(-1)
    assert (counts == 4).all()
    assert (restored.reshape(1, 1, 2, 8, 16)[..., :12] == 0).all()


@pytest.mark.parametrize("length", [0, 1, 7, 33])
def test_no_compression_control_is_exact(length):
    x = torch.randn(1, 2, length, 32)
    cfg = ResKVConfig(sparsity=0, bits=16, kmeans_iterations=2)
    restored, centers = compress_prefill(x, cfg)
    assert torch.equal(restored, x)
    assert torch.equal(compress_decode(x, centers, cfg)[0], x)


@pytest.mark.parametrize(
    "native_cls,accuracy_cls,config_cls",
    [
        (LlamaForCausalLM, ResKVLlamaAccuracyForCausalLM, LlamaConfig),
        (Qwen3ForCausalLM, ResKVQwen3AccuracyForCausalLM, Qwen3Config),
    ],
)
def test_model_dense_control_rope_cache_and_reload(tmp_path, native_cls, accuracy_cls, config_cls):
    torch.manual_seed(31)
    config = config_cls(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=192,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
    )
    config._attn_implementation = "sdpa"
    native = native_cls(config).eval()
    cfg = ResKVConfig(sparsity=0, bits=16, num_clusters=3, recent_size=2, group_size=4, kmeans_iterations=2)
    model = accuracy_cls(config, reskv_config=cfg).eval()
    expected_attention = (
        ResKVQwen3AccuracyAttention if config.model_type == "qwen3" else ResKVLlamaAccuracyAttention
    )
    assert all(isinstance(layer.self_attn, expected_attention) for layer in model.model.layers)
    model.load_state_dict(native.state_dict(), strict=True)
    ids = torch.randint(3, 100, (1, 17))
    with torch.inference_mode():
        a, b = native(ids), model(ids, logits_to_keep=0)
        torch.testing.assert_close(a.logits, b.logits, rtol=0, atol=0)
        for _ in range(8):
            token = a.logits[:, -1].argmax(-1, keepdim=True)
            a = native(token, past_key_values=a.past_key_values)
            b = model(token, past_key_values=b.past_key_values)
            torch.testing.assert_close(a.logits, b.logits, rtol=0, atol=0)
        assert isinstance(b.past_key_values, ResKVAccuracyCache)
        model.save_pretrained(tmp_path)
        loaded = accuracy_cls.from_pretrained(tmp_path).eval()
        assert loaded.reskv_config == cfg
        assert loaded.config.reskv_config == cfg.to_dict()
        assert all(isinstance(layer.self_attn, expected_attention) for layer in loaded.model.layers)
        torch.testing.assert_close(model(ids).logits, loaded(ids).logits, rtol=0, atol=0)
        cfg_sparse = replace(cfg, sparsity=0.75, bits=4)
        compressed = accuracy_cls(config, reskv_config=cfg_sparse).eval()
        compressed.load_state_dict(native.state_dict(), strict=True)
        # Prefill logits must remain identical even when stored KV is lossy.
        torch.testing.assert_close(
            native(ids, logits_to_keep=1).logits, compressed(ids).logits, rtol=0, atol=0
        )
        generated = compressed.generate(
            ids, max_new_tokens=10, do_sample=False, eos_token_id=None, pad_token_id=0
        )
        assert generated.shape == (1, 27)
    with pytest.raises(ValueError, match="unpadded"):
        model(ids, attention_mask=torch.zeros_like(ids))
    with pytest.raises(ValueError, match="contiguous"):
        model(ids, position_ids=torch.arange(1, 18).unsqueeze(0))


def test_invalid_inputs_and_settings():
    with pytest.raises(ValueError):
        ResKVConfig(bits=3)
    with pytest.raises(ValueError):
        ResKVConfig(layout="structured", sparsity=0.5)
    with pytest.raises(ValueError, match="nonfinite"):
        compress_prefill(torch.full((1, 1, 2, 32), float("nan")), ResKVConfig())


@pytest.mark.parametrize(
    "config_cls,model_cls",
    [(LlamaConfig, ResKVLlamaAccuracyForCausalLM), (Qwen3Config, ResKVQwen3AccuracyForCausalLM)],
)
def test_model_input_errors_are_explicit(config_cls, model_cls):
    config = config_cls(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
    )
    model = model_cls(config).eval()
    ids = torch.ones(1, 2, dtype=torch.long)
    with pytest.raises(ValueError, match="exactly one"):
        model()
    with pytest.raises(ValueError, match="exactly one"):
        model(input_ids=ids, inputs_embeds=torch.zeros(1, 2, 64))
    with pytest.raises(ValueError, match="contiguous"):
        model(ids, position_ids=torch.arange(2, device="meta").unsqueeze(0))


@pytest.mark.parametrize(
    "config_cls,model_cls",
    [(LlamaConfig, ResKVLlamaAccuracyForCausalLM), (Qwen3Config, ResKVQwen3AccuracyForCausalLM)],
)
def test_prefill_is_causal_and_uses_dense_kv(config_cls, model_cls):
    torch.manual_seed(19)
    config = config_cls(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
    )
    config._attn_implementation = "sdpa"
    model = model_cls(
        config,
        ResKVConfig(sparsity=0.75, bits=4, num_clusters=3, group_size=4, recent_size=2, kmeans_iterations=2),
    ).eval()
    first = torch.randint(3, 100, (1, 18))
    second = first.clone()
    second[:, 6:] = torch.randint(3, 100, (1, 12))
    with torch.inference_mode():
        a = model(first, logits_to_keep=0).logits[:, :6]
        b = model(second, logits_to_keep=0).logits[:, :6]
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_accuracy_imports_do_not_load_kernel_modules():
    code = (
        "import sys; from reskv import ResKVConfig, ResKVLlamaAccuracyForCausalLM, ResKVQwen3AccuracyForCausalLM; "
        "assert not any(n in sys.modules for n in "
        "('reskv._extension', 'reskv._attention', 'reskv.compression', 'reskv.cache'))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
