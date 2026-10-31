# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import LlamaConfig

from reskv import ResKVForCausalLM, ResKVKernelConfig
from reskv.cache import ResKVCache

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.model,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def model_config():
    return LlamaConfig(
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        vocab_size=256,
        max_position_embeddings=8192,
        rope_theta=500000.0,
    )


@pytest.mark.parametrize(
    "bits,sparsity,layout",
    [(16, 0.5, "auto"), (16, 0.75, "auto"), (4, 0.5, "auto"), (4, 0.75, "auto")],
)
def test_generation_cache_boundary_and_save_reload(tmp_path, bits, sparsity, layout):
    torch.manual_seed(9)
    settings = ResKVKernelConfig(
        bits=bits,
        sparsity=sparsity,
        layout=layout,
        recent_size=16,
        group_size=32,
        num_clusters=4,
        max_clusters=8,
    )
    model = ResKVForCausalLM(model_config(), settings).cuda().half().eval()
    x = torch.randint(10, 200, (1, 96), device="cuda")
    with torch.inference_mode():
        initial = model(x, use_cache=True)
        assert isinstance(initial.past_key_values, ResKVCache)
        assert initial.past_key_values.layers[0].compressed.key.plan == settings.resolve(1)
        result = model.generate(x, max_new_tokens=66, do_sample=False, eos_token_id=None, pad_token_id=0)
    assert result.shape == (1, 162)
    model.save_pretrained(tmp_path)
    restored = ResKVForCausalLM.from_pretrained(tmp_path, dtype=torch.float16).cuda().eval()
    assert restored.kernel_config == settings
    assert restored.config.reskv_kernel_config == settings.to_dict()
    assert "reskv_config" not in restored.config.to_dict()
    with torch.inference_mode():
        torch.testing.assert_close(initial.logits, restored(x, use_cache=True).logits)


def test_padded_batches_are_rejected():
    model = ResKVForCausalLM(model_config()).cuda().half().eval()
    x = torch.ones(1, 16, dtype=torch.long, device="cuda")
    mask = torch.ones_like(x)
    mask[:, :2] = 0
    with pytest.raises(ValueError, match="unpadded"):
        model(x, attention_mask=mask)


@pytest.mark.parametrize("field", ["position_ids", "cache_position"])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_noncontiguous_positions_are_rejected(field, phase):
    model = ResKVForCausalLM(model_config()).cuda().half().eval()
    x = torch.ones(1, 16, dtype=torch.long, device="cuda")
    cache = None
    start = 0
    with torch.inference_mode():
        if phase == "decode":
            cache = model(x).past_key_values
            start = x.shape[-1]
            x = x[:, :1]
        positions = torch.arange(start, start + x.shape[-1], device="cuda") + 1
        if field == "position_ids":
            positions = positions.unsqueeze(0)
        with pytest.raises(ValueError, match="contiguous|start at position zero"):
            model(x, past_key_values=cache, **{field: positions})


def test_decode_padding_is_rejected():
    model = ResKVForCausalLM(model_config()).cuda().half().eval()
    x = torch.ones(1, 16, dtype=torch.long, device="cuda")
    with torch.inference_mode():
        cache = model(x).past_key_values
        mask = torch.ones(1, 17, dtype=torch.long, device="cuda")
        mask[:, 0] = 0
        with pytest.raises(ValueError, match="unpadded"):
            model(x[:, :1], past_key_values=cache, attention_mask=mask)


def test_explicit_full_and_indexed_logits_preserve_transformers_semantics():
    model = ResKVForCausalLM(model_config()).cuda().half().eval()
    x = torch.ones(1, 96, dtype=torch.long, device="cuda")
    with torch.inference_mode():
        last = model(x).logits
        full = model(x, logits_to_keep=0).logits
        selected = model(x, logits_to_keep=torch.tensor([0, 95], device="cuda")).logits
    assert last.shape == (1, 1, 256) and full.shape == (1, 96, 256)
    torch.testing.assert_close(last, full[:, -1:], atol=1e-3, rtol=1e-2)
    torch.testing.assert_close(selected, full[:, [0, 95]], atol=1e-3, rtol=1e-2)
