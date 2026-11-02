# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP

from reskv.models.llama import ResKVMLP


def mlp_pair(config, *, device="cpu", dtype=torch.float32):
    reference = LlamaMLP(config).eval()
    model = ResKVMLP(config).eval()
    model.load_state_dict(reference.state_dict(), strict=True)
    return reference.to(device=device, dtype=dtype), model.to(device=device, dtype=dtype)


@pytest.mark.parametrize("activation", ["silu", "gelu"])
def test_mlp_inference_matches_transformers(activation):
    torch.manual_seed(42)
    config = LlamaConfig(hidden_size=64, intermediate_size=128, hidden_act=activation)
    reference, model = mlp_pair(config)
    x = torch.randn(2, 17, 64).transpose(0, 1)
    with torch.inference_mode():
        torch.testing.assert_close(model(x), reference(x), rtol=0, atol=0)


def test_mlp_preserves_gradient_execution():
    torch.manual_seed(42)
    config = LlamaConfig(hidden_size=64, intermediate_size=128)
    reference, model = mlp_pair(config)
    x = torch.randn(2, 17, 64, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    reference(x).square().sum().backward()
    model(other).square().sum().backward()
    torch.testing.assert_close(x.grad, other.grad, rtol=0, atol=0)
    for left, right in zip(reference.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mlp_fp16_reduces_peak_memory():
    torch.manual_seed(42)
    config = LlamaConfig(hidden_size=512, intermediate_size=1024)
    reference, model = mlp_pair(config, device="cuda", dtype=torch.float16)
    x = torch.randn(2, 2048, 512, device="cuda", dtype=torch.float16)

    def measure(module):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        result = module(x)
        torch.cuda.synchronize()
        return result, torch.cuda.max_memory_allocated() - baseline

    with torch.inference_mode():
        # Initialize GEMM workspaces before counting transient allocations.
        reference(x)
        model(x)
        expected, original_peak = measure(reference)
        actual, optimized_peak = measure(model)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    intermediate_bytes = x.shape[0] * x.shape[1] * config.intermediate_size * x.element_size()
    assert original_peak - optimized_peak >= intermediate_bytes
