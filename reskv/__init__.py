# SPDX-License-Identifier: Apache-2.0
"""ResKV: residual-based channel-wise unstructured pruning for KV cache compression."""

from .config import ResKVConfig, ResKVKernelConfig

__version__ = "0.1.0"
__all__ = [
    "ResKVKernelConfig",
    "ResKVForCausalLM",
    "ResKVConfig",
    "ResKVLlamaAccuracyForCausalLM",
    "ResKVQwen3AccuracyForCausalLM",
    "compress",
    "attention",
]


def __getattr__(name):
    # Configuration imports require neither CUDA initialization nor native builds.
    if name == "ResKVForCausalLM":
        from .models.llama import ResKVForCausalLM

        return ResKVForCausalLM
    if name in ("ResKVLlamaAccuracyForCausalLM", "ResKVQwen3AccuracyForCausalLM"):
        from .models.accuracy import ResKVLlamaAccuracyForCausalLM, ResKVQwen3AccuracyForCausalLM

        return {
            "ResKVLlamaAccuracyForCausalLM": ResKVLlamaAccuracyForCausalLM,
            "ResKVQwen3AccuracyForCausalLM": ResKVQwen3AccuracyForCausalLM,
        }[name]
    if name == "compress":
        from .compression import compress

        return compress
    if name == "attention":
        from ._attention import attention

        return attention
    raise AttributeError(name)
