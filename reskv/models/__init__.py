# SPDX-License-Identifier: Apache-2.0
"""Lazy model exports keep accuracy evaluation independent of kernel modules."""

__all__ = [
    "ResKVForCausalLM",
    "ResKVLlamaAccuracyAttention",
    "ResKVQwen3AccuracyAttention",
    "ResKVLlamaAccuracyForCausalLM",
    "ResKVQwen3AccuracyForCausalLM",
]


def __getattr__(name):
    if name == "ResKVForCausalLM":
        from .llama import ResKVForCausalLM

        return ResKVForCausalLM
    if name in __all__:
        from . import accuracy

        return getattr(accuracy, name)
    raise AttributeError(name)
