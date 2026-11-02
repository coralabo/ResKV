# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys

import pytest

from reskv import ResKVConfig, ResKVKernelConfig


@pytest.mark.parametrize(
    "settings",
    [
        {"sparsity": 0.25},
        {"bits": 8},
        {"layout": "invalid"},
        {"rope_mode": "invalid"},
        {"sparsity": 0.5, "layout": "structured"},
        {"bits": 2, "layout": "structured", "sparsity": 0.75},
        {"bits": 16, "layout": "tiled", "sparsity": 0.75},
        {"num_clusters": 65},
        {"max_clusters": 0},
        {"group_size": 0},
        {"recent_size": -1},
    ],
)
def test_invalid_config(settings):
    with pytest.raises(ValueError):
        ResKVKernelConfig(**settings)


def test_selection_preserves_distinct_semantics():
    assert ResKVKernelConfig(bits=16, layout="tiled").resolve(1).layout == "tiled"
    assert ResKVKernelConfig(bits=16, layout="auto").resolve(1).layout == "unstructured"
    assert ResKVKernelConfig(bits=16, layout="auto").resolve(16).layout == "unstructured"
    assert ResKVKernelConfig(bits=16, sparsity=0.75, layout="auto").resolve(1).centroid_fusion == "per_token"
    assert ResKVKernelConfig(bits=16, sparsity=0.75, layout="auto").resolve(16).centroid_fusion == "deferred"
    assert ResKVKernelConfig(bits=16, sparsity=0.75).resolve(1).layout == "structured"
    assert (
        ResKVKernelConfig(bits=4, sparsity=0.75, layout="unstructured").resolve(16).centroid_fusion
        == "per_token"
    )
    assert ResKVKernelConfig(bits=4, sparsity=0.75, layout="structured").resolve(16).tokens_per_warp == 8


def test_config_round_trip():
    cfg = ResKVKernelConfig(bits=16, sparsity=0.75, layout="unstructured", rope_mode="theta")
    assert ResKVKernelConfig(**cfg.to_dict()) == cfg


@pytest.mark.parametrize(
    "changes",
    [
        {"bits": 4.0},
        {"nnz": 64.0},
        {"tokens_per_warp": 2.0},
        {"splits": True},
        {"fixed_shift": 0},
        {"layout": "structured"},
        {"layout": "tiled"},
        {"bits": 4},
        {"fixed_shift": True},
    ],
)
def test_invalid_kernel_plan_is_rejected(changes):
    from reskv.config import KernelPlan

    values = ResKVKernelConfig().resolve(1).to_dict()
    with pytest.raises(ValueError):
        KernelPlan(**{**values, **changes})


@pytest.mark.parametrize("config_type", [ResKVKernelConfig, ResKVConfig])
def test_bit_width_requires_an_integer(config_type):
    with pytest.raises(ValueError, match="bits"):
        config_type(bits=4.0)


def test_accuracy_sparsity_rejects_boolean():
    from reskv import ResKVConfig

    with pytest.raises(ValueError, match="sparsity"):
        ResKVConfig(sparsity=False)


@pytest.mark.parametrize("batch_size", [True, 1.5])
def test_batch_size_requires_a_positive_integer(batch_size):
    with pytest.raises(ValueError, match="positive integer"):
        ResKVKernelConfig().resolve(batch_size)


@pytest.mark.parametrize("bits", [2, 4, 16])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32])
def test_sparse75_default_plan(bits, batch_size):
    settings = ResKVKernelConfig(sparsity=0.75, bits=bits)
    plan = settings.resolve(batch_size)
    assert settings.layout == "auto"
    assert plan.layout == ("unstructured" if bits == 2 else "structured")
    assert plan.centroid_fusion == ("per_token" if bits == 2 or batch_size <= 2 else "deferred")
    assert ResKVKernelConfig(**settings.to_dict()).resolve(batch_size) == plan


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32])
def test_sparse50_default_and_accuracy_configuration(batch_size):
    from reskv import ResKVConfig

    settings = ResKVKernelConfig(sparsity=0.5, bits=16)
    plan = settings.resolve(batch_size)
    assert settings.layout == "auto"
    assert (plan.layout, plan.centroid_fusion) == ("unstructured", "deferred")
    assert ResKVKernelConfig(**settings.to_dict()).resolve(batch_size) == plan
    assert ResKVKernelConfig(sparsity=0.5, bits=16, layout="unstructured").resolve(batch_size) == plan
    tiled = ResKVKernelConfig(sparsity=0.5, bits=16, layout="tiled").resolve(batch_size)
    assert (tiled.layout, tiled.centroid_fusion) == ("tiled", "per_token")
    assert ResKVConfig(sparsity=0.5).layout == "unstructured"


def test_accuracy_defaults_for_python_api():
    from reskv import ResKVConfig

    settings = ResKVConfig()
    assert (settings.recent_size, settings.kmeans_iterations) == (32, 15)
    assert ResKVConfig(**settings.to_dict()) == settings


def test_accuracy_preserves_explicit_and_saved_parameters():
    from reskv import ResKVConfig

    settings = ResKVConfig(recent_size=0, kmeans_iterations=3)
    assert settings.recent_size == 0 and settings.kmeans_iterations == 3
    restored = ResKVConfig(**{**settings.to_dict(), "recent_size": 32, "kmeans_iterations": 10})
    assert restored.recent_size == 32 and restored.kmeans_iterations == 10


def test_import_does_not_initialize_cuda_or_compile():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import reskv; assert 'torch' not in sys.modules; "
            "assert reskv.ResKVConfig().bits == 16; "
            "assert reskv.ResKVKernelConfig().bits == 16; "
            "assert 'torch' not in sys.modules",
        ],
        check=True,
    )


def test_kernel_config_round_trips_through_checkpoint(tmp_path):
    from transformers import LlamaConfig

    from reskv import ResKVForCausalLM

    config = LlamaConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=32,
    )
    settings = ResKVKernelConfig(
        sparsity=0.75,
        bits=4,
        num_clusters=4,
        max_clusters=8,
        rope_mode="theta",
    )
    config.reskv_kernel_config = settings.to_dict()
    model = ResKVForCausalLM(config).eval()
    assert model.kernel_config == settings
    model.save_pretrained(tmp_path)
    restored = ResKVForCausalLM.from_pretrained(tmp_path)
    assert restored.kernel_config == settings


@pytest.mark.parametrize("implementation", ["accuracy", "kernel"])
def test_model_rejects_configuration_of_other_implementation(implementation):
    from transformers import LlamaConfig

    from reskv import ResKVConfig, ResKVForCausalLM, ResKVLlamaAccuracyForCausalLM

    config = LlamaConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=32,
    )
    model_cls, keyword, stored_key, wrong, expected = (
        (ResKVLlamaAccuracyForCausalLM, "reskv_config", "reskv_config", ResKVKernelConfig(), "ResKVConfig")
        if implementation == "accuracy"
        else (ResKVForCausalLM, "kernel_config", "reskv_kernel_config", ResKVConfig(), "ResKVKernelConfig")
    )
    with pytest.raises(TypeError, match=f"expected {expected}"):
        model_cls(config, **{keyword: wrong})
    # Stored settings must not be silently interpreted as the other method.
    setattr(config, stored_key, wrong.to_dict())
    with pytest.raises(ValueError, match="checkpoint compression settings"):
        model_cls(config)
    # An explicit configuration permits reusing checkpoint weights across implementations.
    correct = ResKVConfig() if implementation == "accuracy" else ResKVKernelConfig()
    model = model_cls(config, **{keyword: correct})
    assert getattr(model, keyword) == correct


@pytest.mark.parametrize("settings", [ResKVConfig(), False, 0, {}])
def test_kernel_compression_rejects_invalid_configuration_types(settings):
    from reskv import compress

    with pytest.raises(TypeError, match="compress requires ResKVKernelConfig"):
        compress(None, None, settings)


@pytest.mark.parametrize("implementation", ["kernel", "llama_accuracy", "qwen3_accuracy"])
def test_model_rejects_configuration_of_other_architecture(implementation):
    from transformers import LlamaConfig, Qwen3Config

    from reskv import ResKVForCausalLM, ResKVLlamaAccuracyForCausalLM, ResKVQwen3AccuracyForCausalLM

    model_cls, config_cls = {
        "kernel": (ResKVForCausalLM, Qwen3Config),
        "llama_accuracy": (ResKVLlamaAccuracyForCausalLM, Qwen3Config),
        "qwen3_accuracy": (ResKVQwen3AccuracyForCausalLM, LlamaConfig),
    }[implementation]
    with pytest.raises(ValueError, match="configuration"):
        model_cls(config_cls())


@pytest.mark.parametrize(
    "config_type,key", [(ResKVConfig, "reskv_config"), (ResKVKernelConfig, "reskv_kernel_config")]
)
@pytest.mark.parametrize("stored", [False, 0, "", []])
def test_invalid_saved_configuration_is_not_replaced_with_defaults(config_type, key, stored):
    from types import SimpleNamespace

    from reskv.config import _model_settings

    model_config = SimpleNamespace(**{key: stored})
    with pytest.raises(ValueError, match="checkpoint compression settings"):
        _model_settings(model_config, None, config_type, key)


def test_public_functions_survive_model_import_order():
    # Loading the model also imports internal attention; it must not shadow
    # the public callable with a Python submodule of the same name.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from reskv import ResKVForCausalLM; from reskv import attention, compress; "
            "assert callable(attention) and callable(compress)",
        ],
        check=True,
    )
