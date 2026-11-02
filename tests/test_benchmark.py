# SPDX-License-Identifier: Apache-2.0
"""Verify dense attention phases, repeated calls and shared continuation counts."""

import argparse
import importlib
import sys
import weakref
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from benchmarks import timing


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_kernel_sweep_releases_previous_case(tmp_path, monkeypatch):
    from benchmarks import kernels

    tracked = []
    original_randn, original_compress = torch.randn, kernels.compress

    def randn(*shape, **kwargs):
        if shape == ((1, 8, 64, 128),):
            assert all(ref() is None for ref in tracked)
        return original_randn(*shape, **kwargs)

    def compress(*args, **kwargs):
        cache = original_compress(*args, **kwargs)
        tracked.extend(weakref.ref(x) for x in (*args[:2], cache.key.payload, cache.value.payload))
        return cache

    monkeypatch.setattr(torch, "randn", randn)
    monkeypatch.setattr(kernels, "compress", compress)
    monkeypatch.setattr(kernels, "validate_gpu", lambda parser: None)
    monkeypatch.setattr(kernels, "event_ms", lambda fn: 0.0)
    monkeypatch.setattr(
        sys,
        "argv",
        ["kernels", "--batch-sizes", "1", "--seq-len", "64", "--output", str(tmp_path / "kernels.json")],
    )
    kernels.main()
    assert all(ref() is None for ref in tracked)


@pytest.mark.parametrize("name", [timing.BENCHMARK_GPU, "  nvidia RTX PRO 6000 Blackwell Server Edition  "])
def test_benchmark_gpu_accepts_server_edition(monkeypatch, name):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: name)
    timing.validate_gpu(argparse.ArgumentParser())


@pytest.mark.parametrize(
    "name",
    [
        "NVIDIA GeForce RTX 4080 SUPER",
        "NVIDIA RTX 6000 Ada Generation",
        "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
    ],
)
def test_benchmark_gpu_rejects_other_models(monkeypatch, capsys, name):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: name)
    with pytest.raises(SystemExit) as error:
        timing.validate_gpu(argparse.ArgumentParser())
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert timing.BENCHMARK_GPU in message and f"detected {name}" in message


def test_benchmark_gpu_rejects_unavailable_cuda(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: pytest.fail("CUDA must not be queried"))
    with pytest.raises(SystemExit) as error:
        timing.validate_gpu(argparse.ArgumentParser())
    assert error.value.code == 2
    assert "CUDA is unavailable" in capsys.readouterr().err


@pytest.mark.parametrize(
    "module,arguments",
    [
        ("inference", ["--model", "/missing-model", "--method", "reskv"]),
        ("inference", ["--model", "/missing-model", "--method", "dense"]),
        ("kernels", []),
        ("compare", ["--model", "/missing-model", "--methods", "reskv", "dense"]),
        ("mustafar", ["--model", "/missing-model", "--mustafar-root", "/missing-mustafar"]),
    ],
)
def test_benchmark_entry_points_reject_gpu_before_output(tmp_path, monkeypatch, capsys, module, arguments):
    entry = importlib.import_module("benchmarks." + module)
    target = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", [module, *arguments, "--output", str(target)])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "NVIDIA GeForce RTX 4080 SUPER")
    if module == "compare":
        monkeypatch.setattr(entry.subprocess, "run", lambda *a, **kw: pytest.fail("child process launched"))
    with pytest.raises(SystemExit) as error:
        entry.main()
    assert error.value.code == 2
    assert "detected NVIDIA GeForce RTX 4080 SUPER" in capsys.readouterr().err
    assert not target.exists()


@pytest.mark.parametrize("new_tokens", [1, 4])
def test_dense_attention_phases_and_decode_count(monkeypatch, new_tokens):
    flash_active = False

    @contextmanager
    def flash_context(backend):
        nonlocal flash_active
        assert backend == timing.SDPBackend.FLASH_ATTENTION
        flash_active = True
        try:
            yield
        finally:
            flash_active = False

    class Model:
        implementation = "sdpa"

        def __init__(self):
            self.calls = []

        def set_attn_implementation(self, implementation):
            self.implementation = implementation

        def __call__(self, ids, **kwargs):
            self.calls.append((ids.shape[-1], self.implementation, flash_active))
            return SimpleNamespace(
                logits=torch.tensor([[[0.0, 1.0, 0.0]]]),
                past_key_values=SimpleNamespace(layers=[]),
            )

    monkeypatch.setattr(timing, "sdpa_kernel", flash_context)
    for name in ("synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    for name in ("max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    model = Model()
    for _ in range(2):
        model.calls.clear()
        record = timing.run(model, torch.ones(1, 8, dtype=torch.long), new_tokens, dense=True)
        assert len(model.calls) == new_tokens
        assert model.calls[0] == (8, "sdpa", True)
        assert model.calls[1:] == [(1, "eager", False)] * (new_tokens - 1)
        assert record["finite_last_logits"]
        assert (record["tpot_ms"] is None) == (new_tokens == 1)


def test_cache_storage_deduplication():
    from dataclasses import dataclass

    @dataclass
    class Payload:
        data: torch.Tensor
        native_view: torch.Tensor

    storage = torch.zeros(64, dtype=torch.float16)
    other = torch.ones(8)
    packed = Payload(storage[:16], storage.view(8, 8))
    layer_cache = SimpleNamespace(layers=[SimpleNamespace(compressed=packed, keys=other, values=other[:2])])
    tuple_cache = (([storage[:2], storage[3:4]], {"values": other}, 8, None),)
    expected = storage.untyped_storage().nbytes() + other.untyped_storage().nbytes()
    assert timing.cache_bytes(layer_cache) == timing.cache_bytes(tuple_cache) == expected


def test_nonfinite_report_does_not_replace_existing_results(tmp_path):
    output = tmp_path / "result.json"
    output.write_text('{"complete": true}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        timing.write_report(output, {"throughput_tps": float("nan")})
    assert output.read_text(encoding="utf-8") == '{"complete": true}\n'


@pytest.mark.parametrize("new_tokens", [1, 4])
def test_tuple_cache_continuation(monkeypatch, new_tokens):
    """External Mustafar receives the preceding tuple cache and no newer HF kwargs."""
    calls = []
    cache = ((torch.zeros(1), [torch.ones(2)], 7),)

    def model(ids, *, use_cache, past_key_values=None):
        assert use_cache
        if calls:
            assert past_key_values is cache and ids.tolist() == [[2]]
        else:
            assert past_key_values is None and ids.shape[-1] == 8
        calls.append(ids)
        return SimpleNamespace(logits=torch.tensor([[[0.0, 1.0, 2.0]]]), past_key_values=cache)

    for name in ("synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    for name in ("max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    record = timing.run(model, torch.ones(1, 8, dtype=torch.long), new_tokens, logits_to_keep=False)
    assert len(calls) == new_tokens
    assert record["decode_steps"] == new_tokens - 1
    assert record["cache_bytes"] == 12
    assert (record["tpot_ms"] is None) == (new_tokens == 1)


def test_deterministic_inputs():
    first, digest = timing.make_inputs(32000, 2, 64, 42, device="cpu")
    second, same = timing.make_inputs(32000, 2, 64, 42, device="cpu")
    changed, different = timing.make_inputs(32000, 2, 64, 43, device="cpu")
    assert torch.equal(first, second) and digest == same
    assert not torch.equal(first, changed) and digest != different
    tiny, _ = timing.make_inputs(12, 1, 16, 42, device="cpu")
    assert tiny.min() >= 0 and tiny.max() < 12


@pytest.mark.parametrize(
    "field",
    [
        "input_ids_sha256",
        "git_commit",
        "source_sha256",
        "model_dtype",
        "gpu",
        "torch",
        "cuda",
        "decode_steps",
        "finite_last_logits",
    ],
)
def test_comparison_rejects_incompatible_measurements(field):
    from copy import deepcopy

    from benchmarks.compare import validate_reports

    report = {
        "model": "model",
        "model_dtype": "fp16",
        "batch_size": 1,
        "input_length": 8,
        "new_tokens": 4,
        "seed": 42,
        "input_ids_sha256": "abc",
        "git_commit": "def",
        "source_sha256": "abc123",
        "gpu": "gpu",
        "torch": "2.8.0+cu128",
        "cuda": "12.8",
        "repeats": 1,
        "warmup": 1,
        "records": [{"decode_steps": 3, "finite_last_logits": True}],
    }
    other = deepcopy(report)
    validate_reports([report, other])
    if field in other:
        other[field] = "changed"
    else:
        other["records"][0][field] = False if field == "finite_last_logits" else 4
    with pytest.raises(ValueError):
        validate_reports([report, other])
