# SPDX-License-Identifier: Apache-2.0
"""Prompt, scoring and executable accuracy evaluation checks."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

import evaluation.run as evaluation_runner
from evaluation.prompts import RULER_GENERATION, generation_options, prepare_input
from evaluation.ruler_metrics import scorer as ruler_scorer
from evaluation.run import main


@pytest.fixture(params=["llama", "qwen3"])
def toy_checkpoint(tmp_path, request):
    torch.set_num_threads(2)
    torch.manual_seed(27)
    root = tmp_path / "model"
    config_cls, model_cls = (
        (LlamaConfig, LlamaForCausalLM) if request.param == "llama" else (Qwen3Config, Qwen3ForCausalLM)
    )
    config = config_cls(
        vocab_size=12,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model_cls(config).save_pretrained(root)
    tokens = {
        "[PAD]": 0,
        "[BOS]": 1,
        "[EOS]": 2,
        "[UNK]": 3,
        "word": 4,
        "answer": 5,
        "one": 6,
        "two": 7,
        "question": 8,
        "user": 9,
        "assistant": 10,
        "extra": 11,
    }
    tokenizer = Tokenizer(WordLevel(tokens, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]"
    )
    fast.chat_template = "{{ messages[0]['content'] }} assistant"
    fast.save_pretrained(root)
    return root, fast


def test_prompt_limits_and_task_stopping(toy_checkpoint):
    _, tok = toy_checkpoint
    text = "word " * 40
    ids, truncated = prepare_input(tok, text, "longbench", "qasper", "llama", "plain", 8)
    assert ids.shape[-1] == 8 and truncated
    ids, truncated = prepare_input(tok, text, "longbench", "qasper", "llama", "dataset", 8)
    plain, _ = prepare_input(tok, text, "longbench", "trec", "llama", "dataset", 8)
    assert ids.shape[-1] == plain.shape[-1] + 1 and truncated
    # Decode/concatenate/retokenize may itself change the exact token count.
    newline_tokenizer = SimpleNamespace(eos_token_id=2, encode=lambda *args, **kwargs: [11])
    options = generation_options(newline_tokenizer, "longbench", "samsum", 7, 12)
    assert options["min_length"] == 13 and len(options["eos_token_id"]) == 2
    assert RULER_GENERATION["vt"] == 30 and RULER_GENERATION["qa_1"] == 32


def test_ruler_metric_definitions():
    assert ruler_scorer("qa_1", ["Only ONE"], [["one", "two"]]) == 100
    assert ruler_scorer("niah_multivalue", ["ONE\n"], [["one", "two"]]) == 50
    assert ruler_scorer("qa_1", [" O\x00NE\t "], [["one"]]) == 100


@pytest.mark.parametrize(
    "prediction,reference,expected",
    [
        ("tide", "diet", 0.25),
        ("```python\n# comment\ntide\n```", "diet", 0.25),
        ("", "", 1.0),
        ("valid()", "", 0.0),
    ],
)
def test_code_similarity_is_independent_of_optional_extensions(prediction, reference, expected):
    from evaluation.longbench_metrics import code_sim_score

    # Levenshtein-based fuzz.ratio returns 0.50 for tide/diet; Python's
    # SequenceMatcher returns 0.25. Installing an extension must not change it.
    assert code_sim_score(prediction, reference) == expected


@pytest.mark.parametrize("suite,dataset", [("ruler", "qa_1"), ("longbench", "qasper")])
def test_qwen3_dataset_prompt(suite, dataset):
    class Tokenizer:
        chat_template = "template"

        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=torch.ones(1, 3, dtype=torch.long))

        def apply_chat_template(self, messages, **kwargs):
            self.messages, self.options = messages, kwargs
            return "formatted input"

    tokenizer = Tokenizer()
    prepare_input(tokenizer, "task input", suite, dataset, "qwen3", "dataset", 100)
    assert tokenizer.messages[-1] == {"role": "user", "content": "task input"}
    assert tokenizer.options["enable_thinking"] is False
    assert tokenizer.options["add_generation_prompt"] is True
    if suite == "ruler":
        assert tokenizer.messages[0] == {
            "role": "system",
            "content": "Answer directly and concisely. Do not explain your reasoning.",
        }
    else:
        assert len(tokenizer.messages) == 1


@pytest.mark.parametrize("model_type", ["llama", "qwen3"])
@pytest.mark.parametrize("style", ["dataset", "chat", "plain"])
def test_ruler_answer_prefix_follows_chat_formatting(model_type, style):
    class Tokenizer:
        chat_template = "template"

        def get_vocab(self):
            return {}

        def apply_chat_template(self, messages, **kwargs):
            assert messages[-1]["content"] == "question"
            return "user: question\nassistant:"

        def __call__(self, text, **kwargs):
            self.text = text
            return SimpleNamespace(input_ids=torch.ones(1, len(text), dtype=torch.long))

    tokenizer = Tokenizer()
    ids, truncated = prepare_input(
        tokenizer, "question", "ruler", "qa_1", model_type, style, 100, answer_prefix=" Answer:"
    )
    expected = "question Answer:" if style == "plain" else "user: question\nassistant: Answer:"
    assert tokenizer.text == expected
    assert ids.shape[-1] == len(expected) and not truncated


@pytest.mark.parametrize("model_type,preamble", [("llama", "<|begin_of_text|>"), ("qwen3", "<|im_start|>")])
def test_ruler_existing_answer_prefix_is_not_duplicated(model_type, preamble):
    class Tokenizer:
        chat_template = "template"

        def __call__(self, text, **kwargs):
            self.text = text
            return SimpleNamespace(input_ids=torch.ones(1, 10, dtype=torch.long))

    tokenizer = Tokenizer()
    prompt = preamble + "question\nassistant: Answer:"
    prepare_input(tokenizer, prompt, "ruler", "qa_1", model_type, "dataset", 100, answer_prefix=" Answer:")
    assert tokenizer.text == prompt


@pytest.mark.parametrize("frontend", ["reskv", "mustafar"])
@pytest.mark.parametrize("alias", ["same", "symlink", "summary", "output_alias"])
def test_accuracy_outputs_cannot_overwrite_inputs(tmp_path, monkeypatch, capsys, frontend, alias):
    from evaluation import mustafar

    source = tmp_path / ("data.summary.json" if alias == "summary" else "data.jsonl")
    text = json.dumps({"input": "question", "outputs": ["answer"]}) + "\n"
    source.write_text(text, encoding="utf-8")
    output = tmp_path / "data.jsonl" if alias == "summary" else source
    if alias == "output_alias":
        output = tmp_path / "predictions.jsonl"
        output.write_text("", encoding="utf-8")
        output.with_suffix(".summary.json").symlink_to(output)
    if alias == "symlink":
        output = tmp_path / "predictions.jsonl"
        output.symlink_to(source)
    arguments = [
        "evaluate",
        "--model",
        "/missing-model",
        "--suite",
        "ruler",
        "--dataset",
        "qa_1",
        "--input",
        str(source),
        "--output",
        str(output),
    ]
    if frontend == "mustafar":
        arguments += ["--mustafar-root", "/missing-mustafar"]
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(
        evaluation_runner.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail("model loading must not start"),
    )
    with pytest.raises(SystemExit) as error:
        (mustafar.main if frontend == "mustafar" else main)()
    expected = "output paths must be distinct" if alias == "output_alias" else "overwrite input"
    assert error.value.code == 2 and expected in capsys.readouterr().err
    assert source.read_text(encoding="utf-8") == text


def test_interrupted_generation_removes_stale_summary(tmp_path, toy_checkpoint):
    from evaluation import pipeline

    _, tokenizer = toy_checkpoint
    source, output = tmp_path / "data.jsonl", tmp_path / "predictions.jsonl"
    text = json.dumps({"input": "question", "outputs": ["answer"]}) + "\n"
    source.write_text(text, encoding="utf-8")
    summary = output.with_suffix(".summary.json")
    summary.write_text('{"samples": 100}\n', encoding="utf-8")

    def fail_generation(*args, **kwargs):
        raise RuntimeError("generation interrupted")

    model = SimpleNamespace(config=SimpleNamespace(model_type="llama"), generate=fail_generation)
    args = SimpleNamespace(
        suite="ruler",
        dataset="qa_1",
        input=source,
        output=output,
        limit=None,
        max_input_tokens=8192,
        max_new_tokens=None,
        prompt_format="plain",
        allow_truncation=False,
        device="cpu",
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        pipeline.evaluate(args, model, tokenizer, metadata={})
    assert not summary.exists()
    assert source.read_text(encoding="utf-8") == text


@pytest.mark.parametrize("suite,dataset", [("longbench", "trec"), ("ruler", "niah_single_1")])
@pytest.mark.parametrize("sparsity,bits", [(0.0, 16), (0.75, 16), (0.75, 4)])
def test_executable_evaluation(tmp_path, toy_checkpoint, monkeypatch, suite, dataset, sparsity, bits):
    # Installed packages may be reached through a symlink. Report source hashes
    # using one canonical root, while still exercising the full evaluation CLI.
    package_link = tmp_path / "package-root"
    package_link.symlink_to(Path(evaluation_runner.__file__).resolve().parents[1], target_is_directory=True)
    monkeypatch.setattr(evaluation_runner, "__file__", str(package_link / "evaluation/run.py"))
    root, _ = toy_checkpoint
    source, target = tmp_path / "input.jsonl", tmp_path / "predictions.jsonl"
    row = {
        "context": "word " * 20,
        "input": "word " * 20,
        "answers": ["answer"],
        "outputs": ["answer"],
        "all_classes": ["answer", "word"],
        "answer_prefix": " extra" if suite == "ruler" else "",
    }
    source.write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--model",
            str(root),
            "--input",
            str(source),
            "--suite",
            suite,
            "--dataset",
            dataset,
            "--sparsity",
            str(sparsity),
            "--bits",
            str(bits),
            "--device",
            "cpu",
            "--dtype",
            "fp32",
            "--max-new-tokens",
            "3",
            "--prompt-format",
            "plain",
            "--group-size",
            "4",
            "--recent-size",
            "2",
            "--kmeans-iterations",
            "2",
            "--output",
            str(target),
        ],
    )
    main()
    report = json.loads(target.with_suffix(".summary.json").read_text())
    prediction = json.loads(target.read_text())
    assert report["samples"] == 1 and 0 <= report["score"] <= 100
    assert report["method"] == "reskv"
    assert report["reskv_config"]["sparsity"] == sparsity
    assert report["reskv_config"]["bits"] == bits
    assert len(prediction["input_ids_sha256"]) == len(report["predictions_sha256"]) == 64
    assert prediction["generated_ids"] and "git_commit" in report
    if suite == "ruler":
        assert prediction["input_tokens"] == 21


def test_python_and_cli_defaults_produce_same_configuration_and_tokens(tmp_path, toy_checkpoint, monkeypatch):
    from evaluation.pipeline import seed_everything
    from reskv import ResKVConfig, ResKVLlamaAccuracyForCausalLM, ResKVQwen3AccuracyForCausalLM

    root, tokenizer = toy_checkpoint
    row = {
        "context": "word " * 193,
        "input": "question",
        "answers": ["answer"],
        "all_classes": ["answer", "word"],
    }
    source, target = tmp_path / "input.jsonl", tmp_path / "predictions.jsonl"
    source.write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--model",
            str(root),
            "--input",
            str(source),
            "--suite",
            "longbench",
            "--dataset",
            "trec",
            "--device",
            "cpu",
            "--dtype",
            "fp32",
            "--prompt-format",
            "plain",
            "--max-input-tokens",
            "256",
            "--max-new-tokens",
            "3",
            "--output",
            str(target),
        ],
    )
    main()
    summary = json.loads(target.with_suffix(".summary.json").read_text())
    prediction = json.loads(target.read_text())
    settings = ResKVConfig()
    assert summary["reskv_config"] == settings.to_dict()
    config = json.loads((root / "config.json").read_text())
    cls = ResKVLlamaAccuracyForCausalLM if config["model_type"] == "llama" else ResKVQwen3AccuracyForCausalLM
    model = cls.from_pretrained(root, dtype=torch.float32, attn_implementation="sdpa", reskv_config=settings)
    templates = json.loads(
        (Path(evaluation_runner.__file__).parent / "config/longbench_prompts.json").read_text()
    )
    ids, _ = prepare_input(
        tokenizer, templates["trec"].format(**row), "longbench", "trec", config["model_type"], "plain", 256
    )
    seed_everything(42)
    with torch.inference_mode():
        generated = model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            **generation_options(tokenizer, "longbench", "trec", 3, ids.shape[-1]),
        )
    assert generated[0, ids.shape[-1] :].tolist() == prediction["generated_ids"]


def test_ruler_rejects_truncated_tasks(tmp_path, toy_checkpoint, monkeypatch):
    root, _ = toy_checkpoint
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps({"input": "word " * 30, "outputs": ["answer"]}) + "\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--model",
            str(root),
            "--input",
            str(source),
            "--suite",
            "ruler",
            "--dataset",
            "niah_single_1",
            "--device",
            "cpu",
            "--dtype",
            "fp32",
            "--max-input-tokens",
            "8",
            "--prompt-format",
            "plain",
            "--output",
            str(tmp_path / "out.jsonl"),
        ],
    )
    with pytest.raises(ValueError, match="truncated"):
        main()
