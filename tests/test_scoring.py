# SPDX-License-Identifier: Apache-2.0
"""Shared scoring contracts for ResKV and external accuracy predictions."""

import json
import sys
from types import SimpleNamespace

import pytest
import torch

from evaluation.mustafar import _last_position, load_model
from evaluation.score import main, normalize_prediction, score_predictions


@pytest.mark.parametrize(
    "row",
    [
        {"prediction": "answer", "answers": ["answer"]},
        {"pred": "answer", "answers": ["answer"]},
        {"predicted_answer": "answer", "answer": "answer"},
        {"prediction": "answer", "outputs": ["answer"]},
    ],
)
def test_prediction_schemas_share_scores(row):
    normalized = normalize_prediction(row)
    assert normalized["prediction"] == "answer" and normalized["answers"] == ["answer"]
    assert score_predictions("ruler", "qa_1", [row]) == 100


@pytest.mark.parametrize(
    "row",
    [
        {"answers": ["answer"]},
        {"prediction": "answer", "answers": []},
        {"prediction": "answer", "answers": [None]},
        {"prediction": "answer", "answers": [""]},
    ],
)
def test_invalid_predictions_are_rejected(row):
    with pytest.raises(ValueError):
        normalize_prediction(row)


def test_empty_predictions_are_rejected():
    with pytest.raises(ValueError, match="empty"):
        score_predictions("ruler", "qa_1", [])


@pytest.mark.parametrize("alias", ["same", "symlink", "directory"])
def test_scoring_cannot_overwrite_predictions(tmp_path, monkeypatch, capsys, alias):
    source = tmp_path / "qa_1.jsonl"
    text = json.dumps({"prediction": "one", "answers": ["one"]}) + "\n"
    source.write_text(text, encoding="utf-8")
    output = source
    if alias == "symlink":
        output = tmp_path / "scores.json"
        output.symlink_to(source)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score",
            "--suite",
            "ruler",
            "--dataset",
            "qa_1",
            "--input",
            str(tmp_path if alias == "directory" else source),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2 and "overwrite input" in capsys.readouterr().err
    assert source.read_text(encoding="utf-8") == text


def test_output_protection_checks_every_target(tmp_path):
    from reskv._io import validate_output_paths

    source = tmp_path / "input.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="overwrite input"):
        validate_output_paths(iter([source]), [tmp_path / "predictions.jsonl", source])


@pytest.mark.parametrize("metric", ["longbench", "ruler_qa", "ruler_all"])
def test_scoring_rejects_mismatched_prediction_counts(metric):
    if metric == "longbench":
        from evaluation.longbench import scorer

        function, arguments = scorer, ("qasper", ["one", "two"], [["one"]], [])
    else:
        from evaluation.ruler_metrics import string_match_all, string_match_part

        function = string_match_part if metric == "ruler_qa" else string_match_all
        arguments = (["one", "two"], [["one"]])
    with pytest.raises(ValueError):
        function(*arguments)


@pytest.mark.parametrize(
    "suite,first,second", [("longbench", "qasper", "hotpotqa"), ("ruler", "qa_1", "qa_2")]
)
def test_directory_task_selection(tmp_path, monkeypatch, suite, first, second):
    inputs = tmp_path / "predictions"
    inputs.mkdir()
    (inputs / f"{first}.jsonl").write_text(json.dumps({"pred": "one", "answers": ["one"]}) + "\n")
    (inputs / f"{second}.jsonl").write_text(json.dumps({"pred": "two", "answers": ["one"]}) + "\n")
    output = tmp_path / "scores.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score",
            "--suite",
            suite,
            "--input",
            str(inputs),
            "--dataset",
            first,
            "--output",
            str(output),
        ],
    )
    main()
    report = json.loads(output.read_text())
    assert report["scores"] == {first: 100}
    assert report["samples_by_task"] == {first: 1}


@pytest.mark.parametrize("directory", [False, True])
def test_ruler_macro_average_weights_tasks_equally(tmp_path, monkeypatch, directory):
    source, output = tmp_path / "predictions.jsonl", tmp_path / "scores.json"
    rows = [{"task": "qa_1", "predicted_answer": "one", "answer": ["one"]}] * 3
    rows.append({"task": "qa_2", "prediction": "two", "outputs": ["one"]})
    if directory:
        source = tmp_path / "predictions"
        source.mkdir()
        for row in rows:
            task = row["task"]
            with (source / f"{task}.jsonl").open("a") as stream:
                stream.write(json.dumps({key: value for key, value in row.items() if key != "task"}) + "\n")
    else:
        source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(
        sys, "argv", ["score", "--suite", "ruler", "--input", str(source), "--output", str(output)]
    )
    main()
    report = json.loads(output.read_text())
    assert report["macro_average"] == 50
    assert report["samples_by_task"] == {"qa_1": 3, "qa_2": 1}
    assert report["task_count"] == 2
    assert all(len(digest) == 64 for digest in report["input_sha256"].values())


def test_last_position_logits_preserve_final_logits():
    torch.manual_seed(19)
    head = torch.nn.Linear(16, 23)
    hidden = torch.randn(1, 17, 16)
    expected = head(hidden)[:, -1:, :]
    handle = head.register_forward_pre_hook(_last_position)
    try:
        actual = head(hidden)
    finally:
        handle.remove()
    assert actual.shape == (1, 1, 23)
    torch.testing.assert_close(actual, expected)


def test_external_mustafar_rejects_unsupported_models():
    with pytest.raises(ValueError, match="Llama only"):
        load_model(SimpleNamespace(), SimpleNamespace(model_type="qwen3"))
    with pytest.raises(ValueError, match="pretraining_tp"):
        load_model(SimpleNamespace(), SimpleNamespace(model_type="llama", pretraining_tp=2))
