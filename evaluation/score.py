# SPDX-License-Identifier: Apache-2.0
"""Score LongBench or RULER prediction JSONL files without loading a model."""

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

from reskv import _io


def normalize_prediction(row):
    prediction = next((row[k] for k in ("prediction", "pred", "predicted_answer") if k in row), None)
    answers = next((row[k] for k in ("answers", "answer", "outputs") if k in row), None)
    if isinstance(answers, str):
        answers = [answers]
    if not isinstance(prediction, str) or not isinstance(answers, list) or not answers:
        raise ValueError("each prediction requires text and a nonempty reference list")
    if any(not isinstance(answer, str) or not answer for answer in answers):
        raise ValueError("references must be nonempty strings")
    return {**row, "prediction": prediction, "answers": answers}


def known_tasks(suite):
    if suite == "longbench":
        from .longbench import dataset2metric

        return set(dataset2metric)
    from .prompts import RULER_GENERATION

    return set(RULER_GENERATION)


def score_predictions(suite, dataset, rows):
    if not rows:
        raise ValueError("cannot score empty predictions")
    rows = [normalize_prediction(row) for row in rows]
    predictions, answers = [r["prediction"] for r in rows], [r["answers"] for r in rows]
    if suite == "longbench":
        from .longbench import scorer

        return scorer(dataset, predictions, answers, rows[0].get("all_classes", []))
    from .ruler_metrics import scorer

    return scorer(dataset, predictions, answers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["longbench", "ruler"], required=True)
    parser.add_argument(
        "--input", type=Path, required=True, help="Prediction JSONL or directory of task JSONLs"
    )
    parser.add_argument("--dataset", help="Task name for a single file, or task to select")
    parser.add_argument("--output", type=Path, default=Path("results/scores.json"))
    args = parser.parse_args()
    if not args.input.is_file() and not args.input.is_dir():
        parser.error("--input must be a JSONL file or directory")
    paths = sorted(args.input.glob("*.jsonl")) if args.input.is_dir() else [args.input]
    try:
        _io.validate_output_paths(paths, [args.output])
    except ValueError as exc:
        parser.error(str(exc))
    groups, digests = defaultdict(list), {}
    for path in paths:
        for item in _io.read_jsonl(path):
            row = normalize_prediction(item)
            default_task = path.stem if args.input.is_dir() or args.suite == "longbench" else None
            if not args.input.is_dir() and args.dataset is not None:
                default_task = args.dataset
            task = row.get("task", default_task)
            if task is None:
                parser.error("RULER predictions require task fields or --dataset")
            if args.dataset is None or task == args.dataset:
                groups[task].append(row)
        digests[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not groups:
        parser.error("no selected predictions")
    unknown = set(groups) - set(known_tasks(args.suite))
    if unknown:
        parser.error(f"unknown tasks: {', '.join(sorted(unknown))}")
    scores = {task: score_predictions(args.suite, task, rows) for task, rows in sorted(groups.items())}
    report = {
        "suite": args.suite,
        "scores": scores,
        "macro_average": round(sum(scores.values()) / len(scores), 2),
        "samples_by_task": {task: len(rows) for task, rows in sorted(groups.items())},
        "task_count": len(scores),
        "input_sha256": digests,
    }
    print(_io.write_json(args.output, report))


if __name__ == "__main__":
    main()
