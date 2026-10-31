# SPDX-License-Identifier: Apache-2.0
"""Run ResKV, Mustafar and dense comparisons in separate sequential processes."""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from reskv._io import write_json

from .timing import add_arguments, add_model_arguments, validate_arguments, validate_gpu


def validate_reports(reports):
    """Reject comparisons whose runs differ in inputs, environment or decode counts."""
    fields = (
        "model",
        "model_dtype",
        "batch_size",
        "input_length",
        "new_tokens",
        "seed",
        "input_ids_sha256",
        "git_commit",
        "source_sha256",
        "gpu",
        "torch",
        "cuda",
        "repeats",
        "warmup",
    )
    first = reports[0]
    for report in reports:
        for field in fields:
            if report[field] != first[field]:
                raise ValueError(f"comparison mismatch in {field}")
        if len(report["records"]) != report["repeats"]:
            raise ValueError("unexpected record count")
        for record in report["records"]:
            if record["decode_steps"] != report["new_tokens"] - 1 or not record["finite_last_logits"]:
                raise ValueError("decode step count does not match output length, or nonfinite logits")


def summarize(report, path):
    records = report["records"]
    row = {"method": report["method"], "batch_size": report["batch_size"], "result": str(path)}
    for name in ("prefill_ms", "decode_ms", "tpot_ms", "throughput_tps"):
        values = [record[name] for record in records if record[name] is not None]
        row[name] = statistics.median(values) if values else None
    row["peak_allocated_bytes"] = max(record["peak_allocated_bytes"] for record in records)
    return row


def command_for(args, method, batch_size, output):
    common = [
        "--model",
        args.model,
        "--batch-size",
        str(batch_size),
        "--input-length",
        str(args.input_length),
        "--new-tokens",
        str(args.new_tokens),
        "--seed",
        str(args.seed),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--output",
        str(output),
    ]
    if method == "mustafar":
        return [
            args.mustafar_python,
            str(Path(__file__).with_name("mustafar.py").resolve()),
            "--mustafar-root",
            str(args.mustafar_root),
            "--sparsity",
            str(args.sparsity),
            "--group-size",
            str(args.mustafar_group_size),
            *common,
        ]
    command = [
        sys.executable,
        "-m",
        "benchmarks.inference",
        "--method",
        method,
        "--dtype",
        args.dtype,
        *common,
    ]
    if method == "reskv":
        command += ["--sparsity", str(args.sparsity)]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, batch_sizes=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["reskv", "mustafar", "dense"],
        default=["reskv", "mustafar", "dense"],
    )
    add_model_arguments(parser)
    parser.add_argument("--mustafar-root", type=Path)
    parser.add_argument(
        "--mustafar-python", default=sys.executable, help="Python executable in Mustafar's environment"
    )
    parser.add_argument("--mustafar-group-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("results/comparison"))
    args = parser.parse_args()
    validate_arguments(parser, args)
    if args.mustafar_group_size < 1:
        parser.error("--mustafar-group-size must be positive")
    if "mustafar" in args.methods and args.mustafar_root is None:
        parser.error("--mustafar-root is required when comparing Mustafar")
    if args.dtype != "fp16" and any(method != "dense" for method in args.methods):
        parser.error("comparisons including ResKV or Mustafar require --dtype fp16")
    if len(set(args.methods)) != len(args.methods) or len(set(args.batch_sizes)) != len(args.batch_sizes):
        parser.error("methods and batch sizes must not contain duplicates")
    validate_gpu(parser)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"runs": [], "errors": []}
    for batch_size in args.batch_sizes:
        reports = []
        rows = []
        for method in args.methods:
            target = args.output / f"{method}-b{batch_size}.json"
            log = target.with_suffix(".log")
            target.unlink(missing_ok=True)
            command = command_for(args, method, batch_size, target)
            print(f"Running {method}, batch={batch_size}; log: {log}", flush=True)
            try:
                with log.open("w", encoding="utf-8") as stream:
                    subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
                report = json.loads(target.read_text(encoding="utf-8"))
                if report["method"] != method:
                    raise ValueError("result method does not match requested method")
                reports.append(report)
                rows.append(summarize(report, target))
            except (OSError, subprocess.CalledProcessError, ValueError, KeyError) as exc:
                summary["errors"].append(
                    {"method": method, "batch_size": batch_size, "error": str(exc), "log": str(log)}
                )
        if reports:
            try:
                validate_reports(reports)
                summary["runs"].extend(rows)
            except (ValueError, KeyError) as exc:
                summary["errors"].append({"batch_size": batch_size, "error": str(exc)})
    summary_path = args.output / "summary.json"
    write_json(summary_path, summary)
    print(f"Results: {summary_path}")
    if summary["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
