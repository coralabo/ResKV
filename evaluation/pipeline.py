# SPDX-License-Identifier: Apache-2.0
"""Shared accuracy inputs, generation, scoring and result metadata."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import transformers

from reskv import _io

from .prompts import RULER_GENERATION, generation_options, postprocess, prepare_input
from .score import known_tasks, score_predictions


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add_arguments(parser):
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--suite", choices=["longbench", "ruler"], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--prompt-format", choices=["plain", "chat", "dataset"], default="dataset")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-input-tokens", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument(
        "--allow-truncation", action="store_true", help="Allow truncation of generated RULER tasks"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=Path("results/predictions.jsonl"))


def validate_arguments(parser, args):
    if not args.input.is_file():
        parser.error("--input must be a JSONL file")
    try:
        _io.validate_output_paths([args.input], [args.output, args.output.with_suffix(".summary.json")])
    except ValueError as exc:
        parser.error(str(exc))
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.max_input_tokens < 2 or (args.max_new_tokens is not None and args.max_new_tokens < 1):
        parser.error("input cap must be >=2 and generation limit must be positive")
    if args.dataset not in known_tasks(args.suite):
        parser.error(f"unknown {args.suite} task: {args.dataset}")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def evaluate(args, model, tokenizer, *, metadata, last_position_logits=True):
    config_dir = Path(__file__).with_name("config")
    templates = json.loads((config_dir / "longbench_prompts.json").read_text(encoding="utf-8"))
    limits = json.loads((config_dir / "longbench_generation.json").read_text(encoding="utf-8"))
    max_new = args.max_new_tokens or (
        limits[args.dataset] if args.suite == "longbench" else RULER_GENERATION[args.dataset]
    )
    input_digest = sha256(args.input)
    rows = []
    for row in _io.read_jsonl(args.input):
        refs = row["answers"] if args.suite == "longbench" else row["outputs"]
        if isinstance(refs, str):
            refs = [refs]
        if not isinstance(refs, list) or not refs or any(not isinstance(x, str) or not x for x in refs):
            raise ValueError("each sample requires a nonempty list of nonempty reference strings")
        row["answers"] = refs
        rows.append(row)
        if args.limit is not None and len(rows) == args.limit:
            break
    if not rows:
        raise ValueError("input has no selected samples")
    model_type = model.config.model_type
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions = []
    with args.output.open("w", encoding="utf-8") as output, torch.inference_mode():
        args.output.with_suffix(".summary.json").unlink(missing_ok=True)
        for index, row in enumerate(rows):
            prompt = templates[args.dataset].format(**row) if args.suite == "longbench" else row["input"]
            ids, truncated = prepare_input(
                tokenizer,
                prompt,
                args.suite,
                args.dataset,
                model_type,
                args.prompt_format,
                args.max_input_tokens,
                answer_prefix=row.get("answer_prefix", "") if args.suite == "ruler" else "",
            )
            if args.suite == "ruler" and truncated and not args.allow_truncation:
                raise ValueError(
                    "RULER input would be truncated; increase the cap or explicitly allow truncation"
                )
            prompt_hash = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
            ids = ids.to(args.device)
            options = generation_options(tokenizer, args.suite, args.dataset, max_new, ids.shape[-1])
            if not last_position_logits:
                options.pop("logits_to_keep")
            result = model.generate(ids, attention_mask=torch.ones_like(ids), **options)
            generated = result[0, ids.shape[-1] :]
            item = {
                "id": row.get("index", row.get("id", index)),
                "prediction": postprocess(tokenizer.decode(generated, skip_special_tokens=True), model_type),
                "answers": row["answers"],
                "input_tokens": ids.shape[-1],
                "truncated": truncated,
                "input_ids_sha256": prompt_hash,
                "generated_ids": generated.tolist(),
                "all_classes": row.get("all_classes", []),
                "length": row.get("length"),
            }
            output.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()
            predictions.append(item)
    report = {
        "suite": args.suite,
        "dataset": args.dataset,
        "samples": len(predictions),
        "score": score_predictions(args.suite, args.dataset, predictions),
        "model": args.model,
        "model_type": model_type,
        "prompt_format": args.prompt_format,
        "dtype": args.dtype,
        "seed": args.seed,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": max_new,
        "truncated_samples": sum(x["truncated"] for x in predictions),
        "input_sha256": input_digest,
        "predictions_sha256": sha256(args.output),
        "git_commit": _io.git_commit(),
        "source_sha256": _io.source_sha256(),
        "tokenizer_class": type(tokenizer).__name__,
        "eos_token_id": model.generation_config.eos_token_id,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(torch.device(args.device))
        if args.device.startswith("cuda")
        else None,
        **metadata,
    }
    print(_io.write_json(args.output.with_suffix(".summary.json"), report))
    return report
