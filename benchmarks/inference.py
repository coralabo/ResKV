# SPDX-License-Identifier: Apache-2.0
"""Benchmark ResKV or the dense Llama baseline on a local checkpoint."""

import argparse
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig, LlamaForCausalLM

from reskv import ResKVForCausalLM, ResKVKernelConfig

from .timing import (
    add_arguments,
    add_model_arguments,
    make_inputs,
    measure,
    report_metadata,
    validate_arguments,
    validate_gpu,
    write_report,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--method", choices=["reskv", "dense"], default="reskv")
    add_model_arguments(parser)
    parser.add_argument("--output", type=Path, default=Path("results/inference.json"))
    args = parser.parse_args()
    validate_arguments(parser, args)
    if args.method == "reskv" and args.dtype != "fp16":
        parser.error("ResKV CUDA inference requires --dtype fp16")
    validate_gpu(parser)
    config = None
    if args.method == "reskv":
        try:
            config = ResKVKernelConfig(
                sparsity=args.sparsity,
                layout="unstructured" if args.reskv_implementation == "unstructure" else "auto",
            )
            config.resolve(args.batch_size)
        except ValueError as exc:
            parser.error(str(exc))
    torch.manual_seed(args.seed)
    model_config = AutoConfig.from_pretrained(args.model)
    if model_config.model_type != "llama":
        parser.error("performance benchmarks support Llama checkpoints only")
    kwargs = {
        "config": model_config,
        "dtype": torch.float16 if args.dtype == "fp16" else torch.bfloat16,
        "attn_implementation": "sdpa",
    }
    if args.method == "dense":
        model = LlamaForCausalLM.from_pretrained(args.model, **kwargs)
    else:
        model = ResKVForCausalLM.from_pretrained(args.model, kernel_config=config, **kwargs)
    model = model.cuda().eval()
    ids, ids_hash = make_inputs(model.config.vocab_size, args.batch_size, args.input_length, args.seed)
    records = measure(model, ids, args, dense=args.method == "dense")
    report = {
        **report_metadata(args, ids_hash, transformers.__version__),
        "method": args.method,
        "model_dtype": args.dtype,
        "kernel_config": config.to_dict() if config else None,
        "kernel_plan": config.resolve(args.batch_size).to_dict() if config else None,
        "records": records,
    }
    write_report(args.output, report)


if __name__ == "__main__":
    main()
