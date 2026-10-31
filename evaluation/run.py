# SPDX-License-Identifier: Apache-2.0
"""Evaluate ResKV accuracy on LongBench or generated RULER JSONL."""

import argparse

import torch
from transformers import AutoConfig, AutoTokenizer

from reskv.config import ResKVConfig

from . import pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    pipeline.add_arguments(parser)
    parser.add_argument("--sparsity", type=float, default=0.75)
    parser.add_argument(
        "--bits", type=int, choices=[2, 4, 16], default=16, help="Residual bits; 16 disables quantization"
    )
    parser.add_argument("--num-clusters", type=int, default=32)
    parser.add_argument("--recent-size", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--kmeans-iterations", type=int, default=15)
    args = parser.parse_args()
    pipeline.validate_arguments(parser, args)
    try:
        config = ResKVConfig(
            sparsity=args.sparsity,
            bits=args.bits,
            num_clusters=args.num_clusters,
            recent_size=args.recent_size,
            group_size=args.group_size,
            sink_tokens=args.sink_tokens,
            kmeans_iterations=args.kmeans_iterations,
        )
    except ValueError as exc:
        parser.error(str(exc))
    pipeline.seed_everything(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_type = AutoConfig.from_pretrained(args.model).model_type
    from reskv.models import accuracy as models

    classes = {"llama": models.ResKVLlamaAccuracyForCausalLM, "qwen3": models.ResKVQwen3AccuracyForCausalLM}
    if model_type not in classes:
        parser.error("accuracy evaluation supports Llama and Qwen3")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    model = (
        classes[model_type]
        .from_pretrained(args.model, dtype=dtype, attn_implementation="sdpa", reskv_config=config)
        .to(args.device)
        .eval()
    )
    pipeline.evaluate(
        args,
        model,
        tokenizer,
        metadata={"method": "reskv", "reskv_config": config.to_dict()},
    )


if __name__ == "__main__":
    main()
