# SPDX-License-Identifier: Apache-2.0
"""Benchmark the external Mustafar Llama model using its sparse CUDA kernels."""

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path


def main():
    if __package__:
        from . import timing
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from benchmarks import timing

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mustafar-root", type=Path, required=True)
    timing.add_arguments(parser)
    parser.add_argument("--sparsity", type=float, choices=[0.5, 0.75], default=0.75)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("results/mustafar.json"))
    args = parser.parse_args()
    timing.validate_arguments(parser, args)
    if args.group_size < 1:
        parser.error("group size must be positive")
    timing.validate_gpu(parser)
    root = args.mustafar_root.resolve()
    if not (root / "models/llama_mustafar_kernel.py").is_file():
        parser.error("--mustafar-root must contain the patched external Mustafar checkout")
    os.environ["MUSTAFAR_PREFILL_ATTN"] = "sdpa"
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "kernel/kernel_wrapper"))

    import torch
    import transformers
    from transformers import AutoConfig

    if transformers.__version__ != "4.44.2":
        parser.error("Mustafar compatibility environment requires transformers==4.44.2; see docs/mustafar.md")
    torch.manual_seed(args.seed)
    importlib.import_module("mustafar_package")  # fail early if the sparse kernels are not built
    module = importlib.import_module("models.llama_mustafar_kernel")
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    if config.model_type != "llama":
        parser.error("performance benchmarks support Llama checkpoints only")
    config.k_sparsity = config.v_sparsity = args.sparsity
    config.group_size = config.residual_length = args.group_size
    config.use_flash = config.use_cache = True
    model = (
        module.LlamaForCausalLM_MUSTAFAR.from_pretrained(
            args.model,
            config=config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        .cuda()
        .eval()
    )
    ids, ids_hash = timing.make_inputs(config.vocab_size, args.batch_size, args.input_length, args.seed)
    records = timing.measure(model, ids, args, logits_to_keep=False)
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    report = {
        **timing.report_metadata(args, ids_hash, transformers.__version__),
        "method": "mustafar",
        "model_dtype": "fp16",
        "sparsity": args.sparsity,
        "group_size": args.group_size,
        "attention": {"prefill": "sdpa", "decode": "mustafar_sparse_kernels"},
        "mustafar_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "records": records,
    }
    timing.write_report(args.output, report)


if __name__ == "__main__":
    main()
