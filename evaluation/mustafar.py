# SPDX-License-Identifier: Apache-2.0
"""Evaluate the external Mustafar Llama accuracy model on LongBench or RULER."""

import argparse
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

# Also run by absolute path inside Mustafar's separate environment.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import pipeline

MODEL_SOURCE = "models/llama_mustafar_Kt_Mag_Vt_Mag.py"


def _last_position(module, inputs):
    return (inputs[0][:, -1:, :], *inputs[1:])


def load_model(args, config):
    """Load the requested checkout under a unique name; retain its pruning code."""
    import torch

    if config.model_type != "llama":
        raise ValueError("the external Mustafar accuracy implementation supports Llama only")
    if config.pretraining_tp != 1:
        raise ValueError("Mustafar accuracy requires pretraining_tp=1")
    source = args.mustafar_root.resolve() / MODEL_SOURCE
    if not source.is_file():
        raise ValueError(f"--mustafar-root must contain {MODEL_SOURCE}")
    name = "_reskv_external_mustafar_" + hashlib.sha256(str(source).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    config.k_sparsity = config.v_sparsity = args.sparsity
    config.group_size, config.residual_length = args.group_size, args.recent_size
    config.use_flash = config.use_cache = True
    # Keep the 4D causal mask used by its auxiliary prefill attention and decode.
    # The external forward separately calls FlashAttention with no padding mask.
    config._flash_attn_2_enabled = False
    cls = module.LlamaForCausalLM_MUSTAFAR
    cls._supports_cache_class = False
    model = cls.from_pretrained(args.model, config=config, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model = model.to(args.device).eval()
    model.lm_head.register_forward_pre_hook(_last_position)
    commit = subprocess.run(
        ["git", "-C", str(args.mustafar_root), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    metadata = {
        "method": "mustafar",
        "mustafar_config": {
            "sparsity": args.sparsity,
            "group_size": args.group_size,
            "recent_size": args.recent_size,
        },
        "external_model_class": cls.__name__,
        "mustafar_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "logits_policy": "final-position lm_head input",
        "cache_implementation": "legacy_tuple",
    }
    return model, metadata


def main():
    import torch
    import transformers
    from transformers import AutoConfig, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    pipeline.add_arguments(parser)
    parser.add_argument("--mustafar-root", type=Path, required=True)
    parser.add_argument("--sparsity", type=float, choices=[0.5, 0.75, 0.875], default=0.75)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--recent-size", type=int, default=64)
    args = parser.parse_args()
    pipeline.validate_arguments(parser, args)
    if min(args.group_size, args.recent_size) < 1:
        parser.error("group size and recent size must be positive")
    if args.dtype != "fp16" or not args.device.startswith("cuda"):
        parser.error("the external Mustafar accuracy model requires CUDA and --dtype fp16")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if transformers.__version__ != "4.44.2":
        parser.error("use the Mustafar environment with transformers==4.44.2; see docs/mustafar.md")
    pipeline.seed_everything(args.seed)
    config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    try:
        model, metadata = load_model(args, config)
    except ValueError as exc:
        parser.error(str(exc))
    pipeline.evaluate(args, model, tokenizer, metadata=metadata, last_position_logits=False)


if __name__ == "__main__":
    main()
