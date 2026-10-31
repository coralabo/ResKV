# SPDX-License-Identifier: Apache-2.0
"""CUDA event benchmark of the unified compression and attention APIs."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from reskv import ResKVKernelConfig, attention, compress
from reskv._io import write_json
from reskv.reference import reference_attention

from .timing import validate_gpu


def event_ms(fn, warmup=5, repeats=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) / repeats)
    return sorted(values)[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--sparsities", type=float, nargs="+", choices=[0.5, 0.75], default=[0.5, 0.75])
    parser.add_argument("--output", type=Path, default=Path("results/kernels.json"))
    args = parser.parse_args()
    if args.seq_len < 1 or min(args.batch_sizes) < 1:
        parser.error("sequence length and batch sizes must be positive")
    validate_gpu(parser)
    records = []
    for b in args.batch_sizes:
        for sparsity in args.sparsities:
            cfg = ResKVKernelConfig(sparsity=sparsity)
            torch.manual_seed(42)
            shape = (b, 8, args.seq_len, 128)
            k = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.5
            v = torch.randn_like(k) * 0.5
            kc = torch.randn(b, 8, 32, 128, device="cuda", dtype=torch.float16) * 0.1
            vc = torch.randn_like(kc) * 0.1
            ki = torch.randint(32, shape[:3], device="cuda", dtype=torch.int32)
            vi = torch.randint(32, shape[:3], device="cuda", dtype=torch.int32)
            q = torch.randn(b, 32, 1, 128, device="cuda", dtype=torch.float16) * 0.5

            def make(k=k, v=v, cfg=cfg, kc=kc, ki=ki, vc=vc, vi=vi):
                return compress(k, v, cfg, clustering=((kc, ki), (vc, vi)))

            cache = make()

            def fn(q=q, cache=cache):
                return attention(q, cache, position=args.seq_len)

            out = fn()
            ref = reference_attention(q, cache, position=args.seq_len)
            torch.testing.assert_close(out.float(), ref, atol=3e-3, rtol=3e-2)
            record = {
                "kernel_config": cfg.to_dict(),
                "kernel_plan": cache.key.plan.to_dict(),
                "batch_size": b,
                "seq_len": args.seq_len,
                "decode_ms": event_ms(fn),
                "compression_ms": event_ms(make),
                "max_abs_error": (out.float() - ref).abs().max().item(),
                "cache_bytes": cache.nbytes,
            }
            records.append(record)
            print(json.dumps(record), flush=True)
            del k, v, cache, ref
    write_json(
        args.output,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "records": records,
        },
    )


if __name__ == "__main__":
    main()
