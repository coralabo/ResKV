# SPDX-License-Identifier: Apache-2.0
"""Lazy, content-addressed builds outside the source tree."""

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import torch

from .config import KernelPlan


@lru_cache(maxsize=32)
def load_extension(family: str, plan: KernelPlan | None = None, device_index: int = 0):
    from torch.utils.cpp_extension import load

    if family not in ("compression", "attention", "cluster_mass"):
        raise ValueError(f"unknown native family: {family}")
    capability = torch.cuda.get_device_capability(device_index)
    if capability < (8, 0):
        raise RuntimeError("ResKV fused kernels require NVIDIA compute capability >= 8.0")
    source_dir = Path(__file__).parent / "csrc"
    flags = [
        "-O3",
        "-std=c++17",
        "--use_fast_math",
        f"-gencode=arch=compute_{capability[0]}{capability[1]},code=sm_{capability[0]}{capability[1]}",
    ]
    if plan and family == "attention":
        flags += [
            f"-DFK16_QTILE_STRUCT_INDEX_4X16={int(plan.layout == 'structured')}",
            f"-DRQ_QTILE_TOKENS_PER_WARP={plan.tokens_per_warp if plan.bits != 16 and plan.layout != 'structured' else 16}",
        ]
    if plan and family == "cluster_mass":
        flags += [
            f"-DCMASS_TOKENS_PER_WARP={plan.tokens_per_warp}",
            f"-DCMASS_INT4_FIXED_SHIFT_SOFTMAX={int(plan.fixed_shift)}",
            f"-DCMASS_INT4_ROTATE_K_ONCE={int(plan.bits == 4)}",
            f"-DCMASS_INT4_DIRECT_PACKED={int(plan.bits == 4)}",
        ]
    # Hash sources, headers, flags and PyTorch/CUDA versions for the build cache.
    digest = hashlib.sha256()
    for p in [source_dir / f"{family}.cu", *sorted(source_dir.glob("*.cuh"))]:
        digest.update(p.name.encode())
        digest.update(p.read_bytes())
    digest.update(str((flags, torch.__version__, torch.version.cuda)).encode())
    name = f"reskv_{family}_{digest.hexdigest()[:16]}"
    cache_root = Path(os.environ.get("RESKV_CACHE_DIR", Path.home() / ".cache" / "reskv"))
    build_dir = cache_root / name
    build_dir.mkdir(parents=True, exist_ok=True)
    # PyTorch honours MAX_JOBS; do not overwrite an explicitly configured value.
    os.environ.setdefault("MAX_JOBS", os.environ.get("RESKV_BUILD_JOBS", "4"))
    with torch.cuda.device(device_index):
        return load(
            name=name,
            sources=[str(source_dir / f"{family}.cu")],
            build_directory=str(build_dir),
            extra_cuda_cflags=flags,
            verbose=os.environ.get("RESKV_VERBOSE_BUILD") == "1",
        )
