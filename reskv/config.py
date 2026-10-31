# SPDX-License-Identifier: Apache-2.0
"""Accuracy-evaluation and CUDA-kernel configurations."""

from dataclasses import asdict, dataclass, fields
from typing import Literal


@dataclass(frozen=True)
class ResKVConfig:
    """Compression-error simulation for Python/PyTorch accuracy evaluation.

    Pre-RoPE Keys and Values are clustered, the residuals are sparsified (and
    optionally quantized), and dense KV is reconstructed from the result.

    Attributes:
        sparsity: Fraction of residual coordinates discarded (0 disables compression).
        bits: Residual bit width; 16 disables quantization.
        num_clusters: K-means clusters per head at prefill.
        recent_size: Most recent tokens kept uncompressed.
        group_size: Tokens compressed together during decode.
        sink_tokens: Initial tokens kept uncompressed.
        kmeans_iterations: Lloyd iterations at prefill.
        layout: ``unstructured`` top-|r| or ``structured`` 4:16 residual masks.
    """

    sparsity: float = 0.75
    bits: int = 16
    num_clusters: int = 32
    recent_size: int = 32
    group_size: int = 64
    sink_tokens: int = 4
    kmeans_iterations: int = 15
    layout: str = "unstructured"

    def __post_init__(self):
        if isinstance(self.sparsity, bool) or self.sparsity not in (0.0, 0.5, 0.75, 0.875):
            raise ValueError("accuracy sparsity must be 0, 0.5, 0.75, or 0.875")
        if type(self.bits) is not int or self.bits not in (2, 4, 8, 16):
            raise ValueError("accuracy bits must be 2, 4, 8, or 16 (16 disables quantization)")
        for name in ("num_clusters", "group_size", "kmeans_iterations"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("recent_size", "sink_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.layout not in ("unstructured", "structured"):
            raise ValueError("accuracy layout must be unstructured or structured")
        if self.layout == "structured" and self.sparsity != 0.75:
            raise ValueError("4:16 accuracy sparsity requires sparsity=0.75")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ResKVKernelConfig:
    """CUDA compression and fused-decode settings.

    Attributes:
        sparsity: Fraction of residual coordinates discarded per token (0.5 or 0.75).
        bits: Residual bit width; 16 keeps sparse FP16 residuals, 4/2 also quantize them.
        layout: Residual mask layout. ``unstructured`` keeps the top-|r| coordinates
            of each token, ``structured`` keeps 4 of every 16 (sparsity 0.75 only),
            ``tiled`` is a fixed-tile FP16 50% variant; ``auto`` follows ``resolve``.
        centroid_fusion: How decode adds back the Value centroids. ``per_token``
            adds each token's centroid inside the attention loop; ``deferred``
            accumulates attention mass per cluster and applies the centroids once
            after the split-K reduction (Value deferred fusion in the paper).
        num_clusters: K-means clusters per head at prefill.
        max_clusters: Centroid limit; each compressed decode group appends one centroid.
        group_size: Tokens compressed together during decode.
        recent_size: Most recent tokens kept dense (FP16) in the sliding window.
        kmeans_iterations: Lloyd iterations at prefill.
        rope_mode: ``model`` uses the checkpoint's RoPE frequencies; ``theta``
            recomputes them from ``rope_theta``.
        softmax_mode: ``online`` softmax or ``fixed_shift`` (structured INT4 only).
    """

    sparsity: float = 0.5
    bits: Literal[2, 4, 16] = 16
    layout: Literal["auto", "unstructured", "tiled", "structured"] = "auto"
    centroid_fusion: Literal["auto", "per_token", "deferred"] = "auto"
    num_clusters: int = 32
    max_clusters: int = 64
    group_size: int = 64
    recent_size: int = 32
    kmeans_iterations: int = 10
    rope_mode: Literal["model", "theta"] = "model"
    softmax_mode: Literal["online", "fixed_shift"] = "online"

    def __post_init__(self):
        for name in ("num_clusters", "max_clusters", "group_size", "recent_size", "kmeans_iterations"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.sparsity not in (0.5, 0.75):
            raise ValueError("sparsity must be 0.5 or 0.75 (discarded fraction)")
        if type(self.bits) is not int or self.bits not in (2, 4, 16):
            raise ValueError("bits must be 2, 4, or 16 (FP16 sparse-only)")
        if self.layout not in ("auto", "unstructured", "tiled", "structured"):
            raise ValueError("unknown sparsity layout")
        if self.centroid_fusion not in ("auto", "per_token", "deferred"):
            raise ValueError("centroid_fusion must be auto, per_token or deferred")
        if not 1 <= self.num_clusters <= self.max_clusters <= 64:
            raise ValueError("require 1 <= num_clusters <= max_clusters <= 64")
        if self.group_size <= 0 or self.recent_size < 0 or self.kmeans_iterations <= 0:
            raise ValueError("group_size/iterations must be positive; recent_size nonnegative")
        if self.rope_mode not in ("model", "theta"):
            raise ValueError("rope_mode must be 'model' or 'theta'")
        if self.softmax_mode not in ("online", "fixed_shift"):
            raise ValueError("softmax_mode must be 'online' or 'fixed_shift'")
        if self.softmax_mode == "fixed_shift" and not (self.bits == 4 and self.sparsity == 0.75):
            raise ValueError("fixed_shift softmax requires sparse75 INT4")
        if self.layout == "structured" and (self.sparsity != 0.75 or self.bits == 2):
            raise ValueError("structured layout requires sparsity=0.75 and bits=4 or 16")
        if self.layout == "tiled" and (self.bits != 16 or self.sparsity != 0.5):
            raise ValueError("tiled masks support only FP16 residuals at sparsity 0.5")

    @property
    def keep_ratio(self) -> float:
        return 1.0 - self.sparsity

    def to_dict(self) -> dict:
        return asdict(self)

    def resolve(self, batch_size: int) -> "KernelPlan":
        """Select the layout, centroid fusion and launch schedule for a batch size."""
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        layout = self.layout
        fusion = self.centroid_fusion
        if layout == "auto":
            if self.sparsity == 0.75 and self.bits != 2:
                layout = "structured"
            else:
                layout = "unstructured"
        if fusion == "auto":
            if layout == "tiled" or (self.bits != 16 and layout == "unstructured"):
                fusion = "per_token"
            elif layout == "structured" and batch_size <= 2:
                fusion = "per_token"
            else:
                fusion = "deferred"
        if layout == "tiled" and fusion != "per_token":
            raise ValueError("tiled masks require per_token centroid fusion")
        if layout == "unstructured" and self.bits == 16 and fusion != "deferred":
            raise ValueError("unstructured FP16 requires deferred centroid fusion")
        if layout == "unstructured" and self.bits != 16 and fusion != "per_token":
            raise ValueError("unstructured quantized caches require per_token centroid fusion")
        if self.softmax_mode == "fixed_shift" and (layout != "structured" or fusion != "deferred"):
            raise ValueError("fixed_shift requires structured layout with deferred centroid fusion")
        splits = 0
        tpw = 4 if fusion == "per_token" else 2
        if layout == "structured":
            splits, tpw = 20, 4
            if self.bits == 4 and fusion == "per_token" and batch_size >= 8:
                splits = 16
            elif self.bits == 4 and fusion == "deferred":
                tpw, splits = (4, 10) if batch_size >= 32 else (8, 14) if batch_size >= 8 else (4, 20)
        elif self.bits != 16:
            if batch_size <= 2:
                tpw, splits = 16, 40
            elif batch_size <= 4:
                tpw, splits = 8, 20
            elif batch_size == 8:
                tpw = 12
            else:
                tpw = 16
        return KernelPlan(
            layout,
            fusion,
            self.bits,
            int(128 * self.keep_ratio),
            tpw,
            splits,
            self.softmax_mode == "fixed_shift",
        )


def _model_settings(model_config, settings, config_type, key):
    """Use explicit settings, or read them from ``model_config.<key>``; store them back."""
    if settings is not None and not isinstance(settings, config_type):
        raise TypeError(f"expected {config_type.__name__}, got {type(settings).__name__}")
    if settings is None:
        stored = getattr(model_config, key, None)
        stored = {} if stored is None else stored
        if not isinstance(stored, dict) or set(stored) - {field.name for field in fields(config_type)}:
            raise ValueError(
                f"checkpoint compression settings do not match {config_type.__name__}; "
                "pass an explicit configuration"
            )
        settings = config_type(**stored)
    setattr(model_config, key, settings.to_dict())
    return settings


@dataclass(frozen=True)
class KernelPlan:
    """Resolved implementation for one batch size, saved with each benchmark result.

    Attributes:
        layout: ``unstructured``, ``tiled`` or ``structured`` residual masks.
        centroid_fusion: ``per_token`` or ``deferred`` Value centroid fusion.
        bits: Residual bit width (16 means unquantized FP16).
        nnz: Retained residual coordinates per 128-dimensional head vector.
        tokens_per_warp: Compressed tokens each warp processes per iteration.
        splits: Split-K partitions along the sequence; 0 lets the kernel choose.
        fixed_shift: Whether softmax uses a fixed shift instead of a running maximum.
    """

    layout: str
    centroid_fusion: str
    bits: int
    nnz: int
    tokens_per_warp: int
    splits: int
    fixed_shift: bool = False

    def __post_init__(self):
        for name in ("bits", "nnz", "tokens_per_warp", "splits"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if type(self.fixed_shift) is not bool:
            raise ValueError("fixed_shift must be a boolean")
        if self.layout not in ("unstructured", "tiled", "structured") or self.bits not in (2, 4, 16):
            raise ValueError("invalid resolved layout or bit width")
        if self.centroid_fusion not in ("per_token", "deferred") or self.nnz not in (32, 64):
            raise ValueError("invalid resolved centroid fusion or retained dimension count")
        if self.tokens_per_warp not in (2, 4, 8, 12, 16) or self.splits < 0:
            raise ValueError("invalid warp or split schedule")
        if self.layout == "structured" and (self.nnz != 32 or self.bits == 2):
            raise ValueError("invalid indexed residual configuration")
        if self.layout == "tiled" and (
            self.bits != 16 or self.nnz != 64 or self.centroid_fusion != "per_token"
        ):
            raise ValueError("invalid tiled residual configuration")
        if self.layout == "unstructured" and self.centroid_fusion != (
            "deferred" if self.bits == 16 else "per_token"
        ):
            raise ValueError("invalid residual bit width and centroid fusion")
        if self.fixed_shift and not (
            self.layout == "structured" and self.bits == 4 and self.centroid_fusion == "deferred"
        ):
            raise ValueError("fixed shift requires indexed INT4 deferred fusion")

    def to_dict(self) -> dict:
        return asdict(self)
