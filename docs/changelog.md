# Changelog

## 0.1.0

Initial public release.

* CUDA inference for Llama: residual compression of pre-RoPE Keys and Values,
  sparse FP16 residuals and fused Flash
  Decoding kernels with per-token or deferred Value centroid fusion.
* PyTorch accuracy evaluation for Llama and Qwen3 on LongBench and RULER, with a
  model-free `reskv-score` scorer.
* Performance benchmarks against the dense baseline and an external Mustafar
  adapter, plus kernel microbenchmarks.
* Documented model/data preparation, accuracy generation/scoring and performance commands.
