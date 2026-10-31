# Contributing to ResKV

Install `python -m pip install -c constraints.txt -e '.[dev,eval]'` in the
[documented environment](../README.md#installation).
Dependency ranges are declared in `pyproject.toml`; `constraints.txt` pins the
Python 3.12 reference environment. Run these checks before submitting changes:

```bash
ruff check reskv tests benchmarks evaluation
ruff format --check reskv tests benchmarks evaluation
python -m pytest -q
python -m build
```

Use `pytest -q -m 'not cuda'` on a CPU-only machine. CUDA tests require an
NVIDIA GPU and toolkit. See [testing](../docs/testing.md) for installed-wheel checks.

Keep K and V on the shared compression API. Add specializations to the existing
CUDA source families. Changes to packing, clustering, RoPE or softmax should
include an independent numerical reference check.

Run GPU benchmarks serially and record the configuration, input sizes, dtype,
software and device with the output. Keep checkpoints, datasets, binaries and
build caches outside the source tree. Preserve third-party license notices.
