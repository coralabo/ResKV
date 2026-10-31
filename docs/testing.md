# Testing and package builds

Install the development and evaluation dependencies:

```bash
python -m pip install -c constraints.txt -e '.[dev,eval]'
```

## Run tests

Use the [environment setup](../README.md#installation). CPU tests
exercise small local models and generated fixtures; they do not download model
weights or datasets. CUDA tests compile and exercise the native kernels.

```bash
# All tests; CUDA tests require an NVIDIA GPU and toolkit.
python -m pytest -q
# CPU tests.
python -m pytest -q -m 'not cuda'
# CUDA numerical and model checks.
python -m pytest -q -m cuda
# Accuracy reconstruction, model integration and evaluation commands.
python -m pytest -q tests/test_accuracy.py tests/test_evaluation.py tests/test_scoring.py
```

The tests cover configuration, compression/reconstruction, cache updates,
attention, generation, checkpoint loading, and LongBench/RULER command execution.
CUDA checks use the independent FP32 reference in `reskv/reference.py`.
Numerical CUDA tests follow the library's device requirements. The RTX PRO 6000
Server Edition restriction applies to performance commands, not these tests.

## Lint and build

```bash
ruff check reskv tests benchmarks evaluation
ruff format --check reskv tests benchmarks evaluation
python -m build
```

The wheel and source archive are written to `dist/`.

## Test the installed wheel

```bash
reskv_source_dir=$(pwd)
pip install --force-reinstall --no-deps dist/*.whl
cd "$(mktemp -d)"
python -m pytest -q -m 'not cuda' --import-mode=importlib "$reskv_source_dir/tests"
```

The CI workflow runs linting, CPU tests, package builds and installed-wheel
checks. Run CUDA tests locally when changing kernels or device-specific code.
Use the [accuracy guide](evaluation.md) for model/dataset scores and the
[benchmark guide](benchmarks.md) for latency and memory measurements. These
commands are separate from the unit and integration tests above.
