# Mustafar setup

Use a separate environment for the external Mustafar adapters. ResKV uses
Transformers 4.57.x; Mustafar requires 4.44.2. Run both adapters by their absolute
script paths, and keep the ResKV package installed in the ResKV environment.

## Environment and checkout

The following uses Python 3.12, PyTorch 2.8.0, CUDA 12.8 and Triton 3.4.0.
Install the CUDA toolkit and a compatible C++ compiler as described in the
[ResKV setup](../README.md#installation). Replace the example
paths with your checkout and environment locations.

```bash
python3.12 -m venv /path/to/mustafar-env
source /path/to/mustafar-env/bin/activate
python -m pip install --upgrade pip setuptools==80.9.0 wheel
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install transformers==4.44.2 tokenizers==0.19.1 accelerate==1.10.1 \
  numpy==2.3.5 rouge==1.0.1 jieba==0.42.1 ninja packaging
python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip check

git clone https://github.com/dhjoo98/mustafar /path/to/mustafar
git -C /path/to/mustafar checkout 86fa14d6f19ea6cad46567827c209f9d402336f3
```

FlashAttention is needed by the external model's imports, including the
performance model whose prefill uses SDPA. Accuracy evaluation requires a
FlashAttention build compatible with the selected GPU. Only performance runs
need the patch and sparse CUDA extension below.

## Accuracy evaluation

The adapter loads Mustafar's Python Llama model at
`models/llama_mustafar_Kt_Mag_Vt_Mag.py`. It runs FP16 on CUDA with unpadded
batch-one inputs. It does not require the sparse CUDA extension or performance
patch. From the Mustafar environment:

```bash
python /path/to/ResKV/evaluation/mustafar.py --mustafar-root /path/to/mustafar \
  --model /models/llama-3-8b-instruct --suite longbench --dataset qasper \
  --input /data/longbench/data/qasper.jsonl --sparsity 0.75 \
  --max-input-tokens 8192 --output results/mustafar/longbench/sparse75/qasper.jsonl

python /path/to/ResKV/evaluation/mustafar.py --mustafar-root /path/to/mustafar \
  --model /models/llama-3.1-8b-instruct --suite ruler --dataset qa_1 \
  --input /data/ruler/qa_1/validation.jsonl --sparsity 0.5 \
  --max-input-tokens 16384 --output results/mustafar/ruler/sparse50/qa_1.jsonl
```

The external model supports Llama and sparsity-only settings `0.5`, `0.75`
and `0.875`. It uses per-token magnitude pruning with a recent dense window;
`--recent-size` and `--group-size` both default to 64. Qwen3 and quantization
are not supported by this adapter.

Both accuracy entry points use the [same dataset and generation options](evaluation.md).
Use the same checkpoint, task files, prompt format, seed and length limits for
each method. The adapter retains the external pruning, RoPE and attention code,
supplies its causal mask and tuple cache, and computes only the last position's
logits. Predictions and sibling `.summary.json` files record input-ID hashes,
the ResKV git revision, the Mustafar commit, environment and settings. Score
saved predictions with `reskv-score` in the ResKV environment, passing the
actual prediction path.

## Build the performance extension

Apply the compatibility patch to the pinned checkout. The patch provides the
prefill selector, limits inference logits to the last position, and adapts the
extension build to the active environment. Run these commands once in the
Mustafar environment:

```bash
git -C /path/to/mustafar apply --check /path/to/ResKV/third_party/patches/mustafar-compat.patch
git -C /path/to/mustafar apply /path/to/ResKV/third_party/patches/mustafar-compat.patch
mustafar_sm=$(python -c 'import torch; major, minor = torch.cuda.get_device_capability(); print(f"{major}{minor}")')
export TORCH_CUDA_ARCH_LIST=$(python -c 'import torch; major, minor = torch.cuda.get_device_capability(); print(f"{major}.{minor}")')
make -C /path/to/mustafar/kernel/build SMS="$mustafar_sm" \
  CUDA_PATH="${CUDA_HOME:-/usr/local/cuda}" INCLUDES="-I${CUDA_HOME:-/usr/local/cuda}/include/"
python -m pip install --no-build-isolation /path/to/mustafar/kernel/kernel_wrapper
python -c 'import torch, mustafar_package; print(mustafar_package.__file__)'
```

The architecture is read from the visible GPU rather than fixed in the commands.
Use a fresh checkout/build directory when changing the GPU or CUDA toolchain.

## Performance run

The performance adapter requires NVIDIA RTX PRO 6000 Blackwell Server Edition.
From the Mustafar environment:

```bash
python /path/to/ResKV/benchmarks/mustafar.py --mustafar-root /path/to/mustafar \
  --model /models/llama-3.1-8b-instruct --batch-size 1 --input-length 512 \
  --new-tokens 8 --sparsity 0.75 --warmup 1 --repeats 1 \
  --output results/mustafar/check.json
```

Select `--sparsity 0.5` or `0.75`. Prefill uses PyTorch SDPA; decode uses
Mustafar's sparse kernels.
The performance adapter's group size defaults to 32.

For a joint batch sweep, return to the ResKV environment and repository root:

```bash
source /path/to/ResKV/.venv/bin/activate
cd /path/to/ResKV
python -m benchmarks.compare --model /models/llama-3.1-8b-instruct \
  --methods reskv mustafar dense --batch-sizes 1 4 8 16 32 \
  --input-length 8192 --new-tokens 2048 --sparsity 0.75 \
  --mustafar-root /path/to/mustafar \
  --mustafar-python /path/to/mustafar-env/bin/python \
  --output results/performance/comparison
```

The launcher starts the appropriate interpreter for each method. All methods
use the same input generator and serial decode loop: one prefill, N−1 decode
forwards, and TPOT divided by N−1. Output records include timing, memory, input
hashes, settings, the ResKV git revision and the Mustafar commit. See the
[benchmark guide](benchmarks.md) for output fields and memory accounting.
