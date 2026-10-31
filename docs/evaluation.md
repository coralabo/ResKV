# Accuracy evaluation

Use the [ResKV environment](../README.md#installation), then install
`python -m pip install -c constraints.txt -e '.[eval]'` from the repository root.
Supply a local checkpoint and dataset JSONL.
JSONL files use UTF-8. Prediction and score outputs must differ from their inputs.
This interface does not compile ResKV CUDA kernels or require `flash-attn`.
The ResKV evaluator applies compression error in Python/PyTorch and
stores reconstructed dense KV for attention. Keys are compressed before RoPE;
Qwen3 keeps Q/K normalization before that step.

## Models and configuration

| Model | Python class |
| --- | --- |
| Llama | `ResKVLlamaAccuracyForCausalLM` |
| Qwen3 | `ResKVQwen3AccuracyForCausalLM` |

```python
import torch
from reskv import ResKVConfig, ResKVLlamaAccuracyForCausalLM

config = ResKVConfig(sparsity=0.75, bits=4)
model = (
    ResKVLlamaAccuracyForCausalLM.from_pretrained(
        "/models/llama-3-8b-instruct",
        reskv_config=config,
        dtype=torch.float16,
        attn_implementation="sdpa",
    )
    .to("cuda")
    .eval()
)
```

Use `ResKVQwen3AccuracyForCausalLM` for Qwen3. The CLI selects the class from the
checkpoint's `model_type`. Accuracy supports full-attention Llama/Qwen3 models,
FP16/BF16/FP32, unpadded batch-one inputs, and static default/Llama3/linear RoPE.
For CPU execution, pass `--device cpu --dtype fp32`.

`sparsity` is the discarded residual fraction. The common combinations are:

| Settings | Arguments |
| --- | --- |
| 50% sparsity | `--sparsity 0.5 --bits 16` |
| 75% sparsity | `--sparsity 0.75 --bits 16` |
| 50% sparsity + INT4 | `--sparsity 0.5 --bits 4` |
| 75% sparsity + INT4 | `--sparsity 0.75 --bits 4` |

`bits=16` disables quantization without changing the model dtype. Bit width
2 is also available in this accuracy interface. Residuals are sparsified
before quantization/dequantization, then added back to their centroids.

Prefill attention uses the original dense KV. Decode uses the reconstructed
cache, keeping the initial sink tokens and recent window dense. Complete groups
outside those regions are compressed once; a partial group remains dense.
Reconstructed keys keep their original token positions when RoPE is applied.

## LongBench

Use LongBench JSONL rows containing `context`, `input`, `answers` and
`all_classes`. The task name selects its prompt, generation limit and metric.
Download and extract LongBench v1 using the [dataset commands](../README.md#models-and-data).
For example, a task file contains rows of this form:

```json
{"context": "Document text...", "input": "Question?", "answers": ["Answer"], "all_classes": []}
```

```bash
python -m evaluation.run --suite longbench --dataset qasper \
  --input /data/longbench/data/qasper.jsonl --model /models/llama-3-8b-instruct \
  --sparsity 0.75 --bits 16 --max-input-tokens 8192 \
  --output results/longbench/sparse75/qasper.jsonl

python -m evaluation.run --suite longbench --dataset qasper \
  --input /data/longbench/data/qasper.jsonl --model /models/llama-3-8b-instruct \
  --sparsity 0.75 --bits 4 --max-input-tokens 8192 \
  --output results/longbench/sparse75_int4/qasper.jsonl

python -m evaluation.run --suite longbench --dataset qasper \
  --input /data/longbench/data/qasper.jsonl --model /models/llama-3-8b-instruct \
  --sparsity 0 --bits 16 --max-input-tokens 8192 --output results/longbench/dense/qasper.jsonl
```

Use `--sparsity 0 --bits 16` to disable compression within the same ResKV model.
Add `--limit 2` to process two samples with the task's usual generation limit.

The [LongBench loop](../README.md#longbench) runs all 16 tasks (3,750 samples)
and scores their unweighted mean. Use a separate output directory per model and
compression configuration.

## RULER

Prepare JSONL using [NVIDIA/RULER](https://github.com/NVIDIA/RULER).
Rows contain `input`, an `outputs` list and an optional `answer_prefix`.
The answer prefix follows the chat wrapper; inputs that already include it are
kept as supplied. Use the generated task name as `--dataset` and set the input
cap to fit the formatted task.

### Prepare RULER data

Install [Git LFS](https://git-lfs.com/), then set up the upstream generator in a
separate environment:

```bash
git lfs install
git clone https://github.com/NVIDIA/RULER.git
cd RULER
git checkout c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a
git lfs pull
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install transformers==4.57.1 numpy==2.3.5 \
  pyyaml nltk==3.9.2 wonderwords tenacity tqdm html2text beautifulsoup4
python -m nltk.downloader punkt punkt_tab
```

Download the essay and QA corpora, then generate all tasks from the RULER
checkout. The word list is fetched by `git lfs pull` above.

```bash
cd scripts/data/synthetic/json
python download_paulgraham_essay.py
bash download_qa_dataset.sh
cd ../../../..
reskv_tokenizer=/models/llama-3.1-8b-instruct
reskv_data=/data/ruler/llama31
for reskv_task in niah_single_1 niah_single_2 niah_single_3 \
  niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multivalue niah_multiquery \
  vt cwe fwe qa_1 qa_2; do
  python scripts/data/prepare.py --save_dir "$reskv_data" \
    --benchmark synthetic --task "$reskv_task" --subset validation \
    --tokenizer_path "$reskv_tokenizer" --tokenizer_type hf \
    --max_seq_length 16384 --model_template_type base --num_samples 100 --random_seed 42 || exit 1
done
```

For Qwen3, change `reskv_tokenizer` to `/models/qwen3-8b` and use
`reskv_data=/data/ruler/qwen3`. Each task writes `<task>/validation.jsonl`.
The generator's length includes its generation budget; the evaluator's
`--max-input-tokens` caps the prompt. Return to the ResKV environment and
repository before running the [RULER evaluation loop](../README.md#ruler).

RULER truncation is rejected by default; `--allow-truncation` enables it.
QA scoring uses partial string match; other tasks average the fraction of
reference strings recovered. Reference lists must be nonempty.

The [RULER loop](../README.md#ruler) runs all 13 tasks and computes their
unweighted mean. Prepare the data for the same tokenizer at a 16K context length.

## Prompts and generation

| Option | Behavior |
| --- | --- |
| `--prompt-format dataset` (default) | Dataset-specific wrappers and middle truncation; retain already formatted RULER text |
| `--prompt-format chat` | Checkpoint chat template and a cap on final input token IDs |
| `--prompt-format plain` | Use the task text directly |
| `--max-input-tokens` | Input length cap |
| `--max-new-tokens` | Override the dataset's generation limit |
| `--seed` | Random seed |

The default LongBench formatting skips chat wrappers for `trec`, `triviaqa`,
`samsum`, `lsht`, `lcc` and `repobench-p`. Llama3 instruct uses the literal wrapper
and subsequent special-token insertion, which can include a second BOS; the
wrapper can also increase the final input length. Choose `chat` to apply the
checkpoint's template and cap the final token IDs. Qwen3 chat mode disables
thinking. SAMSum includes newline/EOS stopping.
Qwen3 RULER dataset prompts also include the system instruction to answer directly
and concisely without explaining reasoning.

Use the same input, formatting, dtype and generation settings for each configuration.
Generation is greedy and uses one prefill followed by single-token decode.

## Mustafar comparison

Run the external Mustafar Llama accuracy model in its separate environment;
see [setup and commands](mustafar.md#accuracy-evaluation). ResKV and Mustafar
share the dataset reader, prompt formatting, generation settings and scorer.
The adapter loads Mustafar's magnitude-pruning implementation directly from
the supplied checkout and records its commit.

## Score saved predictions

```bash
reskv-score --suite longbench --input results/longbench/llama/sparse75 \
  --output results/longbench/llama/sparse75/scores.json
reskv-score --suite ruler --dataset niah_single_1 \
  --input results/ruler/sparse75_int4/niah_single_1.jsonl \
  --output results/ruler/sparse75_int4/niah_single_1_scores.json
```

A LongBench directory contains one `<dataset>.jsonl` per task. RULER files
may contain multiple tasks if each row includes `task`. A directory can instead
contain one `<task>.jsonl` per task. Use `--dataset` when scoring a single file
without task fields. Prediction fields can be `prediction`, `pred` or `predicted_answer`;
reference fields can be `answers`, `answer` or `outputs`.
The report lists every selected task and its sample count. Its macro average
weights tasks equally, including when their sample counts differ.

## Compression options

| Parameter | Default |
| --- | --- |
| Centroids | 32 |
| Recent window | 32 |
| K-means iterations | 15 |
| Group size / sink tokens | 64 / 4 |
| Centroid storage | BF16 |
| Decode centroid initialization | Prefill centroids |

Override these values with `--num-clusters`, `--recent-size`, `--group-size`,
`--sink-tokens` and `--kmeans-iterations`. Prefill quantization uses per-token affine
min/max scaling; decode uses symmetric absolute-maximum scaling.
The same defaults apply in Python with `ResKVConfig()`.
Explicit parameter values and saved configuration values take precedence over defaults.

## Outputs

The output JSONL contains generated text/token IDs, references, actual input
length, truncation status and input-ID hashes. A sibling `.summary.json` contains
scores, configuration, prompt settings, dtype, seed, environment and git revision.
The summary is written after the task completes; an interrupted rerun retains
partial predictions without a completed summary.
LCC and RepoBench-P use Python's `difflib.SequenceMatcher` for code similarity,
rounded to integer percent per sample before averaging over the dataset.
