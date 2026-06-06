# autofollowdown

[![tests](https://github.com/peetwan/autofollowdown/actions/workflows/tests.yml/badge.svg)](https://github.com/peetwan/autofollowdown/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A unified, simple toolkit for compressing AI models — `quantization`, `pruning`,
and `knowledge distillation` behind one small API — plus a `real benchmark` that
measures the actual impact on size, latency, and accuracy.

No mocks. Every operation changes real weights, and every metric is measured from
a real model running on real data.

## One command does it all

Compress a model every way, benchmark them side by side, and pick the winner —
in a single command:

```bash
autofollowdown auto                       # offline demo (trains a digit CNN)
autofollowdown auto --model facebook/opt-125m --output small.pt
```

```
┌───────────────────────────────┬─────────┬──────────┬────────┬───────┬───────┬────────┐
│ Model                         │ Size MB │ Sparsity │   Acc  │ Size× │ Speed×│  ΔAcc  │
├───────────────────────────────┼─────────┼──────────┼────────┼───────┼───────┼────────┤
│ baseline                      │   1.077 │     0.0% │ 90.4%  │   —   │   —   │   —    │
│ int8 dynamic                  │   0.303 │     0.0% │ 90.4%  │ 3.56× │ 0.60× │ +0.0%  │
│ prune 50%                     │   1.077 │    50.0% │ 91.6%  │ 1.00× │ 1.07× │ +1.1%  │
│ ➤ prune+quantize              │   0.303 │    18.4% │ 91.6%  │ 3.56× │ 0.60× │ +1.1%  │
│ distilled student (1/4 width) │   0.293 │     0.0% │ 74.4%  │ 3.67× │ 5.39× │ -16.0% │
└───────────────────────────────┴─────────┴──────────┴────────┴───────┴───────┴────────┘

➤ Recommended: prune+quantize (3.56× smaller, 91.6% acc)
Pick a method to keep:  [1-5, default 4]:
```

It prompts you to choose (or pass `--method 'prune+quantize' --output model.pt`,
or `--yes` to take the recommendation). Same flow in Python:

```python
from autofollowdown import compress_and_benchmark

study = compress_and_benchmark(model, eval_loader=test_loader)
study.show()                              # table + "which to pick"
study.export(study.recommended, "small.pt")   # or study.pick("int8 dynamic")
```

### Or drive each step yourself

```python
from autofollowdown import ModelCompressor

# chainable, framework-agnostic API
ModelCompressor(my_model) \
    .prune(sparsity=0.5, method="unstructured") \
    .quantize(method="int8", approach="dynamic") \
    .export("compressed.pt", format="pt")
```

## Install

```bash
# from a clone
git clone https://github.com/peetwan/autofollowdown
cd autofollowdown
pip install -e ".[examples]"     # core + scikit-learn/datasets for the demos

# or straight from GitHub
pip install "git+https://github.com/peetwan/autofollowdown#egg=autofollowdown[examples]"
```

Requires Python `>=3.9`, PyTorch `>=2.1`. All core deps (torch, onnx, onnxruntime,
transformers, numpy) install automatically.

### 📓 Demo notebook

A runnable walkthrough of everything — with outputs you can see right on GitHub — is in
[`notebooks/autofollowdown_demo.ipynb`](notebooks/autofollowdown_demo.ipynb) (core API,
one-command flow, auto-picker, benchmarks, and which datasets are used).

### Try it in one command

```bash
autofollowdown auto               # ⭐ compress every way, benchmark, pick a winner
autofollowdown info               # version, available backends, benchmark catalog
autofollowdown benchmark-vision   # real CNN compression benchmark (offline)
autofollowdown benchmark-llm      # real LLM perplexity benchmark (downloads a small model)
autofollowdown autopick           # best-library recommendation per model family
```

## What it does

| Technique | API | What actually happens |
|-----------|-----|-----------------------|
| Pruning | `.prune(sparsity, method)` | Global L1 magnitude (`unstructured`) or channel (`structured`) pruning via `torch.nn.utils.prune`, made permanent so zeros are real |
| Quantization | `.quantize(method, approach)` | INT8 `dynamic` (portable) or FX `static` PTQ with calibration; INT8 on the ONNX graph for `.onnx` inputs |
| Distillation | `.distill(teacher, train_loader, epochs)` | A real KD training loop (KL on softened logits + CE on labels) that updates the student |
| Export | `.export(path, format)` | Real `.pt` (torch) or `.onnx` (runnable under onnxruntime) |

Inputs accepted: a PyTorch `nn.Module`, a Hugging Face model id, or a local `.onnx` file.

## The benchmark

The point of the benchmark is honesty: it tells you what compression cost you.

```python
from autofollowdown import Benchmark, ModelCompressor
import copy

bench = Benchmark(example_input, eval_loader=test_loader, reference_model=baseline)
bench.measure(baseline, "baseline (fp32)")

quant = ModelCompressor(copy.deepcopy(baseline)).quantize(approach="dynamic").model
bench.measure(quant, "quantized int8")

print(bench.to_markdown())   # before/after table with size×, speed×, ΔAcc
```

It measures (all real): parameter count, true sparsity, on-disk size (MB), p50
inference latency, throughput, top-1 accuracy, and output fidelity vs the baseline.

### Run the included example (offline, no download)

```bash
python3 examples/benchmark_digits.py --epochs 8
```

It trains a real CNN on the scikit-learn `digits` dataset, then prunes / quantizes /
distills it. Sample output:

```
| Model                         | Size (MB) | Sparsity | Latency (ms) | Acc    | Size× | Speed× | ΔAcc   |
|-------------------------------|-----------|----------|--------------|--------|-------|--------|--------|
| baseline (fp32)               | 1.077     | 0.0%     | 0.64         | 96.00% | —     | —      | —      |
| pruned 50% (unstructured)     | 1.077     | 50.0%    | 0.68         | 96.00% | 1.00× | 0.95×  | +0.00% |
| quantized (int8 dynamic)      | 0.303     | 0.0%     | 1.18         | 96.00% | 3.56× | 0.54×  | +0.00% |
| pruned+quantized              | 0.303     | 17.6%    | 1.17         | 95.78% | 3.56× | 0.55×  | -0.22% |
| distilled student (1/4 width) | 0.293     | 0.0%     | 0.14         | 89.78% | 3.67× | 4.53×  | -6.22% |
```

What this honestly shows: INT8 cuts size `3.56×` with no accuracy loss but is *slower*
on a tiny CPU model (quant/dequant overhead); distillation is `4.5×` faster and
smaller but trades `~6%` accuracy. Real tradeoffs, not marketing.

#### Works on bigger models too (Qwen `< 8B`)

```bash
autofollowdown benchmark-llm --model Qwen/Qwen3-0.6B     # 1.7B / 3B too; GPU for >1B
```

Real measured output (Qwen3-0.6B, 596M params, WikiText-2):

```
| Model            | Size (MB) | Perplexity↓ | Size× | ΔPPL   |
| Qwen3-0.6B fp32  | 2274      | 20.37       |  —    |  —     |
| int8 dynamic     | 1164      | 30.36       | 1.95× | +10.00 |
```

Honest caveat: naive **dynamic INT8 is a quick, portable baseline** — but it costs real
quality on capable LLMs (note the perplexity jump). That's why weight-only, calibrated
methods (GPTQ / AWQ) exist, and why the auto-picker recommends `llm-compressor` or
`NVIDIA ModelOpt` (not native dynamic) for LLMs.

Caveats worth knowing:
- Pruning zeros weights but dense `.pt`/`.onnx` storage does not shrink from zeros
  alone — pair pruning with quantization or a sparse format to save space.
- After torch quantization, packed INT8 weights are not regular `Parameters`, so the
  `Params`/`Sparsity` columns reflect only the remaining float tensors. `Size (MB)`
  is the honest footprint metric.

## Benchmarking compressed LLMs

For language models the field judges compression (quantization / pruning /
distillation) on two pillars — and autofollowdown supports both:

1. Perplexity (lower = better) on held-out text. `WikiText-2` is the universal
   default; `C4` and `PTB` are also common. Implemented for real here via the
   standard sliding-window method (`evaluate_perplexity`).
2. Zero-shot / few-shot task accuracy via EleutherAI's `lm-evaluation-harness`
   (the community standard). `lm_eval_command()` builds the exact CLI.

Standard datasets/tasks (matching `lm-eval-harness` task ids), from the GPTQ,
AWQ, SparseGPT, LLMCBench, LeanQuant, NVIDIA MINITRON and Apple LLM-KICK papers:

| Pillar | Datasets / tasks | Measures |
|--------|------------------|----------|
| Perplexity | `wikitext2`, `c4`, `ptb` | language-modeling quality |
| Commonsense (0-shot) | `arc_easy`, `arc_challenge`, `hellaswag`, `winogrande`, `piqa`, `openbookqa`, `boolq`, `lambada_openai` | reasoning / commonsense |
| Knowledge (5-shot) | `mmlu` | broad factual knowledge |
| Advanced knowledge | `mmlu_pro` | harder 10-choice reasoning |
| Multilingual | `mmlu_prox_{lang}` / `mmlu_prox_lite_{lang}` (29 languages) | cross-lingual reasoning |
| Reasoning | `gsm8k` | math word problems |
| Truthfulness | `truthfulqa` | reliability |
| Multimodal (VLMs) | `mmmu_val` / `mmmu_pro` | image+text college-level reasoning |

`MMMU` evaluates **vision-language models** (e.g. Qwen-VL, LLaVA) on 11.5K college-level
image+text questions across 30 subjects (accuracy). Compress a VLM with autofollowdown,
then evaluate it on MMMU via the multimodal harness — `multimodal_eval_command()` builds
the CLI for EleutherAI `lm-eval` (`--model hf-multimodal`) or `lmms-eval`:

```python
from autofollowdown import multimodal_eval_command, mmmu_tasks
print(multimodal_eval_command("Qwen/Qwen2-VL-2B-Instruct", tasks=mmmu_tasks()))
# lm_eval --model hf-multimodal --model_args pretrained=...,max_images=1,interleave=True \
#   --tasks mmmu_val --apply_chat_template --device cuda:0 --batch_size auto
```

`MMLU-ProX` (EMNLP 2025) extends MMLU-Pro to 29 languages with parallel questions
— a strong "beyond perplexity" check, since compression can hurt reasoning and
low-resource languages more than perplexity shows. Build its task ids with
`mmlu_prox_tasks(["en", "th", "zh"], lite=True)` (the `lite` set is 658 Q/language,
ideal for quickly vetting a compressed model).

Caveat from the literature (Apple LLM-KICK, ACL'24 survey): perplexity is the
quick standard but imperfect — pruning degrades knowledge tasks more than
perplexity suggests, and quantization usually preserves accuracy better than
pruning at equal compression. Report both pillars.

### Run it (real perplexity, real model)

```bash
pip install datasets            # for real WikiText-2
python3 examples/benchmark_llm.py --model facebook/opt-125m
```

Real measured output (OPT-125M, WikiText-2):

```
| Model           | Size (MB) | Perplexity↓ | Latency (ms) | Size× | Speed× | ΔPPL   |
| baseline (fp32) | 477.8     | 32.761      | 16.0         | —     | —      | —      |
| int8 dynamic    | 271.9     | 34.625      | 18.6         | 1.76× | 0.86×  | +1.864 |
```

INT8 cuts size `1.76×` for a small `+1.86` perplexity cost — the kind of honest
tradeoff the benchmark exists to show. (Tip: OPT uses `nn.Linear`, so INT8
dynamic compresses it; GPT-2 uses `Conv1D` and barely shrinks under dynamic quant.)

### Full accuracy suite via lm-evaluation-harness

```python
from autofollowdown import lm_eval_command, STANDARD_LLM_TASKS, mmlu_prox_tasks
print(lm_eval_command("./my-compressed-model",
                      tasks=STANDARD_LLM_TASKS["commonsense_zeroshot"]
                            + mmlu_prox_tasks(["en", "th"], lite=True)))
# lm_eval --model hf --model_args pretrained=./my-compressed-model \
#   --tasks arc_easy,...,hellaswag,mmlu_prox_lite_en,mmlu_prox_lite_th --device cuda:0 --batch_size auto
```

## Auto-picker: best library for your model

There are many compression libraries, each best at something different. autofollowdown
profiles your model and recommends (and can run) the best one for it — falling back
to the always-available native engine when an optional library isn't installed.

```python
from autofollowdown import explain, recommend, auto_compress

print(explain(my_model))                 # ranked backends + why, for this model
compressed, chosen = auto_compress(my_model)   # runs the best runnable backend
print("picked:", chosen.backend, chosen.scheme)
```

```
Model profile: ModelProfile(family=cnn, params=78,442, conv=True, transformer=False, ...)

Ranked compression backends:
    1. [0.90] Microsoft NNI: L1FilterPruner + ModelSpeedup — not installed   (pip install nni)
 →  2. [0.60] autofollowdown (native): unstructured-0.5 + int8-dynamic — runnable
    3. [0.60] NVIDIA TensorRT Model Optimizer: INT8 PTQ — not installed (needs NVIDIA GPU)

Auto-pick (runnable now): autofollowdown (native)
```

Backends and what they're chosen for:

| Backend | Best for | Technique | Requirement |
|---------|----------|-----------|-------------|
| autofollowdown (native) | anything (fallback) | INT8 dynamic / pruning / distillation | built in |
| Microsoft NNI | CNNs / vision | structured filter pruning + `ModelSpeedup` (real shrink) | `pip install nni` |
| llm-compressor (vLLM) | HF LLMs | GPTQ/AWQ 4-bit weight-only (`oneshot`) | `pip install llmcompressor` (GPU) |
| NVIDIA ModelOpt | LLMs / transformers | SmoothQuant/AWQ/NVFP4 PTQ → TensorRT | `pip install nvidia-modelopt` (NVIDIA GPU) |

The ranking always shows the *ideal* backend even if it isn't installed, plus the
best one you can run right now. `recommend()` is advisory; `auto_compress()` executes.

#### Use a specific connected backend in one line

`compress_with(model, backend)` runs the real library (by name or alias) — it executes
the moment the library is installed and the hardware suits it, and otherwise tells you
exactly how to enable it. The same call works on a CNN (NNI) or an LLM like Qwen:

```python
from autofollowdown import compress_with

compress_with(cnn,  "nni", dummy_input=x)                       # NNI structured pruning + ModelSpeedup
compress_with(qwen, "llmcompressor",                            # GPTQ/AWQ 4-bit (vLLM-ready)
              recipe=GPTQModifier(targets="Linear", scheme="W4A16"), dataset="open_platypus")
compress_with(qwen, "modelopt", calibration_data=calib)         # NVIDIA SmoothQuant/NVFP4 PTQ (GPU)
```

Try it: `python3 examples/autopick_demo.py` — and the demo notebook calls `compress_with`
on both a CNN and Qwen so you can see each backend's exact invocation.

## How it works (architecture & flow)

autofollowdown is a thin, honest pipeline: **ingest → profile → compress → measure →
recommend → export**. Everything operates on a real model and every number is measured.

```mermaid
flowchart TD
    A["Model: PyTorch nn.Module · HF id · .onnx file"] --> B["ingestion.load_model()"]
    B --> C["profiler.profile_model() → ModelProfile<br/>(family · #params · has_conv/transformer · CUDA?)"]
    C --> D{"How do you drive it?"}
    D -->|"one command"| E["compress_and_benchmark() / autofollowdown auto"]
    D -->|"step by step"| F["ModelCompressor.prune / quantize / distill"]
    D -->|"best library"| G["recommend() · auto_compress() · compress_with()"]
    E --> H
    F --> H
    G --> H
    subgraph H["Compression backends (registry)"]
      H1["native — torch prune / INT8 / KD (always on)"]
      H2["NNI — structured pruning + ModelSpeedup"]
      H3["llm-compressor — GPTQ/AWQ 4-bit (LLMs)"]
      H4["NVIDIA ModelOpt — PTQ → TensorRT (GPU)"]
    end
    H --> I["metrics: size_mb · latency · sparsity · accuracy · fidelity · perplexity"]
    I --> J["Benchmark / CompressionStudy<br/>before↔after table + ➤ recommended pick"]
    J --> K["export → .pt / .onnx"]
    J --> L["LLM/VLM eval commands:<br/>WikiText-2 PPL · lm-eval · lmms-eval<br/>(MMLU · MMLU-ProX · MMMU · …)"]
```

### The stages

1. **Ingest** (`ingestion.py`) — accepts a PyTorch `nn.Module`, a Hugging Face model id
   (loaded with the right `AutoModel*` class), or a path to a local `.onnx` file, and
   normalizes them to one internal representation.
2. **Profile** (`profiler.py`) — inspects the model and returns a `ModelProfile`: its
   *family* (`llm` / `transformer` / `cnn` / `mlp`), parameter count, whether it has
   conv/attention layers, whether it's a Hugging Face model, and whether CUDA is present.
   This is what lets the toolkit choose sensible defaults automatically.
3. **Compress** (`api.py` `ModelCompressor`) — applies real operations to the weights:
   - `prune()` — global L1 magnitude (unstructured) or per-channel L2 (structured) pruning
     via `torch.nn.utils.prune`, made **permanent** (the mask is folded in, so zeros are real).
   - `quantize()` — INT8 `dynamic` (portable) or FX `static` PTQ; INT8 on the ONNX graph
     for `.onnx` inputs. (Picks the right CPU quant engine automatically — fbgemm/qnnpack.)
   - `distill()` — a real knowledge-distillation training loop. For classifiers it's
     `KL(soft) + CE(hard)`; for **causal LMs** it switches to token-level soft KD over the
     vocab, so it works on models like Qwen.
   - `export()` — real `.pt` (torch) or `.onnx` (runnable under onnxruntime).
4. **Measure** (`metrics.py`) — for any model: on-disk size (MB, serialized to a temp file so
   multi-GB LLMs don't blow up RAM), parameter count, true sparsity, p50 latency, throughput;
   with eval data: top-1 accuracy and output fidelity (agreement with the original); for LMs:
   sliding-window WikiText-2 perplexity (`llm_eval.py`).
5. **Recommend** (`benchmark.py` + `pipeline.py`) — `Benchmark` collects before/after rows;
   `best_picks()` scores variants (accuracy-weighted compression) and marks the **smallest,
   fastest, most-accurate, and recommended** options; `to_table()` renders the box-drawing
   terminal table with the recommended row highlighted.
6. **Export / evaluate** — keep the variant you pick (`study.export(...)`); for LLMs/VLMs,
   `lm_eval_command()` / `multimodal_eval_command()` build the exact harness commands to run
   the full accuracy suite (ARC, MMLU, MMLU-ProX, GSM8K, MMMU, …).

### Three ways to drive it (same engine underneath)

| Entry point | What it does | Use when |
|-------------|--------------|----------|
| `ModelCompressor(m).prune().quantize().export()` | manual, chained control | you know exactly what you want |
| `compress_and_benchmark(m)` / `autofollowdown auto` | runs all methods, benchmarks, recommends, lets you pick | you want the best variant chosen for you |
| `recommend(m)` / `auto_compress(m)` / `compress_with(m, "nni")` | profiles the model and routes to the best *library* | you want the strongest method available on your hardware |

### Backend registry (`backends.py`)

Each backend declares: availability (is the library importable?), hardware fit (`needs_cuda`),
a fitness `score(profile)`, the `plan` it would run, and a real `compress()` that calls the
library's documented API. The **native** backend is always available; `NNI`, `llm-compressor`,
and `NVIDIA ModelOpt` register only when installed. `auto_compress` runs the highest-scoring
*runnable* backend (falling back to native); `compress_with(model, "alias")` forces a specific
one — running the real library when present, or telling you exactly how to enable it.

### Data objects you'll see

- `ModelProfile` — the model's family / size / hardware fingerprint.
- `CompressionStudy` — holds the baseline + every compressed variant, their benchmark, and the
  pick API (`.recommended`, `.pick(name)`, `.best()`, `.export(...)`, `.show()`).
- `Recommendation` — one ranked backend (score, runnable?, rationale, install hint).

## Layout

```
autofollowdown/
  api.py            # ModelCompressor — the unified compression API
  pipeline.py       # compress_and_benchmark() + CompressionStudy (the one-command flow)
  auto.py           # auto-picker: recommend() / explain() / auto_compress()
  backends.py       # backend registry (native + NNI + llm-compressor + ModelOpt)
  profiler.py       # model profiling (family / size / hardware)
  metrics.py        # real measurements (size, latency, accuracy, fidelity)
  benchmark.py      # before/after benchmark engine + "best pick" recommendation
  llm_eval.py       # LLM perplexity + lm-eval-harness catalog (incl. MMLU-ProX)
  ingestion.py      # load PyTorch / Hugging Face / ONNX
  onnx_pipeline.py  # ONNX export + onnxruntime quant/prune
  graph_tracing.py  # torch.fx tracing + Conv/BN/ReLU fusion
  demos.py          # packaged benchmark runners (shared by CLI + examples)
  cli.py            # `autofollowdown` command-line interface
examples/
  benchmark_digits.py   # real before/after benchmark (offline, vision)
  benchmark_llm.py      # real LLM perplexity benchmark (WikiText-2)
  autopick_demo.py      # auto-pick the best backend per model
notebooks/
  autofollowdown_demo.ipynb   # runnable walkthrough of the whole toolkit
tests/              # real tests (assert actual effects, not flags)
```

## Tests

```bash
python3 -m pytest -q
```

## License

MIT — see [LICENSE](LICENSE).
