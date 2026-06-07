# Changelog

All notable changes to autofollowdown are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2026-06-07

### Added — safe `safetensors` export (the follow-on to the 0.5.0 security fix)
- `ModelCompressor.export(path, format="safetensors")` and
  `CompressionStudy.export(..., format="safetensors")` save weights with no pickle
  (handles tied/shared tensors via `safetensors.save_model`). Torch-quantized models
  keep packed non-tensor params, so safetensors fails there with a clear pointer to
  `format="pt"`. `autofollowdown compress --format safetensors` exposes it.
- `profile_checkpoint` (and the `recommend` / `advise` / `diagnose` / `gpu` commands)
  now read `.safetensors` — so a model you exported re-profiles with **zero
  `--allow-pickle`**, closing the friction the security fix introduced.
- `safetensors` is now a (tiny) core dependency.
- +5 tests; full suite 180 passed.

## [0.5.0] - 2026-06-07

A correctness, safety, and honesty release — fixes from a deep self-audit.

### Security
- **Fixed pickle-RCE.** `recommend` / `advise` / `diagnose` / `gpu` used to
  `torch.load(weights_only=False)` a user `.pt`, executing arbitrary code. They now
  profile `.pt` safely from the state_dict (`profiler.profile_checkpoint`, no code
  execution); `ingestion.load_model` refuses pickled checkpoints by default. Opt in
  with `--allow-pickle` / `allow_pickle=True` for files you trust.

### Fixed (correctness)
- **No more unsafe LLM picks.** The benchmark never fabricates `retention=1.0` when no
  quality was measured: it refuses to crown a "recommended" (and says so) instead of
  silently shipping the smallest, possibly-wrecked variant. `--min-retention` /
  `--min-accuracy` now error (`meets=False`) when they can't be checked, instead of
  passing on size alone. The LLM auto flow now measures **WikiText-2 perplexity** and
  picks on it.
- **LLM speed is real.** Latency/throughput for causal LMs is now **tokens/sec from a
  short `generate()`**, not a single ~8-token prefill forward.
- **`compress_and_benchmark("hf-id")`** (the documented headline) no longer raises — it
  loads the model instead of a dict.
- **`from autofollowdown import diagnose`** now returns the function (the module was
  renamed `diagnose.py` → `diagnosis.py` to stop it shadowing the export).

### Changed (honesty & practicality)
- `advise` / `diagnose` no longer recommend things the local machine can't produce:
  on a CPU target they recommend portable INT8 (not GPU-only INT4), and when 4-bit is
  ideal but no GPU backend is runnable they say so explicitly.
- `diagnose` memory math models the KV cache in fp16 (precision-independent) and flags
  big models that "fit" on CPU but run too slowly to be practical; it no longer treats
  "budget > 4 GB" as "has a GPU".
- The LLM auto flow skips the no-op unstructured-prune rows (dense `.pt` doesn't shrink
  from zeros) and the wasted multi-GB deepcopy.
- **Lighter install:** the heavy ONNX stack (onnx/onnxruntime/onnxscript) moved to an
  `[onnx]` extra; `api`/`ingestion` import torch-only for the core path (a plain CNN no
  longer pulls in onnxruntime/transformers). Package version is now single-sourced from
  `__init__.__version__`. `info` shows when a backend is installed but needs a CUDA GPU.
- +11 tests (`test_fixes.py`); full suite 175 passed.

## [0.4.0] - 2026-06-07

### Changed — auto-first flow redesign
- **`autofollowdown <model>` just works.** A bare model id / `.pt` path (no
  subcommand) now runs the full auto flow — profile → compress every way →
  benchmark → auto-pick the best for your goal → save.
- **New `flow.autopilot()` orchestrator** unifies the one-command experience. It is
  fully automatic and asks only at the two genuine decision points — your *goal*
  (size/speed/accuracy/ease/balanced) and which *variant* to keep — and only when a
  human is at a TTY. `--yes`, a pipe, or any of `--goal` / `--method` /
  `--max-size-mb` / `--min-retention` makes it run unattended.
- `compress` / `auto` gained `--goal`, `--max-size-mb`, `--min-retention`; the goal
  drives the automatic variant pick (size→smallest, speed→fastest, else→recommended),
  and size/accuracy constraints auto-select the best variant that satisfies them.
- New `flow.choose()` helper: the reusable "auto by default, menu when it matters"
  primitive. Bare CLI now advertises the auto-first and "stuck?" entry points.
- +12 tests (`test_flow.py`); full suite 164 passed.

## [0.3.0] - 2026-06-07

### Added
- **`diagnose` — symptom-first help.** Start from your *problem*, not the jargon:
  `autofollowdown diagnose <model> --problem won't-fit --vram 8` (or
  `--device raspberry-pi-5 / phone / gpu-8gb`). It estimates whether the model fits
  at fp16/int8/int4, gives a plain verdict, the recommended plan, and the exact next
  command — and honestly says "distill or offload" when it won't fit even at 4-bit.
  Device presets for Raspberry Pi / Jetson / phone / microcontroller / GPU tiers.
- **`advise` — a compression advisor.** Recommends *which* technique(s) (quantize /
  prune / distill) + backend to use for a model and why, in the right order, with
  caveats — driven by a declarative `TECHNIQUES` knowledge base. Honors `--goal`,
  `--max-size-ratio`, `--min-retention`, `--can-retrain`, `--hardware`.
- **Constraint-aware decisions on the benchmark**: `CompressionStudy.pick_best(
  max_size_mb=…, min_accuracy=…, min_retention=…)` returns the best variant meeting
  hard limits (or the closest), and `.frontier()` / `Benchmark.pareto_frontier()`
  flag the non-dominated variants on the size↔quality trade-off.
- The bare CLI and `diagnose` with no model now point newcomers at the "stuck? start
  here" flow. +25 tests (advisor + diagnose); full suite 152 passed.

## [0.2.0] - 2026-06-07

### Added
- **Three new compression backends**: `torchao` (PyTorch-native int8/int4/fp8 +
  `torch.compile`, no calibration), `bitsandbytes` (NF4/INT8, easiest path), and
  `HQQ` (fast 4/3/2-bit, no calibration). The registry now has seven backends.
- **Capability-driven router**: each backend declares a `Capability` (families,
  traits, hardware, calibration need) and a single generic scorer ranks them — no
  per-backend hardcoded rules. Adding a backend is a data entry, not new code.
- **Goal-aware routing**: `recommend --goal {balanced,accuracy,size,speed,ease}`
  re-ranks via the `GOAL_TRAITS` / `GOAL_AVOID` data maps (e.g. `ease` prefers the
  no-calibration backends).
- **GPU memory planner** (`gpu.py`): `cuda_info`, `memory_plan`, `free_memory`,
  `load_balanced`, plus the `autofollowdown gpu` command — sequential onloading so
  big LLMs calibrate on a free/small GPU.
- **Showcase notebook** (`notebooks/autofollowdown_showcase.ipynb`): every CLI
  command in ~2 minutes, with real output.
- OSS scaffolding: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, this changelog, and
  GitHub issue / PR templates. New install extras: `torchao`, `bnb`, `hqq`.

### Changed
- **~70× faster CLI startup**: `import autofollowdown` is now lazy (PEP 562), so
  `--help` / `info` / `recommend` no longer pay the torch/transformers import cost
  (~2.2s → ~0.03s).
- `compress_and_benchmark(...)` now accepts a Hugging Face id or `.pt` path string,
  not just a pre-loaded `nn.Module`.
- The CLI prints a friendly one-line error (with an install hint or "model not
  found") instead of a raw traceback; set `AFD_DEBUG=1` for the full trace.
- The `llm-compressor` backend auto-selects memory-saving settings (sequential
  onloading) based on model size and free VRAM.

### Fixed
- `count_parameters` syncs once instead of per-tensor (faster on GPU / large models).
- Removed a redundant full `profile_model` pass in `compress_and_benchmark`.
- `free_memory()` is now called between benchmarked variants to avoid OOM.

## [0.1.0] - 2026-06-06

### Added
- Initial release: real quantization / pruning / distillation (`ModelCompressor`),
  the one-command `compress_and_benchmark` + `CompressionStudy`, the multi-library
  auto-picker (native + NNI + llm-compressor + NVIDIA ModelOpt), a real benchmark
  engine, LLM perplexity + lm-eval-harness catalog (incl. MMLU-ProX, MMMU), ONNX
  export/quant, the `autofollowdown` CLI, and CPU / Colab demo notebooks.
