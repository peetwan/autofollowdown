# Changelog

All notable changes to autofollowdown are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
