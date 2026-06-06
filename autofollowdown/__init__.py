"""autofollowdown — unified model compression (quantize · prune · distill) with
real benchmarks and a capability-driven backend router.

Quick start (one call does it all):

    from autofollowdown import compress_and_benchmark
    study = compress_and_benchmark("facebook/opt-125m")   # compress, benchmark, pick
    study.show()

Public names are imported lazily (PEP 562): `import autofollowdown` and the CLI
stay near-instant, and the heavy deep-learning stack (torch / transformers / onnx)
is only loaded the first time you actually touch a symbol that needs it.
"""

__version__ = "0.3.0"

# name -> submodule it lives in. The single source of truth for what we export
# and where it resolves; `__getattr__` loads the submodule on first access.
_EXPORTS = {
    # core API
    "ModelCompressor": "api",
    # one-command workflow
    "compress_and_benchmark": "pipeline",
    "CompressionStudy": "pipeline",
    # auto-picker / router
    "auto_compress": "auto",
    "compress_with": "auto",
    "recommend": "auto",
    "recommend_profile": "auto",
    "rank_backends": "auto",
    "explain": "auto",
    "Recommendation": "auto",
    "all_backends": "backends",
    "get_backend": "backends",
    # compression advisor (which technique + backend to use, and why)
    "advise": "advisor",
    "CompressionPlan": "advisor",
    "TECHNIQUES": "advisor",
    # symptom-first help ("I can't run this model")
    "diagnose": "diagnose",
    "Diagnosis": "diagnose",
    "DEVICE_PRESETS": "diagnose",
    "memory_needs": "diagnose",
    # profiling
    "profile_model": "profiler",
    "profile_from_pretrained": "profiler",
    "ModelProfile": "profiler",
    # GPU memory helpers (free / small-GPU friendly)
    "cuda_info": "gpu",
    "memory_plan": "gpu",
    "free_memory": "gpu",
    "estimate_weight_gb": "gpu",
    "load_balanced": "gpu",
    # benchmarking
    "Benchmark": "benchmark",
    "count_parameters": "metrics",
    "evaluate_accuracy": "metrics",
    "measure_latency": "metrics",
    "measure_model": "metrics",
    "model_disk_size_mb": "metrics",
    "output_agreement": "metrics",
    # LLM evaluation + benchmark catalog
    "evaluate_perplexity": "llm_eval",
    "perplexity_from_ids": "llm_eval",
    "load_wikitext2": "llm_eval",
    "lm_eval_command": "llm_eval",
    "mmlu_prox_tasks": "llm_eval",
    "mmmu_tasks": "llm_eval",
    "multimodal_eval_command": "llm_eval",
    "STANDARD_LLM_TASKS": "llm_eval",
    "DEFAULT_ZEROSHOT_SUITE": "llm_eval",
    "MMLU_PROX_LANGS": "llm_eval",
    "MMMU_DISCIPLINES": "llm_eval",
    # ingestion
    "load_model": "ingestion",
    # ONNX
    "ONNXCalibrationDataReader": "onnx_pipeline",
    "export_to_onnx": "onnx_pipeline",
    "prune_onnx": "onnx_pipeline",
    "optimize_onnx": "onnx_pipeline",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    """PEP 562 lazy attribute loader — imports the owning submodule on demand."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value          # cache so subsequent access skips __getattr__
    return value


def __dir__():
    return sorted(list(globals()) + __all__)
