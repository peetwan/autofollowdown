"""GPU memory helpers — make the heavy backends runnable on small / free GPUs.

The specialized LLM backends (llm-compressor, NVIDIA ModelOpt) can need a lot of
VRAM during calibration, which is why they look scary to run for free. These
helpers detect what the current GPU can actually hold and pick memory-saving
settings automatically, so compression still runs on a free Colab / Kaggle T4
(16 GB) instead of OOM-ing.

The techniques come straight from the libraries' own docs:
  • Sequential onloading (llm-compressor `pipeline="sequential"`) keeps only one
    slice of the model on the GPU at a time; the rest stays on CPU/disk.
  • `sequential_targets="Linear"` shrinks that slice further — less VRAM, a bit
    slower — which is what lets a big model calibrate on a tiny GPU.
  • `device_map="auto"` loads onto the GPU and automatically spills the overflow
    to CPU (then disk), so a model larger than VRAM still loads.

Everything here is pure and CPU-safe: pass an explicit `vram_gb` and the planner
works with no GPU present, which is also how it is unit-tested.
"""

import gc


def cuda_info():
    """Return {available, name, total_gb, free_gb} for the current CUDA device.

    Safe to call anywhere — returns a CPU placeholder when torch or CUDA is
    missing, so callers never have to guard the import themselves.
    """
    try:
        import torch
    except ImportError:
        return {"available": False, "name": "cpu", "total_gb": 0.0, "free_gb": 0.0}
    if not torch.cuda.is_available():
        return {"available": False, "name": "cpu", "total_gb": 0.0, "free_gb": 0.0}
    free, total = torch.cuda.mem_get_info()
    props = torch.cuda.get_device_properties(0)
    return {"available": True, "name": props.name,
            "total_gb": total / 1e9, "free_gb": free / 1e9}


def free_memory():
    """Release cached allocations. Call this between compression runs so a second
    method doesn't OOM on memory the first one only *cached* (PyTorch keeps freed
    blocks in a pool; `empty_cache` hands them back to the GPU)."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def estimate_weight_gb(num_params, bytes_per_param=2):
    """Rough VRAM (GB) the weights occupy at a given precision (default fp16 = 2
    bytes). A 0.6B model ≈ 1.2 GB in fp16; a 7B model ≈ 14 GB."""
    return (num_params or 0) * bytes_per_param / 1e9


# Heuristic VRAM floor (GB) for sequential onloading to hold one decoder layer
# plus its GPTQ hessian and a calibration batch. Tuned conservatively for a T4.
_SEQUENTIAL_FLOOR_GB = 3.0
_LINEAR_FLOOR_GB = 1.5


def memory_plan(num_params, vram_gb=None, overhead=1.3):
    """Pick memory-saving load + calibration settings for the current GPU.

    Returns a dict ready to splat into `compress_with(model, "llmcompressor", ...)`
    plus a `device_map` for loading the model and a human `note`:

        plan = memory_plan(profile.num_params)
        model = load_balanced(model_id, device_map=plan["device_map"])
        compress_with(model, "llmcompressor",
                      pipeline=plan["pipeline"],
                      sequential_targets=plan["sequential_targets"], ...)

    The logic, given model weight size W (fp16) and free VRAM V:
      • V ≥ W·overhead  → "basic" pipeline, load straight on GPU (fastest).
      • V ≥ 3 GB        → "sequential", device_map="auto" (CPU spills the rest).
      • V ≥ 1.5 GB      → "sequential" + sequential_targets="Linear" (less VRAM).
      • otherwise / CPU → "sequential" + "Linear" + device_map="auto" (CPU/disk).

    These are deliberately conservative heuristics (like the backend fit scores),
    not exact memory math — the goal is "it runs for free" over "it runs fastest".
    """
    info = cuda_info()
    vram = info["free_gb"] if vram_gb is None else float(vram_gb)
    weights = estimate_weight_gb(num_params)

    if vram <= 0:  # CPU-only: nothing to onload, keep it sequential + tiny
        return {"strategy": "cpu", "pipeline": "sequential",
                "sequential_targets": "Linear", "device_map": "auto",
                "fits_on_gpu": False,
                "note": "No CUDA GPU detected — use the portable native INT8 path "
                        "on CPU, or run the GPU backends on a free Colab/Kaggle T4."}

    if vram >= weights * overhead and weights > 0:
        return {"strategy": "fits", "pipeline": "basic",
                "sequential_targets": None, "device_map": "cuda",
                "fits_on_gpu": True,
                "note": f"Model (~{weights:.1f} GB) fits in {vram:.1f} GB VRAM — "
                        "loading straight on GPU with the fast 'basic' pipeline."}

    if vram >= _SEQUENTIAL_FLOOR_GB:
        return {"strategy": "sequential", "pipeline": "sequential",
                "sequential_targets": None, "device_map": "auto",
                "fits_on_gpu": False,
                "note": f"Model (~{weights:.1f} GB) is larger than {vram:.1f} GB VRAM — "
                        "sequential onloading keeps one layer on GPU and offloads the "
                        "rest to CPU. Runs free on a T4; a bit slower."}

    if vram >= _LINEAR_FLOOR_GB:
        return {"strategy": "sequential-linear", "pipeline": "sequential",
                "sequential_targets": "Linear", "device_map": "auto",
                "fits_on_gpu": False,
                "note": f"Tight VRAM ({vram:.1f} GB) — sequential onloading with "
                        "sequential_targets='Linear' onloads one Linear at a time "
                        "(lowest VRAM, slower runtime)."}

    return {"strategy": "offload", "pipeline": "sequential",
            "sequential_targets": "Linear", "device_map": "auto",
            "fits_on_gpu": False,
            "note": f"Very low VRAM ({vram:.1f} GB) — onloading one Linear at a time "
                    "and offloading everything else to CPU/disk so it still runs."}


def load_balanced(model_id, dtype="auto", device_map=None, max_memory=None,
                  offload_folder="./afd_offload", **kwargs):
    """Load an HF causal LM so a model bigger than VRAM still fits, by letting
    `device_map="auto"` spill the overflow to CPU and then disk.

    This is the robust, free-GPU-friendly way to load before quantizing — it is a
    thin convenience around `AutoModelForCausalLM.from_pretrained`, so any extra
    kwargs pass straight through.
    """
    from transformers import AutoModelForCausalLM
    if device_map is None:
        device_map = "auto" if cuda_info()["available"] else None
    load_kwargs = dict(dtype=dtype, low_cpu_mem_usage=True, **kwargs)
    if device_map is not None:
        load_kwargs["device_map"] = device_map
    if max_memory is not None:
        load_kwargs["max_memory"] = max_memory
    if device_map == "auto_offload" or (max_memory is not None):
        load_kwargs["offload_folder"] = offload_folder
    return AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)


def summary(num_params=None, vram_gb=None):
    """One-paragraph, human-readable GPU + plan report for the CLI / notebooks."""
    info = cuda_info()
    if info["available"]:
        head = (f"GPU: {info['name']} — {info['free_gb']:.1f} GB free / "
                f"{info['total_gb']:.1f} GB total")
    else:
        head = "GPU: none (CPU-only) — native INT8 runs here; GPU backends need a T4+"
    if num_params:
        plan = memory_plan(num_params, vram_gb)
        return (f"{head}\nModel ~{num_params/1e6:.0f}M params "
                f"(~{estimate_weight_gb(num_params):.1f} GB fp16)\n"
                f"Plan: {plan['strategy']} — pipeline='{plan['pipeline']}'"
                + (f", sequential_targets='{plan['sequential_targets']}'"
                   if plan['sequential_targets'] else "")
                + f", device_map='{plan['device_map']}'\n{plan['note']}")
    return head
