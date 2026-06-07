"""Real measurement utilities for benchmarking compressed models.

Every function here measures an actual property of a real model — no mocks,
no hardcoded numbers. These are the primitives the benchmark engine composes.
"""

import os
import tempfile
import time

import torch


def count_parameters(model):
    """Return (total_params, nonzero_params, sparsity_fraction).

    Sparsity is the fraction of weight elements that are exactly zero, which is
    what (unstructured) pruning produces. Counting nonzeros is the honest way to
    report pruning — a mask that is never made permanent would show 0 sparsity.
    """
    total = 0
    nonzero_acc = None
    for p in model.parameters():
        total += p.numel()
        c = torch.count_nonzero(p)
        # Accumulate on-device and sync only once at the end (one host transfer
        # instead of one per tensor — matters on GPU / large models).
        nonzero_acc = c if nonzero_acc is None else nonzero_acc + c.to(nonzero_acc.device)
    nonzero = int(nonzero_acc.item()) if nonzero_acc is not None else 0
    sparsity = 0.0 if total == 0 else 1.0 - (nonzero / total)
    return total, nonzero, sparsity


def model_disk_size_mb(model):
    """Serialize the model to a temp file and return its size in MB.

    We serialize rather than estimate, because quantized tensors, buffers, and
    packed params all change the real on-disk footprint in ways a parameter
    count cannot capture. We write to a temp file (not an in-memory buffer) so
    measuring multi-GB models (e.g. a few-billion-param LLM) doesn't blow up RAM.
    """
    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        torch.save(model, path)
        return os.path.getsize(path) / (1024 * 1024)
    finally:
        if os.path.exists(path):
            os.remove(path)


def _forward(model, batch):
    """Run a forward pass on a batch that may be a tensor, a (x, y) tuple, or a
    dict of named tensors (Hugging Face style). Returns the raw logits tensor."""
    if isinstance(batch, dict):
        out = model(**batch)
    elif isinstance(batch, (list, tuple)):
        out = model(batch[0])
    else:
        out = model(batch)
    # Hugging Face models return objects with a .logits attribute
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, (list, tuple)):
        return out[0]
    return out


def measure_latency(model, example_input, n_warmup=5, n_runs=30, device="cpu"):
    """Return (median_latency_ms, throughput_samples_per_s) for a single forward.

    Warmup runs are discarded so we don't measure lazy-init / cache-cold effects,
    and we report the median (p50) rather than the mean so a single GC pause or
    scheduler hiccup does not dominate the number.
    """
    model = model.to(device).eval()
    if isinstance(example_input, torch.Tensor):
        example_input = example_input.to(device)
        batch_size = example_input.shape[0]
    elif isinstance(example_input, dict):
        example_input = {k: v.to(device) for k, v in example_input.items()}
        batch_size = next(iter(example_input.values())).shape[0]
    else:
        batch_size = 1

    with torch.no_grad():
        for _ in range(n_warmup):
            _forward(model, example_input)

        timings = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _forward(model, example_input)
            timings.append(time.perf_counter() - start)

    timings.sort()
    median_s = timings[len(timings) // 2]
    median_ms = median_s * 1000.0
    throughput = batch_size / median_s if median_s > 0 else float("inf")
    return median_ms, throughput


@torch.no_grad()
def evaluate_accuracy(model, dataloader, device="cpu"):
    """Top-1 accuracy over a dataloader yielding (inputs, labels) batches.

    Returns a fraction in [0, 1]. This is the real task-quality signal: it tells
    you what compression actually cost you.
    """
    model = model.to(device).eval()
    correct = 0
    total = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = _forward(model, inputs)
        preds = logits.argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += labels.shape[0]
    return 0.0 if total == 0 else correct / total


@torch.no_grad()
def output_agreement(model, reference_model, dataloader, device="cpu"):
    """Fraction of samples where `model` and `reference_model` predict the same class.

    This is "fidelity": how faithfully the compressed model mimics the original,
    independent of ground-truth labels. Useful when you have no labels, or to
    separate "compression hurt the model" from "the model was always wrong here".
    """
    model = model.to(device).eval()
    reference_model = reference_model.to(device).eval()
    agree = 0
    total = 0
    for batch in dataloader:
        inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
        inputs = inputs.to(device)
        p1 = _forward(model, inputs).argmax(dim=1)
        p2 = _forward(reference_model, inputs).argmax(dim=1)
        agree += int((p1 == p2).sum().item())
        total += inputs.shape[0]
    return 1.0 if total == 0 else agree / total


def _looks_causal_lm(model, example_input):
    return (hasattr(model, "generate") and isinstance(example_input, dict)
            and "input_ids" in example_input)


@torch.no_grad()
def generation_speed(model, example_input, max_new_tokens=32, n_warmup=1, n_runs=3,
                     device="cpu"):
    """For a causal LM, time `generate()` and return (ms_per_token, tokens_per_s) —
    a meaningful inference-speed metric, unlike a single prefill forward."""
    import statistics
    model = model.to(device).eval()
    ids = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in example_input.items()}
    for _ in range(n_warmup):
        model.generate(**ids, max_new_tokens=4, do_sample=False)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
        times.append(time.perf_counter() - t0)
    med = statistics.median(times) if times else 0.0
    tok_per_s = (max_new_tokens / med) if med > 0 else 0.0
    ms_per_token = (med / max_new_tokens) * 1000 if max_new_tokens else 0.0
    return ms_per_token, tok_per_s


def measure_model(model, name, example_input, eval_loader=None,
                  reference_model=None, device="cpu", latency_runs=30, quality_fn=None):
    """Measure every available metric for one model and return a flat dict.

    `eval_loader` enables accuracy; `reference_model` enables fidelity; `quality_fn`
    (e.g. perplexity for LMs, lower = better) enables a quality signal when there is
    no labelled data. All optional, so size/latency always work.

    For a causal LM, latency is measured from `generate()` as ms/token (throughput =
    tokens/sec) instead of a single, misleading prefill forward.
    """
    total, nonzero, sparsity = count_parameters(model)
    if _looks_causal_lm(model, example_input):
        latency_ms, throughput = generation_speed(model, example_input, device=device)
    else:
        latency_ms, throughput = measure_latency(
            model, example_input, n_runs=latency_runs, device=device)
    result = {
        "name": name,
        "params": total,
        "nonzero_params": nonzero,
        "sparsity": sparsity,
        "size_mb": model_disk_size_mb(model),
        "latency_ms": latency_ms,
        "throughput": throughput,
        "accuracy": None,
        "fidelity": None,
        "perplexity": None,
    }
    if eval_loader is not None:
        result["accuracy"] = evaluate_accuracy(model, eval_loader, device=device)
    if reference_model is not None and eval_loader is not None:
        result["fidelity"] = output_agreement(
            model, reference_model, eval_loader, device=device)
    if quality_fn is not None:
        result["perplexity"] = float(quality_fn(model))   # lower = better
    return result
