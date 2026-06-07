"""One-command compression workflow.

`compress_and_benchmark(model)` is the whole pipeline in a single call: profile
the model, apply every compression method, benchmark them side by side, recommend
the best, and hand back a `CompressionStudy` you can pick a variant from and export.

    study = compress_and_benchmark(model, eval_loader=test_loader)
    study.show()                         # table + "which to pick"
    best = study.best()                  # the recommended compressed model
    chosen = study.pick("prune+quantize")
    study.export("prune+quantize", "model.pt")
"""

import copy

import torch

from ._term import color
from .api import ModelCompressor
from .benchmark import Benchmark

# Universal methods that need no training data — applied to every model.
DEFAULT_METHODS = ["int8 dynamic", "prune 50%", "prune+quantize"]


def _apply(method, model, calibration_data=None):
    """Return a compressed copy of `model` for the named method."""
    c = ModelCompressor(model)
    if method == "int8 dynamic":
        return c.quantize(method="int8", approach="dynamic").model
    if method == "prune 50%":
        return c.prune(sparsity=0.5, method="unstructured").model
    if method == "prune+quantize":
        return c.prune(sparsity=0.5, method="unstructured") \
                .quantize(method="int8", approach="dynamic").model
    if method == "fp16":
        return c.quantize(method="fp16").model
    raise ValueError(f"Unknown method: {method!r}")


class CompressionStudy:
    """Holds the baseline + compressed variants, their benchmark, and the pick."""

    def __init__(self, baseline, example_input, eval_loader=None, device="cpu",
                 latency_runs=20, quality_fn=None):
        self.device = device
        self.models = {}
        self._bench = Benchmark(example_input, eval_loader=eval_loader,
                                reference_model=baseline, device=device,
                                latency_runs=latency_runs, quality_fn=quality_fn)
        self._add("baseline", baseline)

    def _add(self, name, model):
        self.models[name] = model
        self._bench.measure(model, name)

    def add(self, name, model):
        """Add an externally-built variant (e.g. a distilled student)."""
        self._add(name, model)
        return self

    @property
    def names(self):
        return list(self.models)

    @property
    def recommended(self):
        pick = self._bench.best_picks().get("recommended")
        return pick["name"] if pick else "baseline"

    def pick(self, name):
        """Return the model for a named variant (raises if the name is unknown)."""
        if name not in self.models:
            raise KeyError(f"No variant {name!r}. Choose from {self.names}")
        return self.models[name]

    def best(self):
        """Return the recommended compressed model."""
        return self.pick(self.recommended)

    def pick_best(self, max_size_mb=None, min_accuracy=None, min_retention=None,
                  prefer="recommended"):
        """Pick the best variant meeting hard constraints (size budget / accuracy
        floor). Returns (name, model, info); info.meets says whether it satisfied
        them. e.g. study.pick_best(min_retention=0.98) → smallest near-lossless one."""
        info = self._bench.pick_best(max_size_mb=max_size_mb, min_accuracy=min_accuracy,
                                     min_retention=min_retention, prefer=prefer)
        row = info.get("row")
        name = row["name"] if row else None
        model = self.pick(name) if name in self.models else None
        return name, model, info

    def frontier(self):
        """Names of variants on the size↔quality Pareto frontier (the rest are
        dominated — bigger *and* less accurate than one of these)."""
        return self._bench.pareto_frontier()

    def report(self):
        return self._bench.report()

    def show(self):
        print(self._bench.to_table())
        print("\n" + self._bench.summary())

    def to_markdown(self):
        """Markdown table of the study — handy for notebooks (display(Markdown(...)))."""
        return self._bench.to_markdown()

    def export(self, name, path, format="pt"):
        """Save a chosen variant. `safetensors` saves weights safely (no pickle);
        `pt` saves the full torch model; `onnx` exports a graph (float models only —
        see ModelCompressor.export for the caveat)."""
        model = self.pick(name)
        if format == "safetensors":
            from .api import save_safetensors
            save_safetensors(model, path)
        elif format == "pt":
            torch.save(model, path)
        elif format == "onnx":
            from .onnx_pipeline import export_to_onnx
            export_to_onnx(model, "pytorch", path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        return path

    def interactive_pick(self):
        """Prompt the user to choose a variant; returns (name, model)."""
        rows = self.report()
        rec = self.recommended
        print(color("\nPick a method to keep:", "bold"))
        for i, r in enumerate(rows, 1):
            tag = color("  (recommended)", "green") if r["name"] == rec else ""
            print(f"  {i}. {r['name']}{tag}")
        default_idx = next((i for i, r in enumerate(rows, 1)
                            if r["name"] == rec), 1)
        try:
            raw = input(f"Choice [1-{len(rows)}, default {default_idx}]: ").strip()
        except EOFError:
            raw = ""
        idx = int(raw) if raw.isdigit() and 1 <= int(raw) <= len(rows) else default_idx
        name = rows[idx - 1]["name"]
        return name, self.pick(name)


def compress_and_benchmark(model, example_input=None, eval_loader=None,
                           methods=None, calibration_data=None, device="cpu",
                           latency_runs=20, input_shape=None, quality_fn=None):
    """Run the full workflow and return a CompressionStudy.

    Applies each method in `methods` (default: INT8 dynamic, 50% pruning, and
    both stacked) to a copy of `model`, benchmarks every variant against the
    baseline, and recommends the best size/quality trade-off. `eval_loader`
    enables accuracy/fidelity; if `example_input` is omitted it's inferred.
    """
    if isinstance(model, str):
        from .ingestion import load_model            # accept a HF id / path directly
        loaded = load_model(model)
        if loaded["type"] == "onnx":
            raise ValueError("compress_and_benchmark works on torch models; for ONNX use "
                             "optimize_onnx/prune_onnx in onnx_pipeline.")
        model = loaded["model"]
    if not isinstance(model, torch.nn.Module):
        raise ValueError("compress_and_benchmark expects a torch.nn.Module or a model id/path")

    if example_input is None:
        from .onnx_pipeline import get_working_dummy_input
        example_input = get_working_dummy_input(model, input_shape)

    methods = methods or DEFAULT_METHODS
    study = CompressionStudy(model, example_input, eval_loader=eval_loader,
                             device=device, latency_runs=latency_runs, quality_fn=quality_fn)
    from .gpu import free_memory
    for method in methods:
        try:
            variant = _apply(method, copy.deepcopy(model), calibration_data)
            study.add(method, variant)
        except Exception as e:  # a method may not apply to every model — keep going
            print(color(f"  (skipped {method}: {e})", "yellow"))
        free_memory()           # release cached VRAM between methods so the next won't OOM
    return study
