"""Compression advisor — decide *which* technique(s) to use, all in one place.

`recommend()` answers "which library"; this answers the bigger question users
actually start with: given my model and what I care about (plus any hard limits
like a size budget or an accuracy floor), should I quantize, prune, distill — or
combine them, in what order, with which backend, and what should I expect?

The domain knowledge (effect ranges, requirements, caveats, when-to-use) lives as
**data** in `TECHNIQUES`, grounded in published practitioner guidance; `advise()`
composes a concrete, ordered plan from it + the model profile + the capability
router. The closing advice is always the honest one: measure it with
`compress_and_benchmark` before you ship.
"""

from dataclasses import dataclass, field


# Declarative knowledge about each technique. Numbers are typical ranges from
# practitioner guides, not promises — the plan tells users to verify by measuring.
TECHNIQUES = {
    "quantize-int8": {
        "label": "INT8 quantization",
        "shrink": "~4× vs FP32 (≈2× vs FP16)",
        "accuracy_cost": "usually <1–2%",
        "speed": "up to ~2–4× on CPU",
        "retrain": False,
        "best_for": "the safe default — general-purpose, no retraining, CPU-friendly",
        "caveats": ["accuracy depends on calibration data matching real traffic"],
    },
    "quantize-int4": {
        "label": "INT4 weight-only quantization (GPTQ/AWQ)",
        "shrink": "~8× vs FP32 (≈4× vs FP16)",
        "accuracy_cost": "~2–5%",
        "speed": "fast with optimized kernels (vLLM/TensorRT)",
        "retrain": False,
        "best_for": "the smallest weight-only LLM on a GPU",
        "caveats": ["INT4 can degrade reasoning/code tasks — safest for recall/summarization",
                    "best quality needs calibrated GPTQ/AWQ on a GPU"],
    },
    "prune-structured": {
        "label": "structured pruning (channels/filters)",
        "shrink": "~1.3–2×",
        "accuracy_cost": "~2–5% (recoverable by fine-tuning)",
        "speed": "real speedup on commodity hardware",
        "retrain": True,
        "best_for": "CNNs and any model where you need an actual latency win",
        "caveats": ["fine-tune after pruning to recover accuracy"],
    },
    "prune-unstructured": {
        "label": "unstructured (magnitude) pruning",
        "shrink": "high sparsity → smaller storage",
        "accuracy_cost": "~2–5%",
        "speed": "little/no CPU speedup without sparse kernels",
        "retrain": False,
        "best_for": "shrinking storage / a pre-conditioner before quantization",
        "caveats": ["it reduces size, not latency, on standard CPUs/GPUs"],
    },
    "distill": {
        "label": "knowledge distillation (smaller student)",
        "shrink": "10–100× (a genuinely smaller architecture)",
        "accuracy_cost": "~5–15% (task-dependent)",
        "speed": "large — fewer parameters, fewer ops",
        "retrain": True,
        "best_for": "a narrow, stable task where you can retrain and need a much smaller model",
        "caveats": ["needs training data + compute (a full training run)",
                    "the student specializes — weaker on out-of-domain inputs"],
    },
}

# Recommended order when stacking, per practitioner guidance
# (distill rebuilds capability in a smaller net → prune trims structure → quantize last).
_ORDER = ["distill", "prune-structured", "prune-unstructured", "quantize-int4", "quantize-int8"]


@dataclass
class Step:
    technique: str      # key into TECHNIQUES
    backend: str        # backend alias to run it with (e.g. "native", "llmcompressor")
    why: str

    @property
    def info(self):
        return TECHNIQUES[self.technique]


@dataclass
class CompressionPlan:
    goal: str
    family: str
    steps: list = field(default_factory=list)
    backend_pick: str = ""        # the ideal library for this model
    runnable_pick: str = ""       # the best one runnable here right now
    caveats: list = field(default_factory=list)
    constraints: dict = field(default_factory=dict)

    @property
    def headline(self):
        if not self.steps:
            return "Keep the model as-is — no compression needed."
        labels = " → ".join(TECHNIQUES[s.technique]["label"] for s in self.steps)
        return labels

    def to_text(self, color=lambda s, *a: s):
        out = [color("Recommended plan: ", "bold", "cyan") + self.headline, ""]
        for i, s in enumerate(self.steps, 1):
            t = s.info
            out.append(color(f"  {i}. {t['label']}  ", "bold") + f"[via {s.backend}]")
            out.append(f"     why     : {s.why}")
            out.append(f"     expect  : {t['shrink']}, accuracy cost {t['accuracy_cost']}; "
                       f"speed {t['speed']}")
        out.append("")
        out.append(color("Best library for this model: ", "green", "bold") + self.backend_pick
                   + (f"   (runnable here: {self.runnable_pick})"
                      if self.runnable_pick and self.runnable_pick != self.backend_pick else ""))
        if self.caveats:
            out.append("")
            out.append(color("Watch out for:", "yellow", "bold"))
            for c in self.caveats:
                out.append(f"  • {c}")
        out.append("")
        out.append(color("→ ", "green") + "Verify on YOUR data before shipping: "
                   "compress_and_benchmark(model)  (or  autofollowdown compress <model>).")
        return "\n".join(out)


def _resolve_profile(model, allow_pickle=False):
    """Accept a ModelProfile, an nn.Module, a HF id, or a .pt path.

    `.pt`/`.pth` files are profiled safely (no pickle execution) unless
    `allow_pickle=True` — see profiler.profile_checkpoint."""
    from .profiler import (ModelProfile, profile_checkpoint,
                           profile_from_pretrained, profile_model)
    if isinstance(model, ModelProfile):
        return model
    if isinstance(model, str):
        if model.endswith((".pt", ".pth", ".safetensors")):
            return profile_checkpoint(model, allow_pickle=allow_pickle)
        return profile_from_pretrained(model)
    return profile_model(model)


def advise(model, goal="balanced", max_size_ratio=None, min_accuracy_retention=None,
           can_retrain=False, hardware=None, allow_pickle=False):
    """Return a `CompressionPlan` — which technique(s) + backend to use, and why.

    Parameters
    - goal: balanced | accuracy | size | speed | ease (what you care about most).
    - max_size_ratio: target fraction of the original size (e.g. 0.25 = 4× smaller).
    - min_accuracy_retention: accuracy floor as a fraction (e.g. 0.98 = keep ≥98%).
    - can_retrain: True unlocks pruning fine-tuning and distillation.
    - hardware: "gpu" | "cpu" | None (auto-detected from the profile if None).
    """
    from .auto import rank_backends

    profile = _resolve_profile(model, allow_pickle=allow_pickle)
    fam = profile.family
    on_gpu = profile.cuda_available if hardware is None else (hardware == "gpu")

    recs = sorted(rank_backends(profile, goal), key=lambda r: r.score, reverse=True)
    ideal = recs[0] if recs else None
    runnable = next((r for r in recs if r.runnable), None)
    backend_alias = _alias_for(ideal.backend) if ideal else "native"
    runnable_alias = _alias_for(runnable.backend) if runnable else "native"

    aggressive = goal == "size" or (max_size_ratio is not None and max_size_ratio <= 0.25)
    want_speed = goal == "speed"

    chosen = {}     # technique -> why

    # 1) Distillation — only when a big reduction is needed AND retraining is allowed.
    if can_retrain and (aggressive or (max_size_ratio is not None and max_size_ratio <= 0.5)):
        chosen["distill"] = ("you allowed retraining and want a big reduction — a smaller "
                             "student gives a genuinely smaller, faster architecture.")

    # 2) Pruning — structured for real speed/CNNs; unstructured only as a size/pre-quant step.
    if fam == "cnn":
        chosen["prune-structured"] = ("CNNs prune well structurally, and structured pruning "
                                      "gives a real latency win on ordinary hardware.")
    elif want_speed and can_retrain:
        chosen["prune-structured"] = ("you want speed and can fine-tune — structured pruning "
                                      "removes whole units for an actual speedup.")
    elif aggressive:
        chosen["prune-unstructured"] = ("extra size reduction and a good pre-conditioner before "
                                        "quantization (won't speed up CPU on its own).")

    # 3) Quantization — almost always the foundation. INT4 when squeezing an LLM on a GPU.
    if fam in ("llm", "transformer") and (aggressive or want_speed) and on_gpu:
        chosen["quantize-int4"] = ("for an LLM on a GPU, weight-only 4-bit (GPTQ/AWQ) is the "
                                   "biggest no-retrain size win with strong quality.")
    else:
        chosen["quantize-int8"] = ("the safe default — ~4× smaller with little accuracy loss "
                                   "and no retraining; portable to CPU.")

    # Route each technique to a backend that actually performs it (the overall
    # router pick is only used for low-bit quant, where it's a real quantizer).
    _QUANTIZERS = {"llmcompressor", "modelopt", "torchao", "bnb", "hqq"}

    def backend_for(tech):
        if tech == "quantize-int4":
            return backend_alias if backend_alias in _QUANTIZERS else "native"
        if tech == "quantize-int8":
            return "native"           # portable INT8 always works (torchao if you want compile)
        if tech == "prune-structured":
            return "nni"              # structured pruning + ModelSpeedup is NNI's specialty
        return "native"               # distillation / unstructured pruning are native

    steps = [Step(t, backend_for(t), chosen[t]) for t in _ORDER if t in chosen]

    # Caveats: union of chosen techniques' caveats + goal/constraint notes.
    caveats = []
    for s in steps:
        caveats.extend(s.info["caveats"])
    if min_accuracy_retention:
        caveats.append(f"you asked for ≥{min_accuracy_retention:.0%} accuracy — measure each "
                       "step and stop if it dips below that.")
    if max_size_ratio:
        caveats.append(f"target is {1/max_size_ratio:.1f}× smaller — stack steps in this order "
                       "and check the size after each.")
    # Honesty: if the plan recommends 4-bit but no low-bit quantizer is actually
    # runnable here, say so — don't recommend something the local machine can't produce.
    has_int4 = any(s.technique == "quantize-int4" for s in steps)
    runnable_quantizer = next(
        (r for r in recs if r.runnable and _alias_for(r.backend) in _QUANTIZERS), None)
    if has_int4 and runnable_quantizer is None:
        caveats.insert(0, "INT4 GPTQ/AWQ needs a GPU + a backend like llm-compressor / "
                          "torchao — none is runnable here, so locally you'd only get "
                          "portable INT8 (native). Run the 4-bit step on a GPU (e.g. Colab).")
    elif has_int4 and not on_gpu:
        caveats.append("4-bit quality is best on a GPU; on CPU prefer portable INT8 (native).")
    # de-duplicate, keep order
    caveats = list(dict.fromkeys(caveats))

    return CompressionPlan(
        goal=goal, family=fam, steps=steps,
        backend_pick=ideal.backend if ideal else "autofollowdown (native)",
        runnable_pick=runnable.backend if runnable else "autofollowdown (native)",
        caveats=caveats,
        constraints={"max_size_ratio": max_size_ratio,
                     "min_accuracy_retention": min_accuracy_retention,
                     "can_retrain": can_retrain, "hardware": "gpu" if on_gpu else "cpu"},
    )


def _alias_for(backend_name):
    from .backends import all_backends
    for b in all_backends():
        if b.name == backend_name:
            return b.alias
    return backend_name
