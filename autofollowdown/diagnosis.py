"""Symptom-first help — "I can't run this model", solved.

Most people don't arrive saying "I'd like INT4 GPTQ". They arrive with a *problem*:
"it won't fit on my GPU", "it OOMs", "it's too slow", "I need it on a Raspberry Pi".
This module turns that plain-language pain into a concrete answer: does it fit, at
what precision, what to do about it, and the exact command to run next.

The memory math is the well-known rule of thumb (weights = params × bytes, where
fp16=2 / int8=1 / int4=0.5 bytes, plus KV-cache + framework overhead), and the
prescription follows the standard OOM ladder: shrink precision → if even 4-bit
won't fit, distill to a smaller model or offload. It's deliberately a rough
estimate — the closing advice is always to measure with `compress`/`benchmark`.
"""

from dataclasses import dataclass, field

from .advisor import advise
from .gpu import estimate_weight_gb


# Rough usable memory (GB) for the model on common targets — the device's RAM/VRAM
# minus what the OS / runtime / display already take. Grounded in edge-LLM guides.
DEVICE_PRESETS = {
    "raspberry-pi-4": (2.5, "Raspberry Pi 4 (8 GB, ~2.5 GB usable)"),
    "raspberry-pi-5": (6.0, "Raspberry Pi 5 (8 GB, ~6 GB usable)"),
    "jetson-orin-nano": (6.0, "Jetson Orin Nano (8 GB, ~6 GB usable)"),
    "phone": (2.5, "a phone (~2.5 GB usable)"),
    "microcontroller": (0.25, "a microcontroller (TinyML, ~256 MB)"),
    "laptop-cpu": (12.0, "a 16 GB laptop on CPU (~12 GB usable)"),
    "gpu-8gb": (6.5, "an 8 GB GPU (~6.5 GB usable)"),
    "gpu-12gb": (10.5, "a 12 GB GPU (~10.5 GB usable)"),
    "gpu-16gb": (14.5, "a 16 GB GPU (~14.5 GB usable)"),
    "gpu-24gb": (22.5, "a 24 GB GPU (~22.5 GB usable)"),
}

# Plain-language symptoms → how to frame the advice.
PROBLEMS = {
    "won't-fit": ("it won't fit / runs out of memory", "size"),
    "oom": ("it runs out of memory (OOM)", "size"),
    "too-slow": ("it's too slow", "speed"),
    "too-big": ("the file is too big to ship/store", "size"),
    "edge": ("you need it on a small edge device", "size"),
    "cost": ("it costs too much to serve", "size"),
}

_PRECISIONS = [("fp16", 2.0), ("int8", 1.0), ("int4", 0.5)]
_OVERHEAD_GB = 1.0      # framework / runtime + activation workspace (rough)
# Rough interactive-CPU throughput ceiling: above this many params, a CPU "fits" is
# still impractically slow (~a couple tok/s), so we flag it.
_CPU_SLOW_PARAMS = 3e9


def memory_needs(num_params, context_length=4096):
    """Rough GB to *run* the model at fp16 / int8 / int4 = weights(precision) + KV
    cache + overhead.

    Deliberately conservative and clearly approximate. The KV cache is modeled in
    fp16 and scaled by params × context (it does NOT shrink with weight precision —
    a common mistake that makes 4-bit look rosier than it is). It ignores GQA vs MHA,
    so treat it as a guide, not a guarantee — verify by actually loading the model.
    """
    p = num_params or 0
    kv_gb = (p / 1e9) * (context_length / 4096) * 0.25   # fp16 KV, precision-independent
    out = {}
    for name, bytes_pp in _PRECISIONS:
        out[name] = estimate_weight_gb(p, bytes_pp) + kv_gb + _OVERHEAD_GB
    return out


@dataclass
class Diagnosis:
    problem: str
    model: str
    num_params: int
    budget_gb: float
    budget_label: str
    needs: dict                 # precision -> GB needed
    fits: dict                  # precision -> bool
    verdict: str
    plan: object = None         # CompressionPlan
    commands: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def to_text(self, color=lambda s, *a: s):
        out = [color("Your problem: ", "bold", "cyan") + self.problem]
        if self.num_params:
            out.append(f"Model ~{self.num_params/1e6:.0f}M params; "
                       f"target: {self.budget_label}")
        out.append("")
        if self.budget_gb:
            out.append(color("Will it fit?", "bold"))
            for name, _ in _PRECISIONS:
                mark = color("✓ fits", "green") if self.fits[name] else color("✗ too big", "red")
                out.append(f"  {name:5} needs ~{self.needs[name]:5.1f} GB   {mark}")
            out.append("")
        out.append(color("→ ", "green", "bold") + self.verdict)
        if self.plan is not None and self.plan.steps:
            out.append("")
            out.append(color("Do this:", "bold") + " " + self.plan.headline
                       + f"  (best library: {self.plan.backend_pick})")
        if self.notes:
            out.append("")
            for n in self.notes:
                out.append(color("  • ", "yellow") + n)
        if self.commands:
            out.append("")
            out.append(color("Run next:", "bold"))
            for c in self.commands:
                out.append(color("  $ ", "cyan") + c)
        return "\n".join(out)


def _resolve_profile(model, allow_pickle=False):
    from .profiler import (ModelProfile, profile_checkpoint,
                           profile_from_pretrained, profile_model)
    if isinstance(model, ModelProfile):
        return model
    if isinstance(model, str):
        if model.endswith((".pt", ".pth", ".safetensors")):
            return profile_checkpoint(model, allow_pickle=allow_pickle)
        return profile_from_pretrained(model)
    return profile_model(model)


def _budget(vram_gb, device):
    if device:
        if device not in DEVICE_PRESETS:
            raise ValueError(f"Unknown device {device!r}. "
                             f"Options: {sorted(DEVICE_PRESETS)}")
        return DEVICE_PRESETS[device]
    if vram_gb is not None:
        return float(vram_gb), f"your {vram_gb:.0f} GB target"
    from .gpu import cuda_info
    info = cuda_info()
    if info["available"]:
        return info["free_gb"], f"this GPU ({info['name']}, {info['free_gb']:.1f} GB free)"
    return None, "your device"


def diagnose(model, problem="won't-fit", vram_gb=None, device=None,
             can_retrain=False, target_size_mb=None, allow_pickle=False):
    """Diagnose a real-world problem and prescribe a fix.

    problem : won't-fit | oom | too-slow | too-big | edge | cost
    vram_gb : how much memory you actually have (GB), or
    device  : a preset target (e.g. "raspberry-pi-5", "gpu-8gb", "phone").

    Returns a `Diagnosis` with a fit table, a plain verdict, the recommended plan,
    and the exact commands to run next.
    """
    profile = _resolve_profile(model, allow_pickle=allow_pickle)
    label, goal = PROBLEMS.get(problem, PROBLEMS["won't-fit"])
    budget_gb, budget_label = _budget(vram_gb, device)
    # an edge device or microcontroller usually implies you can/should retrain a student
    if device in ("microcontroller", "phone", "raspberry-pi-4"):
        can_retrain = True

    needs = memory_needs(profile.num_params or 0)
    fits = {n: (budget_gb is not None and needs[n] <= budget_gb) for n, _ in _PRECISIONS}

    notes = []
    commands = []
    model_ref = model if isinstance(model, str) else "<your-model>"

    # --- Verdict for memory/size problems (the common case) ---
    if goal == "size" and budget_gb is not None and (profile.num_params or 0) > 0:
        if fits["fp16"] and problem not in ("too-big", "edge"):
            verdict = (f"It already fits at fp16 (~{needs['fp16']:.1f} GB ≤ "
                       f"{budget_gb:.1f} GB). If you still OOM, it's usually the context "
                       "window / KV cache or other apps — and you can shrink further with INT8.")
            notes.append("Quantize to INT8 for ~2× headroom; shorten the context window if "
                         "OOM hits mid-conversation (KV cache grows as you talk).")
            max_ratio = 0.5
        elif fits["int8"]:
            verdict = (f"fp16 is too big (~{needs['fp16']:.1f} GB) but INT8 fits "
                       f"(~{needs['int8']:.1f} GB ≤ {budget_gb:.1f} GB). Quantize to INT8 "
                       "— ~2× smaller, usually <1–2% quality loss, no retraining.")
            max_ratio = 0.5
        elif fits["int4"]:
            verdict = (f"Only 4-bit fits (~{needs['int4']:.1f} GB ≤ {budget_gb:.1f} GB; "
                       f"INT8 needs ~{needs['int8']:.1f} GB). Quantize to INT4 (GPTQ/AWQ) "
                       "— ~4× smaller.")
            notes.append("INT4 can degrade reasoning/code tasks — verify on your task.")
            max_ratio = 0.25
        else:
            shortfall = needs["int4"]
            verdict = (f"This model is too big for {budget_label} even at 4-bit "
                       f"(~{shortfall:.1f} GB needed). Realistic options: distill to a smaller "
                       "student, offload layers to CPU/disk (slower), or pick a smaller base model.")
            notes.append("Distillation gives a genuinely smaller architecture (10–100× fewer "
                         "params) — the only way to truly fit a much-too-big model.")
            notes.append("Or run it on a free GPU with sequential onloading: autofollowdown gpu.")
            can_retrain = True
            max_ratio = max(0.1, budget_gb / max(needs["fp16"], 1e-6))
    elif goal == "speed":
        verdict = ("To go faster: structured pruning for a real op-count cut, then INT8 (or "
                   "INT4 on a GPU with optimized kernels). Unstructured pruning shrinks size "
                   "but won't speed up a normal CPU.")
        notes.append("Measure latency before/after — on CPU, INT8 usually gives the biggest win.")
        max_ratio = None
    else:
        verdict = ("Shrink it: quantize first (cheapest), and add pruning/distillation if you "
                   "need more. Set a size budget and check after each step.")
        max_ratio = (target_size_mb and 0.25) or None

    # Honest hardware: a GPU preset or a real CUDA device → gpu; otherwise cpu.
    # (Budget size alone does NOT imply a GPU — a 12 GB laptop is still CPU.)
    if device and device.startswith("gpu"):
        hw = "gpu"
    elif profile.cuda_available:
        hw = "gpu"
    else:
        hw = "cpu"

    # Build the concrete technique plan + the commands to run next.
    plan = advise(profile, goal=goal, max_size_ratio=max_ratio,
                  can_retrain=can_retrain, hardware=hw)

    # Throughput reality check: a big model on CPU may "fit" yet be too slow to use.
    if hw == "cpu" and (profile.num_params or 0) > _CPU_SLOW_PARAMS:
        notes.append("Even if it fits, a model this big on CPU runs at ~a couple tok/s — "
                     "impractical for interactive use; prefer a smaller/distilled model.")
    # Surface advise's runnable caveats (e.g. "INT4 needs a GPU backend") so diagnose
    # never recommends something the local machine can't actually produce.
    notes += [c for c in plan.caveats if ("INT4" in c or "needs a GPU" in c)
              and c not in notes]

    if budget_gb is not None:
        commands.append(f"autofollowdown gpu {model_ref}".strip())
    commands.append(f"autofollowdown advise {model_ref} --goal {goal}".strip()
                    + (" --can-retrain" if can_retrain else ""))
    commands.append(f"autofollowdown compress {model_ref} -o small.pt".strip())

    return Diagnosis(
        problem=label, model=model_ref, num_params=profile.num_params or 0,
        budget_gb=budget_gb, budget_label=budget_label,
        needs=needs, fits=fits, verdict=verdict, plan=plan,
        commands=commands, notes=notes,
    )
