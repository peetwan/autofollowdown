"""Compression backends and a capability-driven registry for the auto-picker.

Each backend wraps one library and **declares its capabilities as data** — which
model families it suits, what it's good at (traits), whether it needs a GPU or a
calibration set — instead of burying routing logic in hand-written if/else rules.
A single generic scorer (`Backend.score`) turns those declarations + the model
profile + the user's goal into a fitness number, so the router stays honest and
**adding a backend is data, not new branching code**.

The native backend (this toolkit) always works. The others — NNI, llm-compressor,
NVIDIA ModelOpt, torchao, bitsandbytes, HQQ — are optional: they're detected at
runtime and only ever delegate to the real library API when genuinely available.
"""

import importlib.util
from dataclasses import dataclass


def _installed(module_name):
    return importlib.util.find_spec(module_name) is not None


# What each user goal cares about, expressed as the backend traits that satisfy it.
# This is the only place "goal → preference" lives; the scorer just matches sets,
# so a new goal or trait is a one-line data change, not a routing rewrite.
GOAL_TRAITS = {
    "size": frozenset({"smallest", "4bit", "weight-only"}),
    "speed": frozenset({"fast", "kernel-optimized", "compile"}),
    "accuracy": frozenset({"accurate", "calibrated"}),
    "ease": frozenset({"easy", "no-calibration", "portable"}),
    "balanced": frozenset(),
}

# Traits a goal actively steers *away* from (e.g. "ease" should avoid methods that
# need a calibration dataset). Also pure data — keeps the scorer free of if/else.
GOAL_AVOID = {
    "ease": frozenset({"calibrated"}),
}


@dataclass(frozen=True)
class Capability:
    """A backend's declared fitness — pure data the scorer reads, no logic."""

    technique: str                       # e.g. "weight-only-quant"
    scheme: str                          # human label, e.g. "GPTQ W4A16"
    rationale: str                       # why it ranks where it does
    families: dict                       # model family -> base fit in [0, 1]
    traits: frozenset = frozenset()      # tags matched against GOAL_TRAITS
    needs_cuda: bool = False
    needs_calibration: bool = False
    hf_bonus: float = 0.0                # added when the model is a HF model


class Backend:
    """Base backend. Subclasses set `capability` (data) and implement `compress`.

    Scoring and planning are derived generically from `capability`, so the router
    never special-cases a backend by name.
    """

    name = "backend"
    alias = "backend"          # short, typeable handle (e.g. "nni", "modelopt")
    library = None             # pip importable module name, or None for built-in
    install_hint = ""
    capability = None

    @property
    def needs_cuda(self):
        return bool(self.capability and self.capability.needs_cuda)

    def is_available(self):
        if self.library is None:
            return True
        return _installed(self.library)

    def device_ok(self, profile):
        return (not self.needs_cuda) or profile.cuda_available

    def score(self, profile, goal="balanced"):
        """Fitness in [0, 1], derived from declared capabilities + the goal.

        base family fit  (+ HF bonus if applicable)  + small per-goal trait nudge.
        Returns 0 when the backend simply doesn't apply to this family.
        """
        cap = self.capability
        if cap is None:
            return 0.0
        base = cap.families.get(profile.family, 0.0)
        if base <= 0:
            return 0.0
        score = base
        if profile.is_huggingface:
            score += cap.hf_bonus
        # Goal alignment: reward declared traits the goal wants, penalize ones it
        # steers away from. Both sides are read from the GOAL_* data maps, so the
        # routing has no per-backend special-casing.
        wanted = GOAL_TRAITS.get(goal, frozenset())
        score += 0.06 * len(cap.traits & wanted)
        avoid = GOAL_AVOID.get(goal, frozenset())
        score -= 0.10 * len(cap.traits & avoid)
        return max(0.0, min(1.0, score))

    def plan(self, profile):
        """Return (technique, scheme, rationale). Backends whose scheme depends on
        the family override this; the rest read it straight from `capability`."""
        cap = self.capability
        return (cap.technique, cap.scheme, cap.rationale)

    def compress(self, model, profile, **kwargs):
        raise NotImplementedError


class NativeBackend(Backend):
    name = "autofollowdown (native)"
    alias = "native"
    library = None
    install_hint = "(built in)"
    # Universal fallback: applies everywhere with a modest fit, so a specialized
    # *available* backend wins, but it still beats an unavailable one.
    capability = Capability(
        technique="prune+quantize",
        scheme="int8-dynamic",
        rationale="Portable INT8 (and pruning/distillation) with no extra deps — "
                  "the always-runnable fallback.",
        families={"cnn": 0.6, "mlp": 0.6, "transformer": 0.5, "llm": 0.4, "unknown": 0.5},
        traits=frozenset({"portable", "no-calibration", "easy"}),
    )

    def plan(self, profile):
        if profile.family == "cnn":
            return ("prune+quantize", "unstructured-0.5 + int8-dynamic",
                    "Global magnitude pruning then portable INT8 — no GPU needed.")
        if profile.family in ("llm", "transformer"):
            return ("quantize", "int8-dynamic",
                    "Portable weight-only INT8 on Linear layers; for 4-bit LLM "
                    "quality prefer llm-compressor or ModelOpt.")
        return ("quantize", "int8-dynamic", "Portable INT8 dynamic quantization.")

    def compress(self, model, profile, calibration_data=None, **kwargs):
        from .api import ModelCompressor
        c = ModelCompressor(model)
        if profile.family == "cnn":
            c.prune(sparsity=0.5, method="unstructured")
        c.quantize(method="int8", approach="dynamic")
        return c.model


class NNIBackend(Backend):
    name = "Microsoft NNI"
    alias = "nni"
    library = "nni"
    install_hint = "pip install nni"
    capability = Capability(
        technique="structured-prune",
        scheme="L1FilterPruner + ModelSpeedup",
        rationale="Channel/filter pruning that ModelSpeedup turns into a genuinely "
                  "smaller, faster model (real FLOP reduction).",
        families={"cnn": 0.9, "transformer": 0.5, "llm": 0.2},
        traits=frozenset({"fast", "structured", "smallest"}),
    )

    def compress(self, model, profile, sparsity=0.5, dummy_input=None, **kwargs):
        if not self.is_available():
            raise RuntimeError("NNI is not installed. " + self.install_hint)
        # Real delegation to NNI filter pruning + ModelSpeedup. NNI reorganized its
        # compression API between v2 and v3, so try the modern path first, then v2.
        try:  # NNI >= 3.x
            from nni.compression.pruning import L1NormPruner
            from nni.compression.speedup import ModelSpeedup
            config_list = [{"op_types": ["Conv2d"], "sparse_ratio": sparsity}]
            pruner = L1NormPruner(model, config_list)
            _, masks = pruner.compress()
            pruner.unwrap_model()
            if dummy_input is not None:
                ModelSpeedup(model, dummy_input, masks).speedup_model()
            return model
        except ImportError:
            pass
        try:  # NNI 2.x
            from nni.algorithms.compression.pytorch.pruning import L1FilterPruner
            from nni.compression.pytorch import ModelSpeedup
            pruner = L1FilterPruner(model, [{"sparsity": sparsity, "op_types": ["Conv2d"]}])
            pruner.compress()
            pruner._unwrap_model()
            if dummy_input is not None:
                ModelSpeedup(model, dummy_input, masks_file=pruner.mask_dict).speedup_model()
            return model
        except ImportError as e:
            raise RuntimeError(
                "Installed NNI exposes a different compression API than expected; "
                "see https://nni.readthedocs.io for your version."
            ) from e


class LLMCompressorBackend(Backend):
    name = "llm-compressor (vLLM)"
    alias = "llmcompressor"
    library = "llmcompressor"
    install_hint = "pip install llmcompressor"
    capability = Capability(
        technique="weight-only-quant",
        scheme="GPTQ W4A16",
        rationale="4-bit weight-only PTQ (GPTQ/AWQ) for HF LLMs, deployable in vLLM. "
                  "Sequential onloading runs it on a small/free GPU.",
        families={"llm": 0.6, "transformer": 0.3},
        traits=frozenset({"smallest", "4bit", "weight-only", "accurate", "calibrated",
                          "kernel-optimized"}),
        needs_calibration=True,
        hf_bonus=0.35,     # GPTQ/AWQ on a HF LLM is the sweet spot (-> 0.95)
    )

    def compress(self, model, profile, dataset=None, recipe=None,
                 num_calibration_samples=512, output_dir=None,
                 pipeline=None, sequential_targets=None, **kwargs):
        if not self.is_available():
            raise RuntimeError("llm-compressor is not installed. " + self.install_hint)
        # Real delegation to llmcompressor.oneshot (API per vLLM llm-compressor docs).
        from llmcompressor import oneshot
        from llmcompressor.modifiers.gptq import GPTQModifier

        from .gpu import memory_plan

        recipe = recipe or GPTQModifier(targets="Linear", scheme="W4A16",
                                        ignore=["lm_head"])
        # Memory-saving defaults so this runs on a free/small GPU instead of OOM:
        # sequential onloading keeps one slice of the model on the GPU at a time.
        # Caller-supplied values always win over the auto plan.
        plan = memory_plan(profile.num_params or 0)
        one = dict(model=model, dataset=dataset, recipe=recipe,
                   num_calibration_samples=num_calibration_samples,
                   output_dir=output_dir)
        chosen_pipeline = pipeline or plan["pipeline"]
        if chosen_pipeline:
            one["pipeline"] = chosen_pipeline
        chosen_targets = sequential_targets or plan["sequential_targets"]
        if chosen_targets:
            one["sequential_targets"] = chosen_targets
        one.update(kwargs)
        oneshot(**one)
        return model


class ModelOptBackend(Backend):
    name = "NVIDIA TensorRT Model Optimizer"
    alias = "modelopt"
    library = "modelopt"
    install_hint = "pip install nvidia-modelopt"
    capability = Capability(
        technique="ptq",
        scheme="INT8 SmoothQuant",
        rationale="Calibrated PTQ (SmoothQuant/AWQ/NVFP4) via mtq.quantize, exportable "
                  "to TensorRT. Requires an NVIDIA GPU.",
        families={"llm": 0.9, "transformer": 0.9, "cnn": 0.6, "unknown": 0.3},
        traits=frozenset({"fast", "kernel-optimized", "accurate", "calibrated", "4bit"}),
        needs_cuda=True,
        needs_calibration=True,
    )

    def plan(self, profile):
        scheme = "INT8 SmoothQuant" if profile.family != "cnn" else "INT8 PTQ"
        return ("ptq", scheme, self.capability.rationale)

    def compress(self, model, profile, calibration_data=None, quant_cfg=None, **kwargs):
        if not self.is_available():
            raise RuntimeError("nvidia-modelopt is not installed. " + self.install_hint)
        if not profile.cuda_available:
            raise RuntimeError("NVIDIA ModelOpt requires a CUDA GPU.")
        # Real delegation to modelopt.torch.quantization (API per NVIDIA docs).
        import modelopt.torch.quantization as mtq
        quant_cfg = quant_cfg or mtq.INT8_SMOOTHQUANT_CFG

        def forward_loop(m):
            for batch in (calibration_data or []):
                m(batch)

        return mtq.quantize(model, quant_cfg, forward_loop)


class TorchAOBackend(Backend):
    name = "torchao (PyTorch native)"
    alias = "torchao"
    library = "torchao"
    install_hint = "pip install torchao"
    capability = Capability(
        technique="weight-only-quant",
        scheme="int8/int4 weight-only (+ torch.compile)",
        rationale="PyTorch-native int8/int4/fp8 with no calibration; pairs with "
                  "torch.compile for fast inference and has real CPU support.",
        families={"llm": 0.7, "transformer": 0.65, "cnn": 0.5, "mlp": 0.5, "unknown": 0.4},
        traits=frozenset({"fast", "compile", "no-calibration", "weight-only", "portable",
                          "easy"}),
    )

    def compress(self, model, profile, scheme="int8-weight-only", **kwargs):
        if not self.is_available():
            raise RuntimeError("torchao is not installed. " + self.install_hint)
        # Real delegation to torchao.quantization.quantize_ (PyTorch-native API).
        from torchao.quantization import quantize_
        want_int4 = "int4" in scheme
        try:  # newer config-object API
            from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig
            cfg = Int4WeightOnlyConfig() if want_int4 else Int8WeightOnlyConfig()
        except ImportError:  # older factory-function API
            from torchao.quantization import int4_weight_only, int8_weight_only
            cfg = int4_weight_only() if want_int4 else int8_weight_only()
        quantize_(model, cfg)
        return model


class BitsAndBytesBackend(Backend):
    name = "bitsandbytes"
    alias = "bnb"
    library = "bitsandbytes"
    install_hint = "pip install bitsandbytes"
    capability = Capability(
        technique="weight-only-quant",
        scheme="NF4 / INT8 (load-time)",
        rationale="The easiest 4-bit/8-bit path — no calibration, loads quantized "
                  "directly through transformers. Great for a quick free-GPU win.",
        families={"llm": 0.65, "transformer": 0.5},
        traits=frozenset({"easy", "no-calibration", "4bit", "smallest"}),
        needs_cuda=True,
        hf_bonus=0.05,
    )

    def compress(self, model, profile, model_id=None, bits=4, device_map="auto", **kwargs):
        if not self.is_available():
            raise RuntimeError("bitsandbytes is not installed. " + self.install_hint)
        # bitsandbytes quantizes at *load* time (it swaps in bnb Linear layers as the
        # model is built), so we reload the model id with a BitsAndBytesConfig.
        if model_id is None:
            model_id = (profile.detail or {}).get("model_id")
        if model_id is None:
            raise RuntimeError(
                "bitsandbytes quantizes at load time — pass model_id='<hf id>' so it "
                "can reload the model in 4/8-bit.")
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        if bits == 4:
            cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_use_double_quant=True)
        else:
            cfg = BitsAndBytesConfig(load_in_8bit=True)
        return AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=cfg, device_map=device_map)


class HQQBackend(Backend):
    name = "HQQ (Half-Quadratic Quantization)"
    alias = "hqq"
    library = "hqq"
    install_hint = "pip install hqq"
    capability = Capability(
        technique="weight-only-quant",
        scheme="HQQ 4-bit (no calibration)",
        rationale="Fast, calibration-free quantization down to 4/3/2-bit; runs on a "
                  "free GPU and pairs with torch.compile for speed.",
        families={"llm": 0.68, "transformer": 0.55},
        traits=frozenset({"fast", "no-calibration", "4bit", "compile", "easy"}),
        hf_bonus=0.05,
    )

    def compress(self, model, profile, nbits=4, group_size=64, **kwargs):
        if not self.is_available():
            raise RuntimeError("HQQ is not installed. " + self.install_hint)
        # Real delegation to HQQ's in-place model quantizer (API per HQQ docs).
        import torch
        from hqq.core.quantize import BaseQuantizeConfig
        try:
            from hqq.models.hf.base import AutoHQQHFModel
        except ImportError:  # older module path
            from hqq.engine.hf import AutoHQQHFModel
        cfg = BaseQuantizeConfig(nbits=nbits, group_size=group_size)
        dev = "cuda" if profile.cuda_available else "cpu"
        dtype = torch.float16 if profile.cuda_available else torch.float32
        AutoHQQHFModel.quantize_model(model, quant_config=cfg, compute_dtype=dtype, device=dev)
        return model


# Registry — order is only a tiebreak; the router scores them.
_REGISTRY = [
    NativeBackend(),
    NNIBackend(),
    LLMCompressorBackend(),
    ModelOptBackend(),
    TorchAOBackend(),
    BitsAndBytesBackend(),
    HQQBackend(),
]


def all_backends():
    return list(_REGISTRY)


def get_backend(name):
    """Look up a backend by exact name, short alias, or case-insensitive substring."""
    q = name.lower()
    for b in _REGISTRY:
        if name == b.name or q == b.alias or q == b.name.lower():
            return b
    for b in _REGISTRY:  # fall back to substring match (e.g. "nvidia", "vllm")
        if q in b.name.lower() or q in b.alias:
            return b
    raise KeyError(f"No backend matching {name!r}. "
                   f"Options: {[b.alias for b in _REGISTRY]}")
