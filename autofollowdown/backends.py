"""Compression backends and a registry for the auto-picker.

Each backend wraps one library. The native backend (this toolkit) always works.
The others — NNI, llm-compressor, NVIDIA ModelOpt — are optional: they declare
what they're good at, whether they're installed, and whether the hardware suits
them, and they delegate to the real library API only when actually available.

This is what lets the router "auto pick the best library based on your model":
it scores every backend against the model profile and runs the best one that is
genuinely runnable here, falling back to the native engine otherwise.
"""

import importlib.util


def _installed(module_name):
    return importlib.util.find_spec(module_name) is not None


class Backend:
    """Base backend. Subclasses set metadata and implement plan()/compress()."""

    name = "backend"
    library = None          # pip importable module name, or None for built-in
    install_hint = ""
    needs_cuda = False

    def is_available(self):
        if self.library is None:
            return True
        return _installed(self.library)

    def device_ok(self, profile):
        return (not self.needs_cuda) or profile.cuda_available

    def score(self, profile):
        """Fitness in [0, 1] for this model family. 0 means 'not applicable'."""
        return 0.0

    def plan(self, profile):
        """Return (technique, scheme, rationale) — the recommended approach."""
        raise NotImplementedError

    def compress(self, model, profile, **kwargs):
        raise NotImplementedError


class NativeBackend(Backend):
    name = "autofollowdown (native)"
    library = None
    install_hint = "(built in)"

    def score(self, profile):
        # Universal fallback: always applicable, modest score so a specialized
        # available backend wins, but it still beats an unavailable one.
        return {"cnn": 0.6, "mlp": 0.6, "transformer": 0.5,
                "llm": 0.4, "unknown": 0.5}.get(profile.family, 0.5)

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
    library = "nni"
    install_hint = "pip install nni"

    def score(self, profile):
        if profile.family == "cnn":
            return 0.9   # filter pruning + ModelSpeedup physically shrinks CNNs
        if profile.family == "transformer":
            return 0.5   # has TransformerHeadPruner
        return 0.2

    def plan(self, profile):
        return ("structured-prune", "L1FilterPruner + ModelSpeedup",
                "Channel/filter pruning that ModelSpeedup turns into a genuinely "
                "smaller, faster model (real FLOP reduction).")

    def compress(self, model, profile, config_list=None, dummy_input=None, **kwargs):
        if not self.is_available():
            raise RuntimeError("NNI is not installed. " + self.install_hint)
        # Real delegation to NNI's filter pruning + speedup (API per NNI docs).
        try:
            from nni.algorithms.compression.pytorch.pruning import L1FilterPruner
            from nni.compression.pytorch import ModelSpeedup
        except ImportError as e:  # NNI's module paths shifted across versions
            raise RuntimeError(
                "Installed NNI exposes a different compression API than expected; "
                "see https://nni.readthedocs.io for your version."
            ) from e
        config_list = config_list or [{"sparsity": 0.5, "op_types": ["Conv2d"]}]
        pruner = L1FilterPruner(model, config_list)
        pruner.compress()
        pruner._unwrap_model()
        if dummy_input is not None:
            ModelSpeedup(model, dummy_input, masks_file=pruner.mask_dict).speedup_model()
        return model


class LLMCompressorBackend(Backend):
    name = "llm-compressor (vLLM)"
    library = "llmcompressor"
    install_hint = "pip install llmcompressor"

    def score(self, profile):
        if profile.family == "llm" and profile.is_huggingface:
            return 0.95  # GPTQ/AWQ weight-only 4-bit is the LLM sweet spot
        if profile.family == "llm":
            return 0.6
        return 0.0

    def plan(self, profile):
        return ("weight-only-quant", "GPTQ W4A16",
                "4-bit weight-only PTQ (GPTQ/AWQ) for HF LLMs, deployable in vLLM. "
                "GPU strongly recommended.")

    def compress(self, model, profile, dataset=None, recipe=None,
                 num_calibration_samples=512, output_dir=None, **kwargs):
        if not self.is_available():
            raise RuntimeError("llm-compressor is not installed. " + self.install_hint)
        # Real delegation to llmcompressor.oneshot (API per vLLM llm-compressor docs).
        from llmcompressor import oneshot
        from llmcompressor.modifiers.gptq import GPTQModifier
        recipe = recipe or GPTQModifier(targets="Linear", scheme="W4A16",
                                        ignore=["lm_head"])
        oneshot(model=model, dataset=dataset, recipe=recipe,
                num_calibration_samples=num_calibration_samples, output_dir=output_dir)
        return model


class ModelOptBackend(Backend):
    name = "NVIDIA TensorRT Model Optimizer"
    library = "modelopt"
    install_hint = "pip install nvidia-modelopt"
    needs_cuda = True

    def score(self, profile):
        if profile.family in ("llm", "transformer"):
            return 0.9   # SmoothQuant/AWQ/NVFP4 PTQ → TensorRT, best on NVIDIA GPUs
        if profile.family == "cnn":
            return 0.6
        return 0.3

    def plan(self, profile):
        scheme = "INT8 SmoothQuant" if profile.family != "cnn" else "INT8 PTQ"
        return ("ptq", scheme,
                "Calibrated PTQ (SmoothQuant/AWQ/NVFP4) via mtq.quantize, exportable "
                "to TensorRT. Requires an NVIDIA GPU.")

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


# Registry — order is only a tiebreak; the router scores them.
_REGISTRY = [
    NativeBackend(),
    NNIBackend(),
    LLMCompressorBackend(),
    ModelOptBackend(),
]


def all_backends():
    return list(_REGISTRY)


def get_backend(name):
    for b in _REGISTRY:
        if b.name == name:
            return b
    raise KeyError(f"No backend named {name!r}")
