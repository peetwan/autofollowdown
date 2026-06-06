"""ModelCompressor — the unified compression API.

This is a *real* implementation: pruning zeros and removes weights, quantization
produces genuine INT8 modules, distillation runs a real training loop, and export
serializes a real model. It wraps a live PyTorch ``nn.Module`` (loaded directly,
or fetched from a Hugging Face id, or an on-disk ``.onnx`` graph) and applies
compression in place, returning ``self`` so steps can be chained.

    ModelCompressor(model).prune(0.3).quantize(approach="dynamic").export("m.pt")
"""

import copy
import os

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from .ingestion import load_model
from .onnx_pipeline import export_to_onnx, optimize_onnx, prune_onnx

# Module types whose `weight` tensors are worth pruning / quantizing.
_PRUNABLE = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)
_QUANTIZABLE_DYNAMIC = {nn.Linear, nn.LSTM, nn.GRU, nn.RNN}


def _select_qengine():
    """Pick and activate a quantized backend for this CPU.

    PyTorch ships fbgemm (x86) and qnnpack (ARM, e.g. Apple Silicon). If no
    engine is active the quantized ops raise NoQEngine, so we set it explicitly.
    Returns the backend name to use for FX qconfig mappings.
    """
    supported = [e for e in torch.backends.quantized.supported_engines if e != "none"]
    if not supported:
        raise RuntimeError("This PyTorch build has no quantized engine (fbgemm/qnnpack).")
    backend = "fbgemm" if "fbgemm" in supported else supported[0]
    torch.backends.quantized.engine = backend
    return backend


class ModelCompressor:
    def __init__(self, model, input_shape=None):
        if model is None:
            raise ValueError("Model cannot be None")

        if input_shape is not None:
            if not isinstance(input_shape, (tuple, list)):
                raise ValueError("input_shape must be a tuple or list")
            if any(not isinstance(x, int) or x <= 0 for x in input_shape):
                raise ValueError("input_shape elements must be positive integers")

        if isinstance(model, nn.Module):
            self.kind = "pytorch"
            self.model = model
        elif isinstance(model, str):
            loaded = load_model(model, input_shape=input_shape)
            self.kind = loaded["type"]          # 'onnx' or 'huggingface'
            self.model = loaded["path"] if self.kind == "onnx" else loaded["model"]
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")

        self.input_shape = input_shape
        self.history = []
        self._is_quantized = False
        self._is_pruned = False
        self._is_distilled = False
        self._is_compressed = False

    # ------------------------------------------------------------------ pruning
    def prune(self, sparsity=0.3, method="unstructured"):
        """Remove `sparsity` fraction of weights. Pruning is made *permanent*
        (the reparametrization mask is folded into the weight), so the zeros are
        real and measurable, not a runtime trick."""
        if self._is_pruned:
            raise ValueError("Model is already pruned")
        if not (0.0 <= sparsity <= 1.0):
            raise ValueError(f"Sparsity must be between 0.0 and 1.0, got {sparsity}")
        if method not in ("unstructured", "structured"):
            raise ValueError(f"Unsupported pruning method: {method}")

        if self.kind == "onnx":
            self.model = self._prune_onnx(sparsity)
        else:
            self._prune_pytorch(sparsity, method)

        self._is_pruned = True
        self._is_compressed = True
        self.history.append(("prune", {"sparsity": sparsity, "method": method}))
        return self

    def _prune_pytorch(self, sparsity, method):
        modules = [m for m in self.model.modules()
                   if isinstance(m, _PRUNABLE) and hasattr(m, "weight")]
        if not modules or sum(p.numel() for p in self.model.parameters()) == 0:
            raise ValueError("Cannot prune a model with zero prunable parameters")

        if method == "unstructured":
            # Global magnitude pruning: rank all weights together, drop the
            # smallest `sparsity` fraction across the whole network.
            params = [(m, "weight") for m in modules]
            prune.global_unstructured(
                params, pruning_method=prune.L1Unstructured, amount=sparsity
            )
            for m, _ in params:
                prune.remove(m, "weight")
        else:  # structured: drop whole output channels/neurons by L2 norm
            for m in modules:
                dim0 = m.weight.shape[0]
                if dim0 < 2:  # nothing meaningful to drop
                    continue
                prune.ln_structured(m, "weight", amount=sparsity, n=2, dim=0)
                prune.remove(m, "weight")

    def _prune_onnx(self, sparsity):
        out_path = self.model + ".pruned.onnx"
        return prune_onnx(self.model, out_path, sparsity=sparsity)

    # ------------------------------------------------------------- quantization
    def quantize(self, method="int8", approach="dynamic", calibration_data=None):
        """Quantize the model. `dynamic` INT8 is robust and portable (great for
        Linear/RNN-heavy models); `static` runs FX post-training quantization
        using `calibration_data` to record activation ranges."""
        if self._is_quantized:
            raise ValueError("Model is already quantized")
        if method not in ("int8", "fp16"):
            raise ValueError(f"Unsupported quantization method: {method}")
        if approach not in ("static", "dynamic"):
            raise ValueError(f"Unsupported quantization approach: {approach}")
        if approach == "static" and (calibration_data is None or len(calibration_data) == 0):
            raise ValueError("Calibration data is required for static quantization")

        if self.kind == "onnx":
            self.model = self._quantize_onnx(approach, calibration_data)
        elif method == "fp16":
            self.model = self.model.half()
        elif approach == "dynamic":
            _select_qengine()
            self.model = torch.ao.quantization.quantize_dynamic(
                self.model.eval().cpu(), _QUANTIZABLE_DYNAMIC, dtype=torch.qint8
            )
        else:  # static FX post-training quantization
            self.model = self._quantize_static_fx(calibration_data)

        self._is_quantized = True
        self._is_compressed = True
        self.history.append(("quantize", {"method": method, "approach": approach}))
        return self

    def _quantize_static_fx(self, calibration_data):
        from torch.ao.quantization import get_default_qconfig_mapping
        from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

        model = self.model.eval().cpu()
        example = self._example_from_calibration(calibration_data)
        backend = _select_qengine()
        qconfig = get_default_qconfig_mapping(backend)
        prepared = prepare_fx(model, qconfig, example_inputs=(example,))
        with torch.no_grad():
            for batch in calibration_data:
                prepared(self._as_tensor(batch))
        return convert_fx(prepared)

    def _quantize_onnx(self, approach, calibration_data):
        out_path = self.model + ".quant.onnx"
        return optimize_onnx(self.model, out_path, {
            "quantize": True,
            "approach": approach,
            "calibration_data": calibration_data,
        })

    @staticmethod
    def _as_tensor(batch):
        if isinstance(batch, dict):
            return next(iter(batch.values()))
        if isinstance(batch, (list, tuple)):
            return batch[0]
        return batch

    def _example_from_calibration(self, calibration_data):
        first = calibration_data[0]
        return self._as_tensor(first)

    # ------------------------------------------------------------- distillation
    def distill(self, teacher_model, train_loader, epochs=3, optimizer=None,
                loss_fn=None, temperature=4.0, alpha=0.5, device="cpu"):
        """Knowledge distillation: train `self.model` (student) to match the
        teacher's softened logits (KL term) and the true labels (CE term).
        This is a real optimization loop that updates the student's weights."""
        if epochs <= 0:
            raise ValueError("Epochs must be greater than 0")
        if teacher_model is None:
            raise ValueError("Teacher model cannot be None")
        if train_loader is None or len(train_loader) == 0:
            raise ValueError("Train loader cannot be empty")
        if not isinstance(self.model, nn.Module):
            raise ValueError("Distillation requires a PyTorch student model")
        if isinstance(teacher_model, str):
            teacher_model = load_model(teacher_model)["model"]
        if not isinstance(teacher_model, nn.Module):
            raise ValueError("Teacher model must be a PyTorch module")

        student = self.model.to(device)
        teacher = teacher_model.to(device).eval()
        optimizer = optimizer or torch.optim.Adam(student.parameters(), lr=1e-3)
        hard_loss_fn = loss_fn or nn.CrossEntropyLoss()
        kl = nn.KLDivLoss(reduction="batchmean")

        student.train()
        for _ in range(epochs):
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                s_logits = self._logits(student(inputs))
                with torch.no_grad():
                    t_logits = self._logits(teacher(inputs))

                soft = kl(
                    torch.log_softmax(s_logits / temperature, dim=1),
                    torch.softmax(t_logits / temperature, dim=1),
                ) * (temperature ** 2)
                hard = hard_loss_fn(s_logits, labels)
                loss = alpha * soft + (1 - alpha) * hard
                loss.backward()
                optimizer.step()

        self._is_distilled = True
        self._is_compressed = True
        self.history.append(("distill", {"epochs": epochs, "temperature": temperature,
                                          "alpha": alpha}))
        return self

    @staticmethod
    def _logits(out):
        if hasattr(out, "logits"):
            return out.logits
        if isinstance(out, (list, tuple)):
            return out[0]
        return out

    # -------------------------------------------------------------------- export
    def export(self, output_path, format="onnx"):
        """Serialize the compressed model. `pt` saves a real torch object;
        `onnx` exports a real ONNX graph runnable under onnxruntime."""
        if not self._is_compressed:
            raise ValueError("Model must be compressed before export")
        if format not in ("onnx", "pt"):
            raise ValueError(f"Unsupported export format: {format}")
        if os.path.isdir(output_path):
            raise ValueError("Output path cannot be a directory")
        dir_name = os.path.dirname(output_path)
        if dir_name and not os.path.exists(dir_name):
            raise ValueError(f"Parent directory does not exist: {dir_name}")

        if self.kind == "onnx":
            import shutil
            shutil.copy2(self.model, output_path)
        elif format == "pt":
            torch.save(self.model, output_path)
        else:  # onnx
            if self._is_quantized:
                raise ValueError(
                    "Exporting a torch-quantized model to ONNX is not supported; "
                    "for INT8 ONNX, ingest/export to ONNX first then quantize on the "
                    "ONNX graph (see onnx_pipeline.optimize_onnx)."
                )
            export_to_onnx(self.model, "pytorch", output_path,
                           input_shape=self.input_shape)

        self.history.append(("export", {"output_path": output_path, "format": format}))
        return output_path

    def clone(self):
        """Return a deep copy of the current (compressed or not) model — handy for
        benchmarking a baseline against a derived, compressed variant."""
        if isinstance(self.model, nn.Module):
            return copy.deepcopy(self.model)
        return self.model
