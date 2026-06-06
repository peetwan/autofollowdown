"""Real compression tests — these assert *actual effects* (sparsity rises, size
drops, outputs stay valid), not that a boolean flag was set. Compare with the
removed mock-era tests that only checked `flag == True` and `file exists`.
"""

import io
import os
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from autofollowdown import ModelCompressor, count_parameters


class TinyNet(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.conv = nn.Conv2d(1, width, 3, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(width * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = torch.flatten(x, 1)
        return self.fc2(self.relu(self.fc1(x)))


def _size_mb(model):
    buf = io.BytesIO()
    torch.save(model, buf)
    return buf.getbuffer().nbytes / (1024 * 1024)


def _loader(n=20):
    x = torch.randn(n, 1, 8, 8)
    y = torch.randint(0, 10, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=4)


# --------------------------------------------------------------------- pruning
def test_unstructured_pruning_creates_real_sparsity():
    model = TinyNet()
    _, _, before = count_parameters(model)
    ModelCompressor(model).prune(sparsity=0.5, method="unstructured")
    _, _, after = count_parameters(model)
    assert before < 0.01
    assert after >= 0.45  # global 50% prune, allow rounding across layers


def test_structured_pruning_zeros_channels():
    model = TinyNet()
    ModelCompressor(model).prune(sparsity=0.25, method="structured")
    _, _, sparsity = count_parameters(model)
    assert sparsity > 0.1


def test_pruned_model_still_runs():
    model = TinyNet()
    ModelCompressor(model).prune(sparsity=0.5)
    out = model(torch.randn(2, 1, 8, 8))
    assert out.shape == (2, 10)


def test_double_prune_rejected():
    c = ModelCompressor(TinyNet()).prune(0.3)
    with pytest.raises(ValueError):
        c.prune(0.3)


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_invalid_sparsity_rejected(bad):
    with pytest.raises(ValueError):
        ModelCompressor(TinyNet()).prune(sparsity=bad)


# ---------------------------------------------------------------- quantization
def test_dynamic_quant_shrinks_model_and_runs():
    model = TinyNet()
    before = _size_mb(model)
    c = ModelCompressor(model).quantize(approach="dynamic")
    after = _size_mb(c.model)
    assert after < before  # INT8 weights are smaller than FP32
    out = c.model(torch.randn(2, 1, 8, 8))
    assert out.shape == (2, 10)


def test_static_fx_quant_runs():
    model = TinyNet()
    calib = [torch.randn(2, 1, 8, 8) for _ in range(4)]
    c = ModelCompressor(model).quantize(approach="static", calibration_data=calib)
    out = c.model(torch.randn(2, 1, 8, 8))
    assert out.shape == (2, 10)


def test_static_quant_requires_calibration():
    with pytest.raises(ValueError):
        ModelCompressor(TinyNet()).quantize(approach="static")


def test_unsupported_quant_method_rejected():
    with pytest.raises(ValueError):
        ModelCompressor(TinyNet()).quantize(method="int3")


def test_double_quant_rejected():
    c = ModelCompressor(TinyNet()).quantize(approach="dynamic")
    with pytest.raises(ValueError):
        c.quantize(approach="dynamic")


# ---------------------------------------------------------------- distillation
def test_distillation_updates_student_weights():
    student = TinyNet()
    teacher = TinyNet()
    before = [p.clone() for p in student.parameters()]
    ModelCompressor(student).distill(teacher, _loader(), epochs=2)
    after = list(student.parameters())
    changed = any(not torch.equal(a, b) for a, b in zip(after, before))
    assert changed, "distillation must actually train the student"


def test_distillation_rejects_none_teacher():
    with pytest.raises(ValueError):
        ModelCompressor(TinyNet()).distill(None, _loader(), epochs=1)


def test_distillation_rejects_nonpositive_epochs():
    with pytest.raises(ValueError):
        ModelCompressor(TinyNet()).distill(TinyNet(), _loader(), epochs=0)


# ---------------------------------------------------------------------- export
def test_export_pt_roundtrips():
    c = ModelCompressor(TinyNet()).prune(0.3)
    path = tempfile.mktemp(suffix=".pt")
    try:
        c.export(path, "pt")
        reloaded = torch.load(path, weights_only=False)
        out = reloaded(torch.randn(1, 1, 8, 8))
        assert out.shape == (1, 10)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_export_onnx_is_valid_and_runs():
    onnxruntime = pytest.importorskip("onnxruntime")
    c = ModelCompressor(TinyNet(), input_shape=(1, 1, 8, 8)).prune(0.3)
    path = tempfile.mktemp(suffix=".onnx")
    try:
        c.export(path, "onnx")
        sess = onnxruntime.InferenceSession(path)
        out = sess.run(None, {"input": torch.randn(1, 1, 8, 8).numpy()})
        assert out[0].shape == (1, 10)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_export_before_compression_rejected():
    with pytest.raises(ValueError):
        ModelCompressor(TinyNet()).export(tempfile.mktemp(suffix=".pt"), "pt")


def test_none_model_rejected():
    with pytest.raises(ValueError):
        ModelCompressor(None)
