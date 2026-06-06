import os
import tempfile
import numpy as np
import pytest
import torch
import onnx
from onnx import numpy_helper
import onnxruntime as ort
from transformers import AutoConfig, AutoModel, DistilBertConfig, DistilBertModel

from autofollowdown.ingestion import load_model


@pytest.fixture(scope="module")
def tiny_hf_dir(tmp_path_factory):
    """Build a tiny real DistilBert model on disk (no download, no 253MB artifact).

    Self-contained replacement for the old tests that depended on a checked-in
    `temp_hf/` directory holding a full 253MB DistilBERT.
    """
    cfg = DistilBertConfig(
        vocab_size=100, dim=32, hidden_dim=64, n_layers=1, n_heads=2,
        max_position_embeddings=32,
    )
    model = DistilBertModel(cfg)
    out = tmp_path_factory.mktemp("tiny_hf")
    model.save_pretrained(str(out))
    return str(out)
from autofollowdown.onnx_pipeline import (
    export_to_onnx,
    prune_onnx,
    optimize_onnx,
    ONNXCalibrationDataReader,
    get_working_dummy_input
)

class SimpleConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.fc = torch.nn.Linear(8 * 4 * 4, 2)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

def test_load_model_pytorch():
    model = SimpleConvModel()
    res = load_model(model)
    assert res["type"] == "pytorch"
    assert res["model"] == model
    assert res["path"] is None

def test_load_model_huggingface(tiny_hf_dir):
    res = load_model(tiny_hf_dir)
    assert res["type"] == "huggingface"
    assert res["model"] is not None
    assert res["path"] == tiny_hf_dir
    # Ensure it resolved the correct class (DistilBertModel)
    assert res["model"].__class__.__name__ == "DistilBertModel"

def test_load_model_onnx():
    # Create a small dummy ONNX model
    model = SimpleConvModel()
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name
    try:
        x = torch.randn(1, 3, 4, 4)
        torch.onnx.export(
            model, x, onnx_path,
            input_names=["input"],
            output_names=["output"]
        )
        res = load_model(onnx_path)
        assert res["type"] == "onnx"
        assert res["model"] is not None
        assert res["path"] == onnx_path
    finally:
        if os.path.exists(onnx_path):
            os.remove(onnx_path)

def test_export_to_onnx_pytorch():
    model = SimpleConvModel()
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name
    try:
        # Pass input shape (1, 3, 4, 4)
        export_to_onnx(model, "onnx", onnx_path, input_shape=(1, 3, 4, 4))
        assert os.path.exists(onnx_path)
        assert os.path.getsize(onnx_path) > 0
        
        # Load and verify it's a valid ONNX model
        loaded = onnx.load(onnx_path)
        onnx.checker.check_model(loaded)
    finally:
        if os.path.exists(onnx_path):
            os.remove(onnx_path)

def test_export_to_onnx_huggingface(tiny_hf_dir):
    res = load_model(tiny_hf_dir)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name
    try:
        # DistilBertModel expects sequence inputs, let's specify input shape
        export_to_onnx(res["model"], "onnx", onnx_path, input_shape=(1, 8))
        assert os.path.exists(onnx_path)
        assert os.path.getsize(onnx_path) > 0
        
        loaded = onnx.load(onnx_path)
        onnx.checker.check_model(loaded)
    finally:
        if os.path.exists(onnx_path):
            os.remove(onnx_path)

def test_prune_onnx_sparsity():
    model = SimpleConvModel()
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        pruned_path = f.name
        
    try:
        x = torch.randn(1, 3, 4, 4)
        torch.onnx.export(model, x, onnx_path)
        
        # Prune with 40% sparsity
        prune_onnx(onnx_path, pruned_path, sparsity=0.4)
        
        # Verify pruned file exists
        assert os.path.exists(pruned_path)
        
        # Verify weight matrices are indeed sparse
        pruned_model = onnx.load(pruned_path)
        for initializer in pruned_model.graph.initializer:
            if initializer.data_type == onnx.TensorProto.FLOAT:
                arr = numpy_helper.to_array(initializer)
                if len(arr.shape) >= 2:
                    zero_count = np.sum(arr == 0.0)
                    total_count = arr.size
                    actual_sparsity = zero_count / total_count if total_count > 0 else 0
                    # For quantile, sparsity should be exactly or very close to 0.4
                    assert actual_sparsity >= 0.38
    finally:
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
        if os.path.exists(pruned_path):
            os.remove(pruned_path)

def test_optimize_onnx_dynamic():
    model = SimpleConvModel()
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        opt_path = f.name
        
    try:
        x = torch.randn(1, 3, 4, 4)
        torch.onnx.export(model, x, onnx_path, input_names=["input"], output_names=["output"])
        
        config = {
            "quantize": True,
            "approach": "dynamic",
            "optimization_level": 99
        }
        
        optimize_onnx(onnx_path, opt_path, config)
        assert os.path.exists(opt_path)
        
        # Verify it runs in ONNX Runtime
        sess = ort.InferenceSession(opt_path, providers=["CPUExecutionProvider"])
        inputs = {"input": np.random.randn(1, 3, 4, 4).astype(np.float32)}
        outputs = sess.run(None, inputs)
        assert len(outputs) == 1
    finally:
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
        if os.path.exists(opt_path):
            os.remove(opt_path)

def test_optimize_onnx_static():
    model = SimpleConvModel()
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        opt_path = f.name
        
    try:
        x = torch.randn(1, 3, 4, 4)
        torch.onnx.export(model, x, onnx_path, input_names=["input"], output_names=["output"])
        
        calibration_data = [{"input": torch.randn(1, 3, 4, 4)} for _ in range(5)]
        config = {
            "quantize": True,
            "approach": "static",
            "calibration_data": calibration_data,
            "optimization_level": 1
        }
        
        optimize_onnx(onnx_path, opt_path, config)
        assert os.path.exists(opt_path)
        
        sess = ort.InferenceSession(opt_path, providers=["CPUExecutionProvider"])
        inputs = {"input": np.random.randn(1, 3, 4, 4).astype(np.float32)}
        outputs = sess.run(None, inputs)
        assert len(outputs) == 1
    finally:
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
        if os.path.exists(opt_path):
            os.remove(opt_path)

class PureMLPModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 20)
        self.fc2 = torch.nn.Linear(20, 5)

    def forward(self, x):
        return self.fc2(self.fc1(x))

def test_pure_mlp_shape_inference():
    model = PureMLPModel()
    dummy_input = get_working_dummy_input(model, input_shape=None)
    assert dummy_input.shape == (1, 10)

def test_export_to_onnx_creates_parent_dir():
    model = SimpleConvModel()
    with tempfile.TemporaryDirectory() as tmpdir:
        nested_output = os.path.join(tmpdir, "nested", "sub", "model.onnx")
        assert not os.path.exists(os.path.dirname(nested_output))
        export_to_onnx(model, "onnx", nested_output, input_shape=(1, 3, 4, 4))
        assert os.path.exists(nested_output)
        assert os.path.exists(os.path.dirname(nested_output))
