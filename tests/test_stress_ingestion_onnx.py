import os
import tempfile
import numpy as np
import pytest
import torch
import onnx
from onnx import numpy_helper
import onnxruntime as ort
from transformers import AutoConfig, AutoModel

from autofollowdown.ingestion import load_model
from autofollowdown.onnx_pipeline import (
    export_to_onnx,
    prune_onnx,
    optimize_onnx,
    ONNXCalibrationDataReader
)

class DeepConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.relu2 = torch.nn.ReLU()
        self.fc = torch.nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

def test_export_and_check_onnx():
    # 1. Verify model export actually outputs valid ONNX models.
    model = DeepConvModel()
    model.eval()
    
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name
        
    try:
        # Export with shape (1, 3, 8, 8)
        export_to_onnx(model, "onnx", onnx_path, input_shape=(1, 3, 8, 8))
        assert os.path.exists(onnx_path)
        assert os.path.getsize(onnx_path) > 0
        
        # Load and verify validity with ONNX checker
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # Verify execution in ONNX Runtime
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        
        x_np = np.random.randn(1, 3, 8, 8).astype(np.float32)
        outputs = sess.run([output_name], {input_name: x_np})
        assert len(outputs) == 1
        assert outputs[0].shape == (1, 10)
    finally:
        if os.path.exists(onnx_path):
            os.remove(onnx_path)

def test_pruning_sparsity_levels():
    # 2. Verify that ONNX graph pruning zeroed out elements correctly according to requested sparsity
    model = DeepConvModel()
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name
        
    try:
        x = torch.randn(1, 3, 8, 8)
        torch.onnx.export(model, x, onnx_path, input_names=["input"], output_names=["output"])
        
        # Test various sparsity targets
        sparsity_levels = [0.0, 0.1, 0.3, 0.5, 0.8, 0.95, 1.0]
        
        for sp in sparsity_levels:
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as pf:
                pruned_path = pf.name
                
            try:
                prune_onnx(onnx_path, pruned_path, sparsity=sp)
                assert os.path.exists(pruned_path)
                
                pruned_model = onnx.load(pruned_path)
                onnx.checker.check_model(pruned_model)
                
                # Check actual sparsities of 2D+ initializers
                total_zeros = 0
                total_elements = 0
                
                for initializer in pruned_model.graph.initializer:
                    if initializer.data_type == onnx.TensorProto.FLOAT:
                        arr = numpy_helper.to_array(initializer)
                        if len(arr.shape) >= 2:
                            zeros = np.sum(arr == 0.0)
                            total = arr.size
                            total_zeros += zeros
                            total_elements += total
                            
                            layer_sparsity = zeros / total if total > 0 else 0.0
                            # Quantile might not match exactly due to duplicate weights, 
                            # but should be very close, except for boundary cases.
                            if sp == 0.0:
                                assert layer_sparsity == 0.0
                            elif sp == 1.0:
                                # for quantile 1.0, abs_arr < threshold (where threshold is max)
                                # will exclude the max element. So it's (total - 1)/total.
                                assert layer_sparsity >= (total - 1) / total
                            else:
                                # Tolerant check to account for float precision & duplicate values
                                assert layer_sparsity >= (sp - 0.02)
                
                # Verify pruned model still runs
                sess = ort.InferenceSession(pruned_path, providers=["CPUExecutionProvider"])
                x_np = np.random.randn(1, 3, 8, 8).astype(np.float32)
                res = sess.run(None, {"input": x_np})
                assert len(res) == 1
                assert res[0].shape == (1, 10)
                
            finally:
                if os.path.exists(pruned_path):
                    os.remove(pruned_path)
    finally:
        if os.path.exists(onnx_path):
            os.remove(onnx_path)

def test_quantization_precision_and_correctness():
    # 3. Verify static and dynamic quantization work as intended and run successfully inside ONNX Runtime.
    model = DeepConvModel()
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        dyn_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        stat_path = f.name
        
    try:
        x_torch = torch.randn(1, 3, 8, 8)
        torch.onnx.export(model, x_torch, onnx_path, input_names=["input"], output_names=["output"])
        
        # Test input data for validation
        test_inputs = np.random.randn(1, 3, 8, 8).astype(np.float32)
        
        # Get baseline prediction from unquantized ONNX model
        base_sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        base_out = base_sess.run(None, {"input": test_inputs})[0]
        
        # 3a. Dynamic Quantization (with optimization_level=None to avoid shape inference failure)
        dyn_config = {
            "quantize": True,
            "approach": "dynamic"
        }
        optimize_onnx(onnx_path, dyn_path, dyn_config)
        assert os.path.exists(dyn_path)
        
        dyn_sess = ort.InferenceSession(dyn_path, providers=["CPUExecutionProvider"])
        dyn_out = dyn_sess.run(None, {"input": test_inputs})[0]
        assert dyn_out.shape == base_out.shape
        
        # Calculate Cosine Similarity for Dynamic Quantization
        cos_sim_dyn = np.dot(base_out.flatten(), dyn_out.flatten()) / (np.linalg.norm(base_out) * np.linalg.norm(dyn_out))
        assert cos_sim_dyn > 0.85  # Reasonable threshold for similarity
        
        # 3b. Static Quantization
        # Generate 5 calibration inputs
        calibration_data = [{"input": torch.randn(1, 3, 8, 8)} for _ in range(5)]
        stat_config = {
            "quantize": True,
            "approach": "static",
            "calibration_data": calibration_data
        }
        optimize_onnx(onnx_path, stat_path, stat_config)
        assert os.path.exists(stat_path)
        
        stat_sess = ort.InferenceSession(stat_path, providers=["CPUExecutionProvider"])
        stat_out = stat_sess.run(None, {"input": test_inputs})[0]
        assert stat_out.shape == base_out.shape
        
        # Calculate Cosine Similarity for Static Quantization
        cos_sim_stat = np.dot(base_out.flatten(), stat_out.flatten()) / (np.linalg.norm(base_out) * np.linalg.norm(stat_out))
        assert cos_sim_stat > 0.85
        
        # Assert the models were genuinely quantized: their weight initializers
        # are now 8-bit. (Comparing raw .onnx file sizes is unreliable across torch
        # versions — newer exporters store fp32 weights as external data, so the
        # baseline .onnx file can be tiny while the embedded-int8 output looks bigger.)
        def _int8_initializers(path):
            m = onnx.load(path)
            return sum(1 for i in m.graph.initializer
                       if i.data_type in (onnx.TensorProto.INT8, onnx.TensorProto.UINT8))

        assert _int8_initializers(dyn_path) > 0, "dynamic quant produced no int8 weights"
        assert _int8_initializers(stat_path) > 0, "static quant produced no int8 weights"

        print(f"Dynamic quant: {_int8_initializers(dyn_path)} int8 tensors (Cos Sim: {cos_sim_dyn:.4f})")
        print(f"Static quant: {_int8_initializers(stat_path)} int8 tensors (Cos Sim: {cos_sim_stat:.4f})")
        
    finally:
        for p in [onnx_path, dyn_path, stat_path]:
            if os.path.exists(p):
                os.remove(p)

def test_quantization_optimization_failure():
    # Verify that combining optimization_level and quantization succeeds without raising any exceptions
    # and outputs a valid quantized ONNX model.
    model = DeepConvModel()
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        opt_path = f.name
        
    try:
        x_torch = torch.randn(1, 3, 8, 8)
        torch.onnx.export(model, x_torch, onnx_path, input_names=["input"], output_names=["output"])
        
        # Combining optimization_level >= 2 (or 99) and quantization should succeed
        config = {
            "quantize": True,
            "approach": "dynamic",
            "optimization_level": 99
        }
        optimize_onnx(onnx_path, opt_path, config)
        assert os.path.exists(opt_path)
        assert os.path.getsize(opt_path) > 0
        
        # Load and verify it's a valid ONNX model
        loaded = onnx.load(opt_path)
        onnx.checker.check_model(loaded)
        
        # Verify execution in ONNX Runtime
        sess = ort.InferenceSession(opt_path, providers=["CPUExecutionProvider"])
        inputs = {"input": np.random.randn(1, 3, 8, 8).astype(np.float32)}
        outputs = sess.run(None, inputs)
        assert len(outputs) == 1
    finally:
        for p in [onnx_path, opt_path]:
            if os.path.exists(p):
                os.remove(p)


def test_static_quantization_empty_calibration():
    # 4. Stress tests for edge cases: empty calibration data for static quantization
    model = DeepConvModel()
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        opt_path = f.name
        
    try:
        x = torch.randn(1, 3, 8, 8)
        torch.onnx.export(model, x, onnx_path, input_names=["input"], output_names=["output"])
        
        # Empty list
        config = {
            "quantize": True,
            "approach": "static",
            "calibration_data": [],
            "optimization_level": 1
        }
        with pytest.raises(ValueError, match="Calibration data is required"):
            optimize_onnx(onnx_path, opt_path, config)
            
        # None calibration data
        config["calibration_data"] = None
        with pytest.raises(ValueError, match="Calibration data is required"):
            optimize_onnx(onnx_path, opt_path, config)
    finally:
        for p in [onnx_path, opt_path]:
            if os.path.exists(p):
                os.remove(p)

def test_pruning_invalid_sparsity():
    # Stress tests: invalid sparsity values
    model = DeepConvModel()
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        pruned_path = f.name
        
    try:
        x = torch.randn(1, 3, 8, 8)
        torch.onnx.export(model, x, onnx_path, input_names=["input"], output_names=["output"])
        
        # Negative sparsity
        with pytest.raises(ValueError, match="Sparsity must be between 0.0 and 1.0"):
            prune_onnx(onnx_path, pruned_path, sparsity=-0.1)
            
        # Sparsity > 1.0
        with pytest.raises(ValueError, match="Sparsity must be between 0.0 and 1.0"):
            prune_onnx(onnx_path, pruned_path, sparsity=1.1)
    finally:
        for p in [onnx_path, pruned_path]:
            if os.path.exists(p):
                os.remove(p)
