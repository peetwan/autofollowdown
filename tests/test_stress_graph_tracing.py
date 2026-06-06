import torch
import torch.nn as nn
import torch.fx as fx
import pytest
import time

from autofollowdown.graph_tracing import (
    trace_model,
    fuse_layers,
    insert_observer,
    insert_observers_after_layers,
    replace_layer,
    replace_node,
    FusedConvReLU,
    FusedLinearReLU
)

# 1. Model with no layers (Identity model)
class IdentityModel(nn.Module):
    def forward(self, x):
        return x

def test_model_with_no_layers():
    model = IdentityModel().eval()
    gm = trace_model(model)
    assert isinstance(gm, fx.GraphModule)
    
    # Verify no fusions occur and outputs match
    x = torch.randn(2, 3, 8, 8)
    out_orig = model(x)
    
    gm_fused = fuse_layers(gm)
    out_fused = gm_fused(x)
    
    assert torch.allclose(out_orig, out_fused)
    
    # Check node list
    nodes = list(gm_fused.graph.nodes)
    assert len(nodes) == 2  # placeholder and output

# 2. Empty inputs (zero batch size)
def test_empty_inputs_zero_batch():
    class SimpleConvModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)
            self.bn = nn.BatchNorm2d(8)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))
            
    model = SimpleConvModel().eval()
    gm = trace_model(model)
    fused_gm = fuse_layers(gm)
    
    # Zero batch size input: shape (0, 3, 8, 8)
    x = torch.randn(0, 3, 8, 8)
    
    out_orig = model(x)
    out_fused = fused_gm(x)
    
    # Verify outputs match and shape is correct
    assert out_orig.shape == (0, 8, 8, 8)
    assert out_fused.shape == (0, 8, 8, 8)
    assert torch.allclose(out_orig, out_fused, atol=1e-5)

# 3. Incorrect node names for observers
def test_incorrect_node_names_for_observers():
    class DummyObserver(nn.Module):
        def forward(self, x):
            return x
            
    model = IdentityModel().eval()
    gm = trace_model(model)
    
    observer = DummyObserver()
    with pytest.raises(ValueError, match="Target node 'non_existent_node' not found."):
        insert_observer(gm, "non_existent_node", observer)

# 4. Multiple sequential fusions on complex models
def test_multiple_sequential_fusions_complex_model():
    class ComplexResidualModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
            self.relu1 = nn.ReLU()
            
            self.conv2 = nn.Conv2d(16, 16, 3, padding=1, bias=True)
            self.bn2 = nn.BatchNorm2d(16)
            self.relu2 = nn.ReLU()
            
            self.fc1 = nn.Linear(16 * 8 * 8, 64)
            self.relu3 = nn.ReLU()
            self.fc2 = nn.Linear(64, 10)
            
        def forward(self, x):
            out1 = self.relu1(self.bn1(self.conv1(x)))
            out2 = self.relu2(self.bn2(self.conv2(out1)))
            out2_flat = torch.flatten(out2, 1)
            out3 = self.relu3(self.fc1(out2_flat))
            return self.fc2(out3)

    model = ComplexResidualModel().eval()
    gm = trace_model(model)
    
    # First fusion pass
    fused_gm_1 = fuse_layers(gm)
    
    # Verify submodules are fused
    modules_1 = dict(fused_gm_1.named_modules())
    assert isinstance(modules_1["conv1"], FusedConvReLU)
    assert isinstance(modules_1["conv2"], FusedConvReLU)
    assert isinstance(modules_1["fc1"], FusedLinearReLU)
    
    # Second fusion pass (should be a safe no-op)
    fused_gm_2 = fuse_layers(fused_gm_1)
    
    # Verify modules are unchanged and still valid
    modules_2 = dict(fused_gm_2.named_modules())
    assert isinstance(modules_2["conv1"], FusedConvReLU)
    assert isinstance(modules_2["conv2"], FusedConvReLU)
    assert isinstance(modules_2["fc1"], FusedLinearReLU)
    
    # Verify equivalence of outputs on multiple passes
    x = torch.randn(4, 3, 8, 8)
    out_orig = model(x)
    out_fused_1 = fused_gm_1(x)
    out_fused_2 = fused_gm_2(x)
    
    assert torch.allclose(out_orig, out_fused_1, atol=1e-6)
    assert torch.allclose(out_orig, out_fused_2, atol=1e-6)

# 5. Mathematical equivalence of BatchNorm folding on customized Conv+BN networks
@pytest.mark.parametrize("conv_bias", [True, False])
@pytest.mark.parametrize("bn_affine", [True, False])
def test_bn_folding_mathematical_equivalence(conv_bias, bn_affine):
    class CustomConvBN(nn.Module):
        def __init__(self, bias, affine):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, padding=1, bias=bias)
            self.bn = nn.BatchNorm2d(16, affine=affine)
            
            # Initialize weights to arbitrary non-zero values to stress test
            with torch.no_grad():
                self.conv.weight.normal_(mean=0.0, std=1.0)
                if bias:
                    self.conv.bias.normal_(mean=0.5, std=0.5)
                if affine:
                    self.bn.weight.uniform_(0.5, 1.5)
                    self.bn.bias.normal_(mean=0.2, std=0.2)
                # Ensure running stats are non-trivial
                self.bn.running_mean.normal_(mean=0.1, std=0.2)
                self.bn.running_var.uniform_(0.1, 2.0)
                
        def forward(self, x):
            return self.bn(self.conv(x))

    # Float32 model for timing and standard inference check
    model_f32 = CustomConvBN(bias=conv_bias, affine=bn_affine).eval()
    gm_f32 = trace_model(model_f32)
    
    # Trace execution time
    t0 = time.perf_counter()
    out_orig_f32 = model_f32(torch.randn(10, 3, 32, 32))  # Warmup
    t_orig = 0.0
    for _ in range(20):
        x = torch.randn(32, 3, 64, 64)
        t_start = time.perf_counter()
        out_orig_f32 = model_f32(x)
        t_orig += time.perf_counter() - t_start
        
    fused_gm_f32 = fuse_layers(gm_f32)
    
    t_fused = 0.0
    for _ in range(20):
        x = torch.randn(32, 3, 64, 64)
        t_start = time.perf_counter()
        out_fused_f32 = fused_gm_f32(x)
        t_fused += time.perf_counter() - t_start
        
    # Quantify float32 differences
    x_test_f32 = torch.randn(50, 3, 128, 128)
    with torch.no_grad():
        out_orig_test_f32 = model_f32(x_test_f32)
        out_fused_test_f32 = fused_gm_f32(x_test_f32)
    abs_diff_f32 = torch.abs(out_orig_test_f32 - out_fused_test_f32)
    max_diff_f32 = abs_diff_f32.max().item()
    mean_diff_f32 = abs_diff_f32.mean().item()

    # Double precision model for mathematical equivalence verification
    model_f64 = CustomConvBN(bias=conv_bias, affine=bn_affine).double().eval()
    with torch.no_grad():
        model_f64.conv.weight.copy_(model_f32.conv.weight.double())
        if conv_bias:
            model_f64.conv.bias.copy_(model_f32.conv.bias.double())
        if bn_affine:
            model_f64.bn.weight.copy_(model_f32.bn.weight.double())
            model_f64.bn.bias.copy_(model_f32.bn.bias.double())
        model_f64.bn.running_mean.copy_(model_f32.bn.running_mean.double())
        model_f64.bn.running_var.copy_(model_f32.bn.running_var.double())
        
    gm_f64 = trace_model(model_f64)
    fused_gm_f64 = fuse_layers(gm_f64)
    
    x_test_f64 = x_test_f32.double()
    with torch.no_grad():
        out_orig_test_f64 = model_f64(x_test_f64)
        out_fused_test_f64 = fused_gm_f64(x_test_f64)
        
    abs_diff_f64 = torch.abs(out_orig_test_f64 - out_fused_test_f64)
    max_diff_f64 = abs_diff_f64.max().item()
    mean_diff_f64 = abs_diff_f64.mean().item()
    
    print(f"\n[BN Folding Equivalence: conv_bias={conv_bias}, bn_affine={bn_affine}]")
    print(f"Float32 max absolute difference: {max_diff_f32:.10e}")
    print(f"Float32 mean absolute difference: {mean_diff_f32:.10e}")
    print(f"Float64 max absolute difference: {max_diff_f64:.10e}")
    print(f"Float64 mean absolute difference: {mean_diff_f64:.10e}")
    print(f"Timing results (20 runs): Original = {t_orig:.6f}s, Fused = {t_fused:.6f}s")
    
    # Verification condition: max difference under double precision must be less than 1e-7
    assert max_diff_f64 < 1e-7, f"Max double difference {max_diff_f64} is greater than 1e-7"

