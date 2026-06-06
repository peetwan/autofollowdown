import torch
import torch.nn as nn
import torch.fx as fx
import pytest

from autofollowdown.graph_tracing import (
    trace_model,
    fuse_layers,
    FusedConvReLU,
    FusedLinearReLU
)

class ChallengerVerificationModel(nn.Module):
    def __init__(self, conv_bias=True, bn_affine=True):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1, bias=conv_bias)
        self.bn = nn.BatchNorm2d(16, affine=bn_affine)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def run_precision_test(conv_bias, bn_affine):
    # Float32 checks
    model_f32 = ChallengerVerificationModel(conv_bias=conv_bias, bn_affine=bn_affine).eval()
    
    # Initialize parameters to ensure non-trivial values
    with torch.no_grad():
        model_f32.conv.weight.normal_(mean=0.0, std=1.0)
        if conv_bias:
            model_f32.conv.bias.normal_(mean=0.5, std=0.5)
        if bn_affine:
            model_f32.bn.weight.uniform_(0.5, 1.5)
            model_f32.bn.bias.normal_(mean=0.2, std=0.2)
        model_f32.bn.running_mean.normal_(mean=0.1, std=0.2)
        model_f32.bn.running_var.uniform_(0.1, 2.0)
        
    gm_f32 = trace_model(model_f32)
    fused_gm_f32 = fuse_layers(gm_f32)
    
    x_f32 = torch.randn(5, 3, 8, 8)
    with torch.no_grad():
        out_orig_f32 = model_f32(x_f32)
        out_fused_f32 = fused_gm_f32(x_f32)
        
    abs_diff_f32 = torch.abs(out_orig_f32 - out_fused_f32)
    max_diff_f32 = abs_diff_f32.max().item()
    mean_diff_f32 = abs_diff_f32.mean().item()
    
    # Float64 checks
    model_f64 = ChallengerVerificationModel(conv_bias=conv_bias, bn_affine=bn_affine).double().eval()
    with torch.no_grad():
        model_f64.conv.weight.copy_(model_f32.conv.weight.double())
        if conv_bias:
            model_f64.conv.bias.copy_(model_f32.conv.bias.double())
        if bn_affine:
            model_f64.bn.weight.copy_(model_f32.bn.weight.double())
            model_f64.bn.bias.copy_(model_f32.bn.bias.double())
        model_f64.bn.running_mean.copy_(model_f32.bn.running_mean.double())
        model_f64.bn.running_var.copy_(model_f32.bn.running_var.double())
        model_f64.fc.weight.copy_(model_f32.fc.weight.double())
        model_f64.fc.bias.copy_(model_f32.fc.bias.double())
        
    gm_f64 = trace_model(model_f64)
    fused_gm_f64 = fuse_layers(gm_f64)
    
    x_f64 = x_f32.double()
    with torch.no_grad():
        out_orig_f64 = model_f64(x_f64)
        out_fused_f64 = fused_gm_f64(x_f64)
        
    abs_diff_f64 = torch.abs(out_orig_f64 - out_fused_f64)
    max_diff_f64 = abs_diff_f64.max().item()
    mean_diff_f64 = abs_diff_f64.mean().item()
    
    return max_diff_f32, mean_diff_f32, max_diff_f64, mean_diff_f64

def test_challenger_bn_folding_equivalence():
    for conv_bias in [True, False]:
        for bn_affine in [True, False]:
            max_f32, mean_f32, max_f64, mean_f64 = run_precision_test(conv_bias, bn_affine)
            print(f"Bias: {conv_bias}, Affine: {bn_affine} -> Max F32: {max_f32:.5e}, Max F64: {max_f64:.5e}")
            assert max_f32 < 1e-5, f"Float32 tolerance exceeded: {max_f32}"
            assert max_f64 < 1e-7, f"Float64 tolerance exceeded: {max_f64}"

def test_challenger_zero_batch_size():
    model = ChallengerVerificationModel().eval()
    gm = trace_model(model)
    fused_gm = fuse_layers(gm)
    
    x = torch.randn(0, 3, 8, 8)
    with torch.no_grad():
        out_orig = model(x)
        out_fused = fused_gm(x)
        
    assert out_orig.shape == (0, 10)
    assert out_fused.shape == (0, 10)
    assert torch.allclose(out_orig, out_fused, atol=1e-5)
