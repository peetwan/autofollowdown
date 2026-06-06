import torch
import torch.nn as nn
import torch.fx as fx
from autofollowdown.graph_tracing import trace_model, fuse_layers, FusedConvReLU

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class NestedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = Block(3, 8)
        self.block2 = Block(8, 16)

    def forward(self, x):
        return self.block2(self.block1(x))

def test_nested_fusion():
    model = NestedModel().eval()
    x = torch.randn(2, 3, 16, 16)
    
    with torch.no_grad():
        expected = model(x)
        
    gm = trace_model(model)
    fused_gm = fuse_layers(gm)
    
    with torch.no_grad():
        actual = fused_gm(x)
        
    assert torch.allclose(expected, actual, atol=1e-5)
    
    # Check that fusion happened at the nested level
    modules = dict(fused_gm.named_modules())
    assert isinstance(modules["block1.conv"], FusedConvReLU)
    assert isinstance(modules["block2.conv"], FusedConvReLU)
