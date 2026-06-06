import torch
import torch.nn as nn
import torch.fx as fx
from autofollowdown.graph_tracing import trace_model, fuse_layers

class PartialBNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(8)
        
        # Initialize BN with non-default values to ensure bn(y) is very different from y
        with torch.no_grad():
            self.bn.running_mean.fill_(2.5)
            self.bn.running_var.fill_(0.5)
            self.bn.weight.fill_(1.5)
            self.bn.bias.fill_(-1.0)

    def forward(self, x):
        x = self.conv1(x)
        # First call is followed by BN
        x1 = self.bn(self.conv2(x))
        # Second call is NOT followed by BN
        x2 = self.conv2(x)
        return x1 + x2

def test_partial_bn_fusion():
    model = PartialBNModel().eval()
    x = torch.randn(2, 3, 16, 16)
    
    with torch.no_grad():
        expected = model(x)
        
    gm = trace_model(model)
    fused_gm = fuse_layers(gm)
    
    with torch.no_grad():
        actual = fused_gm(x)
        
    # Now that BN parameters are non-trivial, this assertion will catch the bug
    assert torch.allclose(expected, actual, atol=1e-5)
