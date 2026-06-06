import torch
import torch.nn as nn
import torch.fx as fx
from autofollowdown.graph_tracing import trace_model, fuse_layers

class PartialActivationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.conv2(x))
        x = self.conv2(x)
        return x

def test_partial_activation_fusion():
    model = PartialActivationModel().eval()
    x = torch.randn(2, 3, 16, 16)
    
    with torch.no_grad():
        expected = model(x)
        
    gm = trace_model(model)
    fused_gm = fuse_layers(gm)
    
    with torch.no_grad():
        actual = fused_gm(x)
        
    # Check if the output is mathematically identical.
    # If the fusion bug is present, the second call to conv2 will also apply ReLU,
    # causing mathematical deviation.
    assert torch.allclose(expected, actual, atol=1e-5)
