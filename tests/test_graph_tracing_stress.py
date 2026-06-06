import torch
import torch.nn as nn
import torch.fx as fx
import pytest

from autofollowdown.graph_tracing import (
    trace_model,
    fuse_layers,
    insert_observer,
    insert_observers_after_layers,
    FusedConvReLU,
    FusedLinearReLU
)

# Helper function to check if the FX graph contains any cycles
def detect_cycle(gm: fx.GraphModule) -> bool:
    visited = {}
    for node in gm.graph.nodes:
        visited[node] = 0 # 0: unvisited, 1: visiting, 2: visited

    def dfs(node) -> bool:
        visited[node] = 1
        for user in node.users:
            state = visited.get(user, 0)
            if state == 1:
                return True
            elif state == 0:
                if dfs(user):
                    return True
        visited[node] = 2
        return False

    for node in gm.graph.nodes:
        if visited[node] == 0:
            if dfs(node):
                return True
    return False

# Model 1: Sequence of multiple Conv layers and BatchNorm/ReLU layers
class MultiConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1, bias=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x

def test_layer_fusion_multiple_conv():
    model = MultiConvModel().eval()
    x = torch.randn(2, 3, 16, 16)
    
    # Get expected output
    with torch.no_grad():
        expected = model(x)
        
    gm = trace_model(model)
    fused_gm = fuse_layers(gm)
    
    # Run fused model and compare output
    with torch.no_grad():
        actual = fused_gm(x)
        
    assert torch.allclose(expected, actual, atol=1e-5)
    
    # Check structure
    modules = dict(fused_gm.named_modules())
    assert isinstance(modules["conv1"], FusedConvReLU)
    assert isinstance(modules["conv2"], FusedConvReLU)
    assert isinstance(modules["conv3"], nn.Conv2d)
    
    # Ensure BN modules are no longer called in the graph
    for node in fused_gm.graph.nodes:
        if node.op == "call_module":
            assert node.target not in ("bn1", "bn2", "relu1", "relu2")

# Model 2: Sequence of Linear layers and activations
class MultiLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.fc2(x)
        return x

def test_layer_fusion_linear_layers():
    model = MultiLinearModel().eval()
    x = torch.randn(2, 10)
    
    with torch.no_grad():
        expected = model(x)
        
    gm = trace_model(model)
    fused_gm = fuse_layers(gm)
    
    with torch.no_grad():
        actual = fused_gm(x)
        
    assert torch.allclose(expected, actual, atol=1e-5)
    
    modules = dict(fused_gm.named_modules())
    assert isinstance(modules["fc1"], FusedLinearReLU)
    assert isinstance(modules["fc2"], nn.Linear)
    
    for node in fused_gm.graph.nodes:
        if node.op == "call_module":
            assert node.target != "relu1"

# Model 3: Multiple activations in sequence (Conv -> ReLU -> ReLU6)
class DoubleActivationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU6()

    def forward(self, x):
        return self.relu2(self.relu1(self.conv(x)))

def test_layer_fusion_multiple_activations():
    model = DoubleActivationModel().eval()
    x = torch.randn(2, 3, 16, 16)
    
    with torch.no_grad():
        expected = model(x)
        
    gm = trace_model(model)
    fused_gm = fuse_layers(gm)
    
    with torch.no_grad():
        actual = fused_gm(x)
        
    assert torch.allclose(expected, actual, atol=1e-5)
    
    modules = dict(fused_gm.named_modules())
    assert isinstance(modules["conv"], FusedConvReLU)
    
    # relu2 should still remain as a node
    relu2_found = False
    for node in fused_gm.graph.nodes:
        if node.op == "call_module" and node.target == "relu2":
            relu2_found = True
    assert relu2_found

# Model 4: Shared activation module instance
class SharedReLUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

def test_layer_fusion_shared_relu():
    model = SharedReLUModel().eval()
    x = torch.randn(2, 3, 16, 16)
    
    with torch.no_grad():
        expected = model(x)
        
    gm = trace_model(model)
    fused_gm = fuse_layers(gm)
    
    with torch.no_grad():
        actual = fused_gm(x)
        
    assert torch.allclose(expected, actual, atol=1e-5)
    
    modules = dict(fused_gm.named_modules())
    assert isinstance(modules["conv1"], FusedConvReLU)
    assert isinstance(modules["conv2"], FusedConvReLU)

# Model 5: In-place and functional activation calls
class FunctionalReLUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.fc = nn.Linear(8 * 16 * 16, 10)

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.nn.functional.relu(x)
        return x

def test_layer_fusion_functional_relu():
    model = FunctionalReLUModel().eval()
    x = torch.randn(2, 3, 16, 16)
    
    with torch.no_grad():
        expected = model(x)
        
    gm = trace_model(model)
    fused_gm = fuse_layers(gm)
    
    with torch.no_grad():
        actual = fused_gm(x)
        
    assert torch.allclose(expected, actual, atol=1e-5)
    
    modules = dict(fused_gm.named_modules())
    assert isinstance(modules["conv"], FusedConvReLU)
    assert isinstance(modules["fc"], FusedLinearReLU)

# Model 6: Residual connections
class ResidualModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(8)

    def forward(self, x):
        out1 = self.relu1(self.bn1(self.conv1(x)))
        out2 = self.bn2(self.conv2(out1))
        return out1 + out2

def test_layer_fusion_residual():
    model = ResidualModel().eval()
    x = torch.randn(2, 3, 16, 16)
    
    with torch.no_grad():
        expected = model(x)
        
    gm = trace_model(model)
    fused_gm = fuse_layers(gm)
    
    with torch.no_grad():
        actual = fused_gm(x)
        
    assert torch.allclose(expected, actual, atol=1e-5)

# Test 7: Observer insertion cycle checks
class Observer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))

    def forward(self, x):
        if self.training or True:
            self.min_val.copy_(torch.min(x))
            self.max_val.copy_(torch.max(x))
        return x

def test_observer_insertion_cycle_dependency():
    model = MultiConvModel().eval()
    gm = trace_model(model)
    
    # Before insertion, verify there are no cycles
    assert not detect_cycle(gm)
    
    def obs_factory():
        return Observer()
        
    # Insert observers after all layers automatically
    gm_fused = fuse_layers(gm)
    gm_obs = insert_observers_after_layers(
        gm_fused,
        obs_factory,
        layer_types=(nn.Conv2d, nn.Linear, FusedConvReLU, FusedLinearReLU)
    )
    
    # Lint and recompile to verify structure
    gm_obs.graph.lint()
    gm_obs.recompile()
    
    # Run cycle check
    assert not detect_cycle(gm_obs)
    
    # Verify execution succeeds and runs correctly
    x = torch.randn(2, 3, 16, 16)
    with torch.no_grad():
        out = gm_obs(x)
        
    # Verify observer buffers were populated
    modules = dict(gm_obs.named_modules())
    assert "conv1_obs_0" in modules
    assert "conv2_obs_1" in modules
    assert "conv3_obs_2" in modules
    
    assert modules["conv1_obs_0"].min_val.item() != float("inf")
    assert modules["conv1_obs_0"].max_val.item() != float("-inf")

# Test 8: Observer insertion on residual model
def test_observer_insertion_residual():
    model = ResidualModel().eval()
    gm = trace_model(model)
    fused = fuse_layers(gm)
    
    def obs_factory():
        return Observer()
        
    obs_gm = insert_observers_after_layers(
        fused,
        obs_factory,
        layer_types=(nn.Conv2d, nn.Linear, FusedConvReLU, FusedLinearReLU)
    )
    
    obs_gm.graph.lint()
    obs_gm.recompile()
    
    assert not detect_cycle(obs_gm)
    
    x = torch.randn(2, 3, 16, 16)
    with torch.no_grad():
        out = obs_gm(x)
        
    assert out.shape == (2, 8, 16, 16)
