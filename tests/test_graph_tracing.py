import torch
import torch.nn as nn
import torch.fx as fx
import pytest

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

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class LeafModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

class ModelWithLeaf(nn.Module):
    def __init__(self):
        super().__init__()
        self.leaf = LeafModule()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.leaf(x))

def test_trace_model():
    model = SimpleModel()
    gm = trace_model(model)
    assert isinstance(gm, fx.GraphModule)
    
    # Check that nodes exist
    node_ops = [node.op for node in gm.graph.nodes]
    assert "placeholder" in node_ops
    assert "output" in node_ops

def test_trace_model_with_leaf():
    model = ModelWithLeaf()
    gm = trace_model(model, leaf_modules=["LeafModule"])
    assert isinstance(gm, fx.GraphModule)
    
    # The LeafModule should be a single call_module node, not traced inside
    leaf_node_found = False
    for node in gm.graph.nodes:
        if node.op == "call_module" and node.target == "leaf":
            leaf_node_found = True
            break
    assert leaf_node_found

def test_fuse_layers_conv_bn_relu():
    class ConvBNReLUModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, padding=1, bias=False)
            self.bn = nn.BatchNorm2d(16)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    model = ConvBNReLUModel().eval()
    gm = trace_model(model)
    
    # Before fusion, verify nodes are conv -> bn -> relu
    modules_before = dict(gm.named_modules())
    assert isinstance(modules_before["conv"], nn.Conv2d)
    assert isinstance(modules_before["bn"], nn.BatchNorm2d)
    assert isinstance(modules_before["relu"], nn.ReLU)
    
    fused_gm = fuse_layers(gm)
    modules_after = dict(fused_gm.named_modules())
    
    # After fusion, conv should be replaced by FusedConvReLU containing folded Conv2d
    assert isinstance(modules_after["conv"], FusedConvReLU)
    # The BN node should be erased
    for node in fused_gm.graph.nodes:
        if node.op == "call_module":
            assert node.target != "bn"
            assert node.target != "relu"

def test_fuse_layers_linear_relu():
    class LinearReLUModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 5)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.fc(x))

    model = LinearReLUModel().eval()
    gm = trace_model(model)
    
    fused_gm = fuse_layers(gm)
    modules_after = dict(fused_gm.named_modules())
    
    assert isinstance(modules_after["fc"], FusedLinearReLU)
    for node in fused_gm.graph.nodes:
        if node.op == "call_module":
            assert node.target != "relu"

def test_insert_observer():
    class IdentityObserver(nn.Module):
        def forward(self, x):
            return x

    model = SimpleModel()
    gm = trace_model(model)
    
    observer = IdentityObserver()
    gm = insert_observer(gm, "conv", observer)
    
    modules = dict(gm.named_modules())
    assert "conv_observer" in modules
    
    # Verify the observer node is inserted after conv
    observer_node_found = False
    for node in gm.graph.nodes:
        if node.op == "call_module" and node.target == "conv_observer":
            observer_node_found = True
            # Its input should be the conv node
            assert len(node.args) == 1
            assert node.args[0].name == "conv"
            break
    assert observer_node_found

def test_insert_observers_after_layers():
    class DummyObserver(nn.Module):
        def forward(self, x):
            return x

    model = SimpleModel()
    gm = trace_model(model)
    
    def obs_factory():
        return DummyObserver()
        
    gm = insert_observers_after_layers(gm, obs_factory, layer_types=(nn.Conv2d, nn.Linear))
    
    modules = dict(gm.named_modules())
    assert "conv_obs_0" in modules
    assert "fc_obs_1" in modules

def test_replace_layer():
    model = SimpleModel()
    gm = trace_model(model)
    
    new_conv = nn.Conv2d(3, 16, 5, padding=2)
    gm = replace_layer(gm, "conv", new_conv)
    
    modules = dict(gm.named_modules())
    assert modules["conv"] is new_conv

def test_replace_node():
    model = SimpleModel()
    gm = trace_model(model)
    
    # We want to replace the relu node with a call to torch.neg
    def neg_creator(graph, old_node):
        return graph.call_function(torch.neg, args=(old_node.args[0],))
        
    gm = replace_node(gm, "relu", neg_creator)
    
    neg_found = False
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.neg:
            neg_found = True
            break
    assert neg_found
