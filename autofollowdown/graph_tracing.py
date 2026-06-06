import torch
import torch.nn as nn
import torch.fx as fx

class CustomTracer(fx.Tracer):
    """
    Custom PyTorch FX Tracer that allows specifying leaf modules
    by class or name, preventing them from being traced internally.
    """
    def __init__(self, leaf_modules=None, *args):
        super().__init__(*args)
        self.leaf_modules = set(leaf_modules) if leaf_modules else set()

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        if m.__class__ in self.leaf_modules or m.__class__.__name__ in self.leaf_modules:
            return True
        return super().is_leaf_module(m, module_qualified_name)

def trace_model(model: nn.Module, leaf_modules=None) -> fx.GraphModule:
    """
    Dynamically traces a PyTorch model into a GraphModule.
    
    Args:
        model: The PyTorch model to trace.
        leaf_modules: An optional list of module classes or class names 
                      to treat as leaf modules.
                      
    Returns:
        A traced torch.fx.GraphModule.
    """
    tracer = CustomTracer(leaf_modules=leaf_modules)
    graph = tracer.trace(model)
    return fx.GraphModule(tracer.root, graph)

class FusedConvReLU(nn.Module):
    """
    Fused module wrapping a Conv2d layer and a ReLU activation.
    """
    def __init__(self, conv: nn.Conv2d, activation: nn.Module = None):
        super().__init__()
        self.conv = conv
        self.activation = activation if activation is not None else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(x))

class FusedLinearReLU(nn.Module):
    """
    Fused module wrapping a Linear layer and a ReLU activation.
    """
    def __init__(self, linear: nn.Linear, activation: nn.Module = None):
        super().__init__()
        self.linear = linear
        self.activation = activation if activation is not None else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))

def _fuse_conv_bn_pass(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Internal pass to fold BatchNorm2d layers into Conv2d layers.
    """
    modules = dict(gm.named_modules())
    nodes_to_erase = []
    
    for node in list(gm.graph.nodes):
        if node.op == 'call_module':
            conv = modules.get(node.target)
            if isinstance(conv, nn.Conv2d):
                users = list(node.users.keys())
                if len(users) == 1:
                    user = users[0]
                    if user.op == 'call_module':
                        bn = modules.get(user.target)
                        if isinstance(bn, nn.BatchNorm2d):
                            # Check if the module is shared
                            target_count = sum(1 for n in gm.graph.nodes if n.op == 'call_module' and n.target == node.target)
                            
                            if target_count > 1:
                                import copy
                                i = 0
                                fused_target = f"{node.target}_fused_{i}"
                                while fused_target in dict(gm.named_modules()) or hasattr(gm, fused_target):
                                    i += 1
                                    fused_target = f"{node.target}_fused_{i}"
                                conv_copy = copy.deepcopy(conv)
                            else:
                                fused_target = node.target
                                conv_copy = conv
                                
                            with torch.no_grad():
                                # Create fused conv with bias=True
                                fused_conv = nn.Conv2d(
                                    in_channels=conv_copy.in_channels,
                                    out_channels=conv_copy.out_channels,
                                    kernel_size=conv_copy.kernel_size,
                                    stride=conv_copy.stride,
                                    padding=conv_copy.padding,
                                    dilation=conv_copy.dilation,
                                    groups=conv_copy.groups,
                                    bias=True,
                                    padding_mode=conv_copy.padding_mode
                                ).to(device=conv_copy.weight.device, dtype=conv_copy.weight.dtype).eval()
                                
                                # Access parameters
                                w = conv_copy.weight
                                b = conv_copy.bias if conv_copy.bias is not None else torch.zeros(conv_copy.out_channels, device=w.device)
                                
                                mean = bn.running_mean
                                var = bn.running_var
                                eps = bn.eps
                                gamma = bn.weight if bn.weight is not None else torch.ones(bn.num_features, device=w.device)
                                beta = bn.bias if bn.bias is not None else torch.zeros(bn.num_features, device=w.device)
                                
                                # Fold parameters
                                scale = gamma / torch.sqrt(var + eps)
                                fused_conv.weight.copy_(w * scale[:, None, None, None])
                                fused_conv.bias.copy_((b - mean) * scale + beta)
                                
                                # Replace/Add conv submodule in GraphModule
                                gm.add_submodule(fused_target, fused_conv)
                                node.target = fused_target
                                
                                # Re-route users of BN to Conv
                                user.replace_all_uses_with(node)
                                nodes_to_erase.append(user)
                                
    for node in nodes_to_erase:
        gm.graph.erase_node(node)
        
    gm.graph.lint()
    gm.recompile()
    return gm

def _fuse_conv_relu_pass(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Internal pass to fuse Conv2d layers and ReLU activations.
    """
    modules = dict(gm.named_modules())
    nodes_to_erase = []
    
    for node in list(gm.graph.nodes):
        if node.op == 'call_module':
            conv = modules.get(node.target)
            if isinstance(conv, nn.Conv2d):
                users = list(node.users.keys())
                if len(users) == 1:
                    user = users[0]
                    is_relu = False
                    act_module = None
                    if user.op == 'call_module':
                        act_module = modules.get(user.target)
                        if isinstance(act_module, (nn.ReLU, nn.ReLU6)):
                            is_relu = True
                    elif user.op == 'call_function' and user.target in (torch.relu, torch.nn.functional.relu):
                        is_relu = True
                    elif user.op == 'call_method' and user.target in ('relu', 'relu_'):
                        is_relu = True
                        
                    if is_relu:
                        target_count = sum(1 for n in gm.graph.nodes if n.op == 'call_module' and n.target == node.target)
                        
                        if target_count > 1:
                            import copy
                            i = 0
                            fused_target = f"{node.target}_fused_{i}"
                            while fused_target in dict(gm.named_modules()) or hasattr(gm, fused_target):
                                i += 1
                                fused_target = f"{node.target}_fused_{i}"
                            conv_copy = copy.deepcopy(conv)
                        else:
                            fused_target = node.target
                            conv_copy = conv
                            
                        fused_module = FusedConvReLU(conv_copy, act_module)
                        gm.add_submodule(fused_target, fused_module)
                        node.target = fused_target
                        user.replace_all_uses_with(node)
                        nodes_to_erase.append(user)
                        
    for node in nodes_to_erase:
        gm.graph.erase_node(node)
        
    gm.graph.lint()
    gm.recompile()
    return gm

def _fuse_linear_relu_pass(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Internal pass to fuse Linear layers and ReLU activations.
    """
    modules = dict(gm.named_modules())
    nodes_to_erase = []
    
    for node in list(gm.graph.nodes):
        if node.op == 'call_module':
            linear = modules.get(node.target)
            if isinstance(linear, nn.Linear):
                users = list(node.users.keys())
                if len(users) == 1:
                    user = users[0]
                    is_relu = False
                    act_module = None
                    if user.op == 'call_module':
                        act_module = modules.get(user.target)
                        if isinstance(act_module, (nn.ReLU, nn.ReLU6)):
                            is_relu = True
                    elif user.op == 'call_function' and user.target in (torch.relu, torch.nn.functional.relu):
                        is_relu = True
                    elif user.op == 'call_method' and user.target in ('relu', 'relu_'):
                        is_relu = True
                        
                    if is_relu:
                        target_count = sum(1 for n in gm.graph.nodes if n.op == 'call_module' and n.target == node.target)
                        
                        if target_count > 1:
                            import copy
                            i = 0
                            fused_target = f"{node.target}_fused_{i}"
                            while fused_target in dict(gm.named_modules()) or hasattr(gm, fused_target):
                                i += 1
                                fused_target = f"{node.target}_fused_{i}"
                            linear_copy = copy.deepcopy(linear)
                        else:
                            fused_target = node.target
                            linear_copy = linear
                            
                        fused_module = FusedLinearReLU(linear_copy, act_module)
                        gm.add_submodule(fused_target, fused_module)
                        node.target = fused_target
                        user.replace_all_uses_with(node)
                        nodes_to_erase.append(user)
                        
    for node in nodes_to_erase:
        gm.graph.erase_node(node)
        
    gm.graph.lint()
    gm.recompile()
    return gm

def fuse_layers(graph_module: fx.GraphModule) -> fx.GraphModule:
    """
    Scans the GraphModule and fuses layer sequences matching the patterns:
    - Conv2d + BatchNorm2d -> Conv2d (folded)
    - Conv2d + BatchNorm2d + ReLU -> FusedConvReLU (containing folded Conv2d)
    - Conv2d + ReLU -> FusedConvReLU
    - Linear + ReLU -> FusedLinearReLU
    
    Args:
        graph_module: The traced GraphModule.
        
    Returns:
        The optimized GraphModule with fused modules.
    """
    graph_module = _fuse_conv_bn_pass(graph_module)
    graph_module = _fuse_conv_relu_pass(graph_module)
    graph_module = _fuse_linear_relu_pass(graph_module)
    return graph_module

def insert_observer(graph_module: fx.GraphModule, target_node_name: str, observer_module: nn.Module) -> fx.GraphModule:
    """
    Inserts an observer module in the graph immediately after the specified target node.
    
    Args:
        graph_module: The GraphModule to modify.
        target_node_name: The name of the node after which to insert the observer.
        observer_module: The observer nn.Module to insert.
        
    Returns:
        The modified GraphModule.
    """
    modules = dict(graph_module.named_modules())
    target_node = None
    
    for node in graph_module.graph.nodes:
        if node.name == target_node_name:
            target_node = node
            break
            
    if target_node is None:
        raise ValueError(f"Target node '{target_node_name}' not found.")
        
    obs_name = f"{target_node_name}_observer"
    graph_module.add_submodule(obs_name, observer_module)
    
    with graph_module.graph.inserting_after(target_node):
        obs_node = graph_module.graph.call_module(obs_name, args=(target_node,))
        
    for user in list(target_node.users.keys()):
        if user is obs_node:
            continue
        user.replace_input_with(target_node, obs_node)
        
    graph_module.graph.lint()
    graph_module.recompile()
    return graph_module

def insert_observers_after_layers(
    graph_module: fx.GraphModule,
    observer_factory_fn,
    layer_types=(nn.Conv2d, nn.Linear, FusedConvReLU, FusedLinearReLU)
) -> fx.GraphModule:
    """
    Automatically inserts an observer module after every layer of the specified types.
    
    Args:
        graph_module: The GraphModule to modify.
        observer_factory_fn: A zero-argument function that returns a new observer instance.
        layer_types: A tuple of module classes after which to insert observers.
        
    Returns:
        The modified GraphModule.
    """
    modules = dict(graph_module.named_modules())
    targets = []
    
    for node in graph_module.graph.nodes:
        if node.op == 'call_module':
            submodule = modules.get(node.target)
            if isinstance(submodule, layer_types):
                targets.append(node)
                
    for i, target_node in enumerate(targets):
        obs_module = observer_factory_fn()
        obs_name = f"{target_node.name}_obs_{i}"
        graph_module.add_submodule(obs_name, obs_module)
        
        with graph_module.graph.inserting_after(target_node):
            obs_node = graph_module.graph.call_module(obs_name, args=(target_node,))
            
        for user in list(target_node.users.keys()):
            if user is obs_node:
                continue
            user.replace_input_with(target_node, obs_node)
            
    graph_module.graph.lint()
    graph_module.recompile()
    return graph_module

def replace_layer(graph_module: fx.GraphModule, target_node_name: str, new_module: nn.Module) -> fx.GraphModule:
    """
    Replaces a module in the GraphModule at the specified node's target path.
    
    Args:
        graph_module: The GraphModule to modify.
        target_node_name: The name of the node whose module should be replaced.
        new_module: The new nn.Module to replace the old one with.
        
    Returns:
        The modified GraphModule.
    """
    target_node = None
    for node in graph_module.graph.nodes:
        if node.op == 'call_module' and node.name == target_node_name:
            target_node = node
            break
            
    if target_node is None:
        raise ValueError(f"Node '{target_node_name}' not found.")
        
    graph_module.add_submodule(target_node.target, new_module)
    graph_module.graph.lint()
    graph_module.recompile()
    return graph_module

def replace_node(graph_module: fx.GraphModule, target_node_name: str, new_node_creator_fn) -> fx.GraphModule:
    """
    Replaces a node in the graph with a new node created by new_node_creator_fn.
    
    Args:
        graph_module: The GraphModule to modify.
        target_node_name: The name of the node to replace.
        new_node_creator_fn: A callable taking (graph, old_node) and returning a new Node.
        
    Returns:
        The modified GraphModule.
    """
    target_node = None
    for node in graph_module.graph.nodes:
        if node.name == target_node_name:
            target_node = node
            break
            
    if target_node is None:
        raise ValueError(f"Node '{target_node_name}' not found.")
        
    with graph_module.graph.inserting_before(target_node):
        new_node = new_node_creator_fn(graph_module.graph, target_node)
        
    target_node.replace_all_uses_with(new_node)
    graph_module.graph.erase_node(target_node)
    
    graph_module.graph.lint()
    graph_module.recompile()
    return graph_module
