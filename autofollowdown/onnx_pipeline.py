import os
import shutil
import tempfile
import torch
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    quantize_dynamic,
    quantize_static,
    QuantType,
    quant_pre_process
)

class ONNXCalibrationDataReader(CalibrationDataReader):
    def __init__(self, calibration_data, input_names=None):
        super().__init__()
        self.data = list(calibration_data)
        self.input_names = input_names
        self.index = 0

    def get_next(self):
        if self.index >= len(self.data):
            return None
        
        batch = self.data[self.index]
        self.index += 1
        
        # Convert batch elements to numpy arrays
        if isinstance(batch, dict):
            res = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    res[k] = v.cpu().numpy()
                elif isinstance(v, np.ndarray):
                    res[k] = v
                else:
                    res[k] = np.array(v)
            return res
        elif isinstance(batch, (list, tuple)):
            res = {}
            for i, item in enumerate(batch):
                name = self.input_names[i] if (self.input_names and i < len(self.input_names)) else f"input_{i}"
                if isinstance(item, torch.Tensor):
                    res[name] = item.cpu().numpy()
                elif isinstance(item, np.ndarray):
                    res[name] = item
                else:
                    res[name] = np.array(item)
            return res
        else:
            name = self.input_names[0] if self.input_names else "input"
            if isinstance(batch, torch.Tensor):
                val = batch.cpu().numpy()
            elif isinstance(batch, np.ndarray):
                val = batch
            else:
                val = np.array(batch)
            return {name: val}

    def rewind(self):
        self.index = 0


def get_working_dummy_input(model, input_shape=None):
    # If input_shape is provided, try that first
    if input_shape is not None:
        if isinstance(input_shape, dict):
            return {k: torch.tensor(v) for k, v in input_shape.items()}
        
        if hasattr(model, "config"):
            batch_size = input_shape[0] if len(input_shape) > 0 else 1
            seq_len = input_shape[1] if len(input_shape) > 1 else 8
            return {
                "input_ids": torch.ones(batch_size, seq_len, dtype=torch.long),
                "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long)
            }
        else:
            return torch.randn(*input_shape)
            
    # input_shape is None. Let's try to infer.
    if hasattr(model, "config"):
        return {
            "input_ids": torch.ones(1, 8, dtype=torch.long),
            "attention_mask": torch.ones(1, 8, dtype=torch.long)
        }
        
    # Check if model is a pure MLP (no Conv2d, but has Linear layers)
    has_conv2d = False
    first_linear = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            has_conv2d = True
        elif isinstance(m, torch.nn.Linear):
            if first_linear is None:
                first_linear = m
                
    if not has_conv2d and first_linear is not None:
        return torch.randn(1, first_linear.in_features)
        
    channels = 3
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            channels = m.in_channels
            break
            
    for spatial in [8, 224, 28, 32]:
        try:
            x = torch.randn(1, channels, spatial, spatial)
            model(x)
            return x
        except Exception:
            continue
            
    return torch.randn(1, channels, 224, 224)


def export_to_onnx(model, format_type, output_path, input_shape=None) -> str:
    dir_name = os.path.dirname(output_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        
    # Resolve the model if a dict is passed
    if isinstance(model, dict):
        if model["type"] == "onnx":
            shutil.copy2(model["path"], output_path)
            return output_path
        model_obj = model["model"]
    else:
        model_obj = model
        
    if isinstance(model_obj, str) and model_obj.endswith(".onnx"):
        shutil.copy2(model_obj, output_path)
        return output_path

    dummy_input = get_working_dummy_input(model_obj, input_shape)
    
    # Dry run
    model_obj.eval()
    with torch.no_grad():
        if isinstance(dummy_input, dict):
            # Unpack dictionary without using double asterisks
            import inspect
            sig = inspect.signature(model_obj.forward)
            pos_args = []
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                if param_name in dummy_input:
                    pos_args.append(dummy_input[param_name])
                elif param.default is not inspect.Parameter.empty:
                    pos_args.append(param.default)
                else:
                    if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                        continue
                    pos_args.append(None)
            outputs = model_obj(*pos_args)
        else:
            outputs = model_obj(dummy_input)
            
    # Setup input names and dynamic axes
    if isinstance(dummy_input, dict):
        input_names = list(dummy_input.keys())
        dynamic_axes = {}
        for name in input_names:
            dynamic_axes[name] = {0: "batch_size", 1: "seq_len"}
    else:
        input_names = ["input"]
        dynamic_axes = {"input": {0: "batch_size"}}
        
    # Setup output names and dynamic axes
    if hasattr(outputs, "keys"):
        output_names = list(outputs.keys())
        for name in output_names:
            dynamic_axes[name] = {0: "batch_size"}
    elif isinstance(outputs, (list, tuple)):
        output_names = [f"output_{i}" for i in range(len(outputs))]
        for name in output_names:
            dynamic_axes[name] = {0: "batch_size"}
    else:
        output_names = ["output"]
        dynamic_axes["output"] = {0: "batch_size"}
        
    # Export to ONNX
    import inspect
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if isinstance(dummy_input, dict):
            args = tuple(dummy_input.values())
        else:
            args = dummy_input

        export_kwargs = dict(
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
        # On torch>=2.5 the default exporter is dynamo-based (needs onnxscript and
        # renames I/O). Force the legacy TorchScript exporter when available so
        # input/output names stay stable across torch versions.
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            export_kwargs["dynamo"] = False

        torch.onnx.export(model_obj, args, output_path, **export_kwargs)

    return output_path


def prune_onnx(onnx_path, output_path, sparsity=0.3) -> str:
    if not (0.0 <= sparsity <= 1.0):
        raise ValueError(f"Sparsity must be between 0.0 and 1.0, got {sparsity}")
        
    model = onnx.load(onnx_path)
    
    for i, initializer in enumerate(model.graph.initializer):
        if initializer.data_type == onnx.TensorProto.FLOAT:
            arr = numpy_helper.to_array(initializer).copy()
            if len(arr.shape) >= 2:
                abs_arr = np.abs(arr)
                if abs_arr.size > 0:
                    threshold = np.quantile(abs_arr, sparsity)
                    mask = abs_arr < threshold
                    arr[mask] = 0.0
                    
                    new_initializer = numpy_helper.from_array(arr, name=initializer.name)
                    model.graph.initializer[i].CopyFrom(new_initializer)
                    
    onnx.save(model, output_path)
    return output_path


def optimize_onnx(onnx_path, output_path, config) -> str:
    temp_files = []
    try:
        opt_level = config.get("optimization_level", None)
        current_path = onnx_path
        quantize = config.get("quantize", False)
        approach = config.get("approach", "dynamic")
        
        if opt_level is not None and not quantize:
            opt_output_fd, opt_output_path = tempfile.mkstemp(suffix="_opt.onnx")
            os.close(opt_output_fd)
            temp_files.append(opt_output_path)
            
            sess_options = ort.SessionOptions()
            if opt_level == 0:
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            elif opt_level == 1:
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            elif opt_level == 2:
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            elif opt_level >= 99:
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                
            sess_options.optimized_model_filepath = opt_output_path
            
            try:
                _ = ort.InferenceSession(current_path, sess_options, providers=["CPUExecutionProvider"])
                current_path = opt_output_path
            except Exception:
                shutil.copy2(current_path, opt_output_path)
                current_path = opt_output_path
                
        if quantize:
            if approach == "dynamic":
                quantize_dynamic(
                    model_input=current_path,
                    model_output=output_path,
                    weight_type=QuantType.QUInt8
                )
            elif approach == "static":
                calibration_data = config.get("calibration_data", None)
                if not calibration_data:
                    raise ValueError("Calibration data is required for static quantization")
                    
                pre_output_fd, pre_output_path = tempfile.mkstemp(suffix="_pre.onnx")
                os.close(pre_output_fd)
                temp_files.append(pre_output_path)
                
                try:
                    quant_pre_process(current_path, pre_output_path)
                    current_path = pre_output_path
                except Exception:
                    pass
                    
                import onnx
                onnx_model = onnx.load(current_path)
                input_names = [inp.name for inp in onnx_model.graph.input]
                
                reader = ONNXCalibrationDataReader(calibration_data, input_names=input_names)
                
                quantize_static(
                    model_input=current_path,
                    model_output=output_path,
                    calibration_data_reader=reader,
                    weight_type=QuantType.QInt8,
                    activation_type=QuantType.QUInt8
                )
        else:
            shutil.copy2(current_path, output_path)
            
        return output_path
    finally:
        for f in temp_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass
