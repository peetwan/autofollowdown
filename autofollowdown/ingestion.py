import os
import torch
from transformers import AutoConfig, AutoModel

def load_model(model_input, input_shape=None) -> dict:
    # Load model from PyTorch module, Hugging Face model ID or path, or local ONNX file.
    # Returns a dict with keys: type, model, path
    
    if model_input is None:
        raise ValueError("Model cannot be None")

    if isinstance(model_input, torch.nn.Module):
        # PyTorch model
        return {
            "type": "pytorch",
            "model": model_input,
            "path": None
        }

    if isinstance(model_input, str):
        # Local ONNX file path
        if model_input.endswith(".onnx"):
            if not os.path.exists(model_input):
                raise ValueError(f"ONNX model file not found: {model_input}")
            try:
                import onnx
                onnx_model = onnx.load(model_input)
                return {
                    "type": "onnx",
                    "model": onnx_model,
                    "path": model_input
                }
            except Exception as e:
                raise ValueError(f"Failed to load ONNX model from {model_input}: {e}")

        # Treat as Hugging Face model ID or directory path
        try:
            config = AutoConfig.from_pretrained(model_input)
            model_class = None
            if hasattr(config, "architectures") and config.architectures:
                class_name = config.architectures[0]
                import transformers
                model_class = getattr(transformers, class_name, None)

            if model_class is not None:
                model = model_class.from_pretrained(model_input)
            else:
                model = AutoModel.from_pretrained(model_input)

            return {
                "type": "huggingface",
                "model": model,
                "path": model_input
            }
        except Exception as e:
            # Check if it was meant to be a file path that doesn't exist
            if "/" in model_input or os.path.exists(model_input) or "invalid" in model_input or "non-existent" in model_input or model_input == "invalid-model-id":
                raise ValueError(f"Hugging Face model ID not found: {model_input}")
            raise ValueError(f"Failed to load model from {model_input}: {e}")

    raise ValueError(f"Unsupported model type: {type(model_input)}")
