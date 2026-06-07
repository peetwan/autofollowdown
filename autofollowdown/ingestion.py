"""Model ingestion — normalize the inputs the toolkit accepts into one shape.

Accepts a live `torch.nn.Module`, a Hugging Face model id / local dir, a `.onnx`
file, or a `.pt`/`.pth` checkpoint. Heavy libs (transformers, onnx) are imported
lazily inside the branch that needs them, so loading a plain `nn.Module` doesn't
drag in the LLM/ONNX stack.

Security: `.pt`/`.pth` files are pickled and can run arbitrary code on load, so they
are refused by default — pass `allow_pickle=True` only for a file you trust.
"""

import os

import torch


def load_model(model_input, input_shape=None, allow_pickle=False) -> dict:
    """Return {type, model, path}. `type` ∈ pytorch | huggingface | onnx."""
    if model_input is None:
        raise ValueError("Model cannot be None")

    if isinstance(model_input, torch.nn.Module):
        return {"type": "pytorch", "model": model_input, "path": None}

    if not isinstance(model_input, str):
        raise ValueError(f"Unsupported model type: {type(model_input)}")

    # Local ONNX graph.
    if model_input.endswith(".onnx"):
        if not os.path.exists(model_input):
            raise ValueError(f"ONNX model file not found: {model_input}")
        try:
            import onnx
            return {"type": "onnx", "model": onnx.load(model_input), "path": model_input}
        except Exception as e:
            raise ValueError(f"Failed to load ONNX model from {model_input}: {e}") from e

    # Local PyTorch checkpoint — pickled, so gated behind allow_pickle.
    if model_input.endswith((".pt", ".pth")):
        if not os.path.exists(model_input):
            raise ValueError(f"Checkpoint not found: {model_input}")
        if not allow_pickle:
            raise ValueError(
                f"{model_input!r} is a pickled checkpoint — loading it can run arbitrary "
                "code. Pass allow_pickle=True (CLI: --allow-pickle) only if you trust it, "
                "or pass the nn.Module directly.")
        obj = torch.load(model_input, weights_only=False)   # trusted only
        if isinstance(obj, torch.nn.Module):
            return {"type": "pytorch", "model": obj, "path": model_input}
        raise ValueError(
            f"{model_input!r} is a state_dict, not a full model — rebuild the architecture "
            "and pass the nn.Module (a state_dict alone can't be compressed).")

    # Otherwise treat as a Hugging Face model id or local directory.
    try:
        from transformers import AutoConfig, AutoModel
        config = AutoConfig.from_pretrained(model_input)
        model_class = None
        if getattr(config, "architectures", None):
            import transformers
            model_class = getattr(transformers, config.architectures[0], None)
        model = (model_class or AutoModel).from_pretrained(model_input)
        return {"type": "huggingface", "model": model, "path": model_input}
    except Exception as e:
        # Distinguish "not found" from a genuine load failure by the real exception,
        # not by string-matching the id.
        name = e.__class__.__name__
        if name in ("RepositoryNotFoundError", "EntryNotFoundError", "HFValidationError") \
                or isinstance(e, (OSError, FileNotFoundError)):
            raise ValueError(
                f"Could not find model {model_input!r} (check the id/path and your network)."
            ) from e
        raise ValueError(f"Failed to load model from {model_input!r}: {e}") from e
