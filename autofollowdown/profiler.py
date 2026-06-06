"""Model profiling — inspect a model to decide which compression backend fits.

The auto-picker is only as good as what it knows about the model, so this module
extracts a small, honest `ModelProfile`: what family it is (LLM / transformer /
CNN / MLP), how big it is, and what hardware is available.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class ModelProfile:
    family: str               # 'llm' | 'transformer' | 'cnn' | 'mlp' | 'unknown'
    num_params: int
    has_conv: bool
    has_transformer: bool
    is_huggingface: bool
    cuda_available: bool
    detail: dict = field(default_factory=dict)

    def __str__(self):
        return (f"ModelProfile(family={self.family}, params={self.num_params:,}, "
                f"conv={self.has_conv}, transformer={self.has_transformer}, "
                f"hf={self.is_huggingface}, cuda={self.cuda_available})")


_ATTENTION_HINTS = ("attention", "transformerblock", "decoderlayer", "encoderlayer")


def _looks_like_transformer(model):
    if any(isinstance(m, nn.MultiheadAttention) for m in model.modules()):
        return True
    for m in model.modules():
        name = m.__class__.__name__.lower()
        if any(h in name for h in _ATTENTION_HINTS):
            return True
    return False


def _is_causal_lm(model):
    cls = model.__class__.__name__.lower()
    if "forcausallm" in cls or "lmheadmodel" in cls:
        return True
    cfg = getattr(model, "config", None)
    archs = getattr(cfg, "architectures", None) or []
    return any("causallm" in a.lower() or "lmhead" in a.lower() for a in archs)


def profile_model(model):
    """Build a ModelProfile for a PyTorch `nn.Module`.

    `family` heuristics, in priority order:
      - 'llm'         : a transformer with a causal-LM head, or a large transformer
      - 'transformer' : has attention but isn't an LLM (e.g. encoder like BERT)
      - 'cnn'         : has convolutions and no attention
      - 'mlp'         : only linear/activation layers
      - 'unknown'     : none of the above
    """
    if not isinstance(model, nn.Module):
        raise ValueError("profile_model expects a torch.nn.Module")

    num_params = sum(p.numel() for p in model.parameters())
    has_conv = any(isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d))
                   for m in model.modules())
    has_transformer = _looks_like_transformer(model)
    is_hf = hasattr(model, "config") and model.__class__.__module__.startswith(
        "transformers")
    is_llm = (has_transformer and (_is_causal_lm(model) or num_params >= 50_000_000))

    if is_llm:
        family = "llm"
    elif has_transformer:
        family = "transformer"
    elif has_conv:
        family = "cnn"
    elif any(isinstance(m, nn.Linear) for m in model.modules()):
        family = "mlp"
    else:
        family = "unknown"

    return ModelProfile(
        family=family,
        num_params=num_params,
        has_conv=has_conv,
        has_transformer=has_transformer,
        is_huggingface=bool(is_hf),
        cuda_available=torch.cuda.is_available(),
        detail={"is_causal_lm": _is_causal_lm(model) if has_transformer else False},
    )


def profile_from_pretrained(model_id):
    """Build a ModelProfile from a Hugging Face *config* alone — no weight download.

    Lets `recommend` advise on a model id quickly. Parameter count is estimated from
    the config (marked approximate); family is read from the architecture name.
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id)
    archs = list(getattr(cfg, "architectures", []) or [])
    is_causal = any(("CausalLM" in a) or ("LMHeadModel" in a) for a in archs)

    h = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None) \
        or getattr(cfg, "dim", None)
    n_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None) \
        or getattr(cfg, "n_layers", None)
    vocab = getattr(cfg, "vocab_size", None)
    inter = getattr(cfg, "intermediate_size", None) or (4 * h if h else None)

    num = 0
    if h and n_layers and vocab:
        per_layer = 4 * h * h + 3 * h * (inter or 4 * h)   # attn + (SwiGLU-ish) MLP
        num = vocab * h + n_layers * per_layer

    family = "llm" if (is_causal or (num and num >= 50_000_000)) else "transformer"
    return ModelProfile(
        family=family,
        num_params=int(num),
        has_conv=False,
        has_transformer=True,
        is_huggingface=True,
        cuda_available=torch.cuda.is_available(),
        detail={"is_causal_lm": is_causal, "model_id": model_id, "estimated": True},
    )
