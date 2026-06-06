"""LLM evaluation for compressed models — the benchmarks that matter for
quantization / pruning / distillation.

Across the literature (GPTQ, AWQ, SparseGPT, LLMCBench, LeanQuant, NVIDIA
MINITRON, Apple LLM-KICK) compressed LLMs are judged on two pillars:

  1. Perplexity (lower = better) on held-out text — WikiText-2 is the universal
     default, with C4 and PTB also common. This module computes it for real,
     using the standard sliding-window method.
  2. Zero-shot / few-shot task accuracy via EleutherAI's lm-evaluation-harness
     (ARC, HellaSwag, WinoGrande, PIQA, LAMBADA, BoolQ, MMLU, GSM8K, ...). That
     harness is the community standard; `lm_eval_command()` builds the exact CLI
     to point it at a model you compressed with autofollowdown.

Perplexity here targets Hugging Face causal LMs (models returning `.loss` when
given `labels`).
"""

import math
import warnings

import torch

# The standard datasets/tasks used to benchmark compressed LLMs, grouped by what
# they measure. Names match EleutherAI lm-evaluation-harness task ids.
STANDARD_LLM_TASKS = {
    "perplexity": ["wikitext2", "c4", "ptb"],          # language-modeling quality
    "commonsense_zeroshot": [
        "arc_easy", "arc_challenge", "hellaswag",
        "winogrande", "piqa", "openbookqa", "boolq", "lambada_openai",
    ],
    "knowledge": ["mmlu"],                              # 5-shot, broad knowledge
    "advanced_knowledge": ["mmlu_pro"],                 # harder 10-choice reasoning
    "multilingual": ["mmlu_prox_en", "mmlu_prox_lite_en"],  # MMLU-ProX (29 langs)
    "reasoning": ["gsm8k"],                             # math word problems
    "truthfulness": ["truthfulqa_mc2"],                # reliability
}

# MMLU-ProX (EMNLP 2025) — a multilingual, reasoning-focused extension of MMLU-Pro
# with the same 11,829 parallel questions per language. Great for checking whether
# compression hurt reasoning/multilingual ability beyond what perplexity reveals.
# The `lite` version (658 questions/language) is ideal for fast compressed-model checks.
MMLU_PROX_LANGS = [
    "en", "zh", "ja", "ko", "fr", "de", "es", "pt", "ar", "th", "hi", "bn", "sw",
    "id", "it", "vi", "ru", "uk", "cs", "pl", "ne", "mr", "te", "af", "yo", "wo",
    "zu", "ig", "ha",
]


def mmlu_prox_tasks(langs=("en",), lite=True):
    """Build MMLU-ProX lm-eval-harness task ids for the given languages.

    `lite=True` uses the 658-questions/language subset (fast, recommended for
    iterating on compressed models); `lite=False` uses the full 11,829/language.
    """
    bad = [l for l in langs if l not in MMLU_PROX_LANGS]
    if bad:
        raise ValueError(f"Unsupported MMLU-ProX language(s): {bad}. "
                         f"Choose from {MMLU_PROX_LANGS}")
    prefix = "mmlu_prox_lite_" if lite else "mmlu_prox_"
    return [f"{prefix}{l}" for l in langs]


# A sensible default subset that runs reasonably fast yet is widely reported.
DEFAULT_ZEROSHOT_SUITE = [
    "arc_easy", "arc_challenge", "hellaswag", "winogrande", "piqa",
]


@torch.no_grad()
def perplexity_from_ids(model, input_ids, stride=512, max_length=None, device="cpu"):
    """Sliding-window perplexity from a (1, seq_len) tensor of token ids.

    This is the method used by Hugging Face and the quantization papers: slide a
    context window of `max_length` across the sequence in steps of `stride`, only
    scoring the newly revealed `stride` tokens each step so no token is scored
    without full context. Returns a float perplexity (lower is better).
    """
    model = model.to(device).eval()
    if max_length is None:
        cfg = getattr(model, "config", None)
        max_length = getattr(cfg, "n_positions", None) or getattr(
            cfg, "max_position_embeddings", 1024) or 1024
        max_length = min(max_length, 1024)

    seq_len = input_ids.size(1)
    nlls = []
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end                     # tokens newly scored
        ids = input_ids[:, begin:end].to(device)
        target = ids.clone()
        target[:, :-trg_len] = -100                  # ignore the context part
        out = model(ids, labels=target)
        # out.loss is mean over scored tokens; weight by count to aggregate.
        nlls.append(out.loss.float() * trg_len)
        prev_end = end
        if end == seq_len:
            break
    total_tokens = prev_end
    if total_tokens == 0:
        raise ValueError("Empty input — cannot compute perplexity.")
    return math.exp(torch.stack(nlls).sum().item() / total_tokens)


def evaluate_perplexity(model, tokenizer, text, stride=512, max_length=None, device="cpu"):
    """Tokenize `text` and compute its perplexity under `model` (HF causal LM)."""
    enc = tokenizer(text, return_tensors="pt")
    return perplexity_from_ids(model, enc["input_ids"], stride=stride,
                               max_length=max_length, device=device)


_FALLBACK_TEXT = (
    "Model compression reduces the size and latency of neural networks while "
    "trying to preserve their accuracy. Quantization lowers numerical precision, "
    "pruning removes redundant weights, and knowledge distillation trains a "
    "smaller student to imitate a larger teacher. Perplexity on held-out text is "
    "the standard quick measure of whether a compressed language model still "
    "models language well. " * 40
)


def load_wikitext2(split="test"):
    """Return WikiText-2 raw text for perplexity eval.

    Uses Hugging Face `datasets` if available (the real benchmark corpus); if it
    isn't installed or there's no network, returns a small public fallback string
    and warns — so examples still run offline (the number just isn't comparable).
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        return "\n\n".join(t for t in ds["text"] if t.strip())
    except Exception as e:  # no datasets package or no network
        warnings.warn(
            f"Could not load real WikiText-2 ({e}); using a small offline fallback "
            "corpus. Install `datasets` for the comparable benchmark number.")
        return _FALLBACK_TEXT


def lm_eval_command(model_path, tasks=None, device="cuda:0", batch_size="auto",
                    num_fewshot=None):
    """Build the EleutherAI lm-evaluation-harness CLI to evaluate a model.

    After exporting an autofollowdown-compressed model (e.g. to a HF folder),
    run the returned command to get the full zero-shot/few-shot accuracy suite.
    Install with `pip install lm_eval`.
    """
    tasks = tasks or DEFAULT_ZEROSHOT_SUITE
    parts = [
        "lm_eval", "--model hf",
        f"--model_args pretrained={model_path}",
        f"--tasks {','.join(tasks)}",
        f"--device {device}",
        f"--batch_size {batch_size}",
    ]
    if num_fewshot is not None:
        parts.append(f"--num_fewshot {num_fewshot}")
    return " ".join(parts)
