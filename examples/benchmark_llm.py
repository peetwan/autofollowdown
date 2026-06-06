"""Real LLM compression benchmark.

Loads a small Hugging Face causal LM, measures WikiText-2 perplexity + on-disk
size + latency, compresses it with autofollowdown (INT8 dynamic), and reports the
before/after delta. Perplexity on WikiText-2 is the headline metric used by GPTQ /
AWQ / SparseGPT / LeanQuant / LLMCBench; for the full accuracy suite (ARC,
HellaSwag, MMLU, MMLU-ProX, ...) it prints the lm-evaluation-harness command.

Defaults to `facebook/opt-125m` (nn.Linear layers that INT8 dynamic compresses;
GPT-2 uses Conv1D and barely shrinks). Needs network to download the model;
`pip install datasets` for real WikiText-2.

Run:  python3 examples/benchmark_llm.py [--model ID] [--max-chars N]
  or: autofollowdown benchmark-llm --model facebook/opt-125m
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autofollowdown.demos import llm_benchmark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/opt-125m")
    ap.add_argument("--max-chars", type=int, default=6000)
    args = ap.parse_args()
    llm_benchmark(model_id=args.model, max_chars=args.max_chars)


if __name__ == "__main__":
    main()
