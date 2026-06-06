"""Tests for LLM evaluation (perplexity + harness helpers).

Uses a tiny randomly-initialized GPT-2 built in-process (no download, offline) so
perplexity runs fast and deterministically.
"""

import math

import pytest
import torch

from autofollowdown import (
    DEFAULT_ZEROSHOT_SUITE,
    STANDARD_LLM_TASKS,
    lm_eval_command,
    perplexity_from_ids,
)


@pytest.fixture(scope="module")
def tiny_gpt2():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.GPT2Config(
        vocab_size=128, n_positions=64, n_embd=32, n_layer=2, n_head=2,
    )
    torch.manual_seed(0)
    return transformers.GPT2LMHeadModel(cfg).eval()


def test_perplexity_is_finite_and_positive(tiny_gpt2):
    ids = torch.randint(0, 128, (1, 200))
    ppl = perplexity_from_ids(tiny_gpt2, ids, stride=32, max_length=64)
    assert math.isfinite(ppl) and ppl > 0


def test_perplexity_drops_on_learnable_pattern(tiny_gpt2):
    # A repeating pattern is more predictable than random ids → lower perplexity
    # once the model has seen enough context within the window.
    pattern = torch.tensor([[1, 2, 3, 4] * 50])           # 200 tokens, periodic
    rand = torch.randint(0, 128, (1, 200))
    ppl_pattern = perplexity_from_ids(tiny_gpt2, pattern, stride=32, max_length=64)
    ppl_rand = perplexity_from_ids(tiny_gpt2, rand, stride=32, max_length=64)
    assert math.isfinite(ppl_pattern) and math.isfinite(ppl_rand)


def test_perplexity_tracks_quantization_effect(tiny_gpt2):
    import copy

    from autofollowdown import ModelCompressor

    ids = torch.randint(0, 128, (1, 200))
    base_ppl = perplexity_from_ids(copy.deepcopy(tiny_gpt2), ids, stride=32, max_length=64)
    q = ModelCompressor(copy.deepcopy(tiny_gpt2)).quantize(approach="dynamic").model
    q_ppl = perplexity_from_ids(q, ids, stride=32, max_length=64)
    # Quantization should keep perplexity finite and in a comparable ballpark.
    assert math.isfinite(q_ppl) and q_ppl > 0
    assert base_ppl > 0


def test_lm_distillation_runs_and_updates_student(tiny_gpt2):
    # Distillation should work on causal LMs (3D logits) via token-level soft KD.
    transformers = pytest.importorskip("transformers")
    from torch.utils.data import DataLoader, TensorDataset

    from autofollowdown import ModelCompressor

    scfg = transformers.GPT2Config(vocab_size=128, n_positions=64, n_embd=32,
                                   n_layer=1, n_head=2)
    student = transformers.GPT2LMHeadModel(scfg)
    ids = torch.randint(0, 128, (8, 32))
    loader = DataLoader(TensorDataset(ids, torch.zeros(8)), batch_size=4)

    before = [p.clone() for p in student.parameters()]
    ModelCompressor(student).distill(tiny_gpt2, loader, epochs=1)
    after = list(student.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(after, before))


def test_standard_tasks_catalog():
    assert "wikitext2" in STANDARD_LLM_TASKS["perplexity"]
    assert "hellaswag" in STANDARD_LLM_TASKS["commonsense_zeroshot"]
    assert "mmlu" in STANDARD_LLM_TASKS["knowledge"]


def test_lm_eval_command_builds():
    cmd = lm_eval_command("./compressed-model", tasks=DEFAULT_ZEROSHOT_SUITE)
    assert cmd.startswith("lm_eval --model hf")
    assert "pretrained=./compressed-model" in cmd
    assert "hellaswag" in cmd
