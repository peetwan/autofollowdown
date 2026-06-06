"""Tests for MMLU-ProX catalog, best-pick recommendation, CLI, and table render."""

import copy

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from autofollowdown import (
    STANDARD_LLM_TASKS,
    Benchmark,
    ModelCompressor,
    mmlu_prox_tasks,
)
from autofollowdown._term import render_table
from autofollowdown.cli import build_parser


# ----------------------------------------------------------------- MMLU-ProX
def test_mmlu_prox_in_catalog():
    assert "mmlu_prox_en" in STANDARD_LLM_TASKS["multilingual"]
    assert "mmlu_pro" in STANDARD_LLM_TASKS["advanced_knowledge"]


def test_mmlu_prox_tasks_lite_and_full():
    assert mmlu_prox_tasks(["en", "th"], lite=True) == [
        "mmlu_prox_lite_en", "mmlu_prox_lite_th"]
    assert mmlu_prox_tasks(["zh"], lite=False) == ["mmlu_prox_zh"]


def test_mmlu_prox_rejects_unknown_language():
    with pytest.raises(ValueError):
        mmlu_prox_tasks(["klingon"])


# -------------------------------------------------------------- best picks
class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _loader():
    x = torch.randn(40, 64)
    y = torch.randint(0, 10, (40,))
    return DataLoader(TensorDataset(x, y), batch_size=8)


def _bench():
    model = _Net()
    b = Benchmark(torch.randn(8, 64), eval_loader=_loader(), reference_model=model)
    b.measure(model, "baseline")
    q = ModelCompressor(copy.deepcopy(model)).quantize(approach="dynamic").model
    b.measure(q, "int8")
    return b


def test_best_picks_returns_recommended():
    picks = _bench().best_picks()
    for key in ("recommended", "smallest", "fastest"):
        assert key in picks and picks[key] is not None


def test_table_and_summary_render():
    b = _bench()
    table = b.to_table()
    assert "Model" in table and "Size MB" in table and "│" in table
    summary = b.summary()
    assert "Recommended" in summary


# ------------------------------------------------------------------- render
def test_render_table_aligns():
    out = render_table(["A", "B"], [["xx", "1"], ["y", "22"]], ["left", "right"])
    lines = out.splitlines()
    # every rendered line has the same visible width
    assert len({len(l) for l in lines}) == 1


# ---------------------------------------------------------------------- CLI
def test_cli_parser_has_subcommands():
    parser = build_parser()
    for cmd in ("info", "benchmark-vision", "benchmark-llm", "autopick"):
        args = parser.parse_args([cmd])
        assert args.command == cmd
        assert hasattr(args, "func")
