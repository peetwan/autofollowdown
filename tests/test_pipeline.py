"""Tests for the one-command compress_and_benchmark workflow + CompressionStudy."""

import os
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from autofollowdown import CompressionStudy, compress_and_benchmark


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.relu(self.conv(x))
        return self.fc2(self.relu(self.fc1(torch.flatten(x, 1))))


def _loader(n=32):
    x = torch.randn(n, 1, 8, 8)
    y = torch.randint(0, 10, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=8)


def test_compress_and_benchmark_builds_all_variants():
    study = compress_and_benchmark(CNN(), eval_loader=_loader())
    assert isinstance(study, CompressionStudy)
    # baseline + the 3 default methods
    for name in ("baseline", "int8 dynamic", "prune 50%", "prune+quantize"):
        assert name in study.names


def test_recommended_is_a_known_variant():
    study = compress_and_benchmark(CNN(), eval_loader=_loader())
    assert study.recommended in study.names


def test_pick_returns_runnable_model():
    study = compress_and_benchmark(CNN(), eval_loader=_loader())
    model = study.pick("prune+quantize")
    out = model(torch.randn(2, 1, 8, 8))
    assert out.shape == (2, 10)


def test_best_returns_recommended_model():
    study = compress_and_benchmark(CNN(), eval_loader=_loader())
    best = study.best()
    assert best(torch.randn(1, 1, 8, 8)).shape == (1, 10)


def test_pick_unknown_raises():
    study = compress_and_benchmark(CNN(), eval_loader=_loader())
    with pytest.raises(KeyError):
        study.pick("does-not-exist")


def test_export_writes_loadable_model():
    study = compress_and_benchmark(CNN(), eval_loader=_loader())
    path = tempfile.mktemp(suffix=".pt")
    try:
        study.export(study.recommended, path)
        reloaded = torch.load(path, weights_only=False)
        assert reloaded(torch.randn(1, 1, 8, 8)).shape == (1, 10)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_example_input_is_inferred_without_eval_loader():
    # No eval_loader, no example_input → still benchmarks size/latency.
    study = compress_and_benchmark(CNN())
    rows = study.report()
    assert all(r["size_mb"] > 0 for r in rows)
    assert all(r["accuracy"] is None for r in rows)  # no labels provided


def test_add_external_variant():
    study = compress_and_benchmark(CNN(), eval_loader=_loader())
    study.add("my distilled student", CNN())
    assert "my distilled student" in study.names
