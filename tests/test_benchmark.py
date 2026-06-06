"""Tests for the metrics + benchmark engine — verifying the measurements are
real and self-consistent."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from autofollowdown import (
    Benchmark,
    ModelCompressor,
    count_parameters,
    evaluate_accuracy,
    measure_latency,
    model_disk_size_mb,
    output_agreement,
)


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _loader(n=40):
    x = torch.randn(n, 64)
    y = torch.randint(0, 10, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=8)


def test_count_parameters_tracks_sparsity():
    model = TinyNet()
    total, nonzero, sparsity = count_parameters(model)
    assert total > 0 and nonzero == total and sparsity == 0.0
    with torch.no_grad():
        list(model.parameters())[0].zero_()
    _, _, sparsity2 = count_parameters(model)
    assert sparsity2 > 0.0


def test_disk_size_is_positive():
    assert model_disk_size_mb(TinyNet()) > 0


def test_latency_and_throughput_positive():
    lat, thr = measure_latency(TinyNet(), torch.randn(8, 64), n_runs=5)
    assert lat > 0 and thr > 0


def test_accuracy_in_unit_range():
    acc = evaluate_accuracy(TinyNet(), _loader())
    assert 0.0 <= acc <= 1.0


def test_fidelity_self_is_one():
    model = TinyNet()
    assert output_agreement(model, model, _loader()) == 1.0


def test_benchmark_report_has_ratios():
    model = TinyNet()
    example = torch.randn(8, 64)
    bench = Benchmark(example, eval_loader=_loader(), reference_model=model)
    bench.measure(model, "baseline")

    import copy
    q = ModelCompressor(copy.deepcopy(model)).quantize(approach="dynamic").model
    bench.measure(q, "quantized")

    rows = bench.report()
    assert rows[0]["name"] == "baseline"
    assert "size_ratio" in rows[1] and rows[1]["size_ratio"] > 1.0  # quant is smaller
    md = bench.to_markdown()
    assert "Size×" in md and "quantized" in md
