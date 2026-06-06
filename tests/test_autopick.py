"""Tests for the multi-library auto-picker (profiler + recommend + auto_compress).

External backends (NNI, llm-compressor, ModelOpt) aren't installed in CI, so these
verify the routing/profiling logic and that the native fallback always runs.
"""

import copy

import pytest
import torch
import torch.nn as nn

from autofollowdown import auto_compress, profile_model, recommend


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 8 * 8, 10)

    def forward(self, x):
        x = self.relu(self.c2(self.relu(self.c1(x))))
        return self.fc(torch.flatten(x, 1))


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(100, 32)
        self.attn = nn.MultiheadAttention(32, 4, batch_first=True)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        h = self.emb(x)
        a, _ = self.attn(h, h, h)
        return self.fc(a.mean(1))


def _mlp():
    return nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))


def test_profile_detects_cnn():
    p = profile_model(CNN())
    assert p.family == "cnn" and p.has_conv and not p.has_transformer


def test_profile_detects_transformer():
    p = profile_model(TinyTransformer())
    assert p.has_transformer and p.family in ("transformer", "llm")


def test_profile_detects_mlp():
    p = profile_model(_mlp())
    assert p.family == "mlp" and not p.has_conv and not p.has_transformer


def test_recommend_ranks_and_marks_runnable():
    profile, recs = recommend(CNN())
    assert profile.family == "cnn"
    assert len(recs) >= 1
    # Native is always runnable; at least one runnable recommendation must exist.
    assert any(r.runnable for r in recs)
    # NNI is the ideal CNN backend, so it should out-score native on fitness.
    nni = next((r for r in recs if "NNI" in r.backend), None)
    native = next((r for r in recs if "native" in r.backend), None)
    assert native is not None and native.runnable
    if nni is not None:
        assert nni.score >= native.score


def test_recommend_marks_uninstalled_backends_not_runnable():
    _, recs = recommend(TinyTransformer())
    for r in recs:
        if not r.available:
            assert not r.runnable
            assert r.install_hint  # tells the user how to get it


def test_auto_compress_runs_and_shrinks_cnn():
    import io

    def size_mb(m):
        b = io.BytesIO()
        torch.save(m, b)
        return b.getbuffer().nbytes / (1024 * 1024)

    model = CNN()
    before = size_mb(model)
    compressed, chosen = auto_compress(copy.deepcopy(model))
    assert chosen.runnable
    assert size_mb(compressed) < before
    out = compressed(torch.randn(1, 3, 8, 8))
    assert out.shape == (1, 10)


def test_auto_compress_picks_native_when_externals_absent():
    # On a machine without NNI/llmcompressor/modelopt, the runnable pick is native.
    _, chosen = auto_compress(copy.deepcopy(_mlp()))
    assert "native" in chosen.backend


def test_get_backend_by_alias():
    from autofollowdown import get_backend
    assert get_backend("nni").alias == "nni"
    assert get_backend("modelopt").alias == "modelopt"
    assert get_backend("llmcompressor").alias == "llmcompressor"
    assert get_backend("nvidia").alias == "modelopt"      # substring match
    assert get_backend("native").alias == "native"


def test_compress_with_native_runs():
    from autofollowdown import compress_with
    model = compress_with(copy.deepcopy(_mlp()), "native")
    assert model(torch.randn(2, 64)).shape == (2, 10)


def test_compress_with_uninstalled_backend_errors_clearly():
    from autofollowdown import compress_with
    with pytest.raises(RuntimeError, match="not installed"):
        compress_with(copy.deepcopy(_mlp()), "nni")
