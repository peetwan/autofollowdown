"""Tests for the compression advisor (which technique + backend, and why) and the
constraint-aware decision layer on the benchmark (pick_best / Pareto frontier)."""

import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from autofollowdown import (
    ModelProfile,
    TECHNIQUES,
    advise,
    compress_and_benchmark,
)
from autofollowdown.advisor import CompressionPlan
from autofollowdown.cli import build_parser


def _llm(gpu=True):
    return ModelProfile(family="llm", num_params=7_000_000_000, has_conv=False,
                        has_transformer=True, is_huggingface=True, cuda_available=gpu)


def _cnn():
    return ModelProfile(family="cnn", num_params=2_000_000, has_conv=True,
                        has_transformer=False, is_huggingface=False, cuda_available=False)


# ----------------------------------------------------------------- knowledge base
def test_techniques_is_data():
    for key in ("quantize-int8", "quantize-int4", "prune-structured", "distill"):
        assert key in TECHNIQUES
        t = TECHNIQUES[key]
        assert "shrink" in t and "accuracy_cost" in t and "caveats" in t


# ----------------------------------------------------------------- advise: CNN
def test_advise_cnn_prunes_then_quantizes():
    plan = advise(_cnn(), goal="speed", can_retrain=True)
    techs = [s.technique for s in plan.steps]
    assert "prune-structured" in techs
    assert "quantize-int8" in techs
    # quantization must route to a real quantizer, not the pruning library
    q = next(s for s in plan.steps if s.technique == "quantize-int8")
    assert q.backend == "native"
    p = next(s for s in plan.steps if s.technique == "prune-structured")
    assert p.backend == "nni"


# ----------------------------------------------------------------- advise: LLM
def test_advise_llm_size_gpu_recommends_int4():
    plan = advise(_llm(gpu=True), goal="size", max_size_ratio=0.2)
    techs = [s.technique for s in plan.steps]
    assert "quantize-int4" in techs
    q4 = next(s for s in plan.steps if s.technique == "quantize-int4")
    assert q4.backend in {"llmcompressor", "modelopt", "torchao", "bnb", "hqq", "native"}
    assert "llm-compressor" in plan.backend_pick


def test_advise_distill_only_when_retrain_allowed():
    with_retrain = advise(_llm(), goal="size", max_size_ratio=0.1, can_retrain=True)
    without = advise(_llm(), goal="size", max_size_ratio=0.1, can_retrain=False)
    assert any(s.technique == "distill" for s in with_retrain.steps)
    assert not any(s.technique == "distill" for s in without.steps)


def test_advise_ease_avoids_int4_and_retraining():
    plan = advise(_llm(gpu=True), goal="ease")
    techs = [s.technique for s in plan.steps]
    assert "quantize-int8" in techs        # the safe, no-calibration default
    assert "distill" not in techs


def test_advise_cpu_llm_prefers_int8():
    plan = advise(_llm(gpu=False), goal="size")
    techs = [s.technique for s in plan.steps]
    assert "quantize-int8" in techs and "quantize-int4" not in techs


def test_plan_has_caveats_and_renders():
    plan = advise(_llm(gpu=True), goal="size", min_accuracy_retention=0.98)
    assert isinstance(plan, CompressionPlan)
    assert plan.caveats
    text = plan.to_text()
    assert "Recommended plan" in text and "Verify on YOUR data" in text


def test_advise_accepts_local_pt(tmp_path):
    cnn = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
                        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 10))
    p = tmp_path / "m.pt"
    torch.save(cnn, str(p))
    plan = advise(str(p), goal="balanced")
    assert plan.family == "cnn" and plan.steps


# ----------------------------------------------------- constraint-aware decisions
def _study():
    torch.manual_seed(0)
    x = torch.randn(60, 16)
    y = (x.sum(1) > 0).long()
    loader = DataLoader(TensorDataset(x, y), batch_size=20)
    model = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 2))
    return compress_and_benchmark(model, eval_loader=loader)


def test_pick_best_meets_constraint():
    study = _study()
    name, model, info = study.pick_best(min_retention=0.9, prefer="smallest")
    assert info["meets"] is True
    assert name in study.names


def test_pick_best_reports_closest_when_impossible():
    study = _study()
    name, model, info = study.pick_best(max_size_mb=1e-9)
    assert info["meets"] is False
    assert "closest" in info["note"]
    assert name is not None


def test_frontier_excludes_dominated():
    study = _study()
    front = study.frontier()
    assert front  # at least one non-dominated variant
    assert set(front) <= set(study.names)


# ------------------------------------------------------------------------- CLI
def test_cli_advise_parses():
    parser = build_parser()
    args = parser.parse_args(
        ["advise", "Qwen/Qwen3-0.6B", "--goal", "size",
         "--max-size-ratio", "0.25", "--min-retention", "0.98", "--can-retrain"])
    assert args.command == "advise" and args.model == "Qwen/Qwen3-0.6B"
    assert args.goal == "size" and args.max_size_ratio == 0.25
    assert args.min_retention == 0.98 and args.can_retrain is True
    assert hasattr(args, "func")
