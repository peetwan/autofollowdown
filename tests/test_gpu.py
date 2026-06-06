"""Tests for the GPU memory planner (gpu.py) and the `gpu` CLI command.

These run on CPU: memory_plan takes an explicit vram_gb so the heuristics are
exercised without any GPU present.
"""

import pytest

from autofollowdown import cuda_info, estimate_weight_gb, free_memory, memory_plan
from autofollowdown.cli import build_parser
from autofollowdown.gpu import summary


def test_cuda_info_has_expected_shape():
    info = cuda_info()
    for key in ("available", "name", "total_gb", "free_gb"):
        assert key in info
    assert isinstance(info["available"], bool)


def test_estimate_weight_gb():
    # 0.6B params in fp16 (2 bytes) ≈ 1.2 GB
    assert estimate_weight_gb(600e6) == pytest.approx(1.2, abs=0.01)
    assert estimate_weight_gb(0) == 0.0
    assert estimate_weight_gb(None) == 0.0


def test_free_memory_is_safe_without_gpu():
    free_memory()  # must not raise even when no CUDA / torch present


def test_plan_small_model_fits_uses_basic():
    plan = memory_plan(600e6, vram_gb=16)
    assert plan["strategy"] == "fits"
    assert plan["pipeline"] == "basic"
    assert plan["fits_on_gpu"] is True
    assert plan["sequential_targets"] is None


def test_plan_big_model_uses_sequential_onloading():
    plan = memory_plan(7e9, vram_gb=16)  # 14 GB weights > 16 GB after overhead
    assert plan["strategy"] == "sequential"
    assert plan["pipeline"] == "sequential"
    assert plan["device_map"] == "auto"
    assert plan["fits_on_gpu"] is False


def test_plan_tight_vram_onloads_per_linear():
    plan = memory_plan(7e9, vram_gb=2)
    assert plan["strategy"] == "sequential-linear"
    assert plan["sequential_targets"] == "Linear"


def test_plan_cpu_only_is_handled():
    plan = memory_plan(7e9, vram_gb=0)
    assert plan["strategy"] == "cpu"
    assert plan["pipeline"] == "sequential"
    assert "CPU" in plan["note"] or "T4" in plan["note"]


def test_plan_thresholds_are_monotonic():
    # As VRAM shrinks, the plan only ever asks for *more* memory saving, never less.
    order = {"fits": 0, "sequential": 1, "sequential-linear": 2, "offload": 3, "cpu": 3}
    prev = -1
    for vram in (32, 16, 8, 4, 2, 1, 0):
        s = order[memory_plan(7e9, vram_gb=vram)["strategy"]]
        assert s >= prev
        prev = s


def test_summary_renders_with_and_without_model():
    assert "GPU" in summary()
    text = summary(600e6, vram_gb=16)
    assert "Plan" in text and "pipeline=" in text


def test_cli_gpu_command_parses():
    parser = build_parser()
    args = parser.parse_args(["gpu", "Qwen/Qwen3-0.6B", "--vram", "16"])
    assert args.command == "gpu"
    assert args.model == "Qwen/Qwen3-0.6B"
    assert args.vram == 16.0
    assert hasattr(args, "func")

    bare = parser.parse_args(["gpu"])
    assert bare.command == "gpu" and bare.model is None
