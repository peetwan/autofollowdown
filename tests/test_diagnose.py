"""Tests for the symptom-first diagnose feature ("I can't run this model")."""

import pytest

from autofollowdown import ModelProfile, diagnose, memory_needs
from autofollowdown.cli import build_parser
from autofollowdown.diagnose import DEVICE_PRESETS, Diagnosis


def _llm(params, gpu=True):
    return ModelProfile(family="llm", num_params=int(params), has_conv=False,
                        has_transformer=True, is_huggingface=True, cuda_available=gpu)


# --------------------------------------------------------------- memory model
def test_memory_needs_orders_by_precision():
    n = memory_needs(7e9)
    assert n["fp16"] > n["int8"] > n["int4"]
    assert n["int4"] > 0


def test_memory_needs_zero_params():
    n = memory_needs(0)
    assert all(v >= 0 for v in n.values())


# --------------------------------------------------- the prescription logic
def test_7b_on_8gb_recommends_int4():
    d = diagnose(_llm(7e9), problem="won't-fit", vram_gb=8)
    assert d.fits["int4"] and not d.fits["int8"]
    assert "INT4" in d.verdict or "4-bit" in d.verdict
    assert d.commands and any("compress" in c for c in d.commands)


def test_13b_class_on_24gb_fits_int8():
    d = diagnose(_llm(13e9), problem="won't-fit", vram_gb=24)
    assert d.fits["int8"]


def test_70b_on_8gb_is_too_big_even_at_int4_so_distill():
    d = diagnose(_llm(70e9), problem="oom", vram_gb=8)
    assert not d.fits["int4"]
    assert "distill" in d.verdict.lower() or any(
        s.technique == "distill" for s in d.plan.steps)


def test_small_model_already_fits():
    d = diagnose(_llm(0.5e9), problem="won't-fit", vram_gb=16)
    assert d.fits["fp16"]
    assert "fits" in d.verdict.lower() or "already" in d.verdict.lower()


def test_edge_device_preset_used():
    d = diagnose(_llm(7e9, gpu=False), problem="edge", device="raspberry-pi-5")
    assert "Raspberry Pi 5" in d.budget_label
    assert d.budget_gb == DEVICE_PRESETS["raspberry-pi-5"][0]


def test_too_slow_gives_speed_plan():
    d = diagnose(_llm(7e9), problem="too-slow", vram_gb=16)
    assert "fast" in d.verdict.lower() or "speed" in d.verdict.lower()
    assert any("advise" in c and "speed" in c for c in d.commands)


def test_unknown_device_raises():
    with pytest.raises(ValueError):
        diagnose(_llm(7e9), device="my-toaster")


def test_diagnosis_renders_text():
    d = diagnose(_llm(7e9), problem="won't-fit", vram_gb=8)
    assert isinstance(d, Diagnosis)
    text = d.to_text()
    assert "Will it fit?" in text and "Run next:" in text


def test_diagnose_accepts_local_pt(tmp_path):
    import torch
    import torch.nn as nn
    m = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))
    p = tmp_path / "m.pt"
    torch.save(m, str(p))
    d = diagnose(str(p), problem="too-big")
    assert d.commands


# ----------------------------------------------------------------------- CLI
def test_cli_diagnose_parses():
    parser = build_parser()
    args = parser.parse_args(
        ["diagnose", "Qwen/Qwen3-0.6B", "--problem", "won't-fit", "--vram", "8"])
    assert args.command == "diagnose" and args.model == "Qwen/Qwen3-0.6B"
    assert args.problem == "won't-fit" and args.vram == 8.0
    assert hasattr(args, "func")


def test_cli_diagnose_device_and_listing():
    parser = build_parser()
    args = parser.parse_args(["diagnose", "x.pt", "--device", "raspberry-pi-5"])
    assert args.device == "raspberry-pi-5"
    bare = parser.parse_args(["diagnose"])
    assert bare.model is None  # no-model → prints guidance
