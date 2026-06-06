"""Tests for the auto-first flow: choose() (auto vs choice), the bare-model CLI
dispatch, and autopilot() running fully unattended."""

import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from autofollowdown.cli import _normalize_argv, build_parser
from autofollowdown.flow import GOAL_OPTIONS, autopilot, choose


# ------------------------------------------------------------- choose() helper
def test_choose_auto_returns_recommended_without_prompt():
    # Non-interactive (interactive=False) must NOT prompt and must take the default.
    opts = [("balanced", "balanced"), ("size", "size"), ("speed", "speed")]
    assert choose("goal?", opts, recommended_idx=0, interactive=False) == "balanced"
    assert choose("goal?", opts, recommended_idx=1, interactive=False) == "size"


def test_choose_yes_forces_auto_even_if_interactive_flag():
    opts = [("a", "a"), ("b", "b")]
    assert choose("pick", opts, recommended_idx=1, interactive=True, yes=True) == "b"


def test_choose_reads_user_selection(monkeypatch):
    opts = [("a", "a"), ("b", "b"), ("c", "c")]
    monkeypatch.setattr("builtins.input", lambda _: "3")
    assert choose("pick", opts, recommended_idx=0, interactive=True) == "c"


def test_choose_blank_input_takes_recommended(monkeypatch):
    opts = [("a", "a"), ("b", "b")]
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert choose("pick", opts, recommended_idx=1, interactive=True) == "b"


def test_goal_options_are_well_formed():
    values = {o[1] for o in GOAL_OPTIONS}
    assert {"balanced", "size", "speed", "accuracy", "ease"} <= values


# ------------------------------------------------------ bare-model CLI dispatch
def test_normalize_argv_injects_compress_for_bare_model():
    assert _normalize_argv(["facebook/opt-125m"]) == ["compress", "facebook/opt-125m"]
    assert _normalize_argv(["model.pt", "--yes"]) == ["compress", "model.pt", "--yes"]


def test_normalize_argv_leaves_subcommands_and_options_alone():
    assert _normalize_argv(["info"]) == ["info"]
    assert _normalize_argv(["diagnose", "x"]) == ["diagnose", "x"]
    assert _normalize_argv(["--version"]) == ["--version"]
    assert _normalize_argv([]) == []


def test_compress_parser_has_auto_flags():
    parser = build_parser()
    args = parser.parse_args(["compress", "m.pt", "--goal", "size",
                              "--max-size-mb", "5", "--min-retention", "0.98", "--yes"])
    assert args.goal == "size" and args.max_size_mb == 5.0
    assert args.min_retention == 0.98 and args.yes is True


def test_auto_accepts_positional_and_model_flag():
    parser = build_parser()
    a = parser.parse_args(["auto", "facebook/opt-125m"])
    assert a.model == "facebook/opt-125m"
    b = parser.parse_args(["auto", "--model", "facebook/opt-125m"])
    assert b.model_flag == "facebook/opt-125m"


# ------------------------------------------------------ autopilot (unattended)
def _digits_like():
    # tiny trainable classifier so the offline demo path isn't needed
    torch.manual_seed(0)
    x = torch.randn(50, 16)
    y = (x.sum(1) > 0).long()
    return DataLoader(TensorDataset(x, y), batch_size=25), \
        nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 2))


def test_autopilot_full_auto_picks_for_goal():
    # yes=True → no prompts; size goal → keeps the smallest variant.
    result = autopilot(model=None, goal="size", yes=True, epochs=1)
    assert result["chosen"] in result["study"].names
    assert result["goal"] == "size"


def test_autopilot_respects_explicit_method():
    result = autopilot(model=None, method="int8 dynamic", yes=True, epochs=1)
    assert result["chosen"] == "int8 dynamic"


def test_autopilot_constraint_path_reports_meets():
    # An impossible size budget → still returns a choice, flagged as not-met.
    result = autopilot(model=None, goal="size", yes=True, epochs=1, max_size_mb=1e-9)
    assert result["info"] is not None
    assert result["info"]["meets"] is False
