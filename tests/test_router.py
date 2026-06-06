"""Tests for the capability-driven router: new backends, declarative scoring,
and goal-aware routing (no hardcoded per-backend rules)."""

import pytest

from autofollowdown import (
    ModelProfile,
    all_backends,
    get_backend,
    rank_backends,
)
from autofollowdown.backends import GOAL_AVOID, GOAL_TRAITS, Capability


def _llm(hf=True, cuda=True):
    return ModelProfile(family="llm", num_params=7_000_000_000, has_conv=False,
                        has_transformer=True, is_huggingface=hf, cuda_available=cuda)


# ----------------------------------------------------------- new backends exist
def test_new_backends_registered():
    aliases = {b.alias for b in all_backends()}
    assert {"torchao", "bnb", "hqq"} <= aliases
    # full set, native always first (tiebreak order)
    assert all_backends()[0].alias == "native"


def test_new_backend_aliases_resolve():
    assert get_backend("torchao").alias == "torchao"
    assert get_backend("bnb").alias == "bnb"
    assert get_backend("bitsandbytes").alias == "bnb"   # exact name match
    assert get_backend("hqq").alias == "hqq"


def test_every_backend_declares_capability():
    for b in all_backends():
        assert isinstance(b.capability, Capability)
        assert b.capability.families  # non-empty fitness map


# --------------------------------------------------- declarative scoring basics
def test_llm_compressor_tops_llm_for_balanced():
    recs = rank_backends(_llm(), "balanced")
    top = max(recs, key=lambda r: r.score)
    assert "llm-compressor" in top.backend


def test_new_backends_rank_for_llms():
    backends = {r.backend for r in rank_backends(_llm())}
    assert any("torchao" in b for b in backends)
    assert any("HQQ" in b for b in backends)


def test_score_zero_when_family_not_in_capability():
    # bitsandbytes declares only llm/transformer — it should not score on a CNN.
    cnn = ModelProfile(family="cnn", num_params=1000, has_conv=True,
                       has_transformer=False, is_huggingface=False, cuda_available=False)
    bnb = get_backend("bnb")
    assert bnb.score(cnn) == 0.0


# ----------------------------------------------------------- goal-aware routing
def _ranked_aliases(goal):
    recs = sorted(rank_backends(_llm(), goal), key=lambda r: r.score, reverse=True)
    name_to_alias = {b.name: b.alias for b in all_backends()}
    return [name_to_alias[r.backend] for r in recs]


def test_goal_changes_the_ranking():
    # Different goals must produce different orderings — the router actually routes.
    assert _ranked_aliases("size") != _ranked_aliases("ease")


def test_ease_goal_avoids_calibration_backends():
    # For "ease", a no-calibration backend should out-rank llm-compressor (which
    # needs a calibration set) — driven by GOAL_AVOID, not a hardcoded rule.
    order = _ranked_aliases("ease")
    assert order.index("torchao") < order.index("llmcompressor")


def test_goal_trait_maps_are_data():
    # The goal preferences live as plain data the scorer reads.
    assert "ease" in GOAL_TRAITS and "calibrated" in GOAL_AVOID["ease"]
    assert GOAL_TRAITS["size"] and GOAL_TRAITS["speed"]


def test_unknown_goal_falls_back_gracefully():
    recs = rank_backends(_llm(), "nonsense-goal")
    assert recs and max(recs, key=lambda r: r.score).score > 0


# --------------------------------------------------- hardware feasibility intact
def test_cuda_only_backend_not_runnable_without_gpu():
    recs = rank_backends(_llm(cuda=False))
    bnb = next((r for r in recs if "bitsandbytes" in r.backend), None)
    assert bnb is not None and not bnb.runnable   # needs CUDA, none here
