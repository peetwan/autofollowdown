"""Tests for the audit fixes: pickle-safety, quality-aware (no fabricated retention)
recommendations, honest memory math, and the string-input path."""

import copy

import pytest
import torch
import torch.nn as nn

from autofollowdown import Benchmark, ModelCompressor
from autofollowdown.ingestion import load_model
from autofollowdown.profiler import profile_checkpoint


# ---------------------------------------------------------- security: .pt loading
def test_profile_checkpoint_safe_on_state_dict(tmp_path):
    cnn = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(), nn.Flatten(), nn.Linear(8, 4))
    p = tmp_path / "sd.pt"
    torch.save(cnn.state_dict(), str(p))           # safe state_dict
    prof = profile_checkpoint(str(p))              # no pickle execution
    assert prof.family == "cnn" and prof.num_params > 0
    assert prof.detail.get("from_checkpoint") is True


def test_profile_checkpoint_refuses_pickled_module_by_default(tmp_path):
    m = nn.Sequential(nn.Linear(8, 8))
    p = tmp_path / "full.pt"
    torch.save(m, str(p))                          # whole-module pickle
    with pytest.raises(ValueError, match="allow.pickle|pickled"):
        profile_checkpoint(str(p))                 # must NOT silently unpickle
    # opt-in works for a trusted file
    prof = profile_checkpoint(str(p), allow_pickle=True)
    assert prof.num_params > 0


def test_load_model_refuses_pt_without_allow_pickle(tmp_path):
    m = nn.Sequential(nn.Linear(4, 2))
    p = tmp_path / "m.pt"
    torch.save(m, str(p))
    with pytest.raises(ValueError, match="pickled|allow_pickle"):
        load_model(str(p))
    assert load_model(str(p), allow_pickle=True)["type"] == "pytorch"


def test_load_model_passthrough_module():
    m = nn.Linear(4, 2)
    assert load_model(m)["type"] == "pytorch" and load_model(m)["model"] is m


# -------------------------------------------- quality-aware, never-fabricated picks
def _bench(quality_fn=None, eval_loader=None):
    # Linear layers big enough that INT8 dynamic visibly shrinks the model.
    b = Benchmark(torch.randn(4, 64), eval_loader=eval_loader, quality_fn=quality_fn)
    base = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 64))
    b.measure(base, "baseline")
    q = ModelCompressor(copy.deepcopy(base)).quantize(approach="dynamic").model
    b.measure(q, "int8")
    return b


def test_no_quality_signal_means_no_recommendation():
    # No eval_loader and no quality_fn → must NOT stamp the smallest as "recommended".
    b = _bench()
    assert b.best_picks()["recommended"] is None
    assert "No quality measured" in b.summary()


def test_quality_constraint_refuses_when_unmeasurable():
    b = _bench()
    res = b.pick_best(min_retention=0.98)
    assert res["meets"] is False and "cannot check" in res["note"]


def test_quality_fn_recommendation_is_quality_aware():
    # baseline perplexity good (10), the int8 variant "wrecked" (1000) → the recommender
    # must NOT recommend the wrecked-but-smaller variant.
    ppls = iter([10.0, 1000.0])
    b = _bench(quality_fn=lambda m: next(ppls))
    rec = b.best_picks()["recommended"]
    assert rec is not None and rec["name"] == "baseline"
    # and a retention floor now actually filters
    assert b.pick_best(min_retention=0.9)["row"]["name"] == "baseline"


def test_quality_fn_keeps_good_variant():
    # int8 barely changes quality (10 -> 10.2) and is much smaller → it IS recommended.
    ppls = iter([10.0, 10.2])
    b = _bench(quality_fn=lambda m: next(ppls))
    assert b.best_picks()["recommended"]["name"] == "int8"


# ----------------------------------------------------- honest memory / diagnose
def test_memory_needs_kv_is_precision_independent():
    from autofollowdown.diagnosis import memory_needs
    n = memory_needs(7e9)
    assert n["fp16"] > n["int8"] > n["int4"]
    # KV (fp16) + overhead don't shrink with weight bits, so int4 isn't 1/4 of fp16
    assert n["int4"] > n["fp16"] / 4


def test_diagnose_cpu_llm_not_steered_to_gpu_int4():
    from autofollowdown import ModelProfile, diagnose
    p = ModelProfile(family="llm", num_params=7e9, has_conv=False, has_transformer=True,
                     is_huggingface=True, cuda_available=False)
    d = diagnose(p, problem="won't-fit", device="laptop-cpu")
    assert not any(s.technique == "quantize-int4" for s in d.plan.steps)


def test_diagnose_big_model_on_cpu_flags_slow():
    from autofollowdown import ModelProfile, diagnose
    p = ModelProfile(family="llm", num_params=13e9, has_conv=False, has_transformer=True,
                     is_huggingface=True, cuda_available=False)
    d = diagnose(p, problem="won't-fit", device="laptop-cpu")
    assert any("tok/s" in n for n in d.notes)


# ------------------------------------------------------ safetensors (safe) export
def test_safetensors_export_and_safe_reprofile(tmp_path):
    cnn = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
                        nn.Flatten(), nn.Linear(8 * 30 * 30, 10))
    out = tmp_path / "m.safetensors"
    ModelCompressor(copy.deepcopy(cnn)).prune(0.5, "unstructured").export(str(out), "safetensors")
    assert out.exists()
    # re-profiling a safetensors needs NO allow-pickle and executes no code
    prof = profile_checkpoint(str(out))
    assert prof.family == "cnn" and prof.detail.get("format") == "safetensors"
    # the weights load straight back into the architecture
    from safetensors.torch import load_file
    cnn.load_state_dict(load_file(str(out)))


def test_safetensors_refuses_quantized(tmp_path):
    m = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 8))
    q = ModelCompressor(copy.deepcopy(m)).quantize(approach="dynamic")
    with pytest.raises(ValueError, match="safetensors|format='pt'"):
        q.export(str(tmp_path / "q.safetensors"), "safetensors")


def test_study_exports_safetensors(tmp_path):
    import torch as _t
    from autofollowdown import compress_and_benchmark
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
    study = compress_and_benchmark(model, methods=["prune 50%"])
    out = tmp_path / "v.safetensors"
    study.export("prune 50%", str(out), format="safetensors")
    assert out.exists()


def test_ingestion_safetensors_is_clear_error(tmp_path):
    from safetensors.torch import save_model
    from autofollowdown.ingestion import load_model
    p = tmp_path / "w.safetensors"
    save_model(nn.Linear(4, 2), str(p))
    with pytest.raises(ValueError, match="weights-only|architecture"):
        load_model(str(p))


def test_cli_format_accepts_safetensors():
    from autofollowdown.cli import build_parser
    args = build_parser().parse_args(["compress", "m.pt", "--format", "safetensors", "--yes"])
    assert args.format == "safetensors"
