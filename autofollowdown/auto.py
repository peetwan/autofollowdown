"""The auto-picker: given a model, recommend (and optionally run) the best
compression library/technique for it.

`recommend(model)` ranks every backend for the model and explains why.
`auto_compress(model)` runs the best backend that is actually available on this
machine, always falling back to the native engine so it never fails for lack of
an optional dependency.
"""

from dataclasses import dataclass

from .backends import all_backends
from .profiler import profile_model


@dataclass
class Recommendation:
    backend: str
    technique: str
    scheme: str
    score: float
    available: bool       # is the library installed?
    runnable: bool        # installed AND hardware (e.g. CUDA) is suitable
    rationale: str
    install_hint: str

    def __str__(self):
        status = "runnable" if self.runnable else (
            "installed, wrong hardware" if self.available else "not installed")
        return (f"[{self.score:.2f}] {self.backend}: {self.scheme} "
                f"({self.technique}) — {status}")


def _build(model):
    profile = profile_model(model)
    recs = []
    for b in all_backends():
        score = b.score(profile)
        if score <= 0:
            continue
        technique, scheme, rationale = b.plan(profile)
        recs.append(Recommendation(
            backend=b.name, technique=technique, scheme=scheme, score=score,
            available=b.is_available(), runnable=b.is_available() and b.device_ok(profile),
            rationale=rationale, install_hint=b.install_hint,
        ))
    # Rank: prefer what we can run now, then by fitness score.
    recs.sort(key=lambda r: (r.runnable, r.score), reverse=True)
    return profile, recs


def recommend(model):
    """Return (ModelProfile, ranked list[Recommendation]).

    The list is the *ideal* ranking by fitness; each item also says whether it is
    installed and runnable here, so you can see both "what's best" and "what's
    best that you can run right now".
    """
    return _build(model)


def explain(model):
    """Human-readable summary of the recommendation — handy for notebooks/CLI."""
    profile, recs = _build(model)
    lines = [f"Model profile: {profile}", "", "Ranked compression backends:"]
    for i, r in enumerate(recs, 1):
        mark = "→" if r.runnable else " "
        lines.append(f" {mark} {i}. {r}")
        lines.append(f"      {r.rationale}")
        if not r.available and r.install_hint:
            lines.append(f"      install: {r.install_hint}")
    best = next((r for r in recs if r.runnable), None)
    lines += ["", f"Auto-pick (runnable now): {best.backend if best else 'none'}"]
    return "\n".join(lines)


def auto_compress(model, **kwargs):
    """Profile the model, pick the best *runnable* backend, compress, and return
    (compressed_model, chosen_recommendation).

    Pass backend-specific kwargs through (e.g. calibration_data for native/ModelOpt,
    dataset for llm-compressor, dummy_input for NNI). The native backend needs none.
    """
    from .backends import get_backend
    profile, recs = _build(model)
    chosen = next((r for r in recs if r.runnable), None)
    if chosen is None:
        raise RuntimeError("No runnable backend found (native should always exist).")
    backend = get_backend(chosen.backend)
    compressed = backend.compress(model, profile, **kwargs)
    return compressed, chosen
