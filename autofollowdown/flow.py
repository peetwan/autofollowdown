"""The auto-first flow — one call does the whole thing, and only asks when it can't
decide for you.

`autopilot(model)` profiles → compresses every way → benchmarks → picks the best →
saves, fully automatically. The two decisions a tool genuinely *can't* make for you
— what you care about (size vs speed vs accuracy) and which variant to keep — become
a quick menu when you're at a terminal, and sensible auto-defaults otherwise (or with
`--yes`). Every auto choice is announced, and every one is overridable by a flag.

Design rule: **auto by default, choose when it matters.** Nothing here blocks a
non-interactive run; the prompts only appear for a human at a TTY.
"""

import sys

from ._term import color

# The one decision a compression tool can't read your mind on: what matters most.
GOAL_OPTIONS = [
    ("balanced", "balanced", "best size/quality trade-off (a safe default)"),
    ("size", "size", "the smallest model"),
    ("speed", "speed", "the fastest inference"),
    ("accuracy", "accuracy", "keep the most accuracy"),
    ("ease", "ease", "easiest path, no calibration"),
]

# How each goal turns into an automatic variant pick after benchmarking.
_GOAL_PREFER = {
    "size": "smallest",
    "speed": "fastest",
    "accuracy": "recommended",   # accuracy-weighted score (avoids picking the baseline)
    "balanced": "recommended",
    "ease": "recommended",
}


def _interactive(interactive):
    return sys.stdin.isatty() if interactive is None else interactive


def choose(prompt, options, recommended_idx=0, interactive=None, yes=False):
    """Auto-or-choose: return the recommended option automatically when there's no
    human to ask (non-TTY or `yes=True`), otherwise show a numbered menu defaulting
    to the recommendation.

    options: list of (label, value) or (label, value, description). Returns the value.
    """
    rec_label = options[recommended_idx][0]
    if yes or not _interactive(interactive):
        print(color(f"  🤖 {prompt} → {rec_label} (auto)", "dim"))
        return options[recommended_idx][1]

    print(color(prompt, "bold"))
    for i, opt in enumerate(options, 1):
        desc = f" — {opt[2]}" if len(opt) > 2 else ""
        tag = color("  ⟵ recommended", "green") if i - 1 == recommended_idx else ""
        print(f"  {i}. {opt[0]}{desc}{tag}")
    try:
        raw = input(f"Choice [1-{len(options)}, Enter = {recommended_idx + 1}]: ").strip()
    except EOFError:
        raw = ""
    idx = int(raw) - 1 if raw.isdigit() and 1 <= int(raw) <= len(options) else recommended_idx
    return options[idx][1]


def autopilot(model=None, goal=None, output=None, method=None, fmt="pt", epochs=8,
              yes=False, max_size_mb=None, min_retention=None, interactive=None):
    """Run the whole flow automatically; ask only at genuine decision points.

    Stages: pick a goal (auto/choose) → profile + compress every way + benchmark
    (auto) → pick the variant (auto by goal/constraints, or choose) → save.

    Returns {study, chosen, goal, info}.
    """
    from .demos import auto_study

    print(color("\n🤖 autofollowdown autopilot", "bold", "cyan"))
    print(f"   target : {model or 'offline digits demo (no model given)'}")

    # ---- Decision 1: the goal. Auto-default to balanced; ask only at a TTY. ----
    if goal is None:
        goal = choose("What do you care about most?", GOAL_OPTIONS,
                      recommended_idx=0, interactive=interactive, yes=yes)
    print(f"   goal   : {goal}   " + color("(override with --goal)", "dim") + "\n")

    # ---- Auto: profile, compress every way, benchmark. ----
    study = auto_study(model_spec=model, epochs=epochs)
    print("\n" + color("=== Results ===", "bold") + "\n")
    study.show()

    # ---- Decision 2: which variant to keep. ----
    info = None
    names = study.names
    if method:
        if method in names:
            chosen = method
            print(color(f"\n➤ Using your method: {method}", "green", "bold"))
        else:
            chosen = study.recommended
            print(color(f"\nUnknown method {method!r}; using recommended ({chosen}).", "yellow"))
    elif max_size_mb is not None or min_retention is not None:
        # Hard constraints → auto-pick the best variant that satisfies them.
        chosen, _, info = study.pick_best(max_size_mb=max_size_mb,
                                          min_retention=min_retention,
                                          prefer=_GOAL_PREFER.get(goal, "recommended"))
        flag = color("🤖 auto", "green") if info["meets"] else color("⚠ closest", "yellow")
        print(f"\n{flag}: {info['note']}")
    elif _interactive(interactive) and not yes:
        # A human is here and there are no hard constraints → let them choose.
        print(color("\nPick a variant to keep (Enter = recommended):", "bold"))
        chosen, _ = study.interactive_pick()
    else:
        # Full auto: honor the goal.
        prefer = _GOAL_PREFER.get(goal, "recommended")
        chosen, _, _ = study.pick_best(prefer=prefer)
        chosen = chosen or study.recommended
        print(color(f"\n🤖 auto: kept '{chosen}' for goal '{goal}'", "green"))

    print(color(f"\n➤ Selected: {chosen}", "green", "bold"))
    if output:
        study.export(chosen, output, format=fmt)
        print(f"Saved to {output}")
    else:
        print(color("Tip: ", "dim")
              + f"add  --method '{chosen}' --output model.pt  to save this one.")
    return {"study": study, "chosen": chosen, "goal": goal, "info": info}
