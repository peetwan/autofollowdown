"""`autofollowdown` command-line interface.

After `pip install autofollowdown`, run everything without touching the source:

    autofollowdown diagnose Qwen/Qwen3-0.6B --problem won't-fit --vram 8   # 🩺 stuck? start here
    autofollowdown compress facebook/opt-125m -o small.pt   # ⭐ the easy one
    autofollowdown compress                                 # offline demo (no model needed)
    autofollowdown advise Qwen/Qwen3-0.6B --goal size       # which technique(s) to use + why
    autofollowdown recommend Qwen/Qwen3-0.6B --goal accuracy  # best library for your LLM + why
    autofollowdown gpu Qwen/Qwen3-0.6B                      # GPU + memory-saving plan (free-GPU friendly)
    autofollowdown info                                     # version, backends, catalog
    autofollowdown benchmark-vision                         # offline CNN benchmark
    autofollowdown benchmark-llm                            # LLM perplexity benchmark
    autofollowdown autopick                                 # best-library recommendations

`compress` is the headline command: give it a model, it compresses every way,
benchmarks them, recommends the best, and (with -o) saves the one you pick.
"""

import argparse
import sys

from . import __version__
from ._term import color


def _cmd_auto(args):
    """One command: compress every way, benchmark, and pick a method to keep."""
    from .demos import auto_study

    study = auto_study(model_spec=args.model, epochs=args.epochs)
    print("\n=== Compression study ===\n")
    study.show()

    if args.method:
        chosen = args.method
        if chosen not in study.names:
            print(color(f"\nUnknown method {chosen!r}; using recommended.", "yellow"))
            chosen = study.recommended
    elif sys.stdin.isatty() and not args.yes:
        chosen, _ = study.interactive_pick()
    else:
        chosen = study.recommended

    print(color(f"\n➤ Selected: {chosen}", "green", "bold"))
    if args.output:
        study.export(chosen, args.output, format=args.format)
        print(f"Saved to {args.output}")
    else:
        print(f"Re-run with  --method '{chosen}' --output model.pt  to save it.")


_GOAL_NOTE = {
    "accuracy": "weight-only 4-bit GPTQ/AWQ (llm-compressor) preserves accuracy best at high "
                "compression; INT8 (ModelOpt/native) is nearly lossless but larger.",
    "size": "4-bit weight-only GPTQ/AWQ (llm-compressor) gives the smallest model.",
    "speed": "INT8 + TensorRT via NVIDIA ModelOpt is fastest on an NVIDIA GPU; "
             "4-bit (llm-compressor) is fast under vLLM.",
    "balanced": "INT8 — native dynamic for portability/CPU, or ModelOpt/llm-compressor "
                "for better quality on a GPU.",
    "ease": "no-calibration, load-and-go: bitsandbytes (NF4/INT8) or HQQ/torchao are the "
            "least-effort paths and need no calibration data.",
}


def _cmd_recommend(args):
    """Find the best library for a (user-chosen) model and explain *why*, with
    optional measured benchmark evidence."""
    from ._term import render_table
    from .auto import rank_backends
    from .profiler import profile_from_pretrained, profile_model

    model = args.model
    if model.endswith((".pt", ".pth")):
        import torch
        profile = profile_model(torch.load(model, weights_only=False))
    else:
        print(f"Reading config for {model} (no weights downloaded) ...")
        profile = profile_from_pretrained(model)

    recs = rank_backends(profile, args.goal)
    pcount = f"~{profile.num_params/1e6:.0f}M" + (" (est.)" if profile.detail.get("estimated") else "")
    print(color(f"\nModel: {model}", "bold", "cyan"))
    print(f"  family={profile.family} · params={pcount} · "
          f"HF={profile.is_huggingface} · CUDA={profile.cuda_available}\n")

    rows = []
    for i, r in enumerate(sorted(recs, key=lambda r: r.score, reverse=True), 1):
        status = ("runnable here" if r.runnable else
                  ("installed, needs GPU" if r.available else "not installed"))
        rows.append([str(i), r.backend, f"{r.score:.2f}", status, r.scheme])
    print(render_table(["#", "Library", "Fit", "Status", "Method"], rows,
                       ["right", "left", "right", "left", "left"]))

    print(color("\nWhy each library ranks where it does:", "bold"))
    for r in sorted(recs, key=lambda r: r.score, reverse=True):
        hint = "" if r.available else f"   ({r.install_hint})"
        print(f"  • {r.backend}: {r.rationale}{hint}")

    ideal = max(recs, key=lambda r: r.score)
    runnable = next((r for r in recs if r.runnable), None)
    print(color("\n➤ Best library for this model: ", "green", "bold")
          + f"{ideal.backend} — {ideal.scheme}")
    if runnable and runnable.backend != ideal.backend:
        print(f"  Runnable right now: {runnable.backend} — {runnable.scheme} "
              f"(install {ideal.install_hint} for the best result)")
    print(f"  For your goal '{args.goal}': {_GOAL_NOTE.get(args.goal, _GOAL_NOTE['balanced'])}")

    if args.benchmark:
        _recommend_benchmark(model, profile, args.max_chars)


def _recommend_benchmark(model, profile, max_chars):
    """Measured evidence: what the portable native INT8 baseline actually costs on
    this model — the justification for preferring GPTQ/AWQ/ModelOpt on LLMs."""
    if not profile.is_huggingface or model.endswith((".pt", ".pth")):
        print(color("\n(Benchmark evidence supports HF model ids; skipping.)", "yellow"))
        return
    import copy
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .api import ModelCompressor
    from .llm_eval import evaluate_perplexity, load_wikitext2

    print(color("\nGathering evidence (native INT8 vs fp32 on WikiText-2) ...", "bold"))
    tok = AutoTokenizer.from_pretrained(model)
    m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32).eval()
    text = load_wikitext2("test")[:max_chars]
    base = evaluate_perplexity(m, tok, text, stride=512, max_length=1024)
    q = ModelCompressor(copy.deepcopy(m)).quantize(approach="dynamic").model
    qppl = evaluate_perplexity(q, tok, text, stride=512, max_length=1024)
    print(f"  fp32 perplexity        : {base:.2f}")
    print(f"  native INT8 perplexity : {qppl:.2f}  ({qppl - base:+.2f})")
    print(color("→ ", "green") + "the portable native INT8 baseline costs "
          f"{qppl - base:+.2f} perplexity here — which is exactly why, for LLMs, "
          "autofollowdown recommends weight-only GPTQ/AWQ (llm-compressor) or "
          "calibrated ModelOpt over native dynamic INT8.")


def _cmd_diagnose(args):
    """Symptom-first help: 'I can't run this model'. Tell it the problem + your
    hardware and it says exactly what to do, with a fit table and the next command."""
    from .diagnose import DEVICE_PRESETS, PROBLEMS, diagnose

    if args.list_devices:
        print(color("Device presets:", "bold"))
        for k, (gb, label) in DEVICE_PRESETS.items():
            print(f"  {k:18} {label}")
        return

    if not args.model:
        print(color("Stuck? Tell me the model and your problem:", "bold", "cyan"))
        print("  autofollowdown diagnose Qwen/Qwen3-0.6B --problem won't-fit --vram 8")
        print("  autofollowdown diagnose meta-llama/Llama-3.1-8B --device raspberry-pi-5")
        print("  autofollowdown diagnose <model> --problem too-slow")
        print(f"\nProblems: {', '.join(repr(p) for p in PROBLEMS)}")
        print("Devices : autofollowdown diagnose --list-devices")
        return

    d = diagnose(args.model, problem=args.problem, vram_gb=args.vram,
                 device=args.device, can_retrain=args.can_retrain,
                 target_size_mb=args.target_size_mb)
    print(color(f"\nModel: {args.model}\n", "bold", "cyan"))
    print(d.to_text(color=color))


def _cmd_advise(args):
    """Recommend WHICH technique(s) + backend to use for a model, and why — the
    'where do I even start: quantize, prune, or distill?' decision, in one place."""
    from .advisor import advise

    if args.model.endswith((".pt", ".pth")):
        print(f"Profiling {args.model} ...")
    else:
        print(f"Reading config for {args.model} (no weights downloaded) ...")
    plan = advise(args.model, goal=args.goal,
                  max_size_ratio=args.max_size_ratio,
                  min_accuracy_retention=args.min_retention,
                  can_retrain=args.can_retrain,
                  hardware=args.hardware)
    print(color(f"\nModel: {args.model}", "bold", "cyan")
          + f"  (family={plan.family}, goal={plan.goal})\n")
    print(plan.to_text(color=color))


def _cmd_gpu(args):
    """Show the current GPU and the memory-saving plan for a given model — the
    settings that let the heavy backends run on a free/small GPU."""
    from .gpu import cuda_info, memory_plan

    info = cuda_info()
    if info["available"]:
        print(color(f"\nGPU: {info['name']}", "bold", "cyan"))
        print(f"  {info['free_gb']:.1f} GB free / {info['total_gb']:.1f} GB total\n")
    else:
        print(color("\nGPU: none (CPU-only)", "bold", "yellow"))
        print("  Native INT8 runs here; the GPU backends need a free Colab/Kaggle T4.\n")

    if not args.model:
        print("Pass a model id to see its memory-saving plan, e.g.:")
        print("  autofollowdown gpu Qwen/Qwen3-0.6B")
        return

    from .profiler import profile_from_pretrained, profile_model
    if args.model.endswith((".pt", ".pth")):
        import torch
        profile = profile_model(torch.load(args.model, weights_only=False))
    else:
        print(f"Reading config for {args.model} (no weights downloaded) ...")
        profile = profile_from_pretrained(args.model)

    plan = memory_plan(profile.num_params or 0, vram_gb=args.vram)
    from .gpu import estimate_weight_gb
    print(color(f"\nModel: {args.model}", "bold"))
    print(f"  ~{(profile.num_params or 0)/1e6:.0f}M params "
          f"(~{estimate_weight_gb(profile.num_params or 0):.1f} GB in fp16)\n")
    print(color("Memory-saving plan (llm-compressor):", "bold"))
    print(f"  strategy           : {plan['strategy']}")
    print(f"  pipeline           : {plan['pipeline']}")
    print(f"  sequential_targets : {plan['sequential_targets'] or '(default decoder layers)'}")
    print(f"  device_map         : {plan['device_map']}")
    print(color("→ ", "green") + plan["note"])
    print("\nUse it directly:")
    print(color(f"  compress_with(model, 'llmcompressor', pipeline='{plan['pipeline']}', "
                f"sequential_targets={plan['sequential_targets']!r})", "cyan"))


def _cmd_info(args):
    from .backends import all_backends
    from .llm_eval import STANDARD_LLM_TASKS

    print(color(f"autofollowdown {__version__}", "bold", "cyan"))
    print("Unified quantization · pruning · distillation, with real benchmarks.\n")
    print(color("Compression backends:", "bold"))
    for b in all_backends():
        status = "installed" if b.is_available() else f"not installed ({b.install_hint})"
        print(f"  • {b.name:<38} {status}")
    print("\n" + color("LLM benchmark tasks (lm-eval-harness ids):", "bold"))
    for group, tasks in STANDARD_LLM_TASKS.items():
        print(f"  {group:<22} {', '.join(tasks)}")
    print("\nRun `autofollowdown benchmark-vision` for a live demo.")


def _cmd_vision(args):
    from .demos import vision_benchmark
    vision_benchmark(epochs=args.epochs, report=args.report)


def _cmd_llm(args):
    from .demos import llm_benchmark
    llm_benchmark(model_id=args.model, max_chars=args.max_chars)


def _cmd_autopick(args):
    from .demos import autopick_demo
    autopick_demo()


def build_parser():
    p = argparse.ArgumentParser(
        prog="autofollowdown",
        description="Unified model compression (quantize/prune/distill) + real benchmarks.")
    p.add_argument("--version", action="version", version=f"autofollowdown {__version__}")
    sub = p.add_subparsers(dest="command")

    # The easy, headline command: `autofollowdown compress <model>`
    co = sub.add_parser(
        "compress",
        help="EASIEST: compress a model, benchmark every method, and pick the best")
    co.add_argument("model", nargs="?", default=None,
                    help="Hugging Face model id or .pt path (omit for the offline demo)")
    co.add_argument("-o", "--output", default=None, help="save the chosen model here")
    co.add_argument("-m", "--method", default=None,
                    help="variant to keep (default = recommended)")
    co.add_argument("--format", default="pt", choices=["pt", "onnx"])
    co.add_argument("--epochs", type=int, default=8, help="epochs for the offline demo")
    co.add_argument("--yes", action="store_true", help="take the recommendation, no prompt")
    co.set_defaults(func=_cmd_auto)

    a = sub.add_parser(
        "auto",
        help="alias of `compress` (uses --model instead of a positional argument)")
    a.add_argument("--model", default=None,
                   help="Hugging Face model id; omit for the offline digits demo")
    a.add_argument("--method", default=None,
                   help="variant to keep (e.g. 'prune+quantize'); default = recommended")
    a.add_argument("--output", default=None, help="save the chosen model to this path")
    a.add_argument("--format", default="pt", choices=["pt", "onnx"])
    a.add_argument("--epochs", type=int, default=8, help="epochs for the digits demo")
    a.add_argument("--yes", action="store_true",
                   help="don't prompt; take the recommended method")
    a.set_defaults(func=_cmd_auto)

    r = sub.add_parser(
        "recommend",
        help="find the best library for a model (esp. LLMs) and explain why, with evidence")
    r.add_argument("model", help="Hugging Face model id (config only) or .pt path")
    r.add_argument("--goal", default="balanced",
                   choices=["balanced", "accuracy", "size", "speed", "ease"],
                   help="what you care about most")
    r.add_argument("--benchmark", action="store_true",
                   help="download the model and measure native-INT8 vs fp32 perplexity as evidence")
    r.add_argument("--max-chars", type=int, default=3000, help="eval text length for --benchmark")
    r.set_defaults(func=_cmd_recommend)

    dg = sub.add_parser(
        "diagnose",
        help="START HERE if you're stuck: \"I can't run this model\" → exactly what to do")
    dg.add_argument("model", nargs="?", default=None,
                    help="Hugging Face model id (config only) or .pt path")
    dg.add_argument("--problem", default="won't-fit",
                    choices=["won't-fit", "oom", "too-slow", "too-big", "edge", "cost"],
                    help="what's going wrong")
    dg.add_argument("--vram", type=float, default=None,
                    help="how much memory you have, in GB (e.g. 8)")
    dg.add_argument("--device", default=None,
                    help="target preset, e.g. raspberry-pi-5 / gpu-8gb / phone "
                         "(see --list-devices)")
    dg.add_argument("--target-size-mb", type=float, default=None,
                    help="size budget in MB (for --problem too-big)")
    dg.add_argument("--can-retrain", action="store_true",
                    help="allow distillation / pruning fine-tune")
    dg.add_argument("--list-devices", action="store_true", help="list device presets and exit")
    dg.set_defaults(func=_cmd_diagnose)

    ad = sub.add_parser(
        "advise",
        help="recommend WHICH technique (quantize/prune/distill) + backend to use, and why")
    ad.add_argument("model", help="Hugging Face model id (config only) or .pt path")
    ad.add_argument("--goal", default="balanced",
                    choices=["balanced", "accuracy", "size", "speed", "ease"],
                    help="what you care about most")
    ad.add_argument("--max-size-ratio", type=float, default=None,
                    help="target fraction of original size (e.g. 0.25 = 4x smaller)")
    ad.add_argument("--min-retention", type=float, default=None,
                    help="accuracy floor as a fraction of baseline (e.g. 0.98)")
    ad.add_argument("--can-retrain", action="store_true",
                    help="allow techniques that need training (pruning fine-tune, distillation)")
    ad.add_argument("--hardware", default=None, choices=["cpu", "gpu"],
                    help="target hardware (default: auto-detect)")
    ad.set_defaults(func=_cmd_advise)

    g = sub.add_parser(
        "gpu",
        help="show your GPU + the memory-saving plan that runs a model on a free/small GPU")
    g.add_argument("model", nargs="?", default=None,
                   help="HF model id or .pt path to plan for (optional)")
    g.add_argument("--vram", type=float, default=None,
                   help="pretend this many GB of VRAM are free (for planning on CPU)")
    g.set_defaults(func=_cmd_gpu)

    sub.add_parser("info", help="show version, backends, and benchmark catalog"
                   ).set_defaults(func=_cmd_info)

    v = sub.add_parser("benchmark-vision", help="run the offline CNN benchmark (sklearn digits)")
    v.add_argument("--epochs", type=int, default=8)
    v.add_argument("--report", default=None, help="optional path to write JSON report")
    v.set_defaults(func=_cmd_vision)

    l = sub.add_parser("benchmark-llm", help="run the LLM perplexity benchmark")
    l.add_argument("--model", default="facebook/opt-125m")
    l.add_argument("--max-chars", type=int, default=6000)
    l.set_defaults(func=_cmd_llm)

    sub.add_parser("autopick", help="show best-library recommendations per model family"
                   ).set_defaults(func=_cmd_autopick)
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        print(color("\n🩺 Stuck? Start with:  ", "bold", "cyan")
              + "autofollowdown diagnose <model> --problem won't-fit --vram 8")
        return
    try:
        args.func(args)
    except KeyboardInterrupt:
        print(color("\nInterrupted.", "yellow"))
        raise SystemExit(130)
    except Exception as e:           # one friendly line instead of a raw traceback
        raise SystemExit(color("Error: ", "red", "bold") + _friendly_error(e))


def _friendly_error(e):
    """Turn common failures into an actionable one-liner (set AFD_DEBUG=1 for the
    full traceback)."""
    import os
    if os.environ.get("AFD_DEBUG"):
        import traceback
        traceback.print_exc()
    msg = str(e).strip() or e.__class__.__name__
    name = e.__class__.__name__
    if isinstance(e, ImportError) or "not installed" in msg:
        return f"{msg}\n  → install the optional backend shown above, then retry."
    if name in ("RepositoryNotFoundError", "EntryNotFoundError") or "is not a local folder" in msg \
            or "Can't load" in msg or isinstance(e, FileNotFoundError):
        return (f"could not find the model — check the id/path and your network.\n  ({msg})")
    if name in ("OSError", "ConnectionError") or "Connection" in msg:
        return f"network/IO error reaching the model hub.\n  ({msg})"
    return f"{msg}\n  (set AFD_DEBUG=1 for the full traceback)"


if __name__ == "__main__":
    main()
