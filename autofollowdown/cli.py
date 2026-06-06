"""`autofollowdown` command-line interface.

After `pip install autofollowdown`, run everything without touching the source:

    autofollowdown compress facebook/opt-125m -o small.pt   # ⭐ the easy one
    autofollowdown compress                                 # offline demo (no model needed)
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
        return
    args.func(args)


if __name__ == "__main__":
    main()
