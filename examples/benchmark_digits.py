"""Real, self-contained vision compression benchmark.

Trains a real CNN on the real scikit-learn `digits` dataset (no download), then
prunes / quantizes / distills it and reports the *real* impact on size, latency,
and accuracy — and which variant to pick. Every number is measured.

Run:  python3 examples/benchmark_digits.py [--epochs N] [--report out.json]
  or: autofollowdown benchmark-vision   (after pip install)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autofollowdown.demos import vision_benchmark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--report", default=None, help="optional path to write JSON report")
    args = ap.parse_args()
    vision_benchmark(epochs=args.epochs, report=args.report)


if __name__ == "__main__":
    main()
