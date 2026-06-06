"""Demo: let autofollowdown pick the best compression library for your model.

Profiles three model families and prints the ranked backend recommendation for
each (which library is ideal, which is runnable now), then auto-compresses one.

Run:  python3 examples/autopick_demo.py
  or: autofollowdown autopick   (after pip install)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autofollowdown.demos import autopick_demo

if __name__ == "__main__":
    autopick_demo()
