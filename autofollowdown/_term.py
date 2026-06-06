"""Tiny terminal-formatting helpers — aligned tables and optional color.

No third-party dependency: colors auto-disable when output isn't a TTY or when
the NO_COLOR environment variable is set, so piped/CI output stays clean.
"""

import os
import sys

_USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")

_CODES = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m",
    "blue": "\033[34m", "magenta": "\033[35m", "cyan": "\033[36m",
}


def color(text, *styles):
    if not _USE_COLOR or not styles:
        return text
    prefix = "".join(_CODES.get(s, "") for s in styles)
    return f"{prefix}{text}{_CODES['reset']}"


def _visible_len(s):
    # Length ignoring ANSI escape sequences, so alignment stays correct.
    out, i = 0, 0
    while i < len(s):
        if s[i] == "\033":
            while i < len(s) and s[i] != "m":
                i += 1
            i += 1
        else:
            out += 1
            i += 1
    return out


def _pad(s, width, align):
    gap = width - _visible_len(s)
    if gap <= 0:
        return s
    if align == "right":
        return " " * gap + s
    if align == "center":
        left = gap // 2
        return " " * left + s + " " * (gap - left)
    return s + " " * gap


def render_table(headers, rows, aligns=None):
    """Render a bordered, aligned table (box-drawing). `rows` is a list of lists
    of already-stringified cells (may contain ANSI color). Returns a string."""
    aligns = aligns or ["left"] * len(headers)
    cols = len(headers)
    widths = [_visible_len(str(headers[c])) for c in range(cols)]
    for row in rows:
        for c in range(cols):
            widths[c] = max(widths[c], _visible_len(str(row[c])))

    def line(left, mid, right):
        return left + mid.join("─" * (w + 2) for w in widths) + right

    def fmt(cells, header=False):
        out = []
        for c in range(cols):
            cell = str(cells[c])
            if header:
                cell = color(cell, "bold")
            out.append(" " + _pad(cell, widths[c], aligns[c]) + " ")
        return "│" + "│".join(out) + "│"

    parts = [line("┌", "┬", "┐"), fmt(headers, header=True), line("├", "┼", "┤")]
    parts += [fmt(r) for r in rows]
    parts.append(line("└", "┴", "┘"))
    return "\n".join(parts)
