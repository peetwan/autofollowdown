"""Benchmark engine — measure real models before and after compression.

The benchmark holds a fixed evaluation set and example input, measures any model
you hand it, and renders an honest before/after comparison (size, latency,
accuracy, fidelity) — then tells you which variant to pick.
"""

import json

from ._term import color, render_table
from .metrics import measure_model


class Benchmark:
    def __init__(self, example_input, eval_loader=None, reference_model=None,
                 device="cpu", latency_runs=30):
        """
        example_input   : a representative input batch (tensor / dict) for timing.
        eval_loader      : optional (inputs, labels) dataloader → enables accuracy.
        reference_model  : optional baseline model → enables fidelity (agreement).
        """
        self.example_input = example_input
        self.eval_loader = eval_loader
        self.reference_model = reference_model
        self.device = device
        self.latency_runs = latency_runs
        self.results = []

    def measure(self, model, name):
        """Measure one model and record it under `name`."""
        m = measure_model(
            model, name,
            example_input=self.example_input,
            eval_loader=self.eval_loader,
            reference_model=self.reference_model,
            device=self.device,
            latency_runs=self.latency_runs,
        )
        self.results.append(m)
        return m

    def _baseline(self):
        return self.results[0] if self.results else None

    def report(self):
        """Return the full results plus derived ratios vs the first (baseline) row."""
        base = self._baseline()
        rows = []
        for r in self.results:
            row = dict(r)
            if base and r is not base:
                row["size_ratio"] = (base["size_mb"] / r["size_mb"]
                                     if r["size_mb"] else None)
                row["speedup"] = (base["latency_ms"] / r["latency_ms"]
                                  if r["latency_ms"] else None)
                if base["accuracy"] is not None and r["accuracy"] is not None:
                    row["accuracy_delta"] = r["accuracy"] - base["accuracy"]
            rows.append(row)
        return rows

    # ------------------------------------------------------------ recommendation
    def best_picks(self):
        """Identify the standout variants so the user can pick with confidence.

        Returns a dict with the rows that are `smallest`, `fastest`,
        `most_accurate`, and `recommended`. The recommendation favors strong
        compression while retaining accuracy (accuracy-weighted), since on CPU a
        quantized model can be smaller yet not faster — size/quality is what
        usually matters for shipping.
        """
        base = self._baseline()
        if not base:
            return {}
        variants = [r for r in self.results if r is not base]
        if not variants:
            return {"recommended": base, "smallest": base,
                    "fastest": base, "most_accurate": base}

        smallest = min(self.results, key=lambda r: r["size_mb"])
        fastest = min(self.results, key=lambda r: r["latency_ms"])
        accs = [r for r in self.results if r["accuracy"] is not None]
        most_accurate = max(accs, key=lambda r: r["accuracy"]) if accs else None

        def score(r):
            size_ratio = base["size_mb"] / r["size_mb"] if r["size_mb"] else 1.0
            if r["accuracy"] is not None and base["accuracy"]:
                retention = r["accuracy"] / base["accuracy"]
            else:
                retention = 1.0
            # accuracy-first, reward compression sub-linearly
            return (retention ** 2) * (size_ratio ** 0.5)

        recommended = max(self.results, key=score)
        return {"recommended": recommended, "smallest": smallest,
                "fastest": fastest, "most_accurate": most_accurate}

    # ----------------------------------------------- constraint-aware decisions
    def _quality(self, r):
        """The quality signal to trade off against size: accuracy, else fidelity."""
        if r.get("accuracy") is not None:
            return r["accuracy"]
        return r.get("fidelity")

    def pareto_frontier(self):
        """Names of variants on the size↔quality Pareto frontier — the only ones
        worth considering, because every other variant is both bigger AND less
        accurate than one of these. (Smaller size + higher quality = better.)"""
        names = []
        for r in self.results:
            q, s = self._quality(r), r["size_mb"]
            dominated = False
            for other in self.results:
                if other is r:
                    continue
                oq, os_ = self._quality(other), other["size_mb"]
                better_or_equal = os_ <= s and (oq is None or q is None or oq >= q)
                strictly_better = os_ < s or (oq is not None and q is not None and oq > q)
                if better_or_equal and strictly_better:
                    dominated = True
                    break
            if not dominated:
                names.append(r["name"])
        return names

    def pick_best(self, max_size_mb=None, min_accuracy=None, min_retention=None,
                  prefer="recommended"):
        """Pick the best variant that satisfies hard constraints — the decision the
        benchmark exists to support.

        Constraints: `max_size_mb` (size budget), `min_accuracy` (absolute floor),
        `min_retention` (fraction of the baseline's accuracy to keep, e.g. 0.98).
        Among the variants that pass, `prefer` chooses: 'smallest' / 'fastest' /
        'most_accurate' / 'recommended' (best size-quality score).

        Returns {row, meets, note}. If nothing meets the constraints, returns the
        closest passing-most-constraints variant with meets=False and a note.
        """
        base = self._baseline()
        if not base:
            return {"row": None, "meets": False, "note": "no baseline measured"}

        def passes(r):
            if max_size_mb is not None and r["size_mb"] > max_size_mb:
                return False
            acc = r.get("accuracy")
            if min_accuracy is not None and (acc is None or acc < min_accuracy):
                return False
            if min_retention is not None and base.get("accuracy"):
                if acc is None or acc / base["accuracy"] < min_retention:
                    return False
            return True

        candidates = [r for r in self.results if passes(r)]
        if candidates:
            chooser = {
                "smallest": lambda rs: min(rs, key=lambda r: r["size_mb"]),
                "fastest": lambda rs: min(rs, key=lambda r: r["latency_ms"]),
                "most_accurate": lambda rs: max(
                    rs, key=lambda r: (r.get("accuracy") or 0)),
            }.get(prefer)
            if chooser:
                row = chooser(candidates)
            else:  # 'recommended' → reuse the size-quality score, restricted to passers
                names = {r["name"] for r in candidates}
                picks = self.best_picks()
                rec = picks.get("recommended")
                row = rec if rec and rec["name"] in names else min(
                    candidates, key=lambda r: r["size_mb"])
            return {"row": row, "meets": True,
                    "note": f"{row['name']} meets your constraints"}

        # Nothing passes — return the closest (fewest constraint violations, then smallest).
        def violations(r):
            v = 0
            if max_size_mb is not None and r["size_mb"] > max_size_mb:
                v += 1
            acc = r.get("accuracy")
            if min_accuracy is not None and (acc is None or acc < min_accuracy):
                v += 1
            if min_retention is not None and base.get("accuracy") and (
                    acc is None or acc / base["accuracy"] < min_retention):
                v += 1
            return v

        closest = min(self.results, key=lambda r: (violations(r), r["size_mb"]))
        return {"row": closest, "meets": False,
                "note": f"nothing meets all constraints; closest is {closest['name']}"}

    # ---------------------------------------------------------------- rendering
    def to_table(self):
        """Pretty, aligned terminal table with the recommended row highlighted."""
        rows = self.report()
        picks = self.best_picks()
        rec_name = picks.get("recommended", {}).get("name")

        def fmt(v, spec, suffix=""):
            return "—" if v is None else format(v, spec) + suffix

        headers = ["Model", "Size MB", "Params", "Sparsity", "Lat ms",
                   "Acc", "Fidelity", "Size×", "Speed×", "ΔAcc"]
        aligns = ["left"] + ["right"] * 9
        table_rows = []
        for r in rows:
            name = r["name"]
            if name == rec_name:
                name = color("➤ " + name, "green", "bold")
            table_rows.append([
                name,
                f"{r['size_mb']:.3f}",
                f"{r['params']:,}",
                f"{r['sparsity']:.1%}",
                f"{r['latency_ms']:.2f}",
                fmt(r["accuracy"], ".1%"),
                fmt(r["fidelity"], ".1%"),
                fmt(r.get("size_ratio"), ".2f", "×"),
                fmt(r.get("speedup"), ".2f", "×"),
                fmt(r.get("accuracy_delta"), "+.1%"),
            ])
        return render_table(headers, table_rows, aligns)

    def summary(self):
        """Human-readable 'which one should I pick' summary."""
        picks = self.best_picks()
        if not picks:
            return ""
        base = self._baseline()
        lines = [color("Which variant to pick:", "bold")]

        def describe(row):
            sr = base["size_mb"] / row["size_mb"] if row["size_mb"] else 1.0
            acc = f", {row['accuracy']:.1%} acc" if row["accuracy"] is not None else ""
            return f"{row['name']} ({sr:.2f}× smaller{acc})"

        if picks.get("recommended"):
            lines.append("  " + color("➤ Recommended: ", "green", "bold")
                         + describe(picks["recommended"]))
        if picks.get("smallest"):
            lines.append("  • Smallest:    " + describe(picks["smallest"]))
        if picks.get("fastest"):
            f = picks["fastest"]
            spd = base["latency_ms"] / f["latency_ms"] if f["latency_ms"] else 1.0
            lines.append(f"  • Fastest:     {f['name']} ({spd:.2f}× speed)")
        if picks.get("most_accurate"):
            ma = picks["most_accurate"]
            lines.append(f"  • Most accurate: {ma['name']} ({ma['accuracy']:.1%})")
        return "\n".join(lines)

    def to_markdown(self):
        rows = self.report()

        def fmt(v, spec):
            return "—" if v is None else format(v, spec)

        header = ("| Model | Size (MB) | Params | Sparsity | Latency (ms) | "
                  "Acc | Fidelity | Size× | Speed× | ΔAcc |")
        sep = "|" + "|".join(["---"] * 10) + "|"
        lines = [header, sep]
        for r in rows:
            lines.append(
                "| {name} | {size:.3f} | {params:,} | {sp:.1%} | {lat:.2f} | "
                "{acc} | {fid} | {sr} | {spd} | {da} |".format(
                    name=r["name"],
                    size=r["size_mb"],
                    params=r["params"],
                    sp=r["sparsity"],
                    lat=r["latency_ms"],
                    acc=fmt(r["accuracy"], ".2%"),
                    fid=fmt(r["fidelity"], ".2%"),
                    sr=fmt(r.get("size_ratio"), ".2f") + ("×" if r.get("size_ratio") else ""),
                    spd=fmt(r.get("speedup"), ".2f") + ("×" if r.get("speedup") else ""),
                    da=fmt(r.get("accuracy_delta"), "+.2%"),
                )
            )
        return "\n".join(lines)

    def to_json(self, path=None):
        rows = self.report()
        text = json.dumps(rows, indent=2)
        if path:
            with open(path, "w") as f:
                f.write(text)
        return text
