"""Runnable demo benchmarks, packaged so they work after `pip install`.

Heavy / optional dependencies (scikit-learn, datasets, a downloaded HF model) are
imported lazily inside each function, so importing `autofollowdown` stays light.
Both the CLI and the scripts in `examples/` call into here — one source of truth.
"""

import copy
import warnings

import torch
import torch.nn as nn

from ._term import color, render_table
from .api import ModelCompressor
from .benchmark import Benchmark
from .pipeline import compress_and_benchmark


# ----------------------------------------------------------------- vision demo
class _DigitCNN(nn.Module):
    """Small CNN sized so quantization/pruning have something to bite into."""

    def __init__(self, width=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def vision_benchmark(epochs=8, report=None):
    """Train a real CNN on sklearn `digits`, compress it 4 ways, print results."""
    warnings.filterwarnings("ignore")
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(0)
    digits = load_digits()
    X = torch.tensor(digits.images.astype("float32") / 16.0).unsqueeze(1)
    y = torch.tensor(digits.target, dtype=torch.long)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0,
                                           stratify=y)
    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(Xte, yte), batch_size=64)

    def train(model):
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        lossf = nn.CrossEntropyLoss()
        model.train()
        for _ in range(epochs):
            for xb, yb in train_loader:
                opt.zero_grad()
                lossf(model(xb), yb).backward()
                opt.step()
        return model

    print("Training baseline CNN on sklearn digits ...")
    baseline = train(_DigitCNN(32))
    example_input = next(iter(test_loader))[0][:32]
    bench = Benchmark(example_input, eval_loader=test_loader, reference_model=baseline)
    bench.measure(baseline, "baseline (fp32)")
    bench.measure(ModelCompressor(copy.deepcopy(baseline)).prune(0.5).model,
                  "pruned 50%")
    bench.measure(ModelCompressor(copy.deepcopy(baseline)).quantize(approach="dynamic").model,
                  "quantized int8")
    bench.measure(ModelCompressor(copy.deepcopy(baseline)).prune(0.5)
                  .quantize(approach="dynamic").model, "pruned+quantized")
    student = _DigitCNN(8)
    ModelCompressor(student).distill(baseline, train_loader, epochs=epochs, alpha=0.7)
    bench.measure(student, "distilled student (1/4 width)")

    print("\n=== Compression Benchmark (sklearn digits, real measurements) ===\n")
    print(bench.to_table())
    print("\n" + bench.summary())
    if report:
        bench.to_json(report)
        print(f"\nJSON report written to {report}")
    return bench


# -------------------------------------------------------------------- llm demo
def llm_benchmark(model_id="facebook/opt-125m", max_chars=6000):
    """Compress a small HF causal LM and compare WikiText-2 perplexity / size."""
    warnings.filterwarnings("ignore")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .llm_eval import evaluate_perplexity, lm_eval_command, load_wikitext2
    from .metrics import measure_latency, model_disk_size_mb

    print(f"Loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
    text = load_wikitext2("test")[:max_chars]
    example_ids = tokenizer("The quick brown fox", return_tensors="pt")["input_ids"]

    def measure(m, name):
        return {"name": name, "size_mb": model_disk_size_mb(m),
                "ppl": evaluate_perplexity(m, tokenizer, text, stride=256, max_length=512),
                "latency_ms": measure_latency(m, example_ids, n_runs=10)[0]}

    print("Measuring baseline (fp32) ...")
    rows = [measure(model, "baseline (fp32)")]
    print("Compressing with autofollowdown (INT8 dynamic) ...")
    q = ModelCompressor(copy.deepcopy(model)).quantize(approach="dynamic").model
    rows.append(measure(q, "int8 dynamic"))

    base = rows[0]
    recommended = next((r for r in rows[1:] if r["ppl"] <= base["ppl"] * 1.05
                        and r["size_mb"] < base["size_mb"]), base)
    table_rows = []
    for r in rows:
        name = color("➤ " + r["name"], "green", "bold") if r is recommended else r["name"]
        table_rows.append([
            name, f"{r['size_mb']:.1f}", f"{r['ppl']:.3f}", f"{r['latency_ms']:.1f}",
            f"{base['size_mb']/r['size_mb']:.2f}×" if r is not base else "—",
            f"{base['latency_ms']/r['latency_ms']:.2f}×" if r is not base else "—",
            f"{r['ppl']-base['ppl']:+.3f}" if r is not base else "—",
        ])
    print("\n=== LLM Compression Benchmark (WikiText-2 perplexity) ===\n")
    print(render_table(["Model", "Size MB", "Perplexity↓", "Lat ms",
                        "Size×", "Speed×", "ΔPPL"], table_rows, ["left"] + ["right"] * 6))
    sr = base["size_mb"] / recommended["size_mb"] if recommended is not base else 1.0
    print("\n" + color("➤ Recommended: ", "green", "bold")
          + f"{recommended['name']} ({sr:.2f}× smaller, perplexity {recommended['ppl']:.2f})")
    print("\nFor the full zero-shot + MMLU-ProX accuracy suite, run:")
    print("  " + lm_eval_command(model_id, num_fewshot=0))
    return rows


# ---------------------------------------------- one-command compress + pick
def _train_digit_baseline(epochs):
    """Train a baseline digit CNN and return (model, train_loader, test_loader)."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(0)
    digits = load_digits()
    X = torch.tensor(digits.images.astype("float32") / 16.0).unsqueeze(1)
    y = torch.tensor(digits.target, dtype=torch.long)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0,
                                           stratify=y)
    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(Xte, yte), batch_size=64)
    model = _DigitCNN(32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            opt.zero_grad()
            lossf(model(xb), yb).backward()
            opt.step()
    return model, train_loader, test_loader


def auto_study(model_spec=None, epochs=8):
    """Build a CompressionStudy for the one-command `auto` flow.

    With no `model_spec`, trains the offline digits CNN (full accuracy + a
    distilled student). With a Hugging Face id, loads that model and benchmarks
    size/latency across methods.
    """
    warnings.filterwarnings("ignore")
    if model_spec is None:
        print("Training a baseline CNN on sklearn digits (offline demo) ...")
        baseline, train_loader, test_loader = _train_digit_baseline(epochs)
        example_input = next(iter(test_loader))[0][:32]
        print("Applying compression methods + benchmarking ...")
        study = compress_and_benchmark(baseline, example_input=example_input,
                                       eval_loader=test_loader)
        # add a distilled student (needs the training data, so done here)
        student = _DigitCNN(8)
        ModelCompressor(student).distill(baseline, train_loader, epochs=epochs, alpha=0.7)
        study.add("distilled student (1/4 width)", student)
        return study

    print(f"Loading {model_spec} ...")
    from transformers import AutoModel, AutoModelForCausalLM
    try:
        model = AutoModelForCausalLM.from_pretrained(model_spec, dtype=torch.float32)
    except Exception:
        model = AutoModel.from_pretrained(model_spec)
    print("Applying compression methods + benchmarking (size/latency) ...")
    return compress_and_benchmark(model.eval())


# --------------------------------------------------------------- autopick demo
def autopick_demo():
    """Show the auto-picker's recommendation for several model families."""
    warnings.filterwarnings("ignore")
    from .auto import auto_compress, explain

    class _CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(3, 16, 3, padding=1)
            self.c2 = nn.Conv2d(16, 32, 3, padding=1)
            self.relu = nn.ReLU()
            self.fc = nn.Linear(32 * 8 * 8, 10)

        def forward(self, x):
            return self.fc(torch.flatten(self.relu(self.c2(self.relu(self.c1(x)))), 1))

    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(1000, 64)
            self.attn = nn.MultiheadAttention(64, 8, batch_first=True)
            self.fc = nn.Linear(64, 2)

        def forward(self, x):
            a, _ = self.attn(self.emb(x), self.emb(x), self.emb(x))
            return self.fc(a.mean(1))

    models = {
        "Vision CNN": _CNN(),
        "Tiny Transformer": _Tiny(),
        "MLP": nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10)),
    }
    for name, model in models.items():
        print(color("=" * 64, "dim"))
        print(color(name, "bold", "cyan"))
        print(explain(model))
        print()
    compressed, chosen = auto_compress(copy.deepcopy(models["Vision CNN"]))
    print(color("=" * 64, "dim"))
    print(color(f"Auto-compressed CNN with: {chosen.backend} — {chosen.scheme}", "green", "bold"))
    return chosen
