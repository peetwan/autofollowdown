# Contributing to autofollowdown

Thanks for your interest in improving autofollowdown! 🙌 This project keeps things
simple (KISS) and honest — every operation changes real weights and every metric is
measured, never mocked. Contributions are very welcome.

## Quick start

```bash
git clone https://github.com/peetwan/autofollowdown
cd autofollowdown
pip install -e ".[dev,examples]"     # editable install + test/demo deps
python -m pytest -q                  # run the test suite (should be all green)
```

## Ground rules

- Keep it simple — prefer a small, readable change over a clever abstraction.
- No mocks in the toolkit. Tests assert real effects (size shrank, sparsity rose,
  perplexity changed), not flags.
- New code is production-ready: no placeholders, no `# TODO` left behind.
- Add or update tests for anything you change; keep `python -m pytest -q` green.
- Match the surrounding style (lazy heavy imports, clear docstrings, helpful errors).

## Adding a compression backend

The router is capability-driven, so adding a backend is mostly **data**:

1. In `autofollowdown/backends.py`, subclass `Backend`, set `name` / `alias` /
   `library` / `install_hint`, and declare a `Capability` (families, traits,
   `needs_cuda`, `needs_calibration`, `hf_bonus`).
2. Implement `compress(self, model, profile, **kwargs)` — delegate to the real
   library's documented API, guarded by `is_available()`.
3. Add it to `_REGISTRY`, and add a row to the backend table in `README.md`.
4. Add a test in `tests/test_router.py` (it should rank where you'd expect).

No scoring code needs to change — the generic scorer reads your declared capability.

## Pull requests

- Branch off `main`, keep the PR focused, and describe the change + why.
- Make sure CI (tests on Python 3.9 and 3.11) passes.
- Link any related issue.

## Reporting bugs / ideas

Open an issue using the templates. For bugs, include the model, the command, and
what you expected vs. what happened (set `AFD_DEBUG=1` to get a full traceback from
the CLI).
