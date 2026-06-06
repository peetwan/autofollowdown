## What & why

Briefly describe the change and the motivation. Link any related issue (e.g. `Closes #12`).

## Type

- [ ] Bug fix
- [ ] New feature / backend
- [ ] Performance / refactor
- [ ] Docs

## Checklist

- [ ] `python -m pytest -q` passes locally
- [ ] Added/updated tests for the change (real effects, not mocks)
- [ ] No placeholders or leftover `# TODO`
- [ ] Updated `README.md` / `CHANGELOG.md` if user-facing
- [ ] For a new backend: declared its `Capability`, added it to `_REGISTRY`, and added a `tests/test_router.py` case
