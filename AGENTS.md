# Notes

- Do not open pull requests against `tinygrad/tinygrad` unless the user explicitly asks.
- Do not push branches or create GitHub PRs as part of review/planning work.
- Run tests with `-n12` for speed (e.g. `python -m pytest test/null/test_dtype.py -x -q -n12`)
- Run `python -m mypy tinygrad/` to typecheck
- Run `python -m ruff check .` to lint
- Read `./tinygrad/viz/README.md` for profiling and debugging rewrite rules
