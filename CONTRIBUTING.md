# Contributing

Create a short-lived branch for each focused change and open a pull request early.
Keep pull requests small enough to review, explain their scientific or behavioral
goal, and call out intentionally untouched areas.

Install the development environment and run the local checks:

```bash
python3 -m pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest
pre-commit run --all-files
```

Add tests for behavior changes and regression tests for fixes. Keep
provider-specific I/O and assumptions behind isolated adapters so the core model
remains provider-independent. Notebooks may demonstrate workflows, but reusable
logic must live in the package and be tested outside notebooks.

Discuss large data additions before committing them. Changes to atlas contracts,
scientific semantics, public data shapes, or compatibility guarantees require
broad review from both scientific and software maintainers.
