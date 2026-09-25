# Validation Guide

Run the checks from the repository root:

```bash
python scripts/validate_openenv.py
python -m pytest -q
```

For a local smoke test, start the server:

```bash
uv run server --host 0.0.0.0 --port 7860
```

Then verify that the task and state endpoints respond before testing an agent trajectory.

## Before submitting

- Keep `openenv.yaml` and `pyproject.toml` consistent.
- Confirm the baseline inference command still runs.
- Run the validator and test suite.
- Check that new task behavior has regression coverage.
