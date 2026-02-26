# Testing Rules

## Environment
Always use the growth conda environment:
```bash
~/.conda/envs/growth/bin/python -m pytest tests/ -v
```

## Conventions
- Test files: `tests/test_<module_name>.py`
- Test functions: `test_<what_is_being_tested>`
- Each test file must be independently runnable
- Use `-v` flag for verbose output, `--tb=short` for concise tracebacks

## Module Dependencies
Do NOT write tests for Phase N+1 until Phase N tests pass.
Phase order: 1 (LoRA) → 2 (SDP) → 3 (Encoding) → 4 (Growth Prediction)
