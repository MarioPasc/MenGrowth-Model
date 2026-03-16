# Testing Rules

## Environment
Always use the growth conda environment:
```bash
~/.conda/envs/growth/bin/python -m pytest tests/ -v --tb=short
```

## Pytest Markers

Tests are tagged with markers defined in `pyproject.toml`. **Always use markers to run the relevant subset.**

| Marker | Description | Example |
|--------|-------------|---------|
| `phase0` | Data loading, transforms, semantic features | `pytest -m phase0` |
| `phase1` | LoRA encoder adaptation, checkpoints, SwinUNETR | `pytest -m phase1` |
| `phase2` | SDP network, losses, partitioning | `pytest -m phase2` |
| `evaluation` | Probes, DCI, latent quality metrics | `pytest -m evaluation` |
| `experiment` | Experiment scripts (lora analysis, viz) | `pytest -m experiment` |
| `unit` | Fast synthetic-only tests (<1s each) | `pytest -m unit` |
| `slow` | Training convergence tests (>30s) | `pytest -m slow` |
| `gpu` | Requires GPU | `pytest -m gpu` |
| `real_data` | Requires real checkpoint or H5 files | `pytest -m real_data` |

## Which tests to run

**After editing code, run the marker(s) matching the module you changed:**

| Files changed in... | Run |
|---------------------|-----|
| `src/growth/data/` | `pytest -m phase0` |
| `src/growth/models/encoder/`, `src/growth/losses/`, `experiments/lora/` | `pytest -m phase1` |
| `src/growth/models/projection/`, `experiments/sdp/` | `pytest -m phase2` |
| `src/growth/evaluation/` | `pytest -m evaluation` |
| `experiments/lora/analysis/`, `experiments/lora/vis/` | `pytest -m experiment` |
| `experiments/utils/settings.py` | `pytest -m experiment` |
| Unknown / broad changes | `pytest -m "not slow and not real_data"` |

**Default safe run** (all fast tests, ~2 min, 370 tests):
```bash
~/.conda/envs/growth/bin/python -m pytest -m "not slow and not real_data" -v --tb=short
```

**Full suite** (includes slow convergence tests, ~20 min):
```bash
~/.conda/envs/growth/bin/python -m pytest -v --tb=short
```

Markers can be combined: `pytest -m "phase1 and unit"` runs only fast Phase 1 tests.

## Conventions
- Test files: `tests/growth/test_<module_name>.py`
- Test functions: `test_<what_is_being_tested>`
- Each test file must be independently runnable
- Every test file must have a module-level `pytestmark` list with appropriate markers

## Stage Dependencies
The project follows a 3-stage complexity ladder. Stage K+1 tests should only be written after Stage K demonstrates results under LOPO-CV.

**Stage 1** (Volumetric Baseline): ScalarGP, LME, HGP on volume trajectories
**Stage 2** (Severity Model): NLME with latent severity, quantile transform
**Stage 3** (Representation Learning): LoRA → SDP → PCA → GP+ARD

Within Stage 3, the old module order applies:
Data (phase0) → LoRA (phase1) → SDP (phase2) → Encoding → Growth Prediction

## Known Issues
- `TestRealDataForwardPass` in `test_swin_loader.py` fails with H5 transforms on NIfTI data (pre-existing, marked `real_data`)
- `TestTrainingConvergence` and `TestSemanticQuality` in `test_sdp.py` are slow (100-epoch training, marked `slow`)
