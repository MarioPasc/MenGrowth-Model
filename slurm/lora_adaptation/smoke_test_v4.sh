#!/usr/bin/env bash
#SBATCH -J gp_smoke_cpu
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=batch

# =============================================================================
# GP PROBE SMOKE TEST (CPU-only)
#
# Verifies that the GP probe refactoring is correct:
#   Step 1: GP probe unit tests
#   Step 2: Latent quality tests
#   Step 3: Import validation
#   Step 4: Quick end-to-end on synthetic 768-dim data
# =============================================================================

set -euo pipefail

REPO_ROOT="${REPO_SRC:-$(cd "$(dirname "$0")/../.." && pwd)}"
CONDA_ENV="${CONDA_ENV_NAME:-growth}"

echo "========================================"
echo "GP Probe Smoke Test (CPU)"
echo "Repo: ${REPO_ROOT}"
echo "Conda: ${CONDA_ENV}"
echo "Date: $(date)"
echo "========================================"

# Activate conda
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

PASS=0
FAIL=0

# --- Step 1: GP probe unit tests ---
echo ""
echo "=== Step 1: GP probe unit tests ==="
if python -m pytest tests/growth/test_gp_probes.py -v --tb=short; then
    echo "PASS: GP probe tests"
    PASS=$((PASS + 1))
else
    echo "FAIL: GP probe tests"
    FAIL=$((FAIL + 1))
fi

# --- Step 2: Latent quality tests ---
echo ""
echo "=== Step 2: Latent quality tests ==="
if python -m pytest tests/growth/test_latent_quality.py -v --tb=short; then
    echo "PASS: Latent quality tests"
    PASS=$((PASS + 1))
else
    echo "FAIL: Latent quality tests"
    FAIL=$((FAIL + 1))
fi

# --- Step 3: Import validation ---
echo ""
echo "=== Step 3: Import validation ==="
if python -c "
from growth.evaluation.gp_probes import GPProbe, GPProbeResults, GPSemanticProbes, GPSemanticResults, extract_sausage_data
from growth.evaluation import GPProbe, GPSemanticProbes
print('All GP probe imports OK')

# Verify old symbols are gone
import importlib
mod = importlib.import_module('growth.evaluation')
for name in ['EnhancedLinearProbe', 'EnhancedSemanticProbes', 'MLPProbe', 'LinearProbe', 'ProbeResults', 'SemanticProbes']:
    assert not hasattr(mod, name), f'Old symbol {name} still exported!'
print('Old symbols correctly removed')
"; then
    echo "PASS: Import validation"
    PASS=$((PASS + 1))
else
    echo "FAIL: Import validation"
    FAIL=$((FAIL + 1))
fi

# --- Step 4: Quick end-to-end on synthetic data ---
echo ""
echo "=== Step 4: Synthetic end-to-end test ==="
if python -c "
import numpy as np
from growth.evaluation.gp_probes import GPSemanticProbes, extract_sausage_data

np.random.seed(42)
N, D = 200, 768
X = np.random.randn(N, D)
targets = {
    'volume': np.random.randn(N, 4),
    'location': np.random.randn(N, 3),
    'shape': np.random.randn(N, 3),
}

probes = GPSemanticProbes(input_dim=D, n_restarts=1, r2_ci_samples=100)
probes.fit(X[:160], {k: v[:160] for k, v in targets.items()})
results = probes.evaluate(X[160:], {k: v[160:] for k, v in targets.items()})
summary = probes.get_summary(results)

print(f'R² linear mean: {summary[\"r2_mean_linear\"]:.4f}')
print(f'R² RBF mean:    {summary[\"r2_mean_rbf\"]:.4f}')

# Test sausage data extraction
sausage = extract_sausage_data(results.linear['volume'], targets['volume'][160:], target_dim=0)
assert 'y_true' in sausage
assert 'y_pred_mean' in sausage
assert 'y_pred_lo' in sausage
assert 'y_pred_hi' in sausage
print('Sausage data extraction OK')

print('Synthetic end-to-end test PASSED')
"; then
    echo "PASS: Synthetic end-to-end"
    PASS=$((PASS + 1))
else
    echo "FAIL: Synthetic end-to-end"
    FAIL=$((FAIL + 1))
fi

# --- Summary ---
echo ""
echo "========================================"
echo "RESULTS: ${PASS} passed, ${FAIL} failed"
echo "========================================"

if [ "${FAIL}" -gt 0 ]; then
    exit 1
fi
