#!/usr/bin/env bash
# =============================================================================
# EXTRACTION WORKER — Picasso compute node (CPU-only)
#
# Extracts BraTS-MEN test split from H5 archive to NIfTI files.
# Must complete successfully before submitting inference jobs.
#
# Usage:
#   # Blocking (wait for completion):
#   sbatch --wait experiments/uncertainty_segmentation/benchmark/slurm/extract_worker.sh
#
#   # Non-blocking:
#   sbatch experiments/uncertainty_segmentation/benchmark/slurm/extract_worker.sh
#
#   # Debug (extract 2 patients only):
#   sbatch --export=ALL,LIMIT=2 experiments/uncertainty_segmentation/benchmark/slurm/extract_worker.sh
#
# Idempotent: skips if manifest.json already exists. Delete the manifest to
# force re-extraction.
# =============================================================================
#SBATCH --job-name=bm_extract
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --output=/mnt/home/users/tic_163_uma/mpascual/execs/growth/benchmark_segmentation/logs/extract_%j.out
#SBATCH --error=/mnt/home/users/tic_163_uma/mpascual/execs/growth/benchmark_segmentation/logs/extract_%j.err

set -euo pipefail

# ========================================================================
# CONFIGURATION
# ========================================================================
H5_FILE="/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/h5_growth_datasets/BraTS_MEN.h5"
OUTPUT_DIR="/mnt/home/users/tic_163_uma/mpascual/execs/growth/benchmark_segmentation"
EXTRACTION_DIR="${OUTPUT_DIR}/extraction"
MANIFEST="${EXTRACTION_DIR}/manifest.json"
CONDA_ENV_NAME="mengrowth"
LIMIT="${LIMIT:-}"

# SLURM copies scripts to /var/spool/slurmd/, so BASH_SOURCE does not
# resolve to the original repo path on compute nodes. Accept REPO_ROOT
# as an env var (export it via sbatch --export, or set it before sbatch).
REPO_ROOT="${REPO_ROOT:-/mnt/home/users/tic_163_uma/mpascual/execs/growth/MenGrowth-Model}"
BENCHMARK_DIR="${REPO_ROOT}/experiments/uncertainty_segmentation/benchmark"

echo "=============================================="
echo "BENCHMARK EXTRACTION"
echo "=============================================="
echo "  Job ID:     ${SLURM_JOB_ID:-local}"
echo "  Node:       $(hostname)"
echo "  Repo:       ${REPO_ROOT}"
echo "  Script:     ${BENCHMARK_DIR}/extract_h5_to_nifti.py"
echo "  H5:         ${H5_FILE}"
echo "  Output:     ${EXTRACTION_DIR}"
echo "  Limit:      ${LIMIT:-all}"
echo ""

# ========================================================================
# ENVIRONMENT SETUP
# ========================================================================
echo "=========================================="
echo "ENVIRONMENT"
echo "=========================================="

module_loaded=0
for m in miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
    if module avail "$m" 2>&1 | grep -qi "${m}"; then
        module load "$m" && module_loaded=1 && break
    fi
done
[ "$module_loaded" -eq 0 ] && echo "[env] No conda module loaded; assuming conda already in PATH."

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
    source activate "${CONDA_ENV_NAME}"
fi

echo "[python] $(which python)"
python -c "import sys; print('Python', sys.version.split()[0])"

set +e
python -c "import h5py; print('[OK] h5py', h5py.version.version)"
H5PY_RC=$?
python -c "import nibabel; print('[OK] nibabel', nibabel.__version__)"
NIB_RC=$?
set -e

if [ "${H5PY_RC}" -ne 0 ] || [ "${NIB_RC}" -ne 0 ]; then
    echo "ERROR: Missing Python dependencies (h5py or nibabel)"
    echo "       Install: pip install h5py nibabel"
    exit 1
fi
echo ""

# ========================================================================
# PRE-CHECKS
# ========================================================================
echo "=========================================="
echo "PRE-CHECKS"
echo "=========================================="

EXTRACT_SCRIPT="${BENCHMARK_DIR}/extract_h5_to_nifti.py"
if [ ! -f "${EXTRACT_SCRIPT}" ]; then
    echo "ERROR: Extraction script not found: ${EXTRACT_SCRIPT}"
    echo "       Set REPO_ROOT to the MenGrowth-Model checkout on this cluster."
    echo "       Example: sbatch --export=ALL,REPO_ROOT=/path/to/MenGrowth-Model ..."
    exit 1
fi
echo "[OK] Extraction script: ${EXTRACT_SCRIPT}"

if [ ! -f "${H5_FILE}" ]; then
    echo "ERROR: H5 file not found: ${H5_FILE}"
    exit 1
fi
echo "[OK] H5 exists: $(du -h "${H5_FILE}" | cut -f1)"

if [ -f "${MANIFEST}" ]; then
    N_EXISTING=$(python -c "import json; print(json.load(open('${MANIFEST}'))['n_patients'])")
    echo "[OK] Manifest already exists with ${N_EXISTING} patients."
    echo "     Delete ${MANIFEST} to force re-extraction."
    echo ""
    echo "SKIPPED (idempotent) — extraction already complete."
    exit 0
fi
echo "[OK] No prior manifest — proceeding with extraction."
echo ""

# ========================================================================
# EXTRACT H5 → NIfTI
# ========================================================================
echo "=========================================="
echo "EXTRACTING H5 → NIfTI"
echo "=========================================="

mkdir -p "${EXTRACTION_DIR}"

EXTRACT_CMD="python ${EXTRACT_SCRIPT} --h5 ${H5_FILE} --output ${EXTRACTION_DIR}"
if [ -n "${LIMIT}" ]; then
    EXTRACT_CMD="${EXTRACT_CMD} --limit ${LIMIT}"
fi

echo "Command: ${EXTRACT_CMD}"
echo ""

eval "${EXTRACT_CMD}"
EXTRACT_RC=$?

if [ "${EXTRACT_RC}" -ne 0 ]; then
    echo ""
    echo "ERROR: Extraction script exited with code ${EXTRACT_RC}"
    exit "${EXTRACT_RC}"
fi

# ========================================================================
# VERIFICATION
# ========================================================================
echo ""
echo "=========================================="
echo "VERIFICATION"
echo "=========================================="

if [ ! -f "${MANIFEST}" ]; then
    echo "ERROR: Extraction completed but manifest not created at ${MANIFEST}"
    exit 1
fi

N_PATIENTS=$(python -c "import json; print(json.load(open('${MANIFEST}'))['n_patients'])")
N_NIFTI_DIRS=$(find "${EXTRACTION_DIR}/nifti" -mindepth 1 -maxdepth 1 -type d | wc -l)
N_GT_DIRS=$(find "${EXTRACTION_DIR}/ground_truth" -mindepth 1 -maxdepth 1 -type d | wc -l)

echo "  Manifest patients:   ${N_PATIENTS}"
echo "  NIfTI directories:   ${N_NIFTI_DIRS}"
echo "  GT directories:      ${N_GT_DIRS}"

# Verify sample patient has all 4 modalities
SAMPLE_DIR=$(find "${EXTRACTION_DIR}/nifti" -mindepth 1 -maxdepth 1 -type d | head -1)
if [ -z "${SAMPLE_DIR}" ]; then
    echo "ERROR: No NIfTI directories found after extraction"
    exit 1
fi

N_MODALITIES=$(find "${SAMPLE_DIR}" -name "*.nii.gz" | wc -l)
echo "  Sample patient:      $(basename "${SAMPLE_DIR}") → ${N_MODALITIES} modality files"

SAMPLE_GT="${EXTRACTION_DIR}/ground_truth/$(basename "${SAMPLE_DIR}")/seg.nii.gz"
if [ -f "${SAMPLE_GT}" ]; then
    echo "  Sample GT:           $(du -h "${SAMPLE_GT}" | cut -f1)"
else
    echo "  WARNING: Sample GT not found: ${SAMPLE_GT}"
fi

# Consistency checks
FAILED=0
if [ "${N_NIFTI_DIRS}" -ne "${N_PATIENTS}" ]; then
    echo "  WARNING: NIfTI count (${N_NIFTI_DIRS}) != manifest (${N_PATIENTS})"
    FAILED=1
fi
if [ "${N_MODALITIES}" -ne 4 ]; then
    echo "  ERROR: Expected 4 modalities per patient, found ${N_MODALITIES}"
    FAILED=1
fi
if [ "${N_GT_DIRS}" -ne "${N_PATIENTS}" ]; then
    echo "  WARNING: GT count (${N_GT_DIRS}) != manifest (${N_PATIENTS})"
    FAILED=1
fi

echo ""
if [ "${FAILED}" -eq 1 ]; then
    echo "=========================================="
    echo "EXTRACTION COMPLETED WITH WARNINGS"
    echo "=========================================="
    echo "  Check the warnings above. Inference may still work."
else
    echo "=========================================="
    echo "EXTRACTION COMPLETE ✓"
    echo "=========================================="
fi

DISK_USAGE=$(du -sh "${EXTRACTION_DIR}" | cut -f1)
echo "  Patients:    ${N_PATIENTS}"
echo "  Disk usage:  ${DISK_USAGE}"
echo "  Manifest:    ${MANIFEST}"
echo ""
echo "Next: submit inference jobs:"
echo "  bash ${BENCHMARK_DIR}/slurm/benchmark_launcher.sh --skip-extract --sequential"
