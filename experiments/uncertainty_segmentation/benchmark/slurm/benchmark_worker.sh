#!/usr/bin/env bash
# =============================================================================
# DEPRECATED — kept only to fail loudly if old `sbatch ... benchmark_worker.sh`
# invocations are still queued or scripted somewhere.
#
# The single-worker design auto-detected CONTAINER_PWD via `find inference.py`
# and used --writable-tmpfs uniformly. Both broke for specific containers:
#   - BraTS25_2 (mmdp): no inference.py → corrupted PWD value.
#   - BraTS23_2 (blackbean): writes ./inputs into read-only /mlcube_project.
#   - BraTS23_3 (cnmc_pmi2023): extracts a zip into ./ in /mlcube_project.
#
<<<<<<< HEAD
# Replaced by per-model workers under slurm/workers/. The launcher dispatches
# to the right one via the MODELS registry (5th field).
# =============================================================================
echo "ERROR: benchmark_worker.sh is deprecated." >&2
echo "       Use slurm/workers/<model>.sh — see benchmark_launcher.sh." >&2
exit 2
=======
# Expected environment variables (exported by benchmark_launcher.sh):
#   MODEL_ID        — e.g. BraTS25_1, BraTS23_2
#   SIF_PATH        — absolute path to .sif file
#   INTERFACE       — "docker_only" or "mlcube"
#   YEAR            — 2025 or 2023
#   PARAMS_FILE     — "yes" if mlcube container requires --parameters_file
#   OUTPUT_DIR      — root benchmark output directory
#   EXTRACTION_DIR  — path to extracted NIfTIs
#   CONDA_ENV_NAME  — conda environment name
#   REPO_ROOT       — repository root path
#   BENCHMARK_DIR   — benchmark module path
#
# Reference:
#   Existing pattern: MenGrowth/slurm/segmentation/meningioma_seg_worker.sh
# =============================================================================
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --time=0-04:00:00

set -euo
trap 'echo "[TRAP] Script failed at line $LINENO (exit $?)" >&2' ERR

START_TIME=$(date +%s)

echo "=============================================="
echo "BENCHMARK WORKER: ${MODEL_ID}"
echo "=============================================="
echo "  Job ID:       ${SLURM_JOB_ID:-local}"
echo "  Node:         $(hostname)"
echo "  Model:        ${MODEL_ID}"
echo "  SIF:          ${SIF_PATH}"
echo "  Interface:    ${INTERFACE}"
echo "  Year:         ${YEAR}"
echo "  Params file:  ${PARAMS_FILE:-not set}"
echo "  Output:       ${OUTPUT_DIR}"
echo "  Extraction:   ${EXTRACTION_DIR}"
echo "  REPO_ROOT:    ${REPO_ROOT:-not set}"
echo "  BENCHMARK_DIR:${BENCHMARK_DIR:-not set}"
echo ""

# ========================================================================
# ENVIRONMENT SETUP
# ========================================================================
echo "=========================================="
echo "ENVIRONMENT"
echo "=========================================="

# Load singularity module
module load singularity 2>/dev/null || true
echo "[singularity] $(singularity --version 2>/dev/null || echo 'NOT FOUND')"

# Load conda
module_loaded=0
for m in miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
    if module avail "$m" 2>&1 | grep -qi "${m}"; then
        module load "$m" && module_loaded=1 && break
    fi
done

if [ "$module_loaded" -eq 0 ]; then
    echo "[env] No conda module loaded; assuming conda already in PATH."
fi

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
    source activate "${CONDA_ENV_NAME}"
fi

echo "[python] $(which python 2>/dev/null || echo 'NOT FOUND')"
python -c "import sys; print('Python', sys.version.split()[0])" 2>/dev/null || true
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "[WARN] nvidia-smi not available"
echo ""

# ========================================================================
# VERIFY INPUTS
# ========================================================================
echo "=========================================="
echo "INPUT VERIFICATION"
echo "=========================================="

if [ ! -f "${SIF_PATH}" ]; then
    echo "ERROR: SIF file not found: ${SIF_PATH}"
    exit 1
fi
echo "[OK] SIF exists: $(du -h "${SIF_PATH}" | cut -f1)"

NIFTI_DIR="${EXTRACTION_DIR}/nifti"
if [ ! -d "${NIFTI_DIR}" ]; then
    echo "ERROR: Extracted NIfTIs not found: ${NIFTI_DIR}"
    exit 1
fi
N_PATIENTS=$(find "${NIFTI_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "[OK] Extracted NIfTIs: ${N_PATIENTS} patients in ${NIFTI_DIR}"
SAMPLE_PAT=$(find "${NIFTI_DIR}" -mindepth 1 -maxdepth 1 -type d | head -1)
echo "[DBG] Sample patient dir: ${SAMPLE_PAT}"
ls -la "${SAMPLE_PAT}/" 2>/dev/null | head -10 || echo "[DBG] Could not list sample patient"

# Verify GPU is accessible via Singularity
set +e
singularity exec --nv "${SIF_PATH}" nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
GPU_CHECK=$?
set -e
if [ "${GPU_CHECK}" -ne 0 ]; then
    echo "[WARN] GPU not accessible inside container — inference may fail or be slow"
else
    echo "[OK] GPU accessible inside container"
fi
echo ""

# ========================================================================
# SETUP WORK DIRECTORY
# ========================================================================
echo "=========================================="
echo "WORK DIRECTORY SETUP"
echo "=========================================="

MODEL_DIR="${OUTPUT_DIR}/models/${MODEL_ID}"
WORK_OUTPUT="${MODEL_DIR}/work/output"
PRED_DIR="${MODEL_DIR}/predictions"

mkdir -p "${WORK_OUTPUT}" "${PRED_DIR}"

# Bind extraction/nifti directly as the container input — no symlinks.
# Singularity cannot follow symlinks whose targets are outside bind mounts.
echo "[OK] Input (bind-mounted):  ${NIFTI_DIR} (${N_PATIENTS} patients)"
echo "[OK] Output (work dir):     ${WORK_OUTPUT}"
echo ""

# ========================================================================
# DETECT CONTAINER WORKDIR
# ========================================================================
# Singularity does NOT honor Docker WORKDIR — we must set --pwd explicitly.
# Strategy: parse the runscript to find the entrypoint script name, then
# search for it inside the container.
echo "=========================================="
echo "CONTAINER WORKDIR DETECTION"
echo "=========================================="

# --- Step 1: dump the raw runscript ---
echo "[DBG] Inspecting runscript..."
set +e
RUNSCRIPT=$(singularity inspect --runscript "${SIF_PATH}" 2>&1)
INSPECT_RC=$?
set -e
echo "[DBG] inspect exit code: ${INSPECT_RC}"
echo "[DBG] --- RAW RUNSCRIPT START ---"
echo "${RUNSCRIPT}"
echo "[DBG] --- RAW RUNSCRIPT END ---"

# --- Step 2: extract the Python script name ---
ENTRYPOINT_SCRIPT=""
if [ "${INSPECT_RC}" -eq 0 ] && [ -n "${RUNSCRIPT}" ]; then
    ENTRYPOINT_SCRIPT=$(echo "${RUNSCRIPT}" | grep -oP 'python[3]?\s+\K\S+\.py' | head -1)
fi

if [ -z "${ENTRYPOINT_SCRIPT}" ]; then
    if [ "${INTERFACE}" = "mlcube" ]; then
        ENTRYPOINT_SCRIPT="mlcube.py"
    else
        ENTRYPOINT_SCRIPT="inference.py"
    fi
    echo "[WARN] Could not parse entrypoint from runscript, trying ${ENTRYPOINT_SCRIPT}"
else
    echo "[OK]   Entrypoint from runscript: ${ENTRYPOINT_SCRIPT}"
fi

# --- Step 3: search for entrypoint file inside the container ---
echo "[DBG] Searching for ${ENTRYPOINT_SCRIPT} inside container (find / -maxdepth 4)..."
set +e
FIND_OUTPUT=$(singularity exec "${SIF_PATH}" \
    find / -maxdepth 4 -name "${ENTRYPOINT_SCRIPT}" \
    -not -path "*/proc/*" -not -path "*/sys/*" -not -path "*/dev/*" \
    2>&1 || true)
set -e
echo "[DBG] find output: ${FIND_OUTPUT}"

CONTAINER_PWD=$(echo "${FIND_OUTPUT}" | grep -v "Permission denied" | head -1 | xargs -I{} dirname {} 2>/dev/null || true)

if [ -n "${CONTAINER_PWD}" ] && [ "${CONTAINER_PWD}" != "." ]; then
    echo "[OK]   WORKDIR: ${CONTAINER_PWD}"
else
    CONTAINER_PWD="/"
    echo "[WARN] Could not find ${ENTRYPOINT_SCRIPT} — defaulting to /"
    echo "[DBG]  Listing container root for clues..."
    singularity exec "${SIF_PATH}" ls -la / 2>&1 | head -30 || true
fi
echo ""

# ========================================================================
# RUN INFERENCE
# ========================================================================
echo "=========================================="
echo "RUNNING INFERENCE: ${MODEL_ID}"
echo "=========================================="

INFER_START=$(date +%s)

set +e
if [ "${INTERFACE}" = "docker_only" ]; then
    # BraTS25: simple /input → /output interface
    echo "Interface: Docker-only (BraTS25)"
    echo "  Bind: ${NIFTI_DIR} → /input (ro)"
    echo "  Bind: ${WORK_OUTPUT} → /output (rw)"
    echo "  PWD:  ${CONTAINER_PWD}"
    SING_CMD="singularity run --nv --cleanenv --no-home --writable-tmpfs --pwd ${CONTAINER_PWD} --bind ${NIFTI_DIR}:/input:ro --bind ${WORK_OUTPUT}:/output:rw ${SIF_PATH}"
    echo ""
    echo "[DBG] COMMAND: ${SING_CMD}"
    echo ""

    singularity run \
        --nv \
        --cleanenv \
        --no-home \
        --writable-tmpfs \
        --pwd "${CONTAINER_PWD}" \
        --bind "${NIFTI_DIR}:/input:ro" \
        --bind "${WORK_OUTPUT}:/output:rw" \
        "${SIF_PATH}"

elif [ "${INTERFACE}" = "mlcube" ]; then
    # BraTS23: MLCube /mlcube_io0 → /mlcube_io2 interface
    echo "Interface: MLCube (BraTS23)"
    echo "  Bind: ${NIFTI_DIR} → /mlcube_io0 (ro)"
    echo "  Bind: ${WORK_OUTPUT} → /mlcube_io2 (rw)"

    # Some MLCube containers require --parameters_file (dummy, unused).
    # Determined per-model from BraTS Orchestrator meningioma.yml config.
    PARAMS_BIND=""
    PARAMS_ARG=""
    if [ "${PARAMS_FILE:-no}" = "yes" ]; then
        PARAMS_DIR="${MODEL_DIR}/work/params"
        mkdir -p "${PARAMS_DIR}"
        echo "{}" > "${PARAMS_DIR}/params.yaml"
        PARAMS_BIND="--bind ${PARAMS_DIR}:/mlcube_io1:ro"
        PARAMS_ARG="--parameters_file=/mlcube_io1/params.yaml"
        echo "  Bind: ${PARAMS_DIR} → /mlcube_io1 (ro)  [dummy params]"
    fi
    echo "  PWD:  ${CONTAINER_PWD}"
    echo "  PARAMS_FILE env: ${PARAMS_FILE:-not set}"
    SING_CMD="singularity run --nv --cleanenv --no-home --writable-tmpfs --pwd ${CONTAINER_PWD} --bind ${NIFTI_DIR}:/mlcube_io0:ro --bind ${WORK_OUTPUT}:/mlcube_io2:rw ${PARAMS_BIND} ${SIF_PATH} infer --data_path=/mlcube_io0 --output_path=/mlcube_io2 ${PARAMS_ARG}"
    echo ""
    echo "[DBG] COMMAND: ${SING_CMD}"
    echo ""

    singularity run \
        --nv \
        --cleanenv \
        --no-home \
        --writable-tmpfs \
        --pwd "${CONTAINER_PWD}" \
        --bind "${NIFTI_DIR}:/mlcube_io0:ro" \
        --bind "${WORK_OUTPUT}:/mlcube_io2:rw" \
        ${PARAMS_BIND} \
        "${SIF_PATH}" \
        infer --data_path=/mlcube_io0 --output_path=/mlcube_io2 ${PARAMS_ARG}
else
    echo "ERROR: Unknown interface type: ${INTERFACE}"
    exit 1
fi

INFER_EXIT=$?
set -e

INFER_END=$(date +%s)
INFER_ELAPSED=$((INFER_END - INFER_START))

echo ""
echo "Inference exit code: ${INFER_EXIT}"
echo "Inference duration:  $(($INFER_ELAPSED / 60))m $(($INFER_ELAPSED % 60))s"

if [ "${INFER_EXIT}" -ne 0 ]; then
    echo ""
    echo "=========================================="
    echo "FAILURE DIAGNOSTICS: ${MODEL_ID}"
    echo "=========================================="
    echo "Exit code: ${INFER_EXIT}"
    echo ""
    echo "[DBG] Verifying --pwd resolves inside container:"
    singularity exec "${SIF_PATH}" ls -la "${CONTAINER_PWD}/" 2>&1 | head -20 || echo "[DBG] Could not list ${CONTAINER_PWD}"
    echo ""
    echo "[DBG] Container environment (HOME, PWD, PATH):"
    singularity exec --cleanenv --no-home "${SIF_PATH}" env 2>&1 | grep -E '^(HOME|PWD|PATH|WORKDIR)=' || echo "[DBG] Could not read env"
    echo ""
    echo "[DBG] Singularity version details:"
    singularity --version 2>&1 || true
    echo ""
    echo "[DBG] All .py files at container root level:"
    singularity exec "${SIF_PATH}" find / -maxdepth 3 -name "*.py" -not -path "*/proc/*" -not -path "*/sys/*" -not -path "*/lib/*" -not -path "*/site-packages/*" 2>/dev/null | head -30 || true
    echo ""
    echo "Manual debug commands:"
    echo "  singularity shell --nv ${SIF_PATH}"
    echo "  singularity inspect --runscript ${SIF_PATH}"
    echo "  singularity exec ${SIF_PATH} find / -maxdepth 3 -name '*.py' -not -path '*/lib/*'"
    echo ""
    echo "Work directory preserved for debugging."

    # Save partial metadata even on failure
    python3 -c "
import json
meta = {
    'model_id': '${MODEL_ID}',
    'sif_path': '${SIF_PATH}',
    'interface': '${INTERFACE}',
    'year': int('${YEAR}'),
    'exit_code': ${INFER_EXIT},
    'inference_seconds': ${INFER_ELAPSED},
    'status': 'FAILED',
    'n_patients_input': ${N_PATIENTS},
}
with open('${MODEL_DIR}/run_metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)
" 2>/dev/null || true

    exit "${INFER_EXIT}"
fi

# ========================================================================
# COLLECT PREDICTIONS
# ========================================================================
echo ""
echo "=========================================="
echo "COLLECTING PREDICTIONS"
echo "=========================================="

# Move predictions from work/output to predictions/
# Containers write either flat files or nested dirs — handle both
N_PREDS=0

for nii_file in $(find "${WORK_OUTPUT}" -name "*.nii.gz" -type f 2>/dev/null); do
    rel_path="${nii_file#${WORK_OUTPUT}/}"
    dest="${PRED_DIR}/${rel_path}"
    mkdir -p "$(dirname "${dest}")"
    cp "${nii_file}" "${dest}"
    N_PREDS=$((N_PREDS + 1))
done

echo "[OK] Collected ${N_PREDS} prediction files → ${PRED_DIR}"

if [ "${N_PREDS}" -eq 0 ]; then
    echo "[WARN] No predictions found! Listing work/output:"
    find "${WORK_OUTPUT}" -type f | head -20
fi

# ========================================================================
# SAVE METADATA
# ========================================================================
END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))

python3 -c "
import json
meta = {
    'model_id': '${MODEL_ID}',
    'sif_path': '${SIF_PATH}',
    'interface': '${INTERFACE}',
    'year': int('${YEAR}'),
    'exit_code': 0,
    'inference_seconds': ${INFER_ELAPSED},
    'total_seconds': ${TOTAL_ELAPSED},
    'status': 'SUCCESS',
    'n_patients_input': ${N_PATIENTS},
    'n_predictions': ${N_PREDS},
    'node': '$(hostname)',
    'job_id': '${SLURM_JOB_ID:-local}',
}
with open('${MODEL_DIR}/run_metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)
" 2>/dev/null || true

# ========================================================================
# CLEANUP
# ========================================================================
echo ""
echo "=========================================="
echo "CLEANUP"
echo "=========================================="

# Remove work/output (container output already collected to predictions/)
rm -rf "${MODEL_DIR}/work"
echo "[OK] Removed work directory"

echo ""
echo "=========================================="
echo "COMPLETE: ${MODEL_ID}"
echo "=========================================="
echo "  Predictions: ${PRED_DIR} (${N_PREDS} files)"
echo "  Metadata:    ${MODEL_DIR}/run_metadata.json"
echo "  Duration:    $(($TOTAL_ELAPSED / 60))m $(($TOTAL_ELAPSED % 60))s"
echo ""
echo "Next step: python3 ${BENCHMARK_DIR}/evaluate_benchmark.py --output-dir ${OUTPUT_DIR} --models ${MODEL_ID}"
>>>>>>> 4d2de34ac7514a508662c8444b64fe553a91999d
