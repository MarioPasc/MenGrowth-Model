#!/usr/bin/env bash
# =============================================================================
# BENCHMARK LAUNCHER — Picasso login node
#
# Pulls 5 BraTS meningioma segmentation Docker images as Singularity SIFs,
# extracts the BraTS-MEN test split from H5 to NIfTI, and submits one
# SLURM job per model.
#
# Usage (from Picasso login node):
#   bash experiments/uncertainty_segmentation/benchmark/slurm/benchmark_launcher.sh
#   bash experiments/uncertainty_segmentation/benchmark/slurm/benchmark_launcher.sh --dry-run
#   bash experiments/uncertainty_segmentation/benchmark/slurm/benchmark_launcher.sh --models BraTS25_1 BraTS25_2
#
# References:
#   BraTS Orchestrator — Kofler et al. (2025), arXiv:2506.13807
#   Existing pattern: MenGrowth/slurm/segmentation/meningioma_seg.sh
# =============================================================================
set -euo pipefail

# ========================================================================
# CONFIGURATION
# ========================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(dirname "${SCRIPT_DIR}")"
REPO_ROOT="$(cd "${BENCHMARK_DIR}/../../../.." && pwd)"

# Picasso paths
H5_FILE="/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/h5_growth_datasets/BraTS_MEN.h5"
OUTPUT_DIR="/mnt/home/users/tic_163_uma/mpascual/execs/growth/benchmark_segmentation"
SIF_DIR="/mnt/home/users/tic_163_uma/mpascual/fscratch/singularity_images"
LOG_DIR="${OUTPUT_DIR}/logs"
CONDA_ENV_NAME="mengrowth"

# Default raw BraTS-MEN dataset path (override with --raw-dir).
# Required: external containers expect canonical (240,240,155) — H5 has 192³.
RAW_BRATS_MEN_DIR="/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/BraTS_Men_Train"

# Per-model worker scripts (one per container, peculiarities documented inline).
WORKERS_DIR="${SCRIPT_DIR}/workers"

# Model registry: ID|DOCKER_IMAGE|YEAR|INTERFACE|WORKER_FILE
MODELS=(
    "BraTS25_1|brainles/brats25_men_qing:latest|2025|docker_only|brats25_1.sh"
    "BraTS25_2|brainles/brats25_men_mmdp:latest|2025|docker_only|brats25_2.sh"
    "BraTS23_1|brainles/brats23_meningioma_nvauto:latest|2023|mlcube|brats23_1.sh"
    "BraTS23_2|brainles/brats23_meningioma_blackbean:latest|2023|mlcube|brats23_2.sh"
    "BraTS23_3|brainles/brats23_meningioma_cnmc_pmi2023:latest|2023|mlcube|brats23_3.sh"
)

# ========================================================================
# PARSE ARGUMENTS
# ========================================================================
DRY_RUN=0
SELECTED_MODELS=""
SKIP_PULL=0
SKIP_EXTRACT=0
SEQUENTIAL=0
KEEP_SIF=0
H5_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run|-n)
            DRY_RUN=1
            shift
            ;;
        --models)
            shift
            SELECTED_MODELS="$*"
            break
            ;;
        --skip-pull)
            SKIP_PULL=1
            shift
            ;;
        --skip-extract)
            SKIP_EXTRACT=1
            shift
            ;;
        --sequential)
            # Quota-safe mode: pull → submit (sbatch --wait) → delete SIF → next.
            # Run inside tmux/nohup; the launcher blocks until each job finishes.
            SEQUENTIAL=1
            shift
            ;;
        --keep-sif)
            KEEP_SIF=1
            shift
            ;;
        --raw-dir)
            shift
            RAW_BRATS_MEN_DIR="$1"
            shift
            ;;
        --h5-only)
            # Legacy: extract 192³ crops from H5. External containers will reject these.
            H5_ONLY=1
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--dry-run] [--skip-pull] [--skip-extract] [--sequential] [--keep-sif]"
            echo "          [--raw-dir <path>] [--h5-only] [--models MODEL1 MODEL2 ...]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "BraTS Meningioma Segmentation Benchmark"
echo "=========================================="
echo "  Repo:       ${REPO_ROOT}"
echo "  H5:         ${H5_FILE}"
echo "  Output:     ${OUTPUT_DIR}"
echo "  SIF dir:    ${SIF_DIR}"
echo "  Log dir:    ${LOG_DIR}"
echo "  Conda env:  ${CONDA_ENV_NAME}"
echo "  Workers:    ${WORKERS_DIR}"
echo "  Raw dir:    ${RAW_BRATS_MEN_DIR}  (h5_only=${H5_ONLY})"
echo "  Dry run:    ${DRY_RUN}"
echo "  Sequential: ${SEQUENTIAL}  (keep_sif=${KEEP_SIF})"
echo ""

if [ ! -d "${WORKERS_DIR}" ]; then
    echo "ERROR: workers/ directory not found: ${WORKERS_DIR}" >&2
    exit 1
fi

# Create directories
mkdir -p "${SIF_DIR}" "${LOG_DIR}" "${OUTPUT_DIR}"

# ========================================================================
# STEP 1: PULL SINGULARITY IMAGES (login node has internet)
# ========================================================================
echo "=========================================="
echo "STEP 1: Pulling Singularity images"
echo "=========================================="

module load singularity 2>/dev/null || true

# Redirect Singularity's cache + build tmp to fscratch.
# Default /tmp on Picasso login node is small and RAM-backed → squashfs
# build for >10 GB images is OOM-killed ("create command failed: signal: killed").
# /fscratch is large and disk-backed.
export SINGULARITY_CACHEDIR="${SIF_DIR}/.singularity_cache"
export SINGULARITY_TMPDIR="${SIF_DIR}/.singularity_tmp"
mkdir -p "${SINGULARITY_CACHEDIR}" "${SINGULARITY_TMPDIR}"
echo "  SINGULARITY_CACHEDIR=${SINGULARITY_CACHEDIR}"
echo "  SINGULARITY_TMPDIR=${SINGULARITY_TMPDIR}"

# Wipe partial build trees from any prior failed pull (these can hold tens of GB
# under .singularity_tmp/build-temp-*/rootfs and silently eat the fscratch quota).
clean_singularity_tmp() {
    if [ -d "${SINGULARITY_TMPDIR}" ]; then
        find "${SINGULARITY_TMPDIR}" -maxdepth 1 -mindepth 1 -name 'build-*' -exec rm -rf {} + 2>/dev/null || true
    fi
}
clean_singularity_tmp
echo "  Quota: $(quota -s 2>/dev/null | awk 'NR==3 {print $1, $2, $3, $4}' || echo 'n/a')"
echo ""

if [ "${SEQUENTIAL}" -eq 1 ]; then
    echo "[sequential] Step 1 batch pull skipped — pulls happen per-model after Step 2."
    echo ""
fi

for entry in "${MODELS[@]}"; do
    [ "${SEQUENTIAL}" -eq 1 ] && break

    IFS='|' read -r model_id docker_image year interface worker_file <<< "${entry}"

    # Filter by selected models if specified
    if [ -n "${SELECTED_MODELS}" ]; then
        if ! echo "${SELECTED_MODELS}" | grep -qw "${model_id}"; then
            continue
        fi
    fi

    # SIF filename: replace / and : with _
    sif_name="$(echo "${docker_image}" | tr '/:' '_').sif"
    sif_path="${SIF_DIR}/${sif_name}"

    if [ "${SKIP_PULL}" -eq 1 ]; then
        echo "  [SKIP] ${model_id}: --skip-pull"
    elif [ -f "${sif_path}" ]; then
        echo "  [OK]   ${model_id}: SIF exists ($(du -h "${sif_path}" | cut -f1))"
    else
        echo "  [PULL] ${model_id}: ${docker_image}"
        if [ "${DRY_RUN}" -eq 0 ]; then
            singularity pull "${sif_path}" "docker://${docker_image}"
            echo "         → $(du -h "${sif_path}" | cut -f1)"
        else
            echo "         (dry run — skipped)"
        fi
    fi
done
echo ""

# ========================================================================
# STEP 2: EXTRACT H5 → NIfTI
# ========================================================================
echo "=========================================="
echo "STEP 2: Extracting test split from H5"
echo "=========================================="

EXTRACTION_DIR="${OUTPUT_DIR}/extraction"
MANIFEST="${EXTRACTION_DIR}/manifest.json"

if [ "${SKIP_EXTRACT}" -eq 1 ]; then
    if [ ! -f "${MANIFEST}" ]; then
        echo "  [ERROR] --skip-extract but manifest not found: ${MANIFEST}"
        echo "          Run extraction first:"
        echo "            sbatch --wait ${SCRIPT_DIR}/extract_worker.sh"
        exit 1
    fi
    N_VERIFIED=$(python3 -c "import json; print(json.load(open('${MANIFEST}'))['n_patients'])" 2>/dev/null || echo "?")
    echo "  [SKIP] --skip-extract (manifest: ${N_VERIFIED} patients)"
elif [ -f "${MANIFEST}" ]; then
    N_EXTRACTED=$(python3 -c "import json; print(json.load(open('${MANIFEST}'))['n_patients'])" 2>/dev/null || echo "?")
    echo "  [OK]   Manifest exists: ${N_EXTRACTED} patients already extracted"
    echo "         Delete ${MANIFEST} to force re-extraction"
else
    echo "  [RUN]  Extracting to ${EXTRACTION_DIR}"
    if [ "${DRY_RUN}" -eq 0 ]; then
        # Activate conda for extraction
        if command -v conda >/dev/null 2>&1; then
            source "$(conda info --base)/etc/profile.d/conda.sh" || true
            conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
        fi

        EXTRACT_ARGS=(
            --h5 "${H5_FILE}"
            --output "${EXTRACTION_DIR}"
        )
        if [ "${H5_ONLY}" -eq 1 ]; then
            EXTRACT_ARGS+=(--h5-only)
            echo "  [WARN] --h5-only: external containers will reject 192³ inputs."
        else
            if [ ! -d "${RAW_BRATS_MEN_DIR}" ]; then
                echo "ERROR: --raw-dir does not exist: ${RAW_BRATS_MEN_DIR}" >&2
                echo "       Pass --raw-dir <path/to/BraTS_Men_Train> or --h5-only." >&2
                exit 1
            fi
            EXTRACT_ARGS+=(--raw-dir "${RAW_BRATS_MEN_DIR}")
        fi

        python3 "${BENCHMARK_DIR}/extract_h5_to_nifti.py" "${EXTRACT_ARGS[@]}"
    else
        echo "         (dry run — skipped)"
    fi
fi
echo ""

# ========================================================================
# STEP 3: SUBMIT SLURM JOBS
# ========================================================================
echo "=========================================="
echo "STEP 3: Submitting SLURM jobs"
echo "=========================================="

# Each model has its own worker under workers/ — see file headers for the
# container-specific quirks each one handles.
resolve_worker() {
    local wf="$1"
    local path="${WORKERS_DIR}/${wf}"
    if [ ! -f "${path}" ]; then
        echo "ERROR: Worker script not found: ${path}" >&2
        exit 1
    fi
    printf '%s' "${path}"
}

# ------------------------------------------------------------------------
# SEQUENTIAL MODE: pull → sbatch --wait → (optionally) delete SIF → next.
# Avoids fscratch quota overflow by holding at most one large SIF on disk
# at a time. Run inside tmux/nohup — sbatch --wait blocks until the job
# completes (queue + run).
# ------------------------------------------------------------------------
if [ "${SEQUENTIAL}" -eq 1 ]; then
    SEQ_RESULTS=()
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r model_id docker_image year interface worker_file <<< "${entry}"

        if [ -n "${SELECTED_MODELS}" ]; then
            if ! echo "${SELECTED_MODELS}" | grep -qw "${model_id}"; then
                continue
            fi
        fi

        sif_name="$(echo "${docker_image}" | tr '/:' '_').sif"
        sif_path="${SIF_DIR}/${sif_name}"
        meta_path="${OUTPUT_DIR}/models/${model_id}/run_metadata.json"

        echo ""
        echo "------------------------------------------"
        echo "[seq] ${model_id} (${docker_image})"
        echo "------------------------------------------"

        # Skip if a SUCCESS run already exists.
        if [ -f "${meta_path}" ] && grep -q '"status": "SUCCESS"' "${meta_path}" 2>/dev/null; then
            echo "  [DONE] previous SUCCESS in ${meta_path} — skipping"
            SEQ_RESULTS+=("${model_id}:SKIP_DONE")
            continue
        fi

        # PULL (idempotent)
        if [ "${SKIP_PULL}" -eq 1 ] && [ ! -f "${sif_path}" ]; then
            echo "  [WARN] --skip-pull set and SIF missing — skipping ${model_id}"
            SEQ_RESULTS+=("${model_id}:SKIP_NO_SIF")
            continue
        fi
        if [ ! -f "${sif_path}" ]; then
            clean_singularity_tmp
            echo "  [PULL] ${docker_image}"
            if [ "${DRY_RUN}" -eq 0 ]; then
                if ! singularity pull "${sif_path}" "docker://${docker_image}"; then
                    echo "  [FAIL] pull failed for ${model_id}"
                    clean_singularity_tmp
                    SEQ_RESULTS+=("${model_id}:PULL_FAIL")
                    continue
                fi
                echo "         → $(du -h "${sif_path}" | cut -f1)"
            fi
        else
            echo "  [OK]   SIF exists ($(du -h "${sif_path}" | cut -f1))"
        fi

        # SUBMIT (blocking)
        JOB_NAME="bm_${model_id}"
        LOG_OUT="${LOG_DIR}/${model_id}_%j.out"
        LOG_ERR="${LOG_DIR}/${model_id}_%j.err"

        WORKER_PATH="$(resolve_worker "${worker_file}")"
        SBATCH_CMD="sbatch --wait \
            --job-name=${JOB_NAME} \
            --output=${LOG_OUT} \
            --error=${LOG_ERR} \
            --export=ALL,MODEL_ID=${model_id},SIF_PATH=${sif_path},INTERFACE=${interface},YEAR=${year},OUTPUT_DIR=${OUTPUT_DIR},EXTRACTION_DIR=${EXTRACTION_DIR},CONDA_ENV_NAME=${CONDA_ENV_NAME},REPO_ROOT=${REPO_ROOT},BENCHMARK_DIR=${BENCHMARK_DIR},RAW_BRATS_MEN_DIR=${RAW_BRATS_MEN_DIR} \
            ${WORKER_PATH}"

        if [ "${DRY_RUN}" -eq 1 ]; then
            echo "  [DRY]  ${SBATCH_CMD}"
            SEQ_RESULTS+=("${model_id}:DRY")
        else
            echo "  [WAIT] sbatch --wait ${JOB_NAME} (blocking until job completes)"
            set +e
            eval "${SBATCH_CMD}"
            JOB_RC=$?
            set -e
            if [ ${JOB_RC} -eq 0 ]; then
                echo "  [DONE] ${model_id} job exit=0"
                SEQ_RESULTS+=("${model_id}:OK")
                if [ "${KEEP_SIF}" -eq 0 ]; then
                    echo "  [RM]   ${sif_path} (use --keep-sif to retain)"
                    rm -f "${sif_path}"
                fi
            else
                echo "  [FAIL] ${model_id} job exit=${JOB_RC} — keeping SIF for debug"
                SEQ_RESULTS+=("${model_id}:JOB_FAIL_${JOB_RC}")
            fi
        fi

        # Always purge the build scratch between models — even successful pulls
        # may leave several GB under .singularity_tmp.
        clean_singularity_tmp
        echo "  [free] $(df -h "${SIF_DIR}" | awk 'NR==2 {print $4" free on "$6}')"
    done

    echo ""
    echo "=========================================="
    echo "SEQUENTIAL SUMMARY"
    echo "=========================================="
    for r in "${SEQ_RESULTS[@]}"; do echo "  ${r}"; done
    echo ""
    echo "Run evaluation:"
    echo "  python3 ${BENCHMARK_DIR}/evaluate_benchmark.py --output-dir ${OUTPUT_DIR}"
    exit 0
fi

SUBMITTED_JOBS=()

for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_id docker_image year interface worker_file <<< "${entry}"

    # Filter by selected models if specified
    if [ -n "${SELECTED_MODELS}" ]; then
        if ! echo "${SELECTED_MODELS}" | grep -qw "${model_id}"; then
            continue
        fi
    fi

    sif_name="$(echo "${docker_image}" | tr '/:' '_').sif"
    sif_path="${SIF_DIR}/${sif_name}"

    if [ ! -f "${sif_path}" ] && [ "${DRY_RUN}" -eq 0 ]; then
        echo "  [WARN] SIF not found for ${model_id}: ${sif_path} — skipping"
        continue
    fi

    JOB_NAME="bm_${model_id}"
    LOG_OUT="${LOG_DIR}/${model_id}_%j.out"
    LOG_ERR="${LOG_DIR}/${model_id}_%j.err"

    WORKER_PATH="$(resolve_worker "${worker_file}")"
    SBATCH_CMD="sbatch --parsable \
        --job-name=${JOB_NAME} \
        --output=${LOG_OUT} \
        --error=${LOG_ERR} \
        --export=ALL,MODEL_ID=${model_id},SIF_PATH=${sif_path},INTERFACE=${interface},YEAR=${year},OUTPUT_DIR=${OUTPUT_DIR},EXTRACTION_DIR=${EXTRACTION_DIR},CONDA_ENV_NAME=${CONDA_ENV_NAME},REPO_ROOT=${REPO_ROOT},BENCHMARK_DIR=${BENCHMARK_DIR},RAW_BRATS_MEN_DIR=${RAW_BRATS_MEN_DIR} \
        ${WORKER_PATH}"

    if [ "${DRY_RUN}" -eq 0 ]; then
        JOB_ID=$(eval "${SBATCH_CMD}")
        echo "  [SUB]  ${model_id} → Job ${JOB_ID}"
        SUBMITTED_JOBS+=("${model_id}:${JOB_ID}")
    else
        echo "  [DRY]  ${model_id}: ${SBATCH_CMD}"
    fi
done

echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
if [ "${DRY_RUN}" -eq 0 ]; then
    echo "Submitted ${#SUBMITTED_JOBS[@]} jobs:"
    for j in "${SUBMITTED_JOBS[@]}"; do
        echo "  ${j}"
    done
    echo ""
    echo "Monitor: squeue -u \$USER -n bm_BraTS"
    echo "After completion, run evaluation:"
    echo "  python3 ${BENCHMARK_DIR}/evaluate_benchmark.py --output-dir ${OUTPUT_DIR}"
else
    echo "(Dry run — no jobs submitted)"
fi
