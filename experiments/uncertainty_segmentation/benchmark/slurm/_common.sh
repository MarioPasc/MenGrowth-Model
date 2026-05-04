#!/usr/bin/env bash
# =============================================================================
# COMMON BENCHMARK WORKER UTILITIES (sourced, never executed directly)
#
# Provides:
#   bm_setup_env          — load singularity + activate conda
#   bm_check_inputs       — validate SIF + extracted NIfTIs
#   bm_setup_workdir      — create work/input symlinks + work/output
#   bm_stage_writable     — copy a container path to a host writable dir
#   bm_save_metadata      — JSON metadata writer
#   bm_collect_predictions — copy *.nii.gz from work/output to predictions/
#
# Expected (exported) environment variables, set by benchmark_launcher.sh:
#   MODEL_ID SIF_PATH INTERFACE YEAR OUTPUT_DIR EXTRACTION_DIR
#   CONDA_ENV_NAME REPO_ROOT BENCHMARK_DIR
# =============================================================================

# ----------------------------------------------------------------------------
# bm_setup_env — load singularity module + activate conda env.
# ----------------------------------------------------------------------------
bm_setup_env() {
    module load singularity 2>/dev/null || true
    echo "[singularity] $(singularity --version 2>/dev/null || echo 'NOT FOUND')"

    local module_loaded=0
    local m
    for m in miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
        if module avail "$m" 2>&1 | grep -qi "${m}"; then
            module load "$m" && module_loaded=1 && break
        fi
    done
    [ "$module_loaded" -eq 0 ] && echo "[env] No conda module loaded; assuming conda in PATH."

    if command -v conda >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh" || true
        conda activate "${CONDA_ENV_NAME}" 2>/dev/null \
            || source activate "${CONDA_ENV_NAME}"
    else
        # shellcheck disable=SC1091
        source activate "${CONDA_ENV_NAME}"
    fi

    echo "[python] $(which python 2>/dev/null || echo 'NOT FOUND')"
    python -c "import sys; print('Python', sys.version.split()[0])" 2>/dev/null || true
    nvidia-smi --query-gpu=name,memory.total,driver_version \
        --format=csv,noheader 2>/dev/null || echo "[WARN] nvidia-smi missing"
    echo ""
}

# ----------------------------------------------------------------------------
# bm_check_inputs — fail fast if SIF or NIfTI dir missing.
# ----------------------------------------------------------------------------
bm_check_inputs() {
    if [ ! -f "${SIF_PATH}" ]; then
        echo "ERROR: SIF file not found: ${SIF_PATH}" >&2
        exit 1
    fi
    echo "[OK] SIF: $(du -h "${SIF_PATH}" | cut -f1)  ${SIF_PATH}"

    if [ ! -d "${NIFTI_DIR}" ]; then
        echo "ERROR: Extracted NIfTIs not found: ${NIFTI_DIR}" >&2
        exit 1
    fi
    N_PATIENTS=$(find "${NIFTI_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo "[OK] ${N_PATIENTS} patients in ${NIFTI_DIR}"

    set +e
    singularity exec --nv "${SIF_PATH}" \
        nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1
    local rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
        echo "[WARN] GPU not visible inside container — inference may fail."
    else
        echo "[OK]  GPU visible inside container."
    fi
}

# ----------------------------------------------------------------------------
# bm_setup_workdir — declare WORK_INPUT/WORK_OUTPUT/PRED_DIR and build the
# bind-arg array EXTRA_BINDS so containers can resolve symlink targets.
#
# WORK_INPUT is set to NIFTI_DIR directly (no per-patient symlink layer).
# Singularity does not follow symlinks whose absolute targets fall outside
# any bind mount; ``extract_from_raw`` writes symlinks under nifti/<id>/
# pointing to RAW_BRATS_MEN_DIR/<id>/, so we identity-bind that root too.
# ----------------------------------------------------------------------------
bm_setup_workdir() {
    MODEL_DIR="${OUTPUT_DIR}/models/${MODEL_ID}"
    WORK_INPUT="${NIFTI_DIR}"                       # bound directly, read-only
    WORK_OUTPUT="${MODEL_DIR}/work/output"
    PRED_DIR="${MODEL_DIR}/predictions"
    mkdir -p "${WORK_OUTPUT}" "${PRED_DIR}"

    EXTRA_BINDS=()
    if [ -n "${RAW_BRATS_MEN_DIR:-}" ] && [ -d "${RAW_BRATS_MEN_DIR}" ]; then
        # Identity-bind the raw root so symlinks under nifti/ resolve inside
        # the container at the same absolute path. Read-only; harmless.
        EXTRA_BINDS+=(--bind "${RAW_BRATS_MEN_DIR}:${RAW_BRATS_MEN_DIR}:ro")
        echo "[OK] WORK_INPUT  = ${WORK_INPUT}  (bound directly)"
        echo "[OK] EXTRA_BINDS = ${RAW_BRATS_MEN_DIR}:${RAW_BRATS_MEN_DIR}:ro"
    else
        echo "[OK] WORK_INPUT  = ${WORK_INPUT}  (bound directly)"
        echo "[WARN] RAW_BRATS_MEN_DIR not set — symlinks under nifti/ may"
        echo "       not resolve. Re-extract with --copy or set RAW_BRATS_MEN_DIR."
    fi
}

# ----------------------------------------------------------------------------
# bm_stage_writable <container_path> <host_stage_dir>
#
# Copies <container_path> from the SIF (read-only) into <host_stage_dir>
# so the run can bind-mount it back as :rw. Necessary for containers that
# write inside their own project tree (BraTS23 mlcube containers do this:
# blackbean → ./inputs, cnmc_pmi2023 → ./Dataset004_*).
# ----------------------------------------------------------------------------
bm_stage_writable() {
    local container_path="$1"
    local stage_dir="$2"
    if [ -z "${container_path}" ] || [ -z "${stage_dir}" ]; then
        echo "ERROR: bm_stage_writable requires <container_path> <stage_dir>" >&2
        return 1
    fi

    if [ -d "${stage_dir}" ] && [ -n "$(ls -A "${stage_dir}" 2>/dev/null)" ]; then
        echo "[stage] reusing ${stage_dir}"
        return 0
    fi
    mkdir -p "${stage_dir}"
    echo "[stage] copying ${container_path} → ${stage_dir}"
    singularity exec \
        --bind "${stage_dir}:/__bm_stage:rw" \
        "${SIF_PATH}" \
        sh -c "cp -a ${container_path}/. /__bm_stage/"
    echo "[stage] done ($(du -sh "${stage_dir}" | cut -f1))"
}

# ----------------------------------------------------------------------------
# bm_collect_predictions — copy *.nii.gz from WORK_OUTPUT to PRED_DIR.
# Sets N_PREDS globally.
# ----------------------------------------------------------------------------
bm_collect_predictions() {
    N_PREDS=0
    local nii rel dest
    while IFS= read -r -d '' nii; do
        rel="${nii#${WORK_OUTPUT}/}"
        dest="${PRED_DIR}/${rel}"
        mkdir -p "$(dirname "${dest}")"
        cp "${nii}" "${dest}"
        N_PREDS=$((N_PREDS + 1))
    done < <(find "${WORK_OUTPUT}" -name "*.nii.gz" -type f -print0 2>/dev/null)
    echo "[OK] Collected ${N_PREDS} predictions → ${PRED_DIR}"
    if [ "${N_PREDS}" -eq 0 ]; then
        echo "[WARN] No predictions found! work/output listing:"
        find "${WORK_OUTPUT}" -type f | head -20
    fi
}

# ----------------------------------------------------------------------------
# bm_save_metadata <status> <exit_code> <infer_seconds> <total_seconds>
# ----------------------------------------------------------------------------
bm_save_metadata() {
    local status="$1"
    local exit_code="$2"
    local infer_secs="$3"
    local total_secs="$4"
    python3 - <<PY
import json
meta = {
    "model_id": "${MODEL_ID}",
    "sif_path": "${SIF_PATH}",
    "interface": "${INTERFACE}",
    "year": int("${YEAR}"),
    "exit_code": ${exit_code},
    "inference_seconds": ${infer_secs},
    "total_seconds": ${total_secs},
    "status": "${status}",
    "n_patients_input": int("${N_PATIENTS:-0}"),
    "n_predictions": int("${N_PREDS:-0}"),
    "node": "$(hostname)",
    "job_id": "${SLURM_JOB_ID:-local}",
}
with open("${MODEL_DIR}/run_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)
PY
}

# ----------------------------------------------------------------------------
# bm_failure_diag — print debugging hints when inference exits non-zero.
# ----------------------------------------------------------------------------
bm_failure_diag() {
    local rc="$1"
    cat <<EOF

==========================================
FAILURE DIAGNOSTICS: ${MODEL_ID}
==========================================
Exit code: ${rc}

[DBG] Singularity: $(singularity --version 2>/dev/null)
[DBG] Container env (HOME, PWD, PATH):
$(singularity exec --cleanenv --no-home "${SIF_PATH}" env 2>/dev/null \
    | grep -E '^(HOME|PWD|PATH)=' || true)

Manual debug commands:
  singularity shell --nv ${SIF_PATH}
  singularity inspect --runscript ${SIF_PATH}
  singularity exec ${SIF_PATH} find / -maxdepth 3 -name '*.py' -not -path '*/lib/*'

Work directory preserved for debugging: ${MODEL_DIR}/work
EOF
}

# ----------------------------------------------------------------------------
# bm_header — banner + START_TIME and NIFTI_DIR globals.
# ----------------------------------------------------------------------------
bm_header() {
    START_TIME=$(date +%s)
    NIFTI_DIR="${EXTRACTION_DIR}/nifti"
    echo "=============================================="
    echo "BENCHMARK WORKER: ${MODEL_ID}"
    echo "=============================================="
    echo "  Job ID:     ${SLURM_JOB_ID:-local}"
    echo "  Node:       $(hostname)"
    echo "  SIF:        ${SIF_PATH}"
    echo "  Interface:  ${INTERFACE}"
    echo "  Year:       ${YEAR}"
    echo "  Output:     ${OUTPUT_DIR}"
    echo "  Extraction: ${EXTRACTION_DIR}"
    echo ""
}
