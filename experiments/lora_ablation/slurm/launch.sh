#!/usr/bin/env bash
# =============================================================================
# LORA ABLATION — PICASSO LAUNCHER
#
# Login-node script that submits 4 training workers (one per experiment) and
# 1 analysis worker (runs after all 4 complete) on Picasso supercomputer.
#
# 4 experiments (1 A100 GPU each):
#   1. LoRA + semantic heads
#   2. LoRA + no semantic heads
#   3. DoRA + semantic heads
#   4. DoRA + no semantic heads
#
# Usage (from login node):
#   cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth-Model
#   bash experiments/lora_ablation/slurm/picasso/launch.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "LORA ABLATION — PICASSO LAUNCHER"
echo "=========================================="
echo "Time: $(date)"
echo ""

# ========================================================================
# CONFIGURATION — Edit these paths before first run
# ========================================================================
export REPO_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/MenGrowth-Model"
export CONDA_ENV_NAME="growth"
export RESULTS_BASE="/mnt/home/users/tic_163_uma/mpascual/execs/growth/results"

CONFIGS_DIR="${REPO_SRC}/experiments/lora_ablation/config/picasso"

# Config files for each experiment
CONFIGS=(
    "${CONFIGS_DIR}/LoRA_semantic_heads_icai.yaml"
    "${CONFIGS_DIR}/LoRA_no_semantic_heads_icai.yaml"
    "${CONFIGS_DIR}/DoRA_semantic_heads_icai.yaml"
    "${CONFIGS_DIR}/DoRA_no_semantic_heads_icai.yaml"
)

NAMES=(
    "lora_sem"
    "lora_nosem"
    "dora_sem"
    "dora_nosem"
)

echo "Configuration:"
echo "  Repo:        ${REPO_SRC}"
echo "  Configs:     ${CONFIGS_DIR}"
echo "  Results:     ${RESULTS_BASE}"
echo "  Conda env:   ${CONDA_ENV_NAME}"
echo ""

# ========================================================================
# PRE-FLIGHT: Verify all config files exist
# ========================================================================
echo "Pre-flight checks:"
ALL_OK=true
for cfg in "${CONFIGS[@]}"; do
    if [ -f "$cfg" ]; then
        echo "  [OK]   $(basename "$cfg")"
    else
        echo "  [MISS] $cfg"
        ALL_OK=false
    fi
done

if [ "$ALL_OK" != "true" ]; then
    echo ""
    echo "ERROR: Missing config files. Aborting."
    exit 1
fi

# Check for PLACEHOLDER paths
if grep -q "/PLACEHOLDER/" "${CONFIGS[0]}"; then
    echo ""
    echo "ERROR: Config files still contain /PLACEHOLDER/ paths."
    echo "Edit the picasso configs to set correct data/checkpoint/output paths."
    exit 1
fi

echo ""

# ========================================================================
# SUBMIT TRAINING WORKERS
# ========================================================================
echo "Submitting training jobs..."

JOB_IDS=()
for i in "${!CONFIGS[@]}"; do
    JOB_ID=$(sbatch --parsable \
        --job-name="ablation_${NAMES[$i]}" \
        --output="${RESULTS_BASE}/slurm_logs/train_${NAMES[$i]}_%j.out" \
        --error="${RESULTS_BASE}/slurm_logs/train_${NAMES[$i]}_%j.err" \
        --export=ALL,CONFIG_PATH="${CONFIGS[$i]}" \
        "${SCRIPT_DIR}/train_worker.sh")

    JOB_IDS+=("$JOB_ID")
    echo "  [${NAMES[$i]}] Job ID: ${JOB_ID} — config: $(basename "${CONFIGS[$i]}")"
done

echo ""

# ========================================================================
# SUBMIT ANALYSIS WORKER (after all training completes)
# ========================================================================
DEPENDENCY=$(IFS=:; echo "${JOB_IDS[*]}")

# Create SLURM log directory
mkdir -p "${RESULTS_BASE}/slurm_logs"

ANALYSIS_JOB=$(sbatch --parsable \
    --job-name="ablation_report" \
    --dependency="afterok:${DEPENDENCY}" \
    --output="${RESULTS_BASE}/slurm_logs/analysis_%j.out" \
    --error="${RESULTS_BASE}/slurm_logs/analysis_%j.err" \
    --export=ALL,RESULTS_DIR="${RESULTS_BASE}" \
    "${SCRIPT_DIR}/analysis_worker.sh")

echo "Analysis job: ${ANALYSIS_JOB} (runs after all training completes)"
echo ""

# ========================================================================
# MONITORING COMMANDS
# ========================================================================
echo "=========================================="
echo "ALL JOBS SUBMITTED"
echo "=========================================="
echo ""
echo "Monitor all jobs:"
echo "  squeue -u \$USER"
echo ""
echo "Individual job status:"
for i in "${!JOB_IDS[@]}"; do
    echo "  squeue -j ${JOB_IDS[$i]}  # ${NAMES[$i]}"
done
echo "  squeue -j ${ANALYSIS_JOB}  # analysis (dependent)"
echo ""
echo "View logs:"
echo "  tail -f ${RESULTS_BASE}/slurm_logs/train_*_<JOB_ID>.out"
echo ""
echo "Cancel all:"
echo "  scancel ${JOB_IDS[*]} ${ANALYSIS_JOB}"
echo ""
echo "Estimated timeline:"
echo "  Training:  ~27 hours per experiment (48h limit)"
echo "  Analysis:  ~30 min after all training completes"
echo "  Total:     ~28 hours (all 4 train in parallel)"
