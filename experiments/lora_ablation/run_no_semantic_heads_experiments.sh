#!/bin/bash
# experiments/lora_ablation/run_all_experiments.sh
#
# Run all LoRA and DoRA experiments with and without semantic heads.
#
# Experiments:
#   1. LoRA + semantic heads (GPU 0)
#   2. LoRA + no semantic heads (GPU 1)
#   3. DoRA + semantic heads (GPU 0)
#   4. DoRA + no semantic heads (GPU 1)
#
# Usage:
#   # Run all 4 experiments sequentially (one at a time)
#   ./run_all_experiments.sh
#
#   # Run only GPU 0 jobs (experiments 1 and 3) - use in Terminal 1
#   ./run_all_experiments.sh --gpu 0
#
#   # Run only GPU 1 jobs (experiments 2 and 4) - use in Terminal 2
#   ./run_all_experiments.sh --gpu 1
#
#   # Dry run (show commands without executing)
#   ./run_all_experiments.sh --dry-run
#
#   # Resume from a specific experiment (1-4)
#   ./run_all_experiments.sh --start-from 3
#
# For parallel execution across 2 GPUs, open 2 terminals:
#   Terminal 1: ./run_all_experiments.sh --gpu 0
#   Terminal 2: ./run_all_experiments.sh --gpu 1
#
# Requirements:
#   - Conda environment 'growth' activated
#   - CUDA GPUs available
#   - Data paths configured in YAML files

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Config files
CONFIG_LORA_SEMANTIC="$SCRIPT_DIR/config/server/LoRA_semantic_heads_icai.yaml"
CONFIG_LORA_NO_SEMANTIC="$SCRIPT_DIR/config/server/LoRA_no_semantic_heads_icai.yaml"
CONFIG_DORA_SEMANTIC="$SCRIPT_DIR/config/server/DoRA_semantic_heads_icai.yaml"
CONFIG_DORA_NO_SEMANTIC="$SCRIPT_DIR/config/server/DoRA_no_semantic_heads_icai.yaml"

# Conda environment
CONDA_ENV="growth"

# Default options
DRY_RUN=false
START_FROM=1
GPU_FILTER=""  # Empty means run all, "0" or "1" to filter
DOMAIN_FEATURES=true
N_GLIOMA=200
N_MENINGIOMA=200
GLIOMA_TEST_SIZE=200

# =============================================================================
# Parse arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_FILTER="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --start-from)
            START_FROM="$2"
            shift 2
            ;;
        --no-domain-features)
            DOMAIN_FEATURES=false
            shift
            ;;
        --glioma-test-size)
            GLIOMA_TEST_SIZE="$2"
            shift 2
            ;;
        --n-glioma)
            N_GLIOMA="$2"
            shift 2
            ;;
        --n-meningioma)
            N_MENINGIOMA="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpu N              Only run experiments for GPU N (0 or 1)"
            echo "                       GPU 0: LoRA+semantic, DoRA+semantic"
            echo "                       GPU 1: LoRA+no-semantic, DoRA+no-semantic"
            echo "  --dry-run            Show commands without executing"
            echo "  --start-from N       Start from experiment N (1-4)"
            echo "  --no-domain-features Skip domain feature extraction"
            echo "  --n-glioma N         Number of glioma samples for domain UMAP (default: 200)"
            echo "  --n-meningioma N     Number of meningioma samples for domain UMAP (default: 200)"
            echo "  --glioma-test-size N Number of glioma subjects for test Dice (default: 200)"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "For parallel execution, open 2 terminals:"
            echo "  Terminal 1: $0 --gpu 0"
            echo "  Terminal 2: $0 --gpu 1"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Helper functions
# =============================================================================

log_header() {
    echo ""
    echo "========================================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "========================================================================"
}

log_info() {
    echo "[$(date '+%H:%M:%S')] $1"
}

run_experiment() {
    local gpu=$1
    local config=$2
    local name=$3
    local exp_num=$4

    # Check GPU filter
    if [ -n "$GPU_FILTER" ] && [ "$GPU_FILTER" != "$gpu" ]; then
        log_info "Skipping experiment $exp_num ($name) - GPU $gpu not selected"
        return 0
    fi

    # Check start-from
    if [ "$exp_num" -lt "$START_FROM" ]; then
        log_info "Skipping experiment $exp_num ($name) - before start-from"
        return 0
    fi

    local domain_flag=""
    if [ "$DOMAIN_FEATURES" = true ]; then
        domain_flag="--domain-features --n-glioma $N_GLIOMA --n-meningioma $N_MENINGIOMA"
    fi

    log_header "EXPERIMENT $exp_num: $name (GPU $gpu)"
    echo "Config: $config"
    echo ""

    local cmd="CUDA_VISIBLE_DEVICES=$gpu python -m experiments.lora_ablation.run_ablation \
        --config $config \
        run-all $domain_flag --glioma-test-size $GLIOMA_TEST_SIZE"

    echo "Command:"
    echo "  $cmd"
    echo ""

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would execute the above command"
        echo ""
        return 0
    fi

    # Run with live output
    eval "$cmd"

    log_info "Experiment $exp_num complete!"
}

# =============================================================================
# Verify environment and configurations
# =============================================================================

log_header "Setup"

# Check conda environment
if [[ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV" ]]; then
    echo "WARNING: Not in '$CONDA_ENV' conda environment (current: $CONDA_DEFAULT_ENV)"
    echo "Please run: conda activate $CONDA_ENV"
    exit 1
fi
echo "[OK] Conda environment: $CONDA_ENV"
echo "[OK] Python: $(which python)"

# Verify config files
echo ""
echo "Verifying configuration files..."
for config in "$CONFIG_LORA_SEMANTIC" "$CONFIG_LORA_NO_SEMANTIC" \
              "$CONFIG_DORA_SEMANTIC" "$CONFIG_DORA_NO_SEMANTIC"; do
    if [ ! -f "$config" ]; then
        echo "ERROR: Configuration file not found: $config"
        exit 1
    fi
    echo "  [OK] $(basename $config)"
done

# Change to project root
cd "$PROJECT_ROOT"
echo ""
echo "Working directory: $(pwd)"

# Show run configuration
echo ""
echo "Run configuration:"
echo "  - GPU filter: ${GPU_FILTER:-all}"
echo "  - Start from: experiment $START_FROM"
echo "  - Domain features: $DOMAIN_FEATURES"
if [ "$DOMAIN_FEATURES" = true ]; then
    echo "    - n_glioma: $N_GLIOMA"
    echo "    - n_meningioma: $N_MENINGIOMA"
fi
echo "  - Glioma test size: $GLIOMA_TEST_SIZE"
echo "  - Dry run: $DRY_RUN"

# =============================================================================
# Run experiments
# =============================================================================

# Experiment 2: LoRA + No Semantic Heads (GPU 0)
run_experiment 0 "$CONFIG_LORA_NO_SEMANTIC" "LoRA + No Semantic Heads" 2

# Experiment 4: DoRA + No Semantic Heads (GPU 1)
run_experiment 1 "$CONFIG_DORA_NO_SEMANTIC" "DoRA + No Semantic Heads" 4

# =============================================================================
# Summary
# =============================================================================

log_header "COMPLETE"

if [ -n "$GPU_FILTER" ]; then
    echo "Completed all experiments for GPU $GPU_FILTER"
else
    echo "Completed all experiments"
fi

echo ""
echo "Results directories:"
if [ -z "$GPU_FILTER" ] || [ "$GPU_FILTER" = "0" ]; then
    echo "  - LoRA + Semantic:  /media/hddb/mario/results/growth/lora_ablation_semantic_heads/"
    echo "  - DoRA + Semantic:  /media/hddb/mario/results/growth/dora_ablation_semantic_heads/"
fi
if [ -z "$GPU_FILTER" ] || [ "$GPU_FILTER" = "1" ]; then
    echo "  - LoRA + No Semantic: /media/hddb/mario/results/growth/lora_ablation_no_semantic_heads/"
    echo "  - DoRA + No Semantic: /media/hddb/mario/results/growth/dora_ablation_no_semantic_heads/"
fi

echo ""
echo "Key output files:"
echo "  - comprehensive_results.csv"
echo "  - test_dice_summary.csv"
echo "  - comprehensive_table.tex"
