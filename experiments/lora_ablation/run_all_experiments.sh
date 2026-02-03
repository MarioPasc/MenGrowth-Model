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
#   # Run all experiments sequentially (one at a time)
#   ./run_all_experiments.sh
#
#   # Run 2 experiments in parallel (one per GPU)
#   ./run_all_experiments.sh --parallel
#
#   # Dry run (show commands without executing)
#   ./run_all_experiments.sh --dry-run
#
#   # Resume from a specific experiment (1-4)
#   ./run_all_experiments.sh --start-from 3
#
#   # Skip domain features extraction
#   ./run_all_experiments.sh --no-domain-features
#
# Requirements:
#   - CUDA GPUs 0 and 1 available
#   - Python environment with all dependencies
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

# Default options
PARALLEL=false
DRY_RUN=false
START_FROM=1
DOMAIN_FEATURES=true
GLIOMA_TEST_SIZE=200

# =============================================================================
# Parse arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            PARALLEL=true
            shift
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
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --parallel           Run 2 experiments in parallel (one per GPU)"
            echo "  --dry-run            Show commands without executing"
            echo "  --start-from N       Start from experiment N (1-4)"
            echo "  --no-domain-features Skip domain feature extraction"
            echo "  --glioma-test-size N Number of glioma subjects for test (default: 200)"
            echo "  -h, --help           Show this help message"
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

log_info() {
    echo ""
    echo "========================================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "========================================================================"
}

run_experiment() {
    local gpu=$1
    local config=$2
    local name=$3

    local domain_flag=""
    if [ "$DOMAIN_FEATURES" = true ]; then
        domain_flag="--domain-features"
    fi

    local cmd="CUDA_VISIBLE_DEVICES=$gpu python -m experiments.lora_ablation.run_ablation \\
        --config $config \\
        run-all $domain_flag --glioma-test-size $GLIOMA_TEST_SIZE"

    log_info "Running: $name (GPU $gpu)"
    echo "Config: $config"
    echo ""
    echo "Command:"
    echo "$cmd"
    echo ""

    if [ "$DRY_RUN" = false ]; then
        eval "$cmd"
    else
        echo "[DRY RUN] Would execute the above command"
    fi
}

run_experiment_background() {
    local gpu=$1
    local config=$2
    local name=$3
    local logfile=$4

    local domain_flag=""
    if [ "$DOMAIN_FEATURES" = true ]; then
        domain_flag="--domain-features"
    fi

    local cmd="CUDA_VISIBLE_DEVICES=$gpu python -m experiments.lora_ablation.run_ablation \\
        --config $config \\
        run-all $domain_flag --glioma-test-size $GLIOMA_TEST_SIZE"

    log_info "Starting: $name (GPU $gpu) [background]"
    echo "Config: $config"
    echo "Log: $logfile"
    echo ""

    if [ "$DRY_RUN" = false ]; then
        eval "$cmd" > "$logfile" 2>&1 &
        echo $!  # Return PID
    else
        echo "[DRY RUN] Would execute in background"
        echo 0
    fi
}

# =============================================================================
# Verify configurations exist
# =============================================================================

log_info "Verifying configuration files"

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

# =============================================================================
# Run experiments
# =============================================================================

if [ "$PARALLEL" = true ]; then
    # =======================================================================
    # PARALLEL MODE: Run 2 experiments at a time (one per GPU)
    # =======================================================================

    LOG_DIR="$PROJECT_ROOT/logs/lora_ablation_$(date '+%Y%m%d_%H%M%S')"
    mkdir -p "$LOG_DIR"

    # Batch 1: LoRA semantic (GPU 0) + LoRA no-semantic (GPU 1)
    if [ "$START_FROM" -le 2 ]; then
        log_info "BATCH 1: LoRA experiments (parallel on GPUs 0 and 1)"

        if [ "$START_FROM" -le 1 ]; then
            PID1=$(run_experiment_background 0 "$CONFIG_LORA_SEMANTIC" "LoRA + Semantic Heads" "$LOG_DIR/lora_semantic.log")
        fi

        if [ "$START_FROM" -le 2 ]; then
            PID2=$(run_experiment_background 1 "$CONFIG_LORA_NO_SEMANTIC" "LoRA + No Semantic Heads" "$LOG_DIR/lora_no_semantic.log")
        fi

        if [ "$DRY_RUN" = false ]; then
            echo "Waiting for batch 1 to complete..."
            echo "  - LoRA + Semantic Heads (PID: $PID1)"
            echo "  - LoRA + No Semantic Heads (PID: $PID2)"
            echo ""
            echo "Monitor logs with:"
            echo "  tail -f $LOG_DIR/lora_semantic.log"
            echo "  tail -f $LOG_DIR/lora_no_semantic.log"
            echo ""

            wait $PID1 $PID2
            echo "Batch 1 complete."
        fi
    fi

    # Batch 2: DoRA semantic (GPU 0) + DoRA no-semantic (GPU 1)
    if [ "$START_FROM" -le 4 ]; then
        log_info "BATCH 2: DoRA experiments (parallel on GPUs 0 and 1)"

        if [ "$START_FROM" -le 3 ]; then
            PID3=$(run_experiment_background 0 "$CONFIG_DORA_SEMANTIC" "DoRA + Semantic Heads" "$LOG_DIR/dora_semantic.log")
        fi

        if [ "$START_FROM" -le 4 ]; then
            PID4=$(run_experiment_background 1 "$CONFIG_DORA_NO_SEMANTIC" "DoRA + No Semantic Heads" "$LOG_DIR/dora_no_semantic.log")
        fi

        if [ "$DRY_RUN" = false ]; then
            echo "Waiting for batch 2 to complete..."
            echo "  - DoRA + Semantic Heads (PID: $PID3)"
            echo "  - DoRA + No Semantic Heads (PID: $PID4)"
            echo ""

            wait $PID3 $PID4
            echo "Batch 2 complete."
        fi
    fi

else
    # =======================================================================
    # SEQUENTIAL MODE: Run experiments one at a time
    # =======================================================================

    # Experiment 1: LoRA + Semantic Heads (GPU 0)
    if [ "$START_FROM" -le 1 ]; then
        run_experiment 0 "$CONFIG_LORA_SEMANTIC" "LoRA + Semantic Heads"
    fi

    # Experiment 2: LoRA + No Semantic Heads (GPU 1)
    if [ "$START_FROM" -le 2 ]; then
        run_experiment 1 "$CONFIG_LORA_NO_SEMANTIC" "LoRA + No Semantic Heads"
    fi

    # Experiment 3: DoRA + Semantic Heads (GPU 0)
    if [ "$START_FROM" -le 3 ]; then
        run_experiment 0 "$CONFIG_DORA_SEMANTIC" "DoRA + Semantic Heads"
    fi

    # Experiment 4: DoRA + No Semantic Heads (GPU 1)
    if [ "$START_FROM" -le 4 ]; then
        run_experiment 1 "$CONFIG_DORA_NO_SEMANTIC" "DoRA + No Semantic Heads"
    fi
fi

# =============================================================================
# Summary
# =============================================================================

log_info "ALL EXPERIMENTS COMPLETE"

echo ""
echo "Results are saved in the following directories:"
echo "  1. LoRA + Semantic:    See output_dir in $CONFIG_LORA_SEMANTIC"
echo "  2. LoRA + No Semantic: See output_dir in $CONFIG_LORA_NO_SEMANTIC"
echo "  3. DoRA + Semantic:    See output_dir in $CONFIG_DORA_SEMANTIC"
echo "  4. DoRA + No Semantic: See output_dir in $CONFIG_DORA_NO_SEMANTIC"
echo ""
echo "Key output files per experiment:"
echo "  - comprehensive_results.csv (all metrics)"
echo "  - comprehensive_table.tex (LaTeX table)"
echo "  - test_dice_summary.csv (Dice scores)"
echo "  - domain_shift_analysis.csv (MEN vs GLI)"
echo ""

if [ "$PARALLEL" = true ] && [ "$DRY_RUN" = false ]; then
    echo "Log files:"
    echo "  - $LOG_DIR/lora_semantic.log"
    echo "  - $LOG_DIR/lora_no_semantic.log"
    echo "  - $LOG_DIR/dora_semantic.log"
    echo "  - $LOG_DIR/dora_no_semantic.log"
fi
