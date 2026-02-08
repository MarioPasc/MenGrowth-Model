#!/bin/bash
# experiments/lora_ablation/run_analysis.sh
#
# Run COMPLETE analysis pipeline on already-trained LoRA/DoRA conditions.
# This script re-computes features, probes, visualizations, tables, and
# enhanced diagnostics without re-training the models.
#
# The analysis pipeline includes:
#   1. Feature extraction (from trained checkpoints)
#   2. Domain feature extraction (glioma + meningioma for UMAP)
#   3. Probe evaluation (linear + MLP R² scores)
#   4. Test Dice evaluation (BraTS-MEN + BraTS-GLI)
#   5. Visualization generation (scatter plots, UMAP, variance)
#   6. Comprehensive table generation (CSV, LaTeX)
#   7. Statistical analysis (bootstrap CI, Wilcoxon, Holm-Bonferroni)
#   8. Enhanced diagnostics (gradient dynamics, feature quality, issue detection)
#
# Usage:
#   # Run analysis on LoRA + semantic heads experiment
#   ./run_analysis.sh --config server/LoRA_semantic_heads_icai.yaml
#
#   # Run analysis on all 4 experiments
#   ./run_analysis.sh --all
#
#   # Run with specific options
#   ./run_analysis.sh --config server/LoRA_semantic_heads_icai.yaml \
#       --domain-features --glioma-test-size 100
#
#   # Skip slow steps (useful for quick re-analysis)
#   ./run_analysis.sh --config server/LoRA_semantic_heads_icai.yaml \
#       --skip-extraction --skip-probes --skip-dice
#
#   # Dry run (show commands without executing)
#   ./run_analysis.sh --config server/LoRA_semantic_heads_icai.yaml --dry-run
#
# Requirements:
#   - Conda environment 'growth' activated
#   - Training must be complete (checkpoints exist)

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_DIR="$SCRIPT_DIR/config"

# Config files for all experiments
CONFIG_LORA_SEMANTIC="$CONFIG_DIR/server/LoRA_semantic_heads_icai.yaml"
CONFIG_LORA_NO_SEMANTIC="$CONFIG_DIR/server/LoRA_no_semantic_heads_icai.yaml"
CONFIG_DORA_SEMANTIC="$CONFIG_DIR/server/DoRA_semantic_heads_icai.yaml"
CONFIG_DORA_NO_SEMANTIC="$CONFIG_DIR/server/DoRA_no_semantic_heads_icai.yaml"

# Conda environment
CONDA_ENV="growth"

# Default options
DRY_RUN=false
RUN_ALL=false
CONFIG_FILE=""
DEVICE="cuda"
DOMAIN_FEATURES=false
N_GLIOMA=200
N_MENINGIOMA=200
GLIOMA_TEST_SIZE=200
SKIP_EXTRACTION=false
SKIP_PROBES=false
SKIP_DICE=false
DOMAIN_METRICS=false
DOMAIN_METRICS_ONLY=false
GPU_ID=""

# =============================================================================
# Parse arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --all)
            RUN_ALL=true
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --domain-features)
            DOMAIN_FEATURES=true
            shift
            ;;
        --n-glioma)
            N_GLIOMA="$2"
            shift 2
            ;;
        --n-meningioma)
            N_MENINGIOMA="$2"
            shift 2
            ;;
        --glioma-test-size)
            GLIOMA_TEST_SIZE="$2"
            shift 2
            ;;
        --skip-extraction)
            SKIP_EXTRACTION=true
            shift
            ;;
        --skip-probes)
            SKIP_PROBES=true
            shift
            ;;
        --skip-dice)
            SKIP_DICE=true
            shift
            ;;
        --domain-metrics)
            DOMAIN_METRICS=true
            shift
            ;;
        --domain-metrics-only)
            DOMAIN_METRICS_ONLY=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config FILE        Config file to use (relative to config/ or absolute)"
            echo "  --all                Run analysis on all 4 experiments"
            echo "  --device DEVICE      Device to use (default: cuda)"
            echo "  --gpu N              Set CUDA_VISIBLE_DEVICES=N"
            echo "  --domain-features    Extract domain features for UMAP visualization"
            echo "  --n-glioma N         Number of glioma samples for domain UMAP (default: 200)"
            echo "  --n-meningioma N     Number of meningioma samples for domain UMAP (default: 200)"
            echo "  --glioma-test-size N Number of glioma subjects for test Dice (default: 200)"
            echo "  --skip-extraction    Skip feature extraction (use existing)"
            echo "  --skip-probes        Skip probe evaluation (use existing)"
            echo "  --skip-dice          Skip Dice evaluation (use existing)"
            echo "  --domain-metrics     Also run domain gap metrics (MMD, CKA, PAD)"
            echo "  --domain-metrics-only Run ONLY domain gap metrics (skip all other steps)"
            echo "  --dry-run            Show commands without executing"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Run analysis on LoRA + semantic heads"
            echo "  $0 --config server/LoRA_semantic_heads_icai.yaml"
            echo ""
            echo "  # Run analysis on all experiments"
            echo "  $0 --all"
            echo ""
            echo "  # Quick re-analysis (skip slow steps)"
            echo "  $0 --config server/LoRA_semantic_heads_icai.yaml --skip-extraction --skip-probes"
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

run_analysis() {
    local config=$1
    local name=$2

    log_header "ANALYSIS: $name"
    echo "Config: $config"
    echo ""

    # Build command
    local cmd="python -m experiments.lora_ablation.run_ablation --config $config analyze-only"
    cmd="$cmd --device $DEVICE"
    cmd="$cmd --glioma-test-size $GLIOMA_TEST_SIZE"

    if [ "$DOMAIN_FEATURES" = true ]; then
        cmd="$cmd --domain-features --n-glioma $N_GLIOMA --n-meningioma $N_MENINGIOMA"
    fi

    if [ "$SKIP_EXTRACTION" = true ]; then
        cmd="$cmd --skip-extraction"
    fi

    if [ "$SKIP_PROBES" = true ]; then
        cmd="$cmd --skip-probes"
    fi

    if [ "$SKIP_DICE" = true ]; then
        cmd="$cmd --skip-dice"
    fi

    # Add GPU prefix if specified
    if [ -n "$GPU_ID" ]; then
        cmd="CUDA_VISIBLE_DEVICES=$GPU_ID $cmd"
    fi

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

    log_info "Analysis complete for $name!"
}

run_domain_metrics() {
    local config=$1
    local name=$2

    log_header "DOMAIN METRICS: $name"
    echo "Config: $config"
    echo ""

    local cmd="python -m experiments.lora_ablation.compute_domain_metrics --config $config"

    if [ "$RUN_ALL" = true ]; then
        cmd="$cmd --all"
    fi

    echo "Command:"
    echo "  $cmd"
    echo ""

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would execute the above command"
        echo ""
        return 0
    fi

    eval "$cmd"

    log_info "Domain metrics complete for $name!"
}

# =============================================================================
# Verify environment
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

# Change to project root
cd "$PROJECT_ROOT"
echo "[OK] Working directory: $(pwd)"

# Show run configuration
echo ""
echo "Run configuration:"
echo "  - Device: $DEVICE"
if [ -n "$GPU_ID" ]; then
    echo "  - GPU: $GPU_ID"
fi
echo "  - Domain features: $DOMAIN_FEATURES"
echo "  - Skip extraction: $SKIP_EXTRACTION"
echo "  - Skip probes: $SKIP_PROBES"
echo "  - Skip dice: $SKIP_DICE"
echo "  - Domain metrics: $DOMAIN_METRICS"
echo "  - Domain metrics only: $DOMAIN_METRICS_ONLY"
echo "  - Dry run: $DRY_RUN"

# =============================================================================
# Run analysis
# =============================================================================

if [ "$DOMAIN_METRICS_ONLY" = true ]; then
    # Run ONLY domain metrics (no feature extraction, probes, dice, etc.)
    if [ "$RUN_ALL" = true ]; then
        log_info "Running domain metrics on all 4 experiments..."
        # Use one config as entry point; --all auto-discovers siblings
        if [ -f "$CONFIG_LORA_SEMANTIC" ]; then
            run_domain_metrics "$CONFIG_LORA_SEMANTIC" "All experiments"
        fi
    elif [ -n "$CONFIG_FILE" ]; then
        # Handle relative paths
        if [[ ! "$CONFIG_FILE" = /* ]]; then
            if [ -f "$CONFIG_DIR/$CONFIG_FILE" ]; then
                CONFIG_FILE="$CONFIG_DIR/$CONFIG_FILE"
            elif [ -f "$PROJECT_ROOT/$CONFIG_FILE" ]; then
                CONFIG_FILE="$PROJECT_ROOT/$CONFIG_FILE"
            fi
        fi
        if [ ! -f "$CONFIG_FILE" ]; then
            echo "ERROR: Config file not found: $CONFIG_FILE"
            exit 1
        fi
        run_domain_metrics "$CONFIG_FILE" "$(basename $CONFIG_FILE .yaml)"
    else
        echo "ERROR: Must specify --config FILE or --all"
        echo "Run with -h for help"
        exit 1
    fi

elif [ "$RUN_ALL" = true ]; then
    # Run all 4 experiments
    log_info "Running analysis on all 4 experiments..."

    if [ -f "$CONFIG_LORA_SEMANTIC" ]; then
        run_analysis "$CONFIG_LORA_SEMANTIC" "LoRA + Semantic Heads"
    fi

    if [ -f "$CONFIG_LORA_NO_SEMANTIC" ]; then
        run_analysis "$CONFIG_LORA_NO_SEMANTIC" "LoRA + No Semantic Heads"
    fi

    if [ -f "$CONFIG_DORA_SEMANTIC" ]; then
        run_analysis "$CONFIG_DORA_SEMANTIC" "DoRA + Semantic Heads"
    fi

    if [ -f "$CONFIG_DORA_NO_SEMANTIC" ]; then
        run_analysis "$CONFIG_DORA_NO_SEMANTIC" "DoRA + No Semantic Heads"
    fi

    # Run domain metrics after all analysis if flag set
    if [ "$DOMAIN_METRICS" = true ]; then
        if [ -f "$CONFIG_LORA_SEMANTIC" ]; then
            run_domain_metrics "$CONFIG_LORA_SEMANTIC" "All experiments"
        fi
    fi

elif [ -n "$CONFIG_FILE" ]; then
    # Run single config

    # Handle relative paths
    if [[ ! "$CONFIG_FILE" = /* ]]; then
        # Not absolute path, check if it's relative to config dir
        if [ -f "$CONFIG_DIR/$CONFIG_FILE" ]; then
            CONFIG_FILE="$CONFIG_DIR/$CONFIG_FILE"
        elif [ -f "$PROJECT_ROOT/$CONFIG_FILE" ]; then
            CONFIG_FILE="$PROJECT_ROOT/$CONFIG_FILE"
        fi
    fi

    if [ ! -f "$CONFIG_FILE" ]; then
        echo "ERROR: Config file not found: $CONFIG_FILE"
        exit 1
    fi

    run_analysis "$CONFIG_FILE" "$(basename $CONFIG_FILE .yaml)"

    # Run domain metrics after analysis if flag set
    if [ "$DOMAIN_METRICS" = true ]; then
        run_domain_metrics "$CONFIG_FILE" "$(basename $CONFIG_FILE .yaml)"
    fi

else
    echo "ERROR: Must specify --config FILE or --all"
    echo "Run with -h for help"
    exit 1
fi

# =============================================================================
# Summary
# =============================================================================

log_header "COMPLETE"

echo ""
echo "Analysis complete!"
echo ""
echo "Key output files generated:"
echo "  - comprehensive_results.csv (all metrics)"
echo "  - comprehensive_table.tex (LaTeX table)"
echo "  - statistical_recommendation.txt (recommendation)"
echo "  - analysis_report.md (full report)"
echo "  - figures/ (visualizations)"
echo ""
echo "Enhanced diagnostics files:"
echo "  - diagnostics_gradients.csv (gradient dynamics analysis)"
echo "  - diagnostics_features.csv (feature quality metrics)"
echo "  - diagnostics_probes.csv (probe quality analysis)"
echo "  - diagnostics_loss.csv (loss dynamics)"
echo "  - diagnostics_issues.csv (detected issues)"
echo "  - diagnostics_report.txt (comprehensive report)"
echo ""
echo "Domain metrics files (if --domain-metrics used):"
echo "  - conditions/<cond>/domain_metrics.json (per-condition)"
echo "  - domain_metrics_summary.csv (all conditions)"
echo "  - figures/domain_gap_metrics.{pdf,png}"
echo "  - figures/domain_effective_rank.{pdf,png}"
echo "  - figures/domain_adaptation_tradeoff.{pdf,png}"
echo "  - figures/domain_umap_*.{pdf,png}"
