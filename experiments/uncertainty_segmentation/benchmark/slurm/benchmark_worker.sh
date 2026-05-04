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
# Replaced by per-model workers under slurm/workers/. The launcher dispatches
# to the right one via the MODELS registry (5th field).
# =============================================================================
echo "ERROR: benchmark_worker.sh is deprecated." >&2
echo "       Use slurm/workers/<model>.sh — see benchmark_launcher.sh." >&2
exit 2
