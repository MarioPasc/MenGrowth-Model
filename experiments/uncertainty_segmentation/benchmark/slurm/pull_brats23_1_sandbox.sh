#!/bin/bash
# STEP 1 of two-stage build for BraTS23_1 — run this on the LOGIN NODE.
#
# Builds an unpacked sandbox (directory tree) instead of a SIF.
# `singularity build --sandbox` fetches and extracts OCI layers but does
# NOT invoke mksquashfs, so it does not trip the login-node memory cap that
# kills `singularity pull` for this image.
#
# Then submit pull_brats23_1.sbatch on a compute node to convert the
# sandbox to a .sif (mksquashfs step, RAM-bound, no internet needed).
#
# Usage (login node, inside tmux/nohup — fetch takes ~30 min):
#   bash experiments/uncertainty_segmentation/benchmark/slurm/pull_brats23_1_sandbox.sh

set -euo pipefail

SIF_DIR="${SIF_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/singularity_images}"
DOCKER_IMAGE="brainles/brats23_meningioma_nvauto:latest"
SANDBOX_PATH="${SIF_DIR}/_brats23_1.sandbox"

mkdir -p "${SIF_DIR}"

module load singularity 2>/dev/null || true

export SINGULARITY_CACHEDIR="${SIF_DIR}/.singularity_cache"
export SINGULARITY_TMPDIR="${SIF_DIR}/.singularity_tmp"
mkdir -p "${SINGULARITY_CACHEDIR}" "${SINGULARITY_TMPDIR}"

# Wipe the half-finished build tree from the previous OOM'd attempt — it can
# silently consume tens of GB of fscratch.
find "${SINGULARITY_TMPDIR}" -maxdepth 1 -mindepth 1 -name 'build-*' -exec rm -rf {} + 2>/dev/null || true

echo "[sandbox] node=$(hostname)"
echo "[sandbox] image=${DOCKER_IMAGE}"
echo "[sandbox] target=${SANDBOX_PATH}"
echo "[sandbox] CACHEDIR=${SINGULARITY_CACHEDIR}"
echo "[sandbox] TMPDIR=${SINGULARITY_TMPDIR}"

if [ -d "${SANDBOX_PATH}" ]; then
    echo "[sandbox] sandbox already exists — delete it first if you want to re-fetch:"
    echo "          rm -rf ${SANDBOX_PATH}"
    exit 0
fi

# --sandbox skips squashfs entirely; layers land in ${SANDBOX_PATH}/.
# --fix-perms ensures world-readable so the compute node can read it back.
singularity build --sandbox --fix-perms "${SANDBOX_PATH}" "docker://${DOCKER_IMAGE}"

echo "[sandbox] done → $(du -sh "${SANDBOX_PATH}" | cut -f1)"
echo "[sandbox] next: sbatch experiments/uncertainty_segmentation/benchmark/slurm/pull_brats23_1.sbatch"
