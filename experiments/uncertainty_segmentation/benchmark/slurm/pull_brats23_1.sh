#!/bin/bash
#SBATCH --job-name=pull_brats23_1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=cascadelake

# STEP 2 of a two-stage build for BraTS23_1 (brainles/brats23_meningioma_nvauto).
#
# Why two stages:
#   Login node has internet but a small RAM cgroup → `singularity pull` OOM-
#   kills mksquashfs ("signal: killed").
#   Compute nodes have RAM but no internet → cannot reach registry-1.docker.io.
#
# Workflow:
#   STEP 1 (login node, see pull_brats23_1_sandbox.sh):
#     singularity build --sandbox <SANDBOX> docker://...
#     # Fetches OCI layers to a directory tree. Does NOT run mksquashfs,
#     # so it survives the login-node memory cap.
#
#   STEP 2 (this sbatch, compute node):
#     singularity build <SIF> <SANDBOX>
#     # Pure local mksquashfs pass — no internet, plenty of RAM.

set -euo pipefail

SIF_DIR="${SIF_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/singularity_images}"
DOCKER_IMAGE="brainles/brats23_meningioma_nvauto:latest"
SIF_NAME="$(echo "${DOCKER_IMAGE}" | tr '/:' '_').sif"
SIF_PATH="${SIF_DIR}/${SIF_NAME}"
SANDBOX_PATH="${SIF_DIR}/_brats23_1.sandbox"

module load singularity 2>/dev/null || true

export SINGULARITY_CACHEDIR="${SIF_DIR}/.singularity_cache"
export SINGULARITY_TMPDIR="${SIF_DIR}/.singularity_tmp"
mkdir -p "${SINGULARITY_CACHEDIR}" "${SINGULARITY_TMPDIR}"

echo "[build] node=$(hostname)"
echo "[build] sandbox=${SANDBOX_PATH}"
echo "[build] sif=${SIF_PATH}"
echo "[build] free mem: $(free -g | awk '/^Mem:/ {print $7" GiB available"}')"

if [ ! -d "${SANDBOX_PATH}" ]; then
    echo "[build] ERROR: sandbox missing — run pull_brats23_1_sandbox.sh on the login node first" >&2
    exit 1
fi

if [ -f "${SIF_PATH}" ]; then
    echo "[build] SIF already exists ($(du -h "${SIF_PATH}" | cut -f1)) — nothing to do"
    exit 0
fi

singularity build "${SIF_PATH}" "${SANDBOX_PATH}"

echo "[build] done → $(du -h "${SIF_PATH}" | cut -f1)"
echo "[build] sandbox can now be removed: rm -rf ${SANDBOX_PATH}"
