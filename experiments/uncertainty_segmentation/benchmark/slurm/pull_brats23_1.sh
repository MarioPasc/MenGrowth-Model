#!/bin/bash
#SBATCH --job-name=pull_brats23_1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=cascadelake

# Pull BraTS23_1 (brainles/brats23_meningioma_nvauto) on a compute node.
# Login-node pulls of this image OOM-kill mksquashfs ("signal: killed" while
# creating squashfs). Compute nodes have ~64-128 GB RAM and no cgroup memcap.
#
# Submit from the login node:
#   sbatch experiments/uncertainty_segmentation/benchmark/slurm/pull_brats23_1.sbatch
#
# After it completes, re-run the benchmark launcher with --skip-pull (or just
# rerun normally — the launcher will detect the existing .sif).

set -euo pipefail

SIF_DIR="${SIF_DIR:-/mnt/home/users/tic_163_uma/mpascual/fscratch/singularity_images}"
DOCKER_IMAGE="brainles/brats23_meningioma_nvauto:latest"
SIF_NAME="$(echo "${DOCKER_IMAGE}" | tr '/:' '_').sif"
SIF_PATH="${SIF_DIR}/${SIF_NAME}"

mkdir -p "${SIF_DIR}"

module load singularity 2>/dev/null || true

export SINGULARITY_CACHEDIR="${SIF_DIR}/.singularity_cache"
export SINGULARITY_TMPDIR="${SIF_DIR}/.singularity_tmp"
mkdir -p "${SINGULARITY_CACHEDIR}" "${SINGULARITY_TMPDIR}"

# Wipe any partial build trees from the failed login-node attempt so we don't
# resume on top of a half-unpacked rootfs.
find "${SINGULARITY_TMPDIR}" -maxdepth 1 -mindepth 1 -name 'build-*' -exec rm -rf {} + 2>/dev/null || true

echo "[pull] node=$(hostname)  image=${DOCKER_IMAGE}"
echo "[pull] target=${SIF_PATH}"
echo "[pull] CACHEDIR=${SINGULARITY_CACHEDIR}"
echo "[pull] TMPDIR=${SINGULARITY_TMPDIR}"
echo "[pull] free mem: $(free -g | awk '/^Mem:/ {print $7" GiB available"}')"

if [ -f "${SIF_PATH}" ]; then
    echo "[pull] SIF already exists ($(du -h "${SIF_PATH}" | cut -f1)) — nothing to do"
    exit 0
fi

singularity pull "${SIF_PATH}" "docker://${DOCKER_IMAGE}"

echo "[pull] done → $(du -h "${SIF_PATH}" | cut -f1)"
