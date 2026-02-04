# src/growth/utils/reproducibility.py
"""Reproducibility utilities for experiment tracking.

Captures environment, git state, and configuration for full reproducibility.
"""

import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class GitInfo:
    """Git repository state."""
    commit: str
    branch: str
    dirty: bool
    remote_url: Optional[str] = None


@dataclass
class EnvironmentInfo:
    """Runtime environment information."""
    python_version: str
    hostname: str
    platform: str
    cuda_visible_devices: Optional[str]
    pytorch_version: str
    monai_version: Optional[str]


@dataclass
class RunManifest:
    """Complete run manifest for reproducibility."""
    started_at: str
    command: str
    config_path: str
    git: Optional[GitInfo]
    environment: EnvironmentInfo
    config_hash: str


def get_git_info(repo_path: Optional[Path] = None) -> Optional[GitInfo]:
    """Get current git repository state.

    Args:
        repo_path: Path to repository. If None, uses current directory.

    Returns:
        GitInfo or None if not a git repo.
    """
    cwd = str(repo_path) if repo_path else None

    try:
        # Get commit hash
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd, stderr=subprocess.DEVNULL
        ).decode().strip()

        # Get branch name
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd, stderr=subprocess.DEVNULL
        ).decode().strip()

        # Check if dirty
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=cwd, stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = len(status) > 0

        # Get remote URL
        try:
            remote_url = subprocess.check_output(
                ["git", "remote", "get-url", "origin"],
                cwd=cwd, stderr=subprocess.DEVNULL
            ).decode().strip()
        except subprocess.CalledProcessError:
            remote_url = None

        return GitInfo(
            commit=commit,
            branch=branch,
            dirty=dirty,
            remote_url=remote_url,
        )

    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_environment_info() -> EnvironmentInfo:
    """Get current runtime environment information."""
    import platform
    import torch

    try:
        import monai
        monai_version = monai.__version__
    except ImportError:
        monai_version = None

    return EnvironmentInfo(
        python_version=sys.version.split()[0],
        hostname=platform.node(),
        platform=platform.platform(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        pytorch_version=torch.__version__,
        monai_version=monai_version,
    )


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute hash of configuration for quick comparison."""
    import hashlib
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.md5(config_str.encode()).hexdigest()[:12]


def create_run_manifest(
    config: Dict[str, Any],
    config_path: str,
) -> RunManifest:
    """Create a complete run manifest.

    Args:
        config: Experiment configuration dict.
        config_path: Path to config file.

    Returns:
        RunManifest with all reproducibility info.
    """
    return RunManifest(
        started_at=datetime.now().isoformat(),
        command=" ".join(sys.argv),
        config_path=str(config_path),
        git=get_git_info(),
        environment=get_environment_info(),
        config_hash=compute_config_hash(config),
    )


def save_reproducibility_artifacts(
    output_dir: Path,
    config: Dict[str, Any],
    config_path: str,
) -> Path:
    """Save all reproducibility artifacts to meta/ directory.

    Creates:
        meta/
        ├── run_manifest.json     # Complete run info
        ├── config.yaml           # Copy of config used
        ├── git_diff.patch        # Uncommitted changes (if any)
        └── pip_freeze.txt        # Installed packages

    Args:
        output_dir: Experiment output directory.
        config: Experiment configuration.
        config_path: Path to original config file.

    Returns:
        Path to meta directory.
    """
    import shutil
    import yaml

    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Create and save run manifest
    manifest = create_run_manifest(config, config_path)
    manifest_dict = _manifest_to_dict(manifest)

    manifest_path = meta_dir / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_dict, f, indent=2)
    logger.info(f"Saved run manifest to {manifest_path}")

    # Copy config file
    config_copy_path = meta_dir / "config.yaml"
    with open(config_copy_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Saved config copy to {config_copy_path}")

    # Save git diff if dirty
    if manifest.git and manifest.git.dirty:
        try:
            diff = subprocess.check_output(
                ["git", "diff", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode()
            diff_path = meta_dir / "git_diff.patch"
            with open(diff_path, "w") as f:
                f.write(diff)
            logger.info(f"Saved git diff to {diff_path}")
        except subprocess.CalledProcessError:
            pass

    # Save pip freeze
    try:
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            stderr=subprocess.DEVNULL
        ).decode()
        freeze_path = meta_dir / "pip_freeze.txt"
        with open(freeze_path, "w") as f:
            f.write(freeze)
        logger.info(f"Saved pip freeze to {freeze_path}")
    except subprocess.CalledProcessError:
        pass

    return meta_dir


def _manifest_to_dict(manifest: RunManifest) -> Dict[str, Any]:
    """Convert manifest to serializable dict."""
    d = {
        "started_at": manifest.started_at,
        "command": manifest.command,
        "config_path": manifest.config_path,
        "config_hash": manifest.config_hash,
    }

    if manifest.git:
        d["git"] = asdict(manifest.git)
    else:
        d["git"] = None

    d["environment"] = asdict(manifest.environment)

    return d


def load_run_manifest(meta_dir: Path) -> Optional[Dict[str, Any]]:
    """Load run manifest from meta directory.

    Args:
        meta_dir: Path to meta directory.

    Returns:
        Manifest dict or None if not found.
    """
    manifest_path = meta_dir / "run_manifest.json"
    if not manifest_path.exists():
        return None

    with open(manifest_path) as f:
        return json.load(f)


def check_reproducibility(
    meta_dir: Path,
    warn_on_mismatch: bool = True,
) -> Dict[str, bool]:
    """Check if current environment matches saved manifest.

    Args:
        meta_dir: Path to meta directory.
        warn_on_mismatch: Whether to log warnings on mismatches.

    Returns:
        Dict with check results for each component.
    """
    manifest = load_run_manifest(meta_dir)
    if manifest is None:
        return {"manifest_found": False}

    results = {"manifest_found": True}

    # Check git
    current_git = get_git_info()
    if current_git and manifest.get("git"):
        git_match = current_git.commit == manifest["git"]["commit"]
        results["git_commit_match"] = git_match
        if not git_match and warn_on_mismatch:
            logger.warning(
                f"Git commit mismatch: current={current_git.commit[:8]}, "
                f"manifest={manifest['git']['commit'][:8]}"
            )

    # Check Python version
    current_env = get_environment_info()
    saved_env = manifest.get("environment", {})

    py_match = current_env.python_version == saved_env.get("python_version")
    results["python_version_match"] = py_match
    if not py_match and warn_on_mismatch:
        logger.warning(
            f"Python version mismatch: current={current_env.python_version}, "
            f"manifest={saved_env.get('python_version')}"
        )

    # Check PyTorch version
    torch_match = current_env.pytorch_version == saved_env.get("pytorch_version")
    results["pytorch_version_match"] = torch_match
    if not torch_match and warn_on_mismatch:
        logger.warning(
            f"PyTorch version mismatch: current={current_env.pytorch_version}, "
            f"manifest={saved_env.get('pytorch_version')}"
        )

    return results
