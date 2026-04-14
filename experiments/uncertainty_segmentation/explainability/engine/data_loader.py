"""Dataset loading helpers for the explainability analysis.

Wraps the ``BraTSDatasetH5`` class with conveniences for:

- Resolving scan IDs and split indices from H5 metadata.
- Selecting a deterministic subset of scans (``"first"`` or ``"random"``).
- Volume-matched cohort pairing for cross-domain DAD analysis (one MEN scan
  paired with the GLI scan whose tumor volume is closest in log-space).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from omegaconf import DictConfig

from growth.data.bratsmendata import BraTSDatasetH5
from growth.data.transforms import get_h5_val_transforms

logger = logging.getLogger(__name__)


@dataclass
class LoadedDataset:
    """Container for a BraTS H5 dataset together with its scan-ID resolution."""

    dataset: BraTSDatasetH5
    all_scan_ids: list[str]
    test_indices: np.ndarray
    h5_path: Path
    domain: str  # "MEN" or "GLI"


def load_brats_dataset(
    h5_path: str | Path,
    split: str = "test",
    roi_size: tuple[int, int, int] = (128, 128, 128),
    domain: str | None = None,
) -> LoadedDataset:
    """Open a BraTS H5 file and return a fully-resolved dataset wrapper.

    Parameters
    ----------
    h5_path : str | Path
        Path to the BraTS H5 file (MEN or GLI, schema v2.0).
    split : str
        Split key to use, typically ``"test"``.
    roi_size : tuple[int, int, int]
        Spatial ROI size for transforms.
    domain : str | None
        Optional domain label override (``"MEN"`` or ``"GLI"``). When
        ``None``, the file's ``attrs["domain"]`` is read; if absent,
        ``"MEN"`` is assumed (legacy MEN-only files).

    Returns
    -------
    LoadedDataset
    """
    h5_path = Path(h5_path)
    transform = get_h5_val_transforms(roi_size=roi_size)
    dataset = BraTSDatasetH5(
        h5_path=str(h5_path),
        split=split,
        transform=transform,
        compute_semantic=False,
    )

    with h5py.File(h5_path, "r") as f:
        if "scan_ids" in f:
            ids = f["scan_ids"][:]
        else:  # legacy v1.0 MEN
            ids = f["subject_ids"][:]
        all_scan_ids = [s.decode() if isinstance(s, bytes) else s for s in ids]
        if domain is None:
            domain = f.attrs.get("domain", "MEN")
            if isinstance(domain, bytes):
                domain = domain.decode()

    splits = BraTSDatasetH5.load_splits_from_h5(str(h5_path))
    test_indices = splits.get(split, np.arange(len(all_scan_ids)))

    return LoadedDataset(
        dataset=dataset,
        all_scan_ids=all_scan_ids,
        test_indices=np.asarray(test_indices),
        h5_path=h5_path,
        domain=domain,
    )


def select_scan_indices(
    n_available: int,
    n_scans: int,
    selection: str = "random",
    seed: int = 42,
) -> list[int]:
    """Choose dataset indices to include in the analysis.

    Parameters
    ----------
    n_available : int
        Number of scans available in the split.
    n_scans : int
        Desired number of scans to use.
    selection : str
        ``"first"`` for deterministic prefix, ``"random"`` for a seeded sample.
    seed : int
        Seed for ``"random"``.

    Returns
    -------
    list[int]
        Sorted list of dataset indices.
    """
    n_use = min(n_scans, n_available)
    if selection == "first":
        return list(range(n_use))
    if selection == "random":
        rng = np.random.RandomState(seed)
        return sorted(rng.choice(n_available, size=n_use, replace=False).tolist())
    raise ValueError(f"Unknown scan_selection: {selection}")


def get_wt_mask(seg: torch.Tensor) -> torch.Tensor:
    """Whole-tumor binary mask from the integer segmentation tensor.

    Both BraTS-MEN and BraTS-GLI use labels ``{0=BG, 1, 2, 3}``; whole
    tumor is the union of all foreground labels.

    Parameters
    ----------
    seg : torch.Tensor
        Segmentation tensor ``[1, D, H, W]`` of integer labels.

    Returns
    -------
    torch.Tensor
        Binary mask ``[D, H, W]`` of float dtype.
    """
    return (seg.squeeze(0) > 0).float()


def compute_tumor_volumes(
    loaded: LoadedDataset,
    scan_indices: Sequence[int],
) -> np.ndarray:
    """Sum the WT-mask voxels for each selected scan.

    Used to compute the volume-matching for DAD. Operates on the raw
    H5 segmentation tensors and bypasses the dataset transforms.

    Parameters
    ----------
    loaded : LoadedDataset
    scan_indices : Sequence[int]
        Indices into ``loaded.dataset``.

    Returns
    -------
    np.ndarray
        WT voxel counts ``[len(scan_indices)]`` of float dtype.
    """
    volumes = np.empty(len(scan_indices), dtype=np.float64)
    with h5py.File(loaded.h5_path, "r") as f:
        seg_dset = f["segs"]
        for i, idx in enumerate(scan_indices):
            ds_idx = int(loaded.test_indices[idx])
            seg = seg_dset[ds_idx]  # [1, D, H, W] int8
            volumes[i] = float((seg > 0).sum())
    return volumes


def match_by_volume(
    men: LoadedDataset,
    men_indices: Sequence[int],
    gli: LoadedDataset,
    gli_candidate_indices: Sequence[int],
) -> list[tuple[int, int]]:
    """Greedy log-volume matching of MEN scans to GLI counterparts.

    For each MEN scan we pick the unused GLI scan whose log-tumor-volume is
    closest. The matching is greedy and processes MEN scans in input order,
    which is sufficient for the modest cohort sizes (~20 each) used in DAD.

    Parameters
    ----------
    men, gli : LoadedDataset
    men_indices : Sequence[int]
        Selected MEN scan indices (within ``men.dataset``).
    gli_candidate_indices : Sequence[int]
        Pool of GLI scan indices to match against.

    Returns
    -------
    list[tuple[int, int]]
        ``len(men_indices)`` pairs ``(men_idx, gli_idx)``.
    """
    if len(gli_candidate_indices) < len(men_indices):
        raise ValueError(
            "Not enough GLI candidates "
            f"(have {len(gli_candidate_indices)}, need {len(men_indices)})."
        )
    men_vol = compute_tumor_volumes(men, men_indices)
    gli_vol = compute_tumor_volumes(gli, gli_candidate_indices)
    # log1p to compress the dynamic range; BraTS volumes span 4-5 orders of
    # magnitude so a linear distance is dominated by the largest tumors.
    men_log = np.log1p(men_vol)
    gli_log = np.log1p(gli_vol)

    available = list(range(len(gli_candidate_indices)))
    pairs: list[tuple[int, int]] = []
    for i, mi in enumerate(men_indices):
        # Pick the closest unused GLI scan.
        diffs = np.abs(gli_log[available] - men_log[i])
        best_local = int(np.argmin(diffs))
        best_global = available.pop(best_local)
        pairs.append((int(mi), int(gli_candidate_indices[best_global])))
        logger.debug(
            "DAD pair %d: MEN idx=%d (logV=%.2f) <-> GLI idx=%d (logV=%.2f)",
            i, mi, men_log[i], gli_candidate_indices[best_global],
            gli_log[best_global],
        )
    return pairs


def build_men_loader(
    config: DictConfig,
    roi_size: tuple[int, int, int],
) -> LoadedDataset:
    """Convenience wrapper: load the MEN test split using the parent config."""
    return load_brats_dataset(
        h5_path=config.paths.men_h5_file,
        split=config.data.test_split,
        roi_size=roi_size,
        domain="MEN",
    )


def build_gli_loader(
    config: DictConfig,
    roi_size: tuple[int, int, int],
) -> LoadedDataset:
    """Convenience wrapper: load the GLI test split using the parent config."""
    return load_brats_dataset(
        h5_path=config.paths.gli_h5_file,
        split=config.data.test_split,
        roi_size=roi_size,
        domain="GLI",
    )
