"""Discovery and loading of benchmark predictions, ground truth, and T1n volumes.

Supports three on-disk schemas:
    - "ours": <run>/predictions/brats_men_test/<case_id>/segmentation.nii.gz
    - "pred_prefix": <model>/predictions/predBraTS-MEN-XXXXX-YYY.nii.gz
    - "bare":        <model>/predictions/XXXXX-YYY.nii.gz

Each model's filename schema is auto-detected by sniffing one file in its
predictions directory. Case IDs are normalized to the canonical
``BraTS-MEN-XXXXX-YYY`` form.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

# All evaluation is performed in the 192³ ROI frame that the H5 dataset uses.
# - Subject-space NIfTIs are 240×240×155 with spacing (1, 1, 1) mm.
# - The canonical mapping to 192³ is a centered xy-crop ([24:216]) plus a
#   centered z-pad ([18:173]). The H5 GT and the BSF 192³ prediction are
#   already in this frame; external 240×240×155 predictions are remapped on
#   the fly.
ROI_192 = (192, 192, 192)
SUBJECT_SHAPE = (240, 240, 155)
XY_OFFSET = 24
Z_OFFSET = 18  # (192 - 155) // 2


def frame_for_shape(shape: tuple[int, ...]) -> str:
    """Map a prediction array shape to the evaluation frame name."""
    if shape == ROI_192:
        return "h5_192"
    if shape == SUBJECT_SHAPE:
        return "subject"
    raise ValueError(f"Unsupported prediction shape {shape}")

CANONICAL_RE = re.compile(r"BraTS-MEN-\d{5}-\d{3}")
SHORT_RE = re.compile(r"^\d{5}-\d{3}$")

OURS_MODEL_ID = "Ours"
DEFAULT_OUR_RUN = Path(
    "/media/mpascual/Sandisk2TB/research/growth-dynamics/growth/results/"
    "uncertainty_segmentation/frozen_decoder/kqv_proj_fc1_fc2/stages_1234/"
    "r16_M20_s42/predictions/brats_men_test"
)
DEFAULT_MODELS_ROOT = Path(
    "/media/mpascual/Sandisk2TB/research/growth-dynamics/growth/results/"
    "segmentation_benchmarking/models"
)
DEFAULT_GT_ROOT = Path(
    "/media/mpascual/Sandisk2TB/data/mri/meningioma/BraTS_Men_Train"
)
DEFAULT_H5_PATH = Path(
    "/media/mpascual/Sandisk2TB/research/growth-dynamics/growth/data/source/BraTS_MEN.h5"
)
DEFAULT_ANALYSIS_ROOT = Path(
    "/media/mpascual/Sandisk2TB/research/growth-dynamics/growth/results/"
    "segmentation_benchmarking/analysis"
)


@dataclass(frozen=True)
class ModelEntry:
    """A model whose predictions can be loaded by case id."""

    model_id: str
    pred_dir: Path
    schema: str  # "ours" | "pred_prefix" | "bare"

    def case_ids(self) -> list[str]:
        """Return the canonical case IDs available for this model."""
        if self.schema == "ours":
            return sorted(p.name for p in self.pred_dir.iterdir() if p.is_dir() and CANONICAL_RE.fullmatch(p.name))
        cases: list[str] = []
        for p in sorted(self.pred_dir.glob("*.nii.gz")):
            cid = _canonicalize_case_id(p.name)
            if cid is not None:
                cases.append(cid)
        return cases

    def prediction_path(self, case_id: str) -> Path:
        """Return absolute path to the NIfTI prediction for ``case_id``."""
        if self.schema == "ours":
            return self.pred_dir / case_id / "segmentation.nii.gz"
        if self.schema == "pred_prefix":
            return self.pred_dir / f"pred{case_id}.nii.gz"
        if self.schema == "bare":
            short = case_id.replace("BraTS-MEN-", "")
            return self.pred_dir / f"{short}.nii.gz"
        raise ValueError(f"Unknown schema: {self.schema}")

    def dir_mtime(self) -> float:
        """Return the latest mtime under ``pred_dir`` (recursive, 1-level deep)."""
        latest = self.pred_dir.stat().st_mtime
        if self.schema == "ours":
            for child in self.pred_dir.iterdir():
                if child.is_dir():
                    seg = child / "segmentation.nii.gz"
                    if seg.exists():
                        latest = max(latest, seg.stat().st_mtime)
        else:
            for child in self.pred_dir.glob("*.nii.gz"):
                latest = max(latest, child.stat().st_mtime)
        return latest


def _canonicalize_case_id(filename: str) -> str | None:
    """Extract a canonical case id from a NIfTI filename, or None if unrecognized."""
    stem = filename
    if stem.endswith(".nii.gz"):
        stem = stem[: -len(".nii.gz")]
    elif stem.endswith(".nii"):
        stem = stem[: -len(".nii")]

    if stem.startswith("pred"):
        stem = stem[len("pred"):]
    m = CANONICAL_RE.search(stem)
    if m is not None:
        return m.group(0)
    if SHORT_RE.fullmatch(stem):
        return f"BraTS-MEN-{stem}"
    return None


def detect_schema(pred_dir: Path) -> str:
    """Sniff the on-disk naming schema for an external model's predictions dir."""
    files = sorted(pred_dir.glob("*.nii.gz"))
    if not files:
        raise FileNotFoundError(f"No NIfTI predictions under {pred_dir}")
    sample = files[0].name
    if sample.startswith("predBraTS-MEN-"):
        return "pred_prefix"
    if SHORT_RE.fullmatch(sample.replace(".nii.gz", "")):
        return "bare"
    if sample.startswith("BraTS-MEN-"):
        return "pred_prefix"  # treat plain canonical name as pred_prefix without the pred chunk
    raise ValueError(f"Unrecognized naming schema for {sample!r} in {pred_dir}")


def discover_models(
    models_root: Path = DEFAULT_MODELS_ROOT,
    our_run: Path = DEFAULT_OUR_RUN,
) -> list[ModelEntry]:
    """Return the list of ModelEntry, with our LoRA ensemble appended last."""
    entries: list[ModelEntry] = []
    if not models_root.exists():
        logger.warning("Models root does not exist: %s", models_root)
    else:
        for child in sorted(models_root.iterdir()):
            if not child.is_dir():
                continue
            pred_dir = child / "predictions"
            if not pred_dir.exists():
                logger.warning("Skipping %s: no predictions/ subfolder", child)
                continue
            try:
                schema = detect_schema(pred_dir)
            except (FileNotFoundError, ValueError) as exc:
                logger.warning("Skipping %s: %s", child.name, exc)
                continue
            entries.append(ModelEntry(model_id=child.name, pred_dir=pred_dir, schema=schema))

    if our_run.exists():
        entries.append(ModelEntry(model_id=OURS_MODEL_ID, pred_dir=our_run, schema="ours"))
    else:
        logger.error("Our LoRA ensemble run not found at %s", our_run)
    return entries


def load_prediction(entry: ModelEntry, case_id: str) -> tuple[np.ndarray, str]:
    """Return ``(prediction_volume, frame)`` in the prediction's native shape.

    ``frame`` is either ``"h5_192"`` (Our 192³ BSF outputs) or ``"subject"``
    (240×240×155 BraTS-challenge container outputs). The caller is expected
    to load the matching GT.
    """
    path = entry.prediction_path(case_id)
    arr = np.asarray(nib.load(path).get_fdata()).astype(np.int8)
    return arr, frame_for_shape(arr.shape)


_H5_HANDLES: dict[Path, h5py.File] = {}


def _h5_handle(h5_path: Path) -> h5py.File:
    handle = _H5_HANDLES.get(h5_path)
    if handle is None or not handle.id.valid:
        handle = h5py.File(h5_path, "r", swmr=True)
        _H5_HANDLES[h5_path] = handle
    return handle


def _h5_index(h5_path: Path) -> dict[str, int]:
    f = _h5_handle(h5_path)
    ids = f["scan_ids"][:]
    return {(s.decode() if isinstance(s, bytes) else s): i for i, s in enumerate(ids)}


def load_ground_truth(
    case_id: str,
    frame: str = "subject",
    gt_root: Path = DEFAULT_GT_ROOT,
    h5_path: Path = DEFAULT_H5_PATH,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Return the GT segmentation in the chosen ``frame`` plus its mm spacing.

    ``frame``:
        * ``"subject"`` — 240×240×155 NIfTI from the BraTS-MEN raw release
          (1×1×1 mm), aligned to the external models' predictions.
        * ``"h5_192"`` — 192³ H5 ROI crop (1×1×1 mm) used by the BSF
          inference engine, aligned to Our predictions.
    """
    if frame == "subject":
        img = nib.load(gt_root / case_id / f"{case_id}-seg.nii.gz")
        arr = np.asarray(img.get_fdata()).astype(np.int8)
        zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
        return arr, zooms  # type: ignore[return-value]
    if frame == "h5_192":
        f = _h5_handle(h5_path)
        idx = _h5_index(h5_path)
        if case_id not in idx:
            raise FileNotFoundError(f"{case_id} not in H5 {h5_path}")
        seg = np.asarray(f["segs"][idx[case_id]])
        if seg.ndim == 4:
            seg = seg[0]
        spacing_attr = f.attrs.get("spacing", np.array([1.0, 1.0, 1.0]))
        spacing = tuple(float(z) for z in spacing_attr[:3])
        return seg.astype(np.int8), spacing  # type: ignore[return-value]
    raise ValueError(f"Unknown frame: {frame}")


def load_t1n(
    case_id: str,
    frame: str = "subject",
    gt_root: Path = DEFAULT_GT_ROOT,
    h5_path: Path = DEFAULT_H5_PATH,
) -> np.ndarray:
    """Return the T1n channel in the requested ``frame``."""
    if frame == "subject":
        return np.asarray(
            nib.load(gt_root / case_id / f"{case_id}-t1n.nii.gz").get_fdata()
        ).astype(np.float32)
    if frame == "h5_192":
        f = _h5_handle(h5_path)
        idx = _h5_index(h5_path)
        if case_id not in idx:
            raise FileNotFoundError(f"{case_id} not in H5 {h5_path}")
        channel_order = [
            (s.decode() if isinstance(s, bytes) else s) for s in f.attrs["channel_order"]
        ]
        t1n_idx = channel_order.index("t1n")
        return np.asarray(f["images"][idx[case_id], t1n_idx]).astype(np.float32)
    raise ValueError(f"Unknown frame: {frame}")
