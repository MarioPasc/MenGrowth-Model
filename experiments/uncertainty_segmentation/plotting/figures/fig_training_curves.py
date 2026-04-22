"""Fig 1: Training Dynamics.

Two-panel figure showing loss and validation Dice over epochs (mean +/- std
ribbon across all ensemble members).
"""

from __future__ import annotations

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt

from experiments.uncertainty_segmentation.plotting.data_loader import (
    EnsembleResultsData,
)
from experiments.uncertainty_segmentation.plotting.style import (
    C_DELTA_NEG,
    C_ENSEMBLE,
    C_MEMBERS,
)


def plot(
    data: EnsembleResultsData,
    config: dict,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Generate the training curves figure.

    Args:
        data: All loaded experiment data.
        config: Figure-specific config from config.yaml.
        ax: Ignored (two-panel figure creates its own axes).

    Returns:
        The Figure object.
    """
    figsize = config.get("figsize", [7, 2.8])
    fig, (ax_loss, ax_dice) = plt.subplots(1, 2, figsize=figsize)

    curves = data.training_curves
    epochs = curves["epoch"].values

    # --- Panel A: Loss ---
    ax_loss.fill_between(
        epochs,
        curves["train_loss_mean"] - curves["train_loss_std"],
        curves["train_loss_mean"] + curves["train_loss_std"],
        alpha=0.2,
        color=C_MEMBERS,
    )
    ax_loss.plot(epochs, curves["train_loss_mean"], color=C_MEMBERS, lw=1.2, label="Train loss")
    ax_loss.fill_between(
        epochs,
        curves["val_loss_mean"] - curves["val_loss_std"],
        curves["val_loss_mean"] + curves["val_loss_std"],
        alpha=0.2,
        color=C_ENSEMBLE,
    )
    ax_loss.plot(epochs, curves["val_loss_mean"], color=C_ENSEMBLE, lw=1.2, label="Val loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss (Dice + CE)")
    ax_loss.legend(frameon=False)
    ax_loss.set_title("a) Training loss", loc="left", fontweight="bold")

    # --- Panel B: Validation Dice ---
    # BraTS-hierarchical training targets: TC=(1|3), WT=(seg>0), ET=(3).
    # WT is the top-line "whole tumor (incl. edema)" metric; ET the
    # enhancing-only metric; TC the tumor core.
    for col, color, lw, ls, label in [
        ("val_dice_wt", C_ENSEMBLE, 1.2, "-", "WT"),
        ("val_dice_tc", C_MEMBERS, 1.0, "--", "TC"),
        ("val_dice_et", C_DELTA_NEG, 1.0, ":", "ET"),
    ]:
        mean_col = f"{col}_mean"
        std_col = f"{col}_std"
        alpha = 0.2 if col == "val_dice_wt" else 0.15
        ax_dice.fill_between(
            epochs,
            curves[mean_col] - curves[std_col],
            curves[mean_col] + curves[std_col],
            alpha=alpha,
            color=color,
        )
        ax_dice.plot(epochs, curves[mean_col], color=color, lw=lw, ls=ls, label=label)

    ax_dice.set_xlabel("Epoch")
    ax_dice.set_ylabel("Validation Dice")
    ax_dice.set_ylim(0, 1)
    ax_dice.legend(frameon=False, ncol=3)
    ax_dice.set_title(
        "b) Validation Dice (mean +/- std across M members)", loc="left", fontweight="bold"
    )

    fig.tight_layout()
    return fig
