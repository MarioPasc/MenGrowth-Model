"""Registry of inter-LoRA figure and table modules."""

from __future__ import annotations

from experiments.uncertainty_segmentation.plotting.inter_lora.figures import (
    qual1_slice_grid,
    qual2_clustered_heatmap,
    quant1_dice_vs_rank,
    quant2_calib_epistemic,
    quant3_convergence_vs_rank,
)
from experiments.uncertainty_segmentation.plotting.inter_lora.tables import (
    tab1_summary,
    tab2_paired,
    tab3_model_complexity,
)

INTER_LORA_FIGURE_REGISTRY: dict[str, object] = {
    "quant1_dice_vs_rank": quant1_dice_vs_rank,
    "quant2_calib_epistemic": quant2_calib_epistemic,
    "quant3_convergence_vs_rank": quant3_convergence_vs_rank,
    "qual1_slice_grid": qual1_slice_grid,
    "qual2_clustered_heatmap": qual2_clustered_heatmap,
}

INTER_LORA_TABLE_REGISTRY: dict[str, object] = {
    "tab1_summary": tab1_summary,
    "tab2_paired": tab2_paired,
    "tab3_model_complexity": tab3_model_complexity,
}

FIGURE_OUTPUT_NAMES: dict[str, str] = {
    "quant1_dice_vs_rank": "quant1_dice_vs_rank",
    "quant2_calib_epistemic": "quant2_calibration_epistemic_vs_rank",
    "quant3_convergence_vs_rank": "quant3_convergence_vs_rank",
    "qual1_slice_grid": "qual1_slice_grid",
    "qual2_clustered_heatmap": "qual2_clustered_heatmap",
}

_QUAL1_BRATS_CAPTION = (
    r"\caption{"
    r"Qualitative segmentation results for a representative BraTS-MEN test scan "
    r"across LoRA ranks $r \in \{2, 4, 8, 16, 32\}$. "
    r"Each row corresponds to a LoRA rank. "
    r"\textbf{Left column:}~Axial T1c slice with colour overlay showing "
    r"ground-truth-only voxels~(\textcolor{orange}{orange}), "
    r"ensemble-only voxels~(\textcolor{blue}{blue}), and "
    r"overlap~(\textcolor{green}{green}). "
    r"\textbf{Right column:}~Predictive entropy of the ensemble-mean "
    r"meningioma probability "
    r"$\mathcal{H}[\bar{p}_{\mathrm{men}}(\mathbf{x})]$ (nats), "
    r"displayed in the \texttt{magma} colourmap with alpha proportional "
    r"to normalised entropy. Scale bar indicates 10\,mm.}"
)

_QUAL1_MENGROWTH_CAPTION = (
    r"\caption{"
    r"Qualitative segmentation results for a representative MenGrowth cohort scan "
    r"across LoRA ranks $r \in \{2, 4, 8, 16, 32\}$. "
    r"Each row corresponds to a LoRA rank; left column shows the ensemble "
    r"segmentation overlay on T1c; right column shows the predictive entropy "
    r"of the ensemble-mean meningioma probability "
    r"$\mathcal{H}[\bar{p}_{\mathrm{men}}(\mathbf{x})]$ (nats). The scan is "
    r"selected as the cohort minimum~(\emph{low\_uncertainty}) or "
    r"maximum~(\emph{high\_uncertainty}) mean in-mask entropy. No ground-truth "
    r"annotations are available for MenGrowth scans; only ensemble predictions "
    r"are shown (\textcolor{blue}{blue}~=~ensemble).}"
)

FIGURE_CAPTIONS: dict[str, str] = {
    "quant1_dice_vs_rank": (
        r"\caption{"
        r"Segmentation accuracy as a function of LoRA rank. "
        r"\textbf{(Left)}~Grouped boxplots of per-scan Dice coefficients for "
        r"TC, WT, and ET across the frozen BrainSegFounder (BSF) baseline and "
        r"LoRA ranks $r \in \{2, 4, 8, 16, 32\}$, with $M = 20$ ensemble members. "
        r"Jittered points represent individual test scans. "
        r"\textbf{(Right)}~Split pairwise comparison matrix: upper triangle "
        r"reports Wilcoxon signed-rank $p$-values with Holm--Bonferroni correction; "
        r"lower triangle reports Cohen's $d$ effect sizes.}"
    ),
    "quant2_calibration_epistemic_vs_rank": (
        r"\caption{"
        r"Epistemic uncertainty quality as a function of LoRA rank. "
        r"\textbf{(a)}~Expected Calibration Error (ECE, solid) and Brier score "
        r"(dashed); vertical dotted line marks the ECE-minimising rank~$r^{*}$. "
        r"\textbf{(b)}~Coverage deficit $\delta_\alpha = \alpha - \hat{C}_\alpha$ "
        r"at four nominal credible-interval levels "
        r"($\alpha \in \{0.50, 0.80, 0.90, 0.95\}$). "
        r"\textbf{(c)}~Bias dominance fractions: proportion of voxels where the "
        r"dominant ensemble member is $k^{*}=1$, $k^{*}>M$, or degenerate. "
        r"\textbf{(d)}~Inter-member Intraclass Correlation Coefficient (ICC) for "
        r"TC, WT, and ET. Dashed vertical band marks the consensus optimal rank "
        r"$r^{*}_{\mathrm{consensus}}$ across all four panels.}"
    ),
    "quant3_convergence_vs_rank": (
        r"\caption{"
        r"Ensemble convergence and binarisation threshold sensitivity. "
        r"Each row corresponds to a tumour region (TC, WT, ET). "
        r"\textbf{Left column:}~Dice coefficient as a function of ensemble "
        r"size~$k$ (ensemble-of-$k$ prediction, solid; mean per-member, dashed), "
        r"with 95\% bootstrap CI shaded, coloured by LoRA rank on a "
        r"$\log_2(r)$ viridis scale. "
        r"\textbf{Right column:}~Dice as a function of binarisation "
        r"threshold~$\tau$, with vertical reference at $\tau = 0.5$. "
        r"Top row displays the rank-to-colour mapping.}"
    ),
    "qual1_best_brats": _QUAL1_BRATS_CAPTION,
    "qual1_best_brats_horizontal": _QUAL1_BRATS_CAPTION,
    "qual1_worst_brats": _QUAL1_BRATS_CAPTION,
    "qual1_worst_brats_horizontal": _QUAL1_BRATS_CAPTION,
    "qual1_mengrowth_low_uncertainty": _QUAL1_MENGROWTH_CAPTION,
    "qual1_mengrowth_low_uncertainty_horizontal": _QUAL1_MENGROWTH_CAPTION,
    "qual1_mengrowth_high_uncertainty": _QUAL1_MENGROWTH_CAPTION,
    "qual1_mengrowth_high_uncertainty_horizontal": _QUAL1_MENGROWTH_CAPTION,
    "qual2_clustered_heatmap": (
        r"\caption{"
        r"Per-scan Dice heatmap with hierarchical clustering. "
        r"Three vertically stacked panels show TC~\textbf{(a)}, "
        r"WT~\textbf{(b)}, and ET~\textbf{(c)}, sharing a common row "
        r"ordering determined by Ward linkage on the concatenated "
        r"$[\mathrm{TC} \mid \mathrm{WT} \mid \mathrm{ET}]$ Dice matrix "
        r"(Euclidean distance). Columns progress from the frozen BSF baseline "
        r"through ascending LoRA ranks. Colourmap diverges at "
        r"$\mathrm{Dice} = 0.7$ (RdYlGn). White-outlined cells mark failure "
        r"cases ($\mathrm{Dice} < 0.4$). Right strip displays "
        r"$\log_{10}(V_{\mathrm{GT}})$ per scan in the \texttt{cividis} "
        r"colourmap. Dendrogram is shown to the left of panel~\textbf{(a)}.}"
    ),
}

__all__ = [
    "INTER_LORA_FIGURE_REGISTRY",
    "INTER_LORA_TABLE_REGISTRY",
    "FIGURE_OUTPUT_NAMES",
    "FIGURE_CAPTIONS",
]
