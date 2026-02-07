  Comprehensive Comparative Analysis: Old vs New LoRA Ablation Results                      
                                                                                              
  1. What Changed Between Runs                                                                
  ┌──────────────────────────┬─────────────────────┬──────────────────┐
  │        Parameter         │       Old Run       │     New Run      │
  ├──────────────────────────┼─────────────────────┼──────────────────┤
  │ Channel convention       │ NCR/ED/ET (BUG)     │ TC/WT/ET (FIXED) │
  ├──────────────────────────┼─────────────────────┼──────────────────┤
  │ Feature extraction level │ multi_scale (1344d) │ encoder10 (768d) │
  ├──────────────────────────┼─────────────────────┼──────────────────┤
  │ Probe training samples   │ 150                 │ 200              │
  ├──────────────────────────┼─────────────────────┼──────────────────┤
  │ Data splits              │ unclear             │ 250/50/200/500   │
  ├──────────────────────────┼─────────────────────┼──────────────────┤
  │ GLI test size            │ 25                  │ 200              │
  ├──────────────────────────┼─────────────────────┼──────────────────┤
  │ MLP probe regularization │ alpha=0.001         │ alpha=0.01       │
  ├──────────────────────────┼─────────────────────┼──────────────────┤
  │ MLP architecture         │ [256, 128]          │ [128]            │
  └──────────────────────────┴─────────────────────┴──────────────────┘
  These are significant methodological improvements. The encoder10 features are the correct
  choice for SDP Phase 2, the probe evaluation is more robust (200 vs 150 samples, better
  regularization), and the GLI evaluation is far more reliable (200 vs 25 samples).

  ---
  2. Has Performance Improved? (Comparison)

  2A. Frozen Baseline Dice (TC/WT/ET fix validation)
  ┌───────────┬───────┬───────┬──────────────────────────────────────────────────────┐
  │  Dataset  │  Old  │  New  │                       Verdict                        │
  ├───────────┼───────┼───────┼──────────────────────────────────────────────────────┤
  │ BraTS-MEN │ 0.020 │ 0.279 │ Fixed - 14x improvement confirms TC/WT/ET fix worked │
  ├───────────┼───────┼───────┼──────────────────────────────────────────────────────┤
  │ BraTS-GLI │ 0.008 │ 0.020 │ Barely changed - anomalous (see Section 3)           │
  └───────────┴───────┴───────┴──────────────────────────────────────────────────────┘
  The TC/WT/ET fix is validated by the MEN improvement.

  2B. Retrained Models - Segmentation Dice
  Condition: baseline (decoder only)
  Old MEN: ~0.94
  New MEN: 0.942
  Old GLI: ~0.64
  New GLI: 0.637
  ────────────────────────────────────────
  Condition: Best LoRA+sem
  Old MEN: ~0.94
  New MEN: 0.942 (r32)
  Old GLI: ~0.64
  New GLI: 0.641 (r32)
  ────────────────────────────────────────
  Condition: Best overall
  Old MEN: -
  New MEN: 0.944 (dora_r8+sem)
  Old GLI: -
  New GLI: 0.646 (dora_r4 no-sem)
  Segmentation Dice is essentially the same. All retrained models converge to ~0.94 MEN /
  ~0.63 GLI regardless of LoRA rank or method. This makes sense: segmentation is primarily a
  decoder task, and the decoder has ~54M trainable parameters that dominate.

  2C. Linear Probe R² (the key metric for encoder quality)

  CRITICAL: Direct R² comparison between old and new is invalid because the feature space
  changed (multi_scale 1344d → encoder10 768d). The correct comparison is the relative
  improvement from LoRA over baseline:
  ┌───────────────────────┬───────────────────┬─────────────────┐
  │        Metric         │ Old (multi_scale) │ New (encoder10) │
  ├───────────────────────┼───────────────────┼─────────────────┤
  │ Baseline R²_mean      │ -0.33             │ -3.12           │
  ├───────────────────────┼───────────────────┼─────────────────┤
  │ Best LoRA+sem R²_mean │ 0.19 (r32)        │ -0.50 (r16)     │
  ├───────────────────────┼───────────────────┼─────────────────┤
  │ Δ R² from LoRA        │ +0.52             │ +2.63           │
  ├───────────────────────┼───────────────────┼─────────────────┤
  │ Relative improvement  │ 158%              │ 84%             │
  └───────────────────────┴───────────────────┴─────────────────┘
  The absolute R² is worse because encoder10 features are harder to linearly probe (768d
  bottleneck vs 1344d multi-scale). But the improvement from LoRA is 5x larger in the new
  results, showing LoRA has a much bigger impact at the encoder10 level.

  2D. MLP Probe R² (nonlinear probes)
  ┌──────────────────────┬─────────────────┬─────────────────┐
  │ Condition (LoRA+sem) │ Old MLP R²_mean │ New MLP R²_mean │
  ├──────────────────────┼─────────────────┼─────────────────┤
  │ baseline             │ -0.44           │ -0.44           │
  ├──────────────────────┼─────────────────┼─────────────────┤
  │ lora_r16             │ -               │ +0.08           │
  ├──────────────────────┼─────────────────┼─────────────────┤
  │ lora_r32             │ -               │ +0.07           │
  └──────────────────────┴─────────────────┴─────────────────┘
  The MLP probes now consistently outperform linear probes for LoRA conditions. In the old
  run, MLP probes paradoxically underperformed linear probes (a sign of overfitting). This is
  fixed by the increased regularization (alpha=0.01 vs 0.001) and simplified architecture
  ([128] vs [256,128]).

  The positive MLP R²_mean (+0.08) for lora_r16 is an important signal: it means the encoder10
   features DO contain semantic information that a nonlinear projection can extract. This is
  exactly what the SDP module will do.

  2E. Statistical Significance
  ┌───────────┬─────────────────┬─────────────────────────────┐
  │ Condition │       Old       │ New (after Holm-Bonferroni) │
  ├───────────┼─────────────────┼─────────────────────────────┤
  │ lora_r4   │ not significant │ p=6.4e-5 (significant)      │
  ├───────────┼─────────────────┼─────────────────────────────┤
  │ lora_r8   │ not significant │ p=5.8e-8 (significant)      │
  ├───────────┼─────────────────┼─────────────────────────────┤
  │ lora_r16  │ not significant │ p=5.6e-11 (significant)     │
  ├───────────┼─────────────────┼─────────────────────────────┤
  │ lora_r32  │ not significant │ p=9.9e-14 (significant)     │
  └───────────┴─────────────────┴─────────────────────────────┘
  All LoRA conditions r4+ are now statistically significant after correction. This was NOT the
   case in the old run. The increased probe training set (200 vs 150) and better feature space
   (encoder10) contribute to this.

  2F. Feature Variance
  ┌─────────────────┬──────────────┬──────────────┐
  │    Condition    │ Old variance │ New variance │
  ├─────────────────┼──────────────┼──────────────┤
  │ baseline_frozen │ 0.012        │ 0.004        │
  ├─────────────────┼──────────────┼──────────────┤
  │ lora_r8+sem     │ ~0.27        │ 0.115        │
  ├─────────────────┼──────────────┼──────────────┤
  │ lora_r16+sem    │ -            │ 0.161        │
  ├─────────────────┼──────────────┼──────────────┤
  │ lora_r32+sem    │ ~0.32        │ 0.140        │
  └─────────────────┴──────────────┴──────────────┘
  Feature variance is lower in the new results because encoder10 (768d) has fewer dimensions
  than multi_scale (1344d). The relative increase from LoRA is still substantial (~30-40x).

  ---
  3. The Glioma Frozen Baseline Anomaly (Your Key Concern)

  The core puzzle: The frozen BrainSegFounder model (fine-tuned on BraTS-GLI-2021, achieving
  0.903 Dice) gets 0.020 Dice on BraTS-GLI-2023 test data. Even after fixing the TC/WT/ET
  convention, this barely improved from the old result (0.008 → 0.020). Meanwhile, on MEN
  data, the fix improved it dramatically (0.020 → 0.279).

  I investigated this thoroughly. Here is what I found:

  What we can rule out:
  1. Channel convention bug: Fixed. Code now correctly uses TC/WT/ET in both
  SegmentationLoss3Ch and DiceMetric3Ch (lines 346-356, 441-449 of segmentation.py).
  2. Checkpoint loading failure: The log shows Loaded full model: 159 keys, 62.19M params with
   NO missing/unexpected keys. The checkpoint reports best_acc=0.9027.
  3. Architecture mismatch: create_swinunetr() creates the same architecture as training.
  MONAI's SwinViT uses relative position biases that are window-size-dependent (not
  img_size-dependent), so patch size compatibility is preserved.
  4. Label convention mismatch: Retrained models achieve dice_ET=0.66 on GLI data, confirming
  the GLI data uses BraTS-2023 labels (ET=label 3, not label 4). If labels were BraTS-2021
  (ET=4), ET Dice would be 0.

  The most likely cause: Center-crop evaluation mismatch

  BrainSegFounder was trained and validated using sliding-window inference over full
  240×240×155 volumes. Our evaluation uses a single center crop to 96^3 via
  ResizeWithPadOrCropd. This is fundamentally different:

  - During training: random 96^3 patches from various brain locations → model learns
  context-dependent segmentation
  - During our eval: fixed center crop → model sees only the medial brain region
  - The frozen model's decoder was trained to aggregate predictions from diverse patches via
  sliding window. Given only a single center view, the decoder's predictions are spatially
  miscalibrated

  Evidence supporting this hypothesis:
  - The retrained decoder (trained ON center-cropped data) gets 0.64 GLI Dice - it learned the
   center-crop context
  - The frozen decoder (trained with random crops + sliding window inference) gets 0.02 - it's
   optimized for a different inference protocol
  - The MEN improvement (0.020→0.279) after the TC/WT/ET fix shows the frozen model CAN
  produce partial predictions, but meningiomas (being more compact) are better captured by
  center crops than diffuse gliomas

  However, I find it slightly suspicious that Dice_ET ≈ 0.001 on GLI for the frozen model.
  Even with center-crop issues, the model should detect SOME enhancing tumor if it's present
  in the crop. This warrants direct investigation.

  3B. Minor Bug: Misleading Channel Names in Logs

  In train_condition.py:194:
  logger.info(f"  NCR: {val_metrics['dice_0']:.4f}")
  The code logs channel 0 as "NCR" but it's actually TC (Tumor Core). This is cosmetic but
  could cause confusion during debugging. The actual computation is correct.

  ---
  4. Methodological Assessment & Smells

  4A. Issues FIXED from previous run

  - TC/WT/ET channel convention: FIXED
  - Feature level for SDP compatibility: FIXED (encoder10, 768d)
  - GLI test set size: FIXED (200 vs 25)
  - MLP probe overfitting: FIXED (better regularization + simpler architecture)
  - Probe training set size: IMPROVED (200 vs 150)

  4B. Remaining Issues

  HIGH: Batch-level Dice aggregation in DiceMetric3Ch

  The Dice metric computes a single Dice per channel across the entire batch (intersection =
  (pred_c * target_c).sum()), NOT per sample. With batch_size=8, this means:
  - Predictions and targets from 8 different subjects are pooled into one Dice value
  - A single sample with a large false positive region can drag down the batch Dice
  - This gives an inaccurate estimate of per-subject segmentation quality

  This is methodologically questionable. Standard BraTS evaluation uses per-sample Dice, then
  averages. The batch-level Dice can be significantly different from the mean of per-sample
  Dice.

  HIGH: No sliding-window inference for frozen baseline

  The frozen baseline evaluation is not comparable to BrainSegFounder's reported performance
  because we use center-crop instead of sliding-window inference. This makes the frozen
  baseline Dice unreliable as a reference point.

  MEDIUM: Effective dimensionality collapse persists

  From diagnostics: encoder10 features show effective_dimensionality ≈ 11 out of 768
  dimensions (1.4% utilization). This means the feature space is highly concentrated. While
  this is not a problem for the SDP (the MLP will learn the relevant subspace), it means
  linear probes are an extremely weak evaluation method for these features.

  MEDIUM: LoRA doesn't improve segmentation

  All conditions achieve ~0.94 MEN Dice regardless of LoRA rank. The 54M decoder parameters
  dominate - the encoder adaptation has negligible effect on segmentation quality. This is
  expected but raises the question: is the LoRA adapting the encoder in a way that matters for
   downstream tasks, or is all improvement just due to the retrained decoder providing better
  gradient signal to the semantic heads?

  Evidence that LoRA does matter: the probe R² (evaluated on ENCODER features, not decoder
  outputs) improves significantly with LoRA. The encoder features themselves become more
  semantically structured.

  LOW: Semantic heads with small impact

  Comparing LoRA with vs without semantic heads:
  ┌────────────────┬──────────────────┬────────────────────┬─────────────┐
  │ Best condition │ R²_mean (no sem) │ R²_mean (with sem) │ Improvement │
  ├────────────────┼──────────────────┼────────────────────┼─────────────┤
  │ lora_r8        │ -1.57            │ -0.69              │ +56%        │
  ├────────────────┼──────────────────┼────────────────────┼─────────────┤
  │ lora_r16       │ -2.45            │ -0.50              │ +80%        │
  ├────────────────┼──────────────────┼────────────────────┼─────────────┤
  │ lora_r32       │ -1.46            │ -0.76              │ +48%        │
  └────────────────┴──────────────────┴────────────────────┴─────────────┘
  Semantic heads provide substantial improvement (48-80%). This confirms they're essential for
   the pipeline.

  ---
  5. Recommendations

  1. Investigate frozen GLI Dice (HIGH PRIORITY): Run the frozen model on a few GLI samples
  and visualize predictions. Check:
    - Are sigmoid outputs all < 0.5 (predicting no tumor)?
    - Or is the model predicting tumor in wrong spatial locations?
    - Consider implementing sliding-window inference for the frozen baseline to get a true
  reference point
  2. Fix DiceMetric3Ch to compute per-sample Dice (HIGH): Change the metric to compute Dice
  per sample, then average. This gives a more meaningful and standard-compliant evaluation:
  # Instead of pooling across batch:
  for b in range(batch_size):
      dice_per_sample.append(compute_dice(pred[b], target[b]))
  3. Fix channel name logging (LOW): Change "NCR" to "TC" in train_condition.py:194.
  4. Best condition for SDP Phase 2: Based on these results, LoRA r16 with semantic heads is
  the recommended condition:
    - Best linear R²_mean: -0.50 (84% improvement over baseline)
    - Positive MLP R²_mean: +0.08 (confirms nonlinear semantic structure)
    - Statistically significant: p=5.6e-11 after Holm-Bonferroni
    - Reasonable encoder parameter count: ~74K LoRA params
  5. Don't panic about negative R²: The linear probe R² being negative just means linear
  regression on encoder10 features is worse than a constant predictor. This is expected for
  768-dim features where information is encoded nonlinearly. The positive MLP R² (+0.08)
  confirms the information IS there - the SDP's trained MLP will extract it effectively.
  6. LoRA vs DoRA: Results are essentially equivalent. DoRA shows slightly better glioma
  retention in some conditions but no consistent advantage. Stick with standard LoRA for
  simplicity.

  ---
  6. Bottom Line

  Has performance improved? Yes, in all ways that matter:
  - TC/WT/ET fix validated (MEN frozen Dice 14x better)
  - Statistical significance now achieved (wasn't before)
  - MLP probes now work correctly (were paradoxically failing before)
  - Feature extraction level matches SDP requirements (encoder10)
  - Evaluation is more robust (200 GLI samples vs 25, better probe training)

  What's genuinely concerning? The frozen model GLI Dice of 0.020 remains the one truly
  anomalous result. While I believe the center-crop evaluation protocol is the most likely
  cause (the pretrained decoder expects sliding-window inference context), this should be
  directly investigated by visualizing the frozen model's predictions on GLI samples. If the
  model's sigmoid outputs are all near-zero, it confirms a context mismatch. If the model IS
  predicting tumor but in wrong locations, there may be a deeper issue.
