---
description: "Implement a specific stage of the 3-stage complexity ladder"
---

Implement stage $ARGUMENTS of the MenGrowth framework.

**Format:** `/implement-stage <stage_number>` (1=Volumetric Baseline, 2=Severity Model, 3=Representation Learning, 4=Variance Decomposition)

Before invoking the stage-implementer:
1. If stage > 1, verify that the previous stage has LOPO-CV results (read `experiments/stage_{N-1}/verification_report.md`).
2. Stage K+1 is only justified if Stage K shows meaningful results.

If prerequisites are met (or this is Stage 1), launch the `stage-implementer` agent with the stage number: $ARGUMENTS
