---
description: "Run verification tests for a phase and report results"
---

Run verification tests for phase $ARGUMENTS of the MenGrowth project.

**Format:** `/run-tests <phase_number>`

Use the test-runner agent with the phase number: $ARGUMENTS

Phase-to-marker mapping:
- 0 -> phase0 (Data)
- 1 -> phase1 (LoRA)
- 2 -> phase2 (SDP)
- 3 -> evaluation (Encoding metrics)
- 4 -> (Growth prediction tests)
- 5 -> experiment (Experiment scripts)
