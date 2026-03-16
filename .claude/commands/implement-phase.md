---
description: "Implement a specific pipeline phase end-to-end"
---

Implement phase $ARGUMENTS of the MenGrowth pipeline.

**Format:** `/implement-phase <phase_number>` (0=Data, 1=DomainGap, 2=LoRA, 3=SDP, 4=Encoding, 5=Growth, 6=Evaluation)

Before invoking the phase-implementer:
1. Check that all previous phases have passing verification reports by reading `experiments/phase_{N-1}/verification_report.md`.
2. If any prior phase gate is BLOCKED, stop and report which phase is blocking.

If all gates are OPEN (or this is Phase 0), launch the `phase-implementer` agent with the phase number: $ARGUMENTS
