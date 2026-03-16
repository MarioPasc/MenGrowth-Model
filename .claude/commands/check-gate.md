---
description: "Check if a stage gate is open (LOPO-CV results demonstrate improvement)"
---

Check the gate status for stage $ARGUMENTS.

Read `experiments/stage_$ARGUMENTS/verification_report.md`.

If the file exists:
- Parse the test results table
- Count BLOCKING tests that PASS vs FAIL
- For stages 2+: check if R² improves over previous stage
- Report: "Stage $ARGUMENTS gate is OPEN" or "Stage $ARGUMENTS gate is BLOCKED" with details

If the file does not exist:
- Report: "Stage $ARGUMENTS gate is UNKNOWN — no verification report found. Run `/run-tests` first."
