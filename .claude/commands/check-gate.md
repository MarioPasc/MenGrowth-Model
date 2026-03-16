---
description: "Check if a phase gate is open (all critical tests pass)"
---

Check the gate status for phase $ARGUMENTS.

Read `experiments/phase_$ARGUMENTS/verification_report.md`.

If the file exists:
- Parse the test results table
- Count CRITICAL tests that PASS vs FAIL
- Report: "Phase $ARGUMENTS gate is OPEN" or "Phase $ARGUMENTS gate is BLOCKED" with failing tests

If the file does not exist:
- Report: "Phase $ARGUMENTS gate is UNKNOWN -- no verification report found. Run `/run-tests $ARGUMENTS` first."
