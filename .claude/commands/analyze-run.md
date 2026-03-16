---
allowed-tools:
  - Read
  - Glob
  - Grep
  - Bash
  - Write
  - Edit
  - Agent
description: "Analyze completed experiment results against quality targets"
argument-hint: "<results_dir>"
---

# Analyze Experiment Results

Launch a results-analyst agent to analyze completed experiment results.

## Steps

1. The `$ARGUMENTS` should be the path to the results directory.
2. Launch the `results-analyst` agent with the results directory.
3. The agent will:
   - Load metrics from JSON files
   - Compare against quality targets from DECISIONS.md and module specs
   - Identify failure modes with literature citations
   - Propose prioritized improvements
   - Write analysis to the results directory
