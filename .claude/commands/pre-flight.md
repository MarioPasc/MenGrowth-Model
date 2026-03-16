---
allowed-tools:
  - Read
  - Glob
  - Grep
  - Bash
  - Agent
description: "Validate code and config before submitting a SLURM training job to Picasso"
argument-hint: "<config_path, e.g. experiments/lora/config/server/lora_semantic.yaml>"
---

# Pre-Flight Validation

Launch a pre-flight-validator agent to check that code and config are ready for a training run on Picasso.

## Steps

1. If `$ARGUMENTS` is provided, use it as the config path. Otherwise, use the default.
2. Launch the `pre-flight-validator` agent with the config path.
3. The agent will run all validation checks and report READY or BLOCKED.

Training runs on Picasso cost A100 GPU hours. This catches config errors, path issues, shape mismatches, and test regressions BEFORE submitting the SLURM job.
